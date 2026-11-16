# 从前十二课读到 Course13

第十三课把已经验证的 EP 数据流接到真实模型。先沿 Router → Dispatch → A2A → Expert → Return A2A → Combine 阅读，再看 vLLM 如何提供输入、权重和进程组。核心算法仍能通过本课的 `reference/`、`checks/` 和 `benchmarks/` 独立运行。

## 13.1 先找到熟悉的步骤

| 前面课程中的概念 | Course13 的入口 | 阅读时关注什么 |
|---|---|---|
| 第 2 课：Top-K 与记录展开 | `reference/router.py`、`distributed/dispatch.py::pack_by_rank` | `[N,K]` 展开成 N×K 条记录，权重仍与原 Token/slot 配对 |
| 第 5～7 课：Rank packing、变长 A2A | `distributed/topology.py`、`distributed/all_to_all.py` | 按目的 Rank 发送，按来源 Rank 接收，split 单位是行 |
| 第 9 课：整数 metadata 与向量搬运分开 | `kernels/grouped_swiglu.py::_build_device_plan` | 先确定每段边界和 tile 身份，GEMM 再读取浮点向量 |
| 第 10 课：逆排列、权重与 Top-K 合并 | `kernels/combine_triton.py`、`reference/dispatch.py` | 回到原 Token/slot 顺序，每个分支应用一次权重 |
| 第 11 课：Expert Loop 与 Grouped GEMM | `model/expert.py`、`kernels/grouped_swiglu.py` | 稳定打包 → offsets/tile 表 → 两次投影 → 恢复接收顺序 |
| 第 8、12 课：先测量，再讨论流水线 | `profiler/stages.py`、`benchmarks/` | 先区分 Host 等待、通信、计算；当前路径没有 Chunk pipeline |

表中核心路径省略了 `mini_ep/` 前缀。独立 PyTorch Reference 和验证工具放在外围目录，这是本课的组织约定；不要因此把它们导入核心包。前课源码只用于对照阅读，本课不依赖其他课程的 Python 包。

建议阅读顺序：

1. `mini_ep/model/moe.py::ep_forward`：完整 A2A 数据流和固定的 collective 顺序。
2. `reference/expert.py::expert_loop` 与 `mini_ep/model/expert.py::expert_loop`：先理解逐 Expert 计算，再看同接口的 Grouped 路径。
3. `reference/grouped_plan.py`：用 Python 构造 offsets、tile_experts、tile_rows。
4. `mini_ep/kernels/grouped_swiglu.py`：GPU 生成同语义的表，两次投影复用它们。
5. `mini_ep/integration/runner.py` 与 `mini_ep/integration/reduction.py`：最后读 TP/DP 输入布局、权重与原生归约边界。

## 13.2 两次打包，两个恢复关系

第六课的 Rank packing 解决“谁发给谁”，第十一课的 Expert packing 解决“收到的行怎样交给 GEMM”。这两个排列有不同的索引空间。

| 名称 | 索引的是什么 | 在哪里使用 |
|---|---|---|
| `pack_index[p]` | Rank-packed 行 p 对应源端哪个 Token/slot 记录 | 正向 A2A 前打包 |
| `restore_index[b]` | 原始 Token/slot 记录 b 在 Rank-packed 缓冲的哪一行 | 返回 A2A 后 Combine |
| `origin[p]` | Expert-packed 行 p 来自接收缓冲的哪一行 | 本地 Expert 计算前后 |
| `expert_offsets[e:e+2]` | Expert e 在 Expert-packed 缓冲中的连续区间 | 两次 Grouped GEMM |

`Packed.tokens / expert_ids` 对应前课 `DispatchOutput.packed_tokens / packed_expert_ids` 的身份信息。本课按 Rank 形成发送段，接收端再按本地 Expert 排序；返回结构的简短字段名沿用现有接口。

第十课已经说明，两级索引可以组合成一次返回缓冲地址查询。本课的 `restore_index[q]` 直接指向 Rank-packed 返回行，`routing_weights[t,k]` 则保持原始 Token/slot 顺序。因此读取 `returned[restore_index[t*K+k]]` 时，配对的是 `routing_weights[t,k]`；不能把返回行号拿去索引原序权重。分组键和目标缓冲区不同，记录身份、恢复方向以及每个分支只乘一次权重的规则没有改变。

本地计算仍是第十一课的三行关系：

```python
packed_tokens = recv_tokens.index_select(0, origin)
# 两次投影都在同一份 Expert-major 行布局上计算 packed_output。
result.index_copy_(0, origin, packed_output)
```

`origin` 是稳定排列，所以同一个 Expert 内仍按接收先后处理。`index_copy_` 把结果写回 source-major 行，返回 A2A 才能继续使用反向 splits。源端随后使用自己的 `restore_index`，恢复 Token/slot 顺序并 Combine。Expert 阶段不应用 routing weights。

## 13.3 第十一课的 tile 表怎样留在 GPU 上

第十一课给出了 CPU counts → offsets/tile 表的可读构造；这里仍先读 `reference/grouped_plan.py`。例如四个本地 Expert 的 counts 为 `[2,0,1,3]`，BM=16：

```text
expert_offsets = [0,2,2,3,6]
tile_experts   = [0,2,3]
tile_rows      = [0,0,0]
```

空 Expert 的两个 offsets 相等，不生成有效 tile。每个 program 仍读取 `TileExperts`、`TileRows` 和 `Offsets`，计算自己的 BM×BN 输出块；写入者和边界 mask 与第十一课相同。

真实模型的 counts 每层、每步都会变化。把 GPU counts 读回 CPU，再构造三张表，会等待此前的 GPU 工作。因此当前 Grouped 路径保留稳定排序，将 metadata 的生成放在一个 Triton kernel 中：

1. 稳定排序得到 `origin`。逻辑上 `local_ids[origin]` 已递增，无需额外保存一份排序后的 IDs。
2. 对每个 Expert e，查找第一个 ID≥e 的位置，得到 `expert_offsets[e]`。相邻边界之差就是该 Expert 的 counts。
3. 计算 `ceil(counts[e]/BM)`，对它做前缀和，得到各 Expert 的 tile 区间。
4. 为每个 tile 查找所属 Expert，写出 `tile_experts` 和 Expert 段内的 `tile_rows`。

仍用六条接收记录 `local_ids=[3,0,3,2,0,3]`：

```text
origin                      = [1,4,3,0,2,5]
逻辑上的 local_ids[origin]   = [0,0,2,3,3,3]
expert_offsets              = [0,2,2,3,6]
counts                      = [2,0,1,3]
每个 Expert 的 tile 结束位置 = [1,1,2,3]
```

这继承了第九课“整数 metadata 与向量搬运分开”的思路。第九课从未排序的路由记录构造稳定位置；这里利用已经排序的 IDs 查段边界，输入条件不同，具体 kernel 也不同。无需为了外观一致，把第九课的按 Expert 扫描再复制一遍。

Host 不读取有效 tile 总数 Q，而是根据已知形状分配 `ceil(R/BM)+L-1` 个表项。有效表项保持前面的 Expert/tile 顺序，多余项标为 `expert=-1`，GEMM 在读权重之前跳过。上述例子分配 4 项，其中 3 项有效：`tile_experts=[0,2,3,-1]`。这是调度表容量，不是给 Token 增加 padding，也不会丢弃记录。R=0 时包装函数直接返回空输出。

`_plan_kernel` 的三段注释与这四步对应；`_grouped_linear_kernel` 延续第十一课的寻址方式。运行时行数和容量只决定输入范围；编译时保留搜索深度、向量大小和固定 tile 配置。两次 GEMM 在调用者的同一 CUDA stream 上消费这些 metadata。

上面的例子沿用第十一课的 BM=16，便于手算。真实 BF16 路径固定 BM/BN/BK=128/128/64、8 warps；FP32 测试路径仍为 16/32/32、4 warps。BM/BN 只改变输出块的覆盖方式，BK 保持 64，因此 K 维累加顺序、SwiGLU 舍入位置和返回 FP32 路径不变。改变的是已测量过的分块参数，offsets/tile 表的含义不变。本次阅读整理不引入自动调参；Kernel 签名中的 K 是 GEMM 归约维，与路由 Top-K 分开理解。

## 13.4 Qwen 带来的三个必要变化

**激活函数。** 第十一课的教学 MLP 是 `Down(GELU(Up(x)))`。Qwen 使用 `Down(SiLU(Gate(x))*Up(x))`，所以第一份权重是 `[L,2F,H]`，一次 Grouped GEMM 输出 `[R,2F]`；激活后回到 `[R,F]`，再执行 `[L,H,F]` 的 Down。两次投影仍共享同一组 offsets/tile 表。

**舍入位置。** BF16 的 Gate/Up 写回、SiLU、乘法保持各自的舍入边界；Down 保留 FP32 结果，经返回 A2A 后再加权、舍入和合并。这些位置由独立 Reference 和真实生成验收决定。仅把算式写得更短或改换累加写法，可能改变浮点结果，因此这次阅读整理保留了原 GEMM/激活表达式与 tile 配置。

**框架边界。** DP=1 保持本地 FP32 Combine，再 TP_REASSEMBLE。DP=2 的输入行已由 vLLM 切分，Runner 不重复切；返回 A2A 后保留 owner 部分和，再使用原生 reduce_scatter 的 BF16 归约顺序。这里额外 gather 的对象是已计算的部分和；Token 到 Expert 的分发仍由 Mini-EP 的 A2A 完成。完整层数、长提示和独立生成的验收均保留。

## 13.5 按前课的方法评价优化

先比较 `expert_loop` 与 Grouped 的值、shape、dtype、device，再看基准：metadata、准备好计划的 GEMM、完整 EP、整模型是不同的测量范围。元数据单独变快，不等于整模型按同比例变快；`--expert-backend loop` 和独立 Reference 都继续保留。

这次优化的主线仍是“第十一课的显式 tile 调度表 + 第九课的 GPU metadata”，tile 形状调整只是同一调度表上的参数选择，不是新增一种 Expert 计算公式。Profiling 显示稳定排序与 Grouped GEMM 是 GPU 侧主要 op，因此保留了 BM=128 的固定 tile；完整模型的收益小于核心微基准，必须按 [PERFORMANCE.md](PERFORMANCE.md) 中标注的口径阅读。四卡数据不能按两卡收益线性外推。

第十二课的通信/计算重叠仍按原课程的方法理解：有异步发起、有多个 stream，并不自动说明已经重叠。当前 Course13 保持同步 A2A，没有加入 Chunk 或 Double Buffer；只有新的 Benchmark 与 Timeline 指出收益时，才值得改变这部分编排。

## 13.6 动态 splits 与固定容量仍是同一条往返路径

第十二课已经介绍容量、有效行和缓冲区边界。本课的 `MINIEP_FIXED_A2A` 实验沿用这些概念，但让 NCCL 的等长 `all_to_all_single` 负责通信。所有 EP Rank 根据同一份 Host 行数选择 C；每个目的 Rank 占 C 行，padding 用全局 Expert ID=-1 表示。两次通信都使用同样的等长分段，Expert 只消费有效行。

主流程统一把恢复行号称为 `restore_index`，查询方向与第九课相同。已验证的返回结构保留原接口：变长布局为 `packed.restore_index`，固定布局为 `fixed_packed.combine_index`；后者指向含 padding 的返回缓冲。它们都查询原始 Token/slot 分支，但两个布局的具体地址不能混用。`pack_by_rank_fixed` 是可在 CPU 上运行的 PyTorch 布局实现，先用它验证索引，再讨论 GPU 搬运优化。

取 P=2、N=2、K=2、C=4：每来源只产生四条有效分支，却发送八行容量。固定容量消除了 splits 回读，但增加通信和临时缓冲；它是否适合 decode，要用相同路由的测量决定。全组有效分支数 T*K 也不等于本 Rank 接收的有效行数，后者由路由决定。

`MINIEP_OWNER_SIZES=host` 则复用 vLLM 已在 CPU 保存的 DP 行数，省去 owner 归约前的 GPU count 收集与回读。这里复用的是本轮输入行数，不缓存随路由变化的 Expert counts。数据 collective 与原生 BF16 归约顺序仍然保持。

完整补课建议见仓库 `docs/Course13知识衔接与后续补充清单.md`，飞书课程首页下有同名文档。两种通信实验都保持同步调用，不据此宣称通信/计算重叠。

## 13.7 从第九课的稳定行号到小批 decode 打包

固定容量使第九课的位置公式变成一个容易手算的特例。记原始分支 b 的目的 Rank 为 d(b)，每目标容量为 C，则：

```text
packed_row(b) = d(b) * C + 前面目的 Rank 同为 d(b) 的分支数
```

第九课的 `offset[d] + prefix[block,d] + within` 在这里仍然成立：`offset[d]=d*C`；最多 64 条分支放进一个块，所以块间前缀为零。每个 Rank program 对同一份小 ID 向量做比较和 `cumsum`，同时写出 `Pack[packed_row]=b`、恢复索引 `combine_index[b]=packed_row` 与有效行数。随后独立 gather kernel 读取 Token 向量；未分配的 Pack 项为 -1，写出零向量和 -1 Expert ID。

这延续第九课“整数位置与向量搬运分开”的方法。范围明确限制为 N*K≤64，较大固定布局保留 PyTorch 路径；变长 A2A 继续使用原布局。独立的逐 Token/slot Reference 位于 `reference/fixed_dispatch.py`，核心 PyTorch fallback 与 Triton 都与它逐项比较。

## 13.8 让 padding 的边界随读取和回写一起执行

沿用第十一课 counts=`[2,0,1,3]` 的例子，在接收缓冲中加入三条传输 padding：

```text
local_ids = [3,-1,0,3,-1,2,0,-1,3]
origin    = [1,4,7,2,6,5,0,3,8]
sorted IDs = [-1,-1,-1,0,0,2,3,3,3]
expert_offsets = [3,5,5,6,9]
```

有效 Expert 的 counts 没有变，只是前面多了三行。`expert_offsets[0]=3` 已经告诉消费者有效行从哪里开始，无需再构造 `packed_ids < 0` 的全尺寸 mask。

Grouped GEMM 只写有效行。SwiGLU 在读取 Gate/Up 时使用这个边界，跳过未写入的 padding，并输出零；SiLU 后的 BF16 舍入和乘法保持原样。Down 之后，回写 kernel 按 `origin[p]` 恢复接收顺序，在同一次 masked load 中让 padding 变为零。这与第十课通过地址映射和局部 mask 省去中间张量的思路一致。

三个概念要分开：框架 TP/SP 补齐的输入行由外层恢复和裁剪；固定 A2A 的无效向量以 Expert ID=-1 标记；tile 表的 -1 表项只表示没有计算任务。向量为零本身不表示它无效，也不能据此丢弃真实 Token。

这些操作都在调用者的同一 CUDA stream 上执行：plan 先生产 offsets，激活和回写随后消费，返回 A2A 再消费恢复后的缓冲。没有新增异步 stream 或缓冲复用协议。性能保留与默认配置以本轮 PERFORMANCE 的实测结论为准。


## Qwen3.5 扩展：同一条路由主线，增加共享分支

Qwen3.5 的 Routed Experts 仍使用前面的 Rank packing → 两次 A2A → offsets/tile → Combine。Shared Expert 是所有 Token 都会经过的另一条分支，不需要按 Top-K 发送；线性 Attention 的循环状态属于 vLLM 的 Attention 路径，不进入 Mini-EP 的 dispatch metadata。

先区分两类并行：Routed Expert 的完整参数由 EP owner 持有；Shared Expert 在双卡无 SP 时按 TP 分片，在四卡 SP 时为本 Rank 的 Token 行保存完整权重。共享分支和路由分支的加法位置因此不同。`runner.forward_ep` 中新增的 shared/TP 处理只解释框架边界，核心 `model/moe.py` 不依赖 Shared Expert 或线性 Attention。

一个 BF16 算例：两 Rank 的 routed partial 分别为 256、-256，shared partial 为 1、0。先各 Rank 相加并舍入，再归约得到 0；先分别归约两个分支，再相加得到 1。改变加法顺序会改变后续层输入，因此接入新模型仍要依次做原生 smoke、逐层对照、同前缀 logits、自由生成和跨请求状态复用检查。
