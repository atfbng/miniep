# 前十二课的实现如何延伸到 Course13

整理先核对了前课实际源码/README/本地课程正文，再形成函数映射并据此修改代码。原始逐文件 SHA/函数映射保存在实验 artifacts 的 `course-materials/lineage-before.json`，最终映射随报告记录。

| 前课实际入口 | Course13 对应函数 | 保留的算法及必要差异 |
|---|---|---|
| course2 `TopKRouter.forward` / `RouterOutput` | `model/router.py::RouterOutput`、`integration/runner.py::route_tokens` | 同样输出 `[N,K]` IDs 与 FP32 weights；Runner 持有原生 Gate/Top-K，保持 tie/slot 顺序；0.29 的 router_logits 参数可能只是 hidden_states 占位，不能当作已计算 logits |
| course1 `ExpertParallelTopology` | `distributed/topology.py::ExpertParallelTopology` | 同样 E%P 校验、e//L 与 e%L；仅扩展固定容量的 -1 padding 哨兵 |
| course1 `RankPackedTensors` / `pack_by_destination_rank`，course6 Dispatch | `distributed/dispatch.py::pack_by_destination_rank` | 沿用字段/记录身份；直接稳定按 Rank 打包组合源端排列，避免额外 Expert-major 拷贝 |
| course5 `variable_split_all_to_all` / `equal_split_all_to_all` | 同名通信函数、`exchange_splits` | 同步 NCCL，split 单位仍是行；反向交换 roles；计数双向交换后合成一次 Host 读取 |
| course6 `ExpertParallelMoE.forward` | `model/moe.py::ep_forward` | 保留明确六阶段；vLLM 持有参数与 Router，因此使用无状态 tensor 接口而非再建模型 |
| course9 stable prefix / inverse metadata | `kernels/dispatch_triton.py::pack_by_destination_rank_fixed_triton` | 稳定 slot 记录与逆索引；短 decode 用固定容量免 GPU count 回读，权重随同一行搬运 |
| course11 `GroupedGemmPlan` / `compute_grouped_local_experts` / `grouped_expert_mlp` | `kernels/grouped_gemm/plan.py`、`triton_impl.py` 同名函数 | 接收行稳定 Expert packing、共享 plan 的两次投影、恢复接收顺序；plan 用 GPU metadata 构建避免逐层 Host counts |
| course11 `compute_local_experts_loop` | `model/expert.py::compute_local_experts_loop` | 同样逐专家跳过空段；独立 PyTorch 专家循环在外围 reference |
| course10 inverse lookup / Top-K combine | `kernels/combine_triton.py::combine_kernel`、`reference/dispatch.py::combine_reference` | `restore_index` 查回 Token/slot；DeepEP 要求专家 FP32 down 先乘权重后低精度回传，故 Combine 无第二次加权 |
| course8/12 Stages、stream/Chunk 生命周期 | `profiler/stages.py`、独立 benchmarks | 先正确计时和明确 buffer 归属；当前使用同步 A2A，没有移植未经验证的 Chunk/TMA pipeline |

## 两次排列，两个恢复空间

`pack_index[p]`：rank-packed 行 p 对应原始 Token-major/slot-minor 记录。`restore_index[b]`：记录 b 在返回 rank-packed buffer 中的地址。固定容量同样使用这个字段，但地址指向含 padding 的 `[P*C,H]`。权重在 Dispatch 即按同一索引搬运。

接收端 `origin[p]` 表示 Expert-major 行 p 来自 source-major 接收 buffer 的哪一行。`packed_tokens=recv_tokens.index_select(0,origin)`，投影后再 `result.index_copy_(0,origin,packed_output)`；返回 A2A 才能反转原来的 splits。

固定 Triton packer 在搬运 Token/ID 时同时写 FP32 weight，包括 padding 的零权重；它没有改变记录身份或浮点加权，只减少额外的权重清零/散射操作。`checks/test_kernels.py` 以独立逐记录 Reference 校验非连续输入、int32/int64 ID、空行、倾斜及 padding。

## Grouped GEMM 的必要差异

Qwen 用 packed gate/up `[L,2F,H]` 与 SwiGLU，前课示例的单激活两层 MLP 不能直接替代。SiLU 单独舍入到模型 dtype 后再乘 up，down 保持 FP32 accumulator。FP32 routing weight 在 down 写出阶段乘一次，再转传输 dtype。

保留 `acc=tl.dot(a,b,acc)`、BF16 tile 128/128/64（8 warps）、FP32 tile 16/32/32（4 warps）、3 stages 及加权 epilogue 的 fusion 设置。course11 示例的 `acc += tl.dot(a,b)` 会引入不同的 FP32 分块加法顺序；course12 已展示带 accumulator 的投影接口。不能为形式一致交换这两种表达式。适配后的 plan 仍由两次投影共享，没有加入自动调参框架。

## 框架边界集中在 integration

`route_tokens` 与原生 `_forward_impl` 一样，在 Runner 持有 Gate 时先执行 Gate，再调用原生选择器，产生前课形式的 RouterOutput。Qwen3 的 `router_logits` 参数实际传入 hidden_states 占位，不能仅凭参数名省略 Gate。DP=2/SP 已经持有本地输入，不重复切分；DP=1 的复制输入按 `token_range` 唯一分配，必要 TP 输出重组在 `integration/input_layout.py`。

真实 DeepEP 对照固定四 Rank。原生 AGRS 专用 owner gather/reduce-scatter 已移除；Attention、SP 和必要 Shared Expert TP 通信不能因此删除。当前 Qwen3.5 DP=1 的 shared/routed 联合舍入不在唯一 DeepEP 路径范围内，显式拒绝，历史通过结果不迁移到新版本。

## 课件参考范围

用户提供的飞书课件目录只读访问，已读取文档文件夹内正式第 1～10 课，以及题为“草稿”的旧 Course13 全文；未修改远端内容。已核对文档/视频两份目录，均未列出第 11/12 课材料；视频未观看，因此第 11/12 课依据本地源码、README 和正文推进。旧 Course13 草稿描述 FP32/source 返回及原生兼容 owner 归约，已被本次明确方向取代，不能沿用其结果或旧命令。课件抓取清单及原文保存在 artifacts 的 `course-materials/`，没有声称阅读缺失材料。

## 同一条 T0 分支仍可沿代码追踪

沿课件的 E=8/P=4/K=2 小例子，源 Rank0 的 IDs 为 `[[6,1],[2,7]]`，权重为 `[[.75,.25],[.75,.25]]`。`pack_index=[1,2,0,3]`、`restore_index=[2,0,1,3]`，打包权重为 `[.25,.75,.75,.25]`。

T0→E6 的原始分支 b=0，经 restore_index 对应发送行2；它在 Rank3 的接收行0计算，Expert packing 与原位恢复抵消，再返回 Rank0 的行2。不同之处只在于权重已随分支传给 Expert：FP32 down 若为2，先乘.75，再存BF16的1.5。Combine 读回1.5，按owner边界归并，不再乘.75。相同的记录身份与两份逆索引贯穿前课和本课。

变长路径的 collective 顺序因此是 `counts → Tokens → IDs → FP32 weights → returned outputs`，共五次 `all_to_all_single`；固定容量不交换 counts，为四次。逻辑仍为两次 Token/结果 A2A。旧草稿的四次变长调用没有 Expert 权重消息，不能继续用来描述新数值路径。

前课 `RankPackedTensors` 接收已经 Expert-major 的 DispatchOutput；本课的同名打包函数直接接收原始 Token/IDs/weights，省去源端 Expert-major 排序与拷贝。接收端仍按 local expert 稳定排序，所以每个 Expert 的原始记录顺序与两级逆排列语义保持清楚。前课 `GroupedGemmPlan` 在 Reference 模块里保存 Host counts；本课把元数据类型单独放入 `plan.py`，保留 offsets/tile maps/block_m，用已知 num_rows 替代 Host counts，Reference 仍在外围独立构造计划。vLLM 已加载连续权重栈，因此 `compute_grouped_local_experts` 直接接收两个权重 Tensor，不在热路径重复 stack 模型参数。
