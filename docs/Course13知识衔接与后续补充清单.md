# Course13 知识衔接与后续补充清单

本页面向已经读过前十二课的学习者与课程维护者。目标是把真实 Qwen/vLLM 推理放回熟悉的 Token 往返过程，并记录值得补进前课的知识。这里的“待补充”表示课程内容建议，不表示对应功能尚未实现。核对日期：2026-10-19。

## 先沿同一条数据流阅读

Router / Top-K → Rank packing → 正向 All-to-All → Expert packing → Grouped MLP → 恢复接收顺序 → 返回 All-to-All → Unpermute / Weighted Combine。

Course13 没有替换这条主线。vLLM 提供 Attention、KV Cache、模型权重、调度和进程组；Mini-EP 负责 MoE 的路由、Token 分发、完整 Expert 计算与返回。基础课程的 Pure EP 仍是理解本课的前提。

| 已有概念 | 前课阅读位置 | Course13 中的延伸 |
|---|---|---|
| Token/slot 记录与逆排列 | 第 2 课；第 9 课 9.1～9.3 | 分别记录 Rank packing 和 Expert packing 的索引空间 |
| 计数、稳定行号、向量搬运 | 第 9 课 9.2～9.4 | GPU metadata 服务真实模型每层、每步变化的路由 |
| 变长 A2A、按来源拼接 | 第 5～6 课 | 对比变长传输与小批 decode 的固定容量传输 |
| offsets、tile_experts、tile_rows | 第 11 课 11.3～11.6 | 在 GPU 上构造同语义 tile 表，计算 SwiGLU 两次投影 |
| 固定容量、有效行、就绪与复用 | 第 12 课 12.2～12.7 | 分清容量思想与 Peer 信号实现；NCCL 同步路径使用流顺序保证消费时机 |
| 测量边界与最慢 Rank | 第 7～8 课、第 11 课 11.7 | 增加整模型请求、原生 vLLM 对照和动态 metadata 等待 |

建议先读 `code/course13/mini_ep/model/moe.py` 的五步编排，再读独立 `reference/`，最后进入 `mini_ep/integration/`。每课保持可独立运行，不通过跨课导入来复用实现。

## 建议优先补充的知识

### 从教学 MLP 到 Qwen SwiGLU

前课已讲两次投影以及共享 offsets/tile 表。新增部分是激活和第一份权重的布局：教学 MLP 为 `Down(GELU(Up(x)))`；Qwen 为 `Down(SiLU(Gate(x)) * Up(x))`。

对本地 L 个 Expert，Gate/Up 权重合并为 `[L,2F,H]`。第一投影 `[R,H] → [R,2F]`，激活后 `[R,F]`，Down 使用 `[L,H,F]` 产生 `[R,H]`。R 是展开后接收的分支数，不是原始 Token 数。routing weights 仍在返回源 Rank 后应用一次。

**建议补入：第 11 课 11.5 之后，优先级 P0。** 用 H=4、F=3 的单 Expert 例子标出 Gate/Up 两半，再对照 Course13 的 `reference/expert.py`。练习完成标准是说清权重 shape、两次投影和权重应用位置，无需先阅读 vLLM API。

### 舍入顺序为什么会影响生成

前课已讨论 dtype 与精度边界；真实自回归模型增加了“微小误差会影响下一 Token，再改变后续输入”的约束。BF16 Gate/Up 写回、SiLU 写回、乘法和最终归约都有各自的舍入位置，代数等价不意味着浮点逐位等价。

最小例子：`BF16(SiLU(0.5)) * BF16(1.5)` 与先用 FP32 算完整表达式再转 BF16 可能不同。另一个例子是带正负抵消的 Top-K 分支，改变求和顺序可能改变恰在 BF16 舍入中点附近的结果。

Course13 因此保留 FP32 返回 A2A；DP=2 保留 owner 部分和和原生归约顺序。历史 BF16/FP16 回传实验虽然减少字节数，却未通过完整生成一致性。

**建议补入：第 10 课 Combine 和第 11 课精度说明，优先级 P0。** 配合 `checks/test_reference.py` 的舍入回归例子，分别比较层输出、logits 和独立生成，不能只放宽容差让实验通过。

### Attention TP、DP 与 MoE EP 的边界

基础课已经定义 EP rank 与完整 Expert 放置；Course13 新增的是框架中的多种分组同时存在。配置为 Attention TP=2，DP=1/2，MoE EP=2/4。EP rank 按 DP-major、TP-minor 排列，不能拿 local GPU 编号代替 group rank。

DP=1 的两张 TP 卡接收复制输入，适配层只让每个 Token 有一个 EP 来源。DP=2 的 MoE 输入已经由框架做 sequence parallel 切分，不能再按 DP=1 的规则切第二次。返回的 owner partial 是已完成 Expert 计算的部分和，其 gather 不替代 Token 到 Expert 的 A2A。

**建议补入：第 6 课后的扩展阅读，并在第 13 课入口复述，优先级 P0。** 用 TP=2/DP=2 的四行表写出 group rank、DP/TP rank、Expert 范围和输入来源。练习覆盖一个 DP 空闲、两侧长度不同、单 Token，检查无重复发送或遗漏。

### 动态 metadata 与固定容量的代价

第 12 课 12.2.4 已经讲固定容量、计数交换、全局行数和准备成本，不能再把固定容量作为 Course13 新发明。需要补充的是：真实推理每层、每步路由会变，前课可提前准备的 counts/offsets 不一定能跨步复用。

变长 NCCL A2A 要把 splits 作为 Host 整数提供；读取很小的 GPU counts 也可能等待此前的流工作。固定容量路径为每个目的 Rank 预留 C 行，用 `expert_id=-1` 标记无效行，传输 `[P*C,H]`，以额外字节换取不回读 splits。C 的安全上界由本轮所有来源的最大 N 与 Top-K 得到，不能用平均路由量假定不会溢出。

例如 P=2、每来源 N=2、K=2，取 C=4，则每来源发送 8 行，其中只有 4 行有效。收到的有效行数取决于路由，不能用全组 T*K 代替本 Rank 的接收数。源端 Combine 使用固定缓冲中的分支位置，padding 不进入 Expert tile 或加权结果。

**建议补入：第 5 课变长 splits、 第 8 课 Host 等待及第 12 课准备成本，优先级 P0。** 用同一路由分别画有效布局和容量布局；用 Timeline 区分拷贝传输时间与 API 等待时间。

### GPU tile 计划是第十一课的延伸

第 11 课 11.3 从 Host counts 构造 offsets/tile 表。Course13 利用稳定排序后的 Expert ID，通过二分下界找到 offsets，再以前缀和生成相同 tile 任务。为避免读取 GPU 的有效 tile 数，Host 按形状分配上界容量，多余 tile 用 `expert=-1` 跳过。

沿用前课 counts=`[2,0,1,3]`、BM=16：offsets=`[0,2,2,3,6]`，有效 tile_experts=`[0,2,3]`，tile_rows=`[0,0,0]`。调度表的多余项与 Token padding 是两个概念，不能混用。参考构造保留在 `reference/grouped_plan.py`。

**建议补入：第 11 课 11.3 后，优先级 P1。** 先比较表的值，再比较 metadata 成本，最后测完整 EP。保留“整数计划和向量搬运分开”的第九课解释方式。

## 验收与性能表应补充什么

| 层次 | 回答的问题 | 不能替代的验证 |
|---|---|---|
| 独立 Reference / Kernel 测试 | 同一路由下值、布局、dtype 是否一致 | 完整模型误差传播 |
| 逐层比较 | 误差从哪一层、哪个 Rank 开始 | 自由生成序列 |
| Teacher forcing + logits | 相同输入前缀下预测分布是否一致 | 模型自行选择后续输入的行为 |
| 自由生成 | 序列是否完全一致 | 对所有未测 prompt 的保证 |
| 核心 EP 基准 | 路由、传输、Expert、Combine 的成本 | Attention、KV Cache、调度及取回结果 |
| 整模型请求 | 同一批请求的端到端完成时间 | 饱和服务吞吐或逐请求延迟分位数 |

**建议补入：第 8 课测量方法及第 13 课验收章节，优先级 P0。** 保留每轮样本、最慢 Rank、预热、设备负载和配置。ABBA 用于暴露运行顺序漂移，不应删除慢轮次。没有 Timeline 证据就不宣称 overlap。

## 后续补课的完成标准

每项补充均包含前课锚点、一张布局表或最小算例、可独立运行的 Reference、对应回归测试和测量范围说明。前十二课正文的改写另行安排；本次先交付衔接说明与缺口清单，保持课程原有编号和布局语言。

来源：课程首页；云盘“文档”中的第 1～9 课；知识库第 10～12 课；Course13 的 TEACHING、PERFORMANCE、独立 Reference 和检查脚本。具体实验结论以 PERFORMANCE 的日期与配置为准。

## 阅读入口

- [课程首页与统一符号](https://tvle9mq8jh.feishu.cn/wiki/QJp1wvBlgiiYV5kEOqFcpNtNnie)
- [第 8 课：负载均衡与 Profiling](https://tvle9mq8jh.feishu.cn/docx/J8T7dczvboxbIExiVkgcOsSdnYb)
- [第 9 课：从记录编号到连续向量](https://tvle9mq8jh.feishu.cn/docx/Eb12deimeoVaKbxMU50cdMHAnTh)
- [第 10 课：Combine 与舍入语义](https://tvle9mq8jh.feishu.cn/wiki/SxHmwEB7ViSYDsk96JPcYraEnDf)
- [第 11 课：Grouped GEMM](https://tvle9mq8jh.feishu.cn/wiki/Sl4Aw148zigah9kjSTmceqNEnMe)
- [第 12 课：通信、计算与缓冲区](https://tvle9mq8jh.feishu.cn/wiki/WCOQw8Oe8iDvQUkl5gjcXS1Onke)
- [Course13 源码目录](https://gitee.com/hellofss/miniep1/tree/master/code/course13)

## 补充案例：同一方法的新边界条件

本轮两项 Kernel 改动继续使用第九课的稳定前缀、第十课的 masked load 和第十一课的 offsets。建议补充的是这些方法在真实模型中的适用边界。

**单块稳定前缀。** 每来源 N*K≤64 时，可把分支放在一个块中。第九课的位置公式 `offset[d] + prefix[block,d] + within` 变为 `d*C + within`：每目标容量 C 已知，块间前缀为零。整数位置规划和 Token 向量搬运仍然分开，较大输入保留已有路径。建议作为第九课的一个小批 decode 变体，配合独立逐 Token/slot Reference 对照。

| padding 种类 | 表示什么 | 处理边界 |
|---|---|---|
| 框架 TP/SP 补齐 | 为 TP 行切分补齐的输入行 | 外层 TP 收集后裁剪 |
| 固定 A2A 容量 | 不对应原始分支的传输行，Expert ID=-1 | Expert 计划排除；激活与回写 masked load 输出零 |
| tile 表容量 | 不对应有效计算任务的表项，expert=-1 | GEMM 读取权重前跳过 |

向量为零不表示它无效，应跟随身份 metadata 判断。接收 ID 为 `[3,-1,0,3,-1,2,0,-1,3]` 时，稳定排序后的无效行占前三行，`expert_offsets=[3,5,5,6,9]`。此时 offsets 起点不再为零，但有效 counts 仍是第十一课的 `[2,0,1,3]`。Host Reference 用 `padding_rows=3` 表达相同语义。建议补入第十一课的 offsets 说明，并与第十二课的固定容量例子对照。

激活和回写可直接消费 `expert_offsets[0]`，在读取时屏蔽未写入的 padding，省去单独生成 mask 和填零中间张量。SiLU 后的 BF16 舍入、乘法与 FP32 Down 保持原有顺序。这样解释融合时，读者能看清减少的是哪些读写与调用。

**数据依赖形状与隐式读回。** 前课已讲 Host 等待；建议增加一个真实算子的排查例子：`bincount` 的结果长度与输入最大值有关，`minlength` 是下界。使用 GPU Tensor 不等于整个规划过程都留在 GPU，应通过 Timeline 核实是否为了输出大小或 Python splits 读回了标量。固定容量、已验证的 ID 值域和 GPU metadata 必须一起考虑，不能把 padding 当作自动加速开关。

输出长度的定义见 [PyTorch torch.bincount API](https://docs.pytorch.org/docs/2.14/generated/torch.bincount.html)；本项目的拷贝与等待结论以固定实验版本的 Timeline 为依据。

## Qwen3.5 扩展新增的补课点

本节对应 Qwen3.5-35B-A3B 的文本 MoE 接入。官方配置为 40 层、256 个路由 Expert、Top-8 加 1 个 Shared Expert，采用 Gated DeltaNet / Attention 混合结构；完整支持范围与验收状态以项目 README 为准。[官方模型说明](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

| 补充主题 | 前课可复用的知识 | 本次需要新增的说明 |
|---|---|---|
| Shared Expert 与路由分支 | 第 2、10 课的加权与 Combine | Shared Expert 每个 Token 都经过，不加入 256 个路由 Expert 的编号，也不经过路由 A2A |
| Shared Expert 的 TP/SP 布局 | 第 7、13 课的 Rank、行所有权 | 双卡无 SP 时按 TP 分片；四卡 SP 时每 Rank 为自己的 Token 行保存完整共享权重 |
| 加法与归约的舍入位置 | BF16 分支与 owner partial | 双卡原生先把两个分支的 Rank 局部结果相加，再归约，不能移动成两个分支分别归约后再加 |
| 循环状态与 Token 布局 | Token 原序、Expert 序和恢复索引 | Gated DeltaNet 的状态由 Attention 路径维护，状态索引不等于 Expert ID、restore_index 或 A2A splits |
| 多模型验收覆盖 | Reference、逐层、Logits、生成一致性 | 模型层数来自 checkpoint，不能把 Qwen3 的固定 48 层套到 Qwen3.5 的 40 层；缺失元数据应暴露为错误 |

建议先增加一个手算例，再介绍真实模型：两个 Rank 的 routed partial 为 256、-256，shared partial 为 1、0。BF16 中先各 Rank 相加并舍入，再归约得到 0；先分别归约两条分支，再相加得到 1。代码与测试分别在 `reference/shared_expert.py`、`checks/test_model_config.py`。

Gated DeltaNet 的递推公式、状态槽分配与跨请求状态复用值得单独补课。当前 Mini-EP 保留原生 vLLM 的 Attention 和缓存实现，通过连续请求、不同批大小、空 DP 与逐步 Logits 检查接入边界；不把这次接入表述成自研线性 Attention 或 SSM Kernel。
