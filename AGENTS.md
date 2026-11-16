# Mini-EP 项目协作指南

本文件是仓库级 AI/开发协作约定，适用于本目录及所有子目录。项目的完整目标、课程设计和技术路线以 [`docs/计划书.md`](docs/计划书.md) 为准；本文件将其中与日常开发直接相关的要求整理为可执行规则。

## 1. 项目定位

Mini-EP 是面向大模型 MoE 推理教学的最小化 Expert Parallel 实验平台。它强调源码透明、执行链路可观察，并应能在 1～4 张 NVIDIA GPU 上完成真实 NCCL 通信、GPU Kernel 实验、性能分析和通信/计算重叠。

核心主线是：

```text
Router / Top-K
→ Expert-to-Rank Mapping
→ Token Histogram and Dispatch
→ Variable-size NCCL All-to-All
→ Local Expert MLP / Grouped GEMM
→ Return All-to-All
→ Unpermute and Weighted Combine
```

主课程实现 Pure Expert Parallel，以 All-to-All 完成 Token Redistribution。不要用 `AllGather + Tensor Parallel` 模拟 EP。`TP × EP` 仅属于可选扩展，不应混入基础实现。

## 2. 开发优先级

始终遵循以下顺序：

```text
理解数据流
→ 实现 PyTorch Reference
→ 验证正确性
→ 实现分布式版本
→ Benchmark
→ Profile
→ 定位瓶颈
→ 优化 Kernel / Compute
→ 优化系统 Pipeline
```

不得为了提前展示 Triton、CUDA 或异步流水线而跳过 Reference、正确性测试或测量。

计划书定义的里程碑顺序为：

1. M0：PyTorch/CUDA/NCCL/torchrun 环境验证。
2. M1：单 GPU Expert、Router、Top-K、Dispatch、Combine。
3. M2：AllReduce、All-to-All 和 variable-split 通信基准。
4. M3：2 GPU、4 Experts 的 Mini-EP V1，并对齐单 GPU Reference。
5. M4：扩展到 EP=1/2/4 和 4 GPU、8 Experts。
6. M5：完整计时分解、NVTX 和 Nsight Systems Timeline。
7. M6：Triton Histogram / Dispatch。
8. M7：Triton Unpermute + Routing Weight + Top-K Combine 融合。
9. M8：Naive Expert Loop 与 Grouped GEMM 对比。
10. M9：Async A2A、CUDA Stream、Chunking、Pipeline，可选 Double Buffering。

如果任务未说明当前里程碑，先从仓库现状、已有测试和任务依赖判断；不要默认可以直接进入后续优化阶段。

## 3. 按课独立的仓库结构

源码按用户要求放在 `code/course0` 至 `code/course13`。`course0` 是环境检查，`course1～course12` 对应原课程编号；`course13` 是明确授权的 vLLM/Qwen MoE 接入扩展，允许 Attention TP=2、DP=1/2 与 MoE EP=2/4；本课的测试与 Reference 位于外围目录，`mini_ep` 只保留核心推理逻辑。每课实际需要的实现、基准和依赖以实体文件保存在本课内；不得导入其他 course、仓库根目录的实现或使用跨目录符号链接。

```text
code/courseN/mini_ep/      本课完整实现与本地依赖
code/courseN/test/         本课可启动示例
code/courseN/checks/       本课正确性测试
code/courseN/benchmarks/   本课需要的独立基准与分析
code/courseN/README.md     本课条件、目录和启动方式
code/courseN/requirements.txt  本课第三方 Python 依赖
code/courseN/SOURCE_MAP.json   迁移来源与文件校验信息
tools/                    仓库级静态检查工具
docs/                     课程正文和项目文档
```

以下模块路径均相对于当前课程目录。每个示例将当前课程目录加入导入路径；同名 `mini_ep` 包供本课独立进程使用。不要为了匹配规划创建无实际内容的目录。

保持模块职责清晰：

- `model/router.py` 输出 `expert_ids` 和 `routing_weights`。
- `distributed/topology.py` 管理 Expert 与 Rank 的映射。
- `distributed/dispatch.py` 提供 PyTorch Reference Dispatch。
- `distributed/all_to_all.py` 只封装同步、异步和 variable-split A2A 语义。
- `distributed/overlap.py` 负责 Chunk、Buffer、Stream、Event 和 Pipeline 编排。
- `model/moe.py` 负责编排 Router → Dispatch → A2A → Expert → A2A → Combine。
- `kernels/` 中的优化实现不得成为唯一实现，必须保留可独立验证的 Reference。

不要为了匹配规划一次性创建空目录或占位文件；只创建当前任务实际需要的内容。

## 4. 实现规则

### 4.1 Reference First

每个优化模块必须有语义清晰的 PyTorch Reference。Reference 以可读性和正确性为首要目标，不能依赖对应 Triton/CUDA 优化实现。

推荐命名应能直接区分实现，例如：

```text
dispatch_reference / dispatch_triton
combine_reference / combine_triton
expert_loop / grouped_gemm
```

### 4.2 Tensor 与 Kernel 接口

新增核心函数或 Kernel 必须在注释、docstring 或相邻文档中明确：

- 输入与输出的语义。
- Shape 及各维含义。
- dtype 和 device 要求。
- stride、contiguous 要求和 memory layout。
- Top-K 维度是否已展开。
- Token 当前处于原始顺序、Expert 顺序还是 Rank-packed 顺序。
- 返回的 metadata 如何供后续步骤使用。

不要用变量名掩盖布局变化。优先使用 `packed_tokens`、`restore_index`、`expert_offsets`、`send_splits`、`recv_splits` 等带语义的名称。

### 4.3 Expert Placement

基础实现采用均匀放置，并验证：

```text
experts_per_rank = num_experts // ep_size
destination_rank = expert_id // experts_per_rank
local_expert_id = expert_id % experts_per_rank
```

使用该公式前必须检查 `num_experts % ep_size == 0`，错误配置应快速失败并给出清楚信息。将来支持非均匀放置时，应把映射封装在 topology 模块，不能让上述公式散落在代码各处。

### 4.4 分布式语义

任何分布式改动必须明确：

- global rank、local rank、world size、EP group、EP rank 和 EP size。
- 每个 Rank 上的 Expert 集合。
- send/recv buffer 的布局、元素数和 dtype。
- `send_splits`、`recv_splits` 的单位、长度、交换方式和对应 Rank。
- 两次 All-to-All 前后的 Token 顺序与恢复关系。
- collective 的参与者和调用顺序。

基础通信统一使用 `torch.distributed`、NCCL 和 `all_to_all_single()`。不要重新实现 NCCL，也不要把仅适用于 Gloo/CPU 的行为当成 NCCL/GPU 结论。

所有 Rank 必须以一致顺序进入 collective。分布式测试失败时要保留足够的 rank-aware 日志，避免用额外 barrier 掩盖真正的顺序或依赖错误。

### 4.5 异步与 Buffer 生命周期

使用 `async_op=True`、CUDA Stream、CUDA Event、Chunking 或 Double Buffer 时，必须解释并实现：

- 谁生产数据，谁消费数据。
- CPU Work handle 与 GPU 工作完成分别由什么保证。
- 哪个 Stream 等待哪个 Event。
- 数据何时 ready。
- Buffer 何时可以读取、复用或覆盖。
- 循环末尾如何 drain pipeline。

`async_op=True` 只表示异步发起，不能据此声称已经出现通信/计算重叠。避免无意的默认流同步、过早 `wait()`、每 Chunk 全局 barrier 或热路径中的 `torch.cuda.synchronize()`。

### 4.6 代码质量

- 优先小而清晰的函数，保持教学代码可读，不复制工业框架的复杂抽象。
- 对公共接口和复杂布局变换添加类型标注与简洁 docstring。
- 所有配置和输入尽早校验；错误信息应包含实际值和期望条件。
- 可复现实验必须显式设置随机种子，并在结果中记录配置、设备和软件环境。
- 不硬编码 GPU 数、Rank、绝对路径或特定显卡型号。
- 不在库代码导入时初始化 process group、分配大块 GPU 内存或执行 Benchmark。
- 不把完整 Transformer、Serving Engine 或工业框架集成提前引入核心路径。
- 未经任务要求，不做与当前里程碑无关的大规模重构。

## 5. 正确性与测试

正确性优先于性能。至少覆盖与改动相关的以下维度：

- Top-1 与 Top-2。
- 不同 Token 数、Hidden Size 和 Expert 数。
- 空 Expert、单 Token Expert、非均匀路由和高度倾斜路由。
- FP32 Reference；FP16/BF16 使用明确、合理的容差。
- Dispatch/Unpermute 的索引可逆性。
- Routing weights 的应用与 Top-K reduction。
- EP=1 与单 GPU MoE 一致。
- EP=2 必须与单 GPU Reference 一致；条件允许时验证 EP=4。
- Triton/CUDA 与 PyTorch Reference 一致。
- Grouped GEMM 与 Naive Expert Loop 一致。

测试必须验证值、shape、dtype、device 和关键 metadata，不能只验证代码没有抛异常。分布式测试应设置合理超时，失败时让所有子进程正确退出，避免永久 hang。

没有 GPU 或 GPU 数不足时，可以明确 skip 对应测试，但不得伪造通过结果。纯 CPU 可覆盖的路由、布局和 Reference 逻辑应尽量保留 CPU 测试能力。

修复缺陷时，优先添加能复现该缺陷的最小回归测试。

## 6. Benchmark 与 Profiling

Benchmark 与单元测试分离。Benchmark 至少记录：

```text
tokens, batch_size, seq_len, hidden_size, expert_hidden_size,
num_experts, top_k, ep_size, dtype, chunk_size
```

根据实验记录：

```text
router_latency, dispatch_latency, a2a1_latency, expert_latency,
a2a2_latency, combine_latency, total_latency, tokens_per_second,
communication_bytes, effective_bandwidth,
tokens_per_expert, imbalance_ratio,
communication_time, compute_time, exposed_communication_time, overlap_ratio
```

计时规则：

- 包含 warmup 和多次重复，报告统计方法。
- GPU 计时使用 CUDA Event 或等价的正确同步方法。
- 只在测量边界同步，不把不必要同步放入被测路径。
- 比较实现时保持输入、dtype、设备、路由结果和测量区间一致。
- 说明统计的是单 Rank、最大 Rank、平均 Rank还是端到端延迟；分布式端到端性能优先关注最慢 Rank。
- 不把首次编译、JIT 或初始化开销混入 steady-state，除非实验明确测量冷启动。

代码应支持以下 NVTX 范围：

```text
ROUTER, DISPATCH, A2A_FORWARD, EXPERT, A2A_RETURN, COMBINE
```

Chunked pipeline 还应标注 `CHUNK_n_COMM` 和 `CHUNK_n_COMPUTE`。对 overlap 的最终判断必须来自 Nsight Systems Timeline，并同时报告 total latency、exposed communication time 和 overlap ratio。

任何“更快”“降低通信开销”“实现 overlap”等性能结论都必须附带 Benchmark 或 Profiler 证据。没有证据时，只能描述为假设或待验证目标。

## 7. 文档与示例

- 示例按课程顺序保持可单独运行，避免依赖未讲解的后续优化模块。
- 面向学习者解释 Token 在哪里、为什么通信、谁发给谁、布局何时变化以及何时开始计算。
- 文档中的命令应注明 GPU 数、启动方式和关键前置条件。
- 新增公共行为、配置项或性能指标时同步更新相邻文档或示例。
- 不用工业框架 API 代替核心算法解释；工业框架仅用于后期概念映射。

## 8. 完成标准

提交改动前，按任务范围确认：

1. 实现位于正确模块，职责没有越界。
2. Reference 与优化路径可以独立运行和比较。
3. Shape、layout、dtype、device 和分布式通信语义已写清楚。
4. 相关单元测试、分布式测试或最小运行验证已执行。
5. 实际执行的命令与结果被准确报告；未执行项说明原因。
6. 性能改动包含可复现 Benchmark；overlap 改动包含 Timeline 验证方案或结果。
7. 没有把“理论可行”表述成“已经验证”。

## 9. 项目边界

第一阶段不实现：完整 DeepSeek-V3、超大模型训练、大规模集群调度器、自研 NCCL、自研 Tensor Core GEMM、NCCL Transport 修改、内核级 GPUDirect RDMA、完整生产级 LLM Serving Engine，以及完整 TP × PP × DP × EP 框架。

遇到这些方向的需求时，应先说明它超出当前核心范围，并优先提供能服务于 Mini-EP 教学目标的最小化实现或接口边界。

项目的最终判断标准始终是：学习者能否通过代码、测试、Benchmark 和 Timeline，真正理解 MoE Expert Parallel 的 Token 数据流、NCCL All-to-All、Expert Compute、自定义 Kernel、Grouped GEMM，以及通信与计算重叠。
