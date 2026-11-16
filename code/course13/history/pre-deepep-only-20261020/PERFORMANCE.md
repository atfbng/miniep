# vLLM 原生 EP 与 Mini-EP 性能对比

> 版本说明：本文件中的性能证据来自旧 FP32 回传版本。当前默认回传跟随输入 dtype，并在专家端先加权；新路径没有据此宣称加速，精度验收见 [EXACT_ALIGNMENT.md](EXACT_ALIGNMENT.md)。

2026-10-20 的 Qwen3.5 接入、代码整理与两次性能门槛未通过后的回退记录见 [QWEN35.md](QWEN35.md)。以下 Qwen3-30B-A3B 历史测量保持原口径，不作为新模型的性能结论。

## 2026-10-19：固定容量 decode、Host metadata 与 padding 读写

本轮保留两项后续优化：小批稳定前缀打包、padding masked load / 接收顺序回写。生产默认 `MINIEP_FIXED_A2A=auto`、`MINIEP_OWNER_SIZES=host`、`MINIEP_FIXED_PACK=triton`。auto 仅在全组最大来源行数≤8 时选择固定容量；Triton 打包要求每来源 N*K≤64，超过该界限保留 PyTorch fallback。变长路径、Expert loop 和独立 PyTorch Reference 均可对照。

### 完整请求主对照

| 配置 | 每请求输入→输出 Token | 基线→优化快照 / s | 原生 vLLM / s | 快照/原生 |
|---|---|---:|---:|---:|
| 2 GPU / EP=2 | 512→1 | 0.10505 → 0.09064 | 0.07214 | 1.26× |
| 2 GPU / EP=2 | 128→32 | 2.75831 → 2.37710 | 1.60192 | 1.48× |
| 2 GPU / EP=2 | 512→16 | 1.37968 → 1.20891 | 0.79470 | 1.52× |
| 4 GPU / EP=4 | 512→1 | 0.12600 → 0.13468 | 0.09703 | 1.39× |
| 4 GPU / EP=4 | 128→32 | 3.39125 → 2.80883 | 2.28366 | 1.23× |
| 4 GPU / EP=4 | 512→16 | 1.62728 → 1.47064 | 1.08602 | 1.35× |

主测量使用 gpu3 的物理 H100 GPU 4/5 或 4–7，CPU 亲和性为对应 NUMA1 范围。所选卡在每轮启动前空闲，但宿主机共享；其他卡上的任务会变化。模型/软件为 Qwen3-30B-A3B、vLLM 0.29.0、PyTorch 2.13.0+cu130、Triton 3.7.1；H=2048、F=768、E=128、K=8、48 层。Attention TP=2，每个 DP 两条请求，eager、同步调度、禁用前缀缓存、NCCL_NVLS_ENABLE=0。基线为提交 `3bf5df2` 的推理代码，测量驱动统一；所有路径在建模前关闭 BF16 reduced-precision reduction。

每个实现按正反顺序启动两次独立进程，每场景预热 2 次、测量 5 次，报告合并 10 个最大 Rank 墙钟样本的中位数。原生单独取 5 个样本。checkpoint prefetch（2 线程）只影响加载，加载/JIT/预热均在计时之外。所有 Rank、所有重复的输出 Token 都与原生一致。数据、范围和每轮中位数保留，不删除慢轮次。

双卡 decode / mixed 的主对照改善约 13.8% / 12.4%，四卡 decode 改善约 17.2%。prefill 的基线及四卡 mixed 明显漂移，不据此宣称稳定加速。图表使用主对照样本，追加复测单列。

### 四卡消融与反向复测

| 路径 | prefill / s | decode / s | mixed / s |
|---|---:|---:|---:|
| 上一份已验证基线 | 0.12600 | 3.39125 | 1.62728 |
| 变长 + Host sizes | 0.13480 | 3.40677 | 1.59140 |
| 固定 auto + legacy sizes | 0.14420 | 3.47200 | 1.75437 |
| 固定 auto + Host sizes（PyTorch 打包） | 0.12977 | 3.25167 | 1.60427 |
| 再加 Triton 稳定打包 | 0.13604 | 2.87310 | 1.43890 |
| 再融合 padding 读取与回写 | 0.13468 | 2.80883 | 1.47064 |
| 原生 vLLM | 0.09703 | 2.28366 | 1.08602 |

固定容量不是自动加速开关：PyTorch 打包及 padding 的额外调用可能超过省下的同步成本。Host sizes 的计数交换/读回消除由 Timeline 直接验证，单独的端到端收益仍受漂移影响。Triton 打包把第九课的稳定前缀用于已知容量，随后 padding 融合复用 offsets[0]，保持算式与舍入点不变。

第二轮在主对照的四卡 mixed 中一轮慢、一轮快，因此只补一组反向 BAAB；没有开启第三项优化，也没有删掉主对照中的慢样本：

| 场景 | Triton 打包两轮 / s | 加 padding 融合两轮 / s |
|---|---:|---:|
| 512→1 | 0.13400 / 0.10389 | 0.13438 / 0.13667 |
| 128→32 | 2.83052 / 2.91821 | 2.68290 / 2.76930 |
| 512→16 | 1.42351 / 1.46702 | 1.35709 / 1.37257 |

反向复测的 decode / mixed 两轮增益均超过 3%，prefill 没有两轮均超过 3% 的退化；结合双卡两轮稳定改善，保留 padding 融合。prefill 不作为收益依据。复测使用相同模型、输入、精度、GPU、CPU 亲和性与计时方法；但 d3 测前检测到约 16.5～16.7 GiB/卡的常驻显存占用，其余复测轮次测前为空闲。所有输出 Token 仍与原生一致，结果不外推为独占硬件保证。

### 正确性与接口修复

最终默认配置通过 26 项 H100 单元测试及 EP=1/2/4 真正 NCCL 检查；额外覆盖每 Rank 仅一个 Expert（E=P）、NaN padding、非连续输入、int32/int64 ID、FP32/BF16，打包额外覆盖 FP16。修复 L=1 时 fmod(-1,1)=0 丢失 sentinel 的边界，单 Expert 映射使用 clamp_max(0)。本模型 L=32/64 的已测算式不受此边界修复影响。

双卡 32 prompts × batch 1/2/4/8 × 64 Token：128/128 序列一致；20 prompts × batch 2/4 × 128 Token：40/40 一致。四卡 9 类工作负载共 22/22 序列、314/314 Token 一致。每 Rank 覆盖 48 层，层与 logits 最大绝对误差为 0。最终验证改用物理 GPU 0–3（双卡 0/1）并重新生成 native 基线，未复用其他卡组的基线。

新增独立 `reference/fixed_dispatch.py`，逐 Token/slot 验证固定位置；Host tile Reference 支持 leading padding。Reference 不接收优化专用参数。固定路径的 received_branches / recv_padding_rows 在热路径为 None；基准在测量结束后根据真实全局 Expert histogram 统计各 Rank 有效行、padding 与发送字节，禁止把全组分支数当作本 Rank 接收数。固定打包 metadata 区分请求的策略和实际执行的 backend。

### Timeline 与复现

| decode 阶段 | memcpy API/范围：基线→最终 | Stream sync/范围：基线→最终 | Launch API/范围：基线→最终 |
|---|---:|---:|---:|
| DISPATCH | 8.000 → 0.250 | 2.000 → 0.062 | 12.000 → 3.281 |
| A2A_FORWARD | 1.000 → 0.031 | 1.000 → 0.031 | 4.000 → 2.062 |
| EXPERT | 2.801 → 3.000 | 0.000 → 0.000 | 8.384 → 9.000 |
| EP_REDUCE | 2.000 → 0.000 | 2.000 → 0.000 | 4.031 → 3.031 |

上述为 Host API 调用统计；decode 捕获包含首次 prefill，残余计数不是纯 decode 每步计数。Launch 包含 cuda/cu API，不等于物理 GPU Kernel 数。

最终 Timeline 和阶段 CUDA Event 对照在共享服务器 GPU 0–3 上独立重跑，使用对应 NUMA0 CPU 范围。Nsight 2025.3.2 捕获 prefill 512→1 和 decode 128→32，每 DP 一条请求；完整请求表则是每 DP 两条，不能相加或直接互换。CUDA Event 结果另行保存；阶段图显示 Host NVTX 范围均值；Nsight 的 NVTX/API 时间是 Host 范围，不是 GPU kernel 总时长或 overlap 指标。

完整命令、失败记录、主对照与复测见 `artifacts/course13_transport_20261019/`。初次最终模型启动因 GPU 4–7 被其他任务占用而失败，属于资源不足，未计为准确性通过；随后换用空闲 0–3 完成复验。CPU 单元运行有 CUDA skip，GPU 套件执行全部项目。源码隔离、依赖与来源校验通过。

精简公开数据与绘图入口：`docs/benchmarks/course13-20261019.json`、`docs/assets/plot_course13.py`。旧实验保留在下面，不能把不同负载、协议或卡组的绝对延迟直接拼接成趋势。

## 2026-10-19：扩展精度验收、Profiling 与 tile 选择

本轮在 gpu3 的空闲四张 H100 80GB 上重新验证。环境为 Qwen3-30B-A3B BF16、vLLM 0.29.0、PyTorch 2.13.0+cu130、Triton 3.7.1，TP=2、DP=2、EP=4，eager、同步调度、关闭 prefix caching。先用 PyTorch Profiler 的 CPU/CUDA op 表定位，再用 Nsight Systems 2025.3.2 分别采集 prefill 和 decode Timeline；历史 Nsight 报告仍保留在 `artifacts/course13_nsys_20261018/`。

op 表中，`aten::argsort`（Rank packing 与 Expert packing）、`_grouped_linear_kernel` 和 NCCL collective 是主要设备侧项目。据此只保留一项不改变数值路径的调整：BF16 Grouped GEMM 的 tile 从 BM/BN/BK=64/64/64 改为 128/128/64，仍为 8 warps。BK 保持 64，K 维累加顺序、SwiGLU 舍入位置和 FP32 Down 输出不变。

### 完整模型阶段

每个场景 warmup 1 次、测量 3 次；每 Rank 覆盖 48 层，表中为各 Rank 阶段中位数的聚合。CUDA Event 在请求结束后统一结算。

| 阶段 | prefill 前 / ms | prefill 后 / ms | decode 前 / ms | decode 后 / ms |
|---|---:|---:|---:|---:|
| EXPERT | 0.34350 | 0.33562 | 0.29419 | 0.30014 |
| EP_REDUCE | 0.33409 | 0.34482 | 0.29318 | 0.29258 |
| A2A_FORWARD | 0.28698 | 0.27714 | 0.26542 | 0.24934 |
| DISPATCH | 0.26059 | 0.26682 | 0.25611 | 0.26126 |
| 其余阶段合计 | 0.24546 | 0.25168 | 0.23521 | 0.23592 |
| 阶段和 | 1.47062 | 1.47610 | 1.34411 | 1.33925 |

EXPERT 的 prefill 中位数下降约 2.3%，decode 上升约 2.0%；阶段和分别变化约 -0.37%/+0.36%。这说明 tile 改动只触及 Expert 的一部分成本，且会被 A2A、owner-partial 归约和层间等待淹没。不能把核心微基准的局部收益写成端到端加速。

### Nsight Timeline：通信与 Host 同步

本轮 Nsight 使用 gpu3 物理 GPU 0–3，关闭逐阶段 CUDA Event（`--stage-events off`），分别采集一次 prefill（512→1）和一次 decode（128→32）。每层的 NVTX 时长按所有 Rank、全部层和全部 decode step 汇总后取平均：

| 阶段 | decode / ms | prefill / ms |
|---|---:|---:|
| A2A_FORWARD | 0.6072 | 0.8861 |
| EP_REDUCE | 0.5915 | 0.7266 |
| DISPATCH | 0.4030 | 0.5314 |
| EXPERT | 0.3691 | 0.4167 |
| ROUTER | 0.1277 | 0.1364 |
| A2A_RETURN | 0.1016 | 0.1119 |
| COMBINE | 0.0562 | 0.0594 |

decode 的 `A2A_FORWARD + EP_REDUCE` 约占 NVTX 总时长 **53%**，prefill 约 **56%**。GPU Kernel 时间也以通信为主：decode 中 `ncclDevKernel_SendRecv` 约占 **54.0%**、AllGather **21.5%**、ReduceScatter **4.3%**，`_grouped_linear_kernel` 只有 **3.2%**；prefill 分别是 **49.9% / 21.9% / 3.0% / 5.4%**。继续只调 GEMM 不能解释主要延迟。

更重要的是 Host 同步：

| 位置 | decode 每层 | prefill 每层 |
|---|---:|---:|
| A2A count D2H `cudaMemcpyAsync` | 0.260 ms | 0.519 ms |
| EP_REDUCE `cudaStreamSynchronize` | 0.149 ms | 0.241 ms |
| EP_REDUCE D2H `cudaMemcpyAsync` | 0.136 ms | 0.135 ms |
| DISPATCH `cudaMemcpyAsync` | 0.082 ms | 0.208 ms |

这些 `cudaMemcpyAsync` 的实际 DTOH 数据量只有每个 A2A 约 77 字节、每个 owner reduction 约 42 字节，GPU 拷贝本身约 3–5 μs；绝大部分 API 时间是在等待前序流工作完成。decode 每层约有 **45 次 Kernel launch** 和 **15 次 memcpy API 调用**，但通信等待和 count/size 的 Host 同步仍是关键路径。A2A 的计数交换、owner-partial 的 sizes 读取无法继续用“更多 GEMM 调参”解决。

因此下一阶段优先验证 **decode 固定容量 A2A / 固定大小 owner-partial**：用 padding 和有效行 mask 换取移除 count/size 的 D2H 同步；预期先改善 decode，再决定是否扩展到 prefill。在该实验证明收益前，不把 local-row skip 或 TMA 作为第一优先级。

原始 `.nsys-rep` 和 CSV 保存在 gpu3 宿主机 `/SharedData/fuss/miniep-nsight-20261019/`，不纳入仓库；仓库只保留命令和结论。

### 核心 EP 与完整请求

核心 EP 使用四卡、每 Rank 4/128/512 Token 的合成输入，按 ABBA 顺序各启动两轮、每轮 60 次测量，合并每种实现 120 个最大 Rank 的 CUDA Event 样本。核心基准的轮次间波动明显，因此只作为定位信号：

| 每 Rank Token | 总延迟前 / ms | 总延迟后 / ms | EXPERT 前 / ms | EXPERT 后 / ms |
|---:|---:|---:|---:|---:|
| 4 | 1.2077 | 1.1858 | 0.2989 | 0.2997 |
| 128 | 1.2717 | 1.1884 | 0.3456 | 0.3387 |
| 512 | 1.5809 | 1.6284 | 0.5238 | 0.4789 |

核心基准中，T128 的总延迟下降约 6.5%，T512 的 EXPERT 下降约 8.6%，但 T512 总延迟因 A2A 等阶段波动上升约 3.0%。这些样本只能说明 tile 对 EXPERT 的方向性影响，不能替代完整请求。

完整请求在空闲四卡上每场景取 5 个最大 Rank 墙钟样本，包含调度、Attention、MoE 和取回结果，不含模型加载或预热：

| 每条请求输入→输出 | 前 / s | 后 / s | 原生 vLLM / s | 后 / 前 | 后 / 原生 |
|---|---:|---:|---:|---:|---:|
| 512→1 | 0.12199 | 0.11050 | 0.07037 | -9.4% | 1.57× |
| 128→32 | 3.29091 | 3.22686 | 2.14695 | -1.9% | 1.50× |
| 512→16 | 1.66457 | 1.64751 | 1.08358 | -1.0% | 1.52× |

这一次完整请求中，prefill/decode/mixed 分别下降约 9.4%/1.9%/1.0%；但 Stage 分解只显示 EXPERT 的局部变化，其他阶段会抵消或放大它。因此保留 tile 改动的依据是：大 batch 的 EXPERT 有方向性下降，所有输出、层和 logits 仍与原生逐位一致；不是声称整模型稳定加速 9%。owner-partial 的另一种“只发送 owner 列”的变长 All-to-All 也测过，EP_REDUCE 没有稳定收益，已回退，默认仍使用课件中的 `all_gatherv + reduce_scatterv`。

### 扩展正确性验收

| 配置 | 场景 | 结果 |
|---|---|---|
| 双卡 TP=2/DP=1/EP=2 | 32 prompts，batch=1/2/4/8，每条 64 Token；短/长 prompt | 128/128 序列一致；16,384 个 logits，层与 logits 最大误差 0；两 Rank 各 48 层 |
| 双卡 TP=2/DP=1/EP=2 | 20 prompts，batch=2/4，每条 128 Token | 40/40 序列一致；10,240 个 logits，最大误差 0 |
| 四卡 TP=2/DP=2/EP=4 | balanced、uneven、双侧空闲、单 Token Decode、宽 prefill、长 decode、长 prefill、2040 Token max-prefill | 22/22 序列、314/314 Token 一致；四个 Rank 各 48 层；层与 logits 最大误差 0，argmax 一致率 100% |

结构测试为 18 项单元测试、两卡 Grouped/loop 各 48 组每 Rank、四卡 TP/DP 72 组每 Rank。

以下各节是更早的共享环境记录，保留用于对照；其中的绝对延迟和 Mini/native 比值不能与本节空闲四卡结果混用。

## 当前实现与原生 vLLM：先看同配置结果

以下沿用 2026-10-19 本轮 H100 实验的完整请求中位延迟，单位为秒。两者都是 Qwen3-30B-A3B BF16、48 层、H=2048/F=768、128 Experts/Top-8，Attention TP=2，每个 DP 两条请求。使用 eager、同步调度、关闭 prefix caching、Triton Expert 后端。原生 EP 使用 vLLM 默认 allgather_reducescatter；Mini-EP 使用本课 A2A，DP=2 保留 owner-partial 和原生归约。

| 配置 | 每条请求输入→输出 | 原生 vLLM / s | 当前 Mini-EP / s | Mini/native 延迟比 |
|---|---|---:|---:|---:|
| DP=1 / EP=2 | 512→1 | 0.05144 | 0.09014 | 1.75× |
| DP=1 / EP=2 | 128→32 | 1.58806 | 2.77415 | 1.75× |
| DP=1 / EP=2 | 512→16 | 0.79445 | 1.38466 | 1.74× |
| DP=2 / EP=4，共享 GPU | 512→1 | 0.09538 | 0.56113 | 5.88× |
| DP=2 / EP=4，共享 GPU | 128→32 | 15.78139 | 17.52085 | 1.11× |
| DP=2 / EP=4，共享 GPU | 512→16 | 8.24169 | 8.63910 | 1.05× |

比值是 Mini-EP 延迟除以原生延迟，大于 1 表示 Mini-EP 较慢。延迟是每个 DP 两条请求组成的批次全部完成所需的时间，不是逐请求延迟分位数。512→1 仍包含调度、Attention 和输出取回，128→32 则包含一次 prefill 和 32 个生成 Token；这些是完整批次耗时，不是纯 prefill kernel 或单步 decode 耗时。

每轮先预热 2 次、测量 5 次，取最大 Rank 的端到端墙钟。Mini-EP 的优化前后按 ABBA 各测两轮，当前实现合计 10 个样本；原生单独测一轮，共 5 个样本。模型加载、JIT/预热、报告写入和汇总 collective 不计时。两卡选择时空闲；四卡有 29.7 GiB/卡的常驻其他任务，利用率快照为 90%～100%，原生和 Mini-EP 均受争用影响。原始范围、每轮中位数与输出 Token 保存在实验目录。

所有性能样本的输出 Token 都与原生逐项一致。完整严格验收中，两卡 64/64 条、四卡 14/14 条独立生成序列一致，层输出/logits 最大误差均为 0。后面的 6%～55% 是相对旧 Mini-EP 的延迟降低，不能理解为相对原生的加速。四卡 prefill 的剩余差距需要用当前版本逐阶段测量进一步定位，不能直接沿用旧版本的阶段占比定量解释。

课程衔接与源码阅读顺序见 [TEACHING.md](TEACHING.md)。命名、注释和阅读结构的整理保持原有可执行操作、kernel 算式和通信顺序。55 个 Python 文件的可执行 AST 等价，函数操作码/全局访问与 Triton kernel AST 一致；整理后 H100 18 项单元测试、两卡 Grouped/loop 各 48 组每 Rank 的 NCCL 检查通过。两组短 ABBA 核心 EP（128 Token/Rank）前后各 200 个样本，中位数为 1.18910/1.17694 ms；偏慢轮次在整理前后均出现，不把约 1% 差异当作加速。全部样本与等价审计见 `artifacts/course13_teaching_20261019/`。本次没有重复完整模型生成或原生性能实验；性能表与零误差完整模型验收沿用此前已验证、执行等价的版本。

## 2026-10-19：Triton Expert metadata

本轮只改 Expert 的 metadata 规划。稳定排序后，单个 Triton kernel 通过二分下界生成 Expert offsets，再用前缀和生成有界 tile map；不再执行 `bincount().tolist()` 或在 Host 构造三个 GPU metadata 张量。有效 tile 的顺序和大小不变，两次 Grouped GEMM、SwiGLU 舍入、FP32 返回 A2A、DP=1 本地 FP32 Combine/TP_REASSEMBLE、DP=2 owner-partial/原生 reduce_scatter 全部沿用原路径。逐 Expert `expert_loop`、`loop` 后端和外围 PyTorch Reference 保留；Host metadata Reference 位于 `reference/grouped_plan.py`。

两卡环境为 gpu3 的物理 H100 80GB GPU 4、5，选择时为空闲；Python 3.12、PyTorch 2.13.0+cu130、Triton 3.7.1、vLLM 0.29.0，Qwen3-30B-A3B BF16。TP=2、DP=1、EP=2，每个 DP 同时输入两条请求。使用 eager、同步调度、相同 prompt/采样配置，关闭 prefix caching。每轮独立 torchrun 内每个场景完整预热 2 次、测量 5 次，按 baseline → optimized → optimized → baseline 顺序运行；每种实现合并 10 个最大 Rank 端到端墙钟样本取中位数。原生 vLLM 另测 5 次。测量包含请求调度、Attention、MoE 和输出取回，排除模型加载、预热/JIT、报告写入与汇总 collective。

| 双卡每条请求输入→输出 | 优化前 / s | 优化后 / s | 延迟降低 | 原生 vLLM / s |
|---|---:|---:|---:|---:|
| 512→1 | 0.10331 | 0.09014 | 12.75% | 0.05144 |
| 128→32 | 2.97437 | 2.77415 | 6.73% | 1.58806 |
| 512→16 | 1.47827 | 1.38466 | 6.33% | 0.79445 |

所有性能运行的输出 Token 均逐项等于同场景原生输出。优化后仍约为原生延迟的 1.74～1.75 倍。两轮 baseline 的 prefill 中位数分别为 100.51、135.65 ms，存在时间漂移；表格保留全部样本，不删除较慢样本，完整最小/最大值与每轮中位数均写入 `summary.json`。这些数据描述本次小批请求，不代表饱和服务吞吐。

四卡使用 gpu5 的物理 H100 GPU 0–3、TP=2/DP=2/EP=4，每 Rank 32 Experts，默认 owner-partial + 原生 reduce_scatter。各卡在实验前已有约 29.7 GiB 常驻占用、90%～100% 利用率；本次四卡测量在共享环境完成。测量协议同双卡：每个 DP 两条不同请求、每轮预热 2 次、测量 5 次，完整 ABBA 合并每种实现 10 个最大 Rank 墙钟样本；原生另测一轮 5 次。

| 四卡每条请求输入→输出 | 优化前 / s | 优化后 / s | 延迟降低 | 原生 vLLM / s |
|---|---:|---:|---:|---:|
| 512→1 | 1.24678 | 0.56113 | 54.99% | 0.09538 |
| 128→32 | 37.82110 | 17.52085 | 53.67% | 15.78139 |
| 512→16 | 18.70674 | 8.63910 | 53.82% | 8.24169 |

两轮优化的 decode 中位数分别为 17.5149、17.5268 s；两轮 baseline 的中位数为 38.2954、37.7745 s，反向复测重新出现差距。全部优化前后样本及所有 Rank 的输出 Token 都与原生一致。四卡收益在本次共享负载中复现，但不能外推为四卡独占收益。原生参考仅一轮，也受共享负载漂移影响；优化后 prefill 仍约为原生的 5.88 倍，decode/mixed 约为 1.11/1.05 倍，剩余差距需要另行分阶段分析。

| 四卡场景 | 优化前最小～最大 / s | 优化后最小～最大 / s |
|---|---:|---:|
| prefill | 1.1482～1.3262 | 0.5500～0.6351 |
| decode | 36.6372～38.8120 | 17.2872～18.2758 |
| mixed | 18.4185～19.5917 | 8.4432～9.0881 |

单卡 metadata 微基准：5 次预热、40 次交替 A/B；两条路径都不包含稳定排序。R=4096 个接收分支、L=64 时，Host/Triton 的 CUDA Event 中位数为 0.14099/0.04262 ms，同步墙钟为 0.15027/0.05188 ms。五次预热后的 CPU/CUDA trace 中，Host 路径有 15 次 D2H、15 次 H2D 和 30 次 `cudaStreamSynchronize`；Triton 路径有 5 次 `_plan_kernel`，没有这些拷贝或流同步。两者都只在 trace 边界进行 device synchronize。该微基准只证明 metadata 成本下降，不能代替完整 EP 或整模型计时。

最终双卡核心 EP 基准每轮使用 20 次预热、50 次测量，按 ABBA 启动四轮独立 torchrun，每种实现两轮，合并 100 个最大 Rank CUDA Event 样本。输入为种子 13 的合成 BF16 Token/权重，H/F/E/K 与模型相同，每次运行先对齐独立单 GPU Reference。核心 EP 不包含 Attention、KV Cache 或请求调度；总时间直接测量，不累加各阶段最大值。

| 每 Rank Token 数 | 总延迟：前→后 / ms | EXPERT：前→后 / ms | 总延迟降低 |
|---:|---:|---:|---:|
| 4 | 1.26214→1.18403 | 0.40690→0.31131 | 6.19% |
| 128 | 1.32675→1.15854 | 0.55450→0.42626 | 12.68% |
| 512 | 1.51693→1.35950 | 0.69773→0.55712 | 10.38% |

单元测试已在 H100 上执行 18 项；两卡 Grouped 与 loop NCCL 正确性各通过每 Rank 48 组，四卡 TP/DP 布局测试通过每 Rank 72 组。课程隔离、依赖与来源校验通过。四卡完整模型严格验收通过：每 Rank 覆盖 48 层，层输出与 logits 最大误差均为 0，argmax 一致率 100%，独立生成 14/14 条序列、216/216 个 Token 完全一致，包含空闲 DP、不等长与长 Prefill 场景。双卡完整严格验收也通过：32 条 prompt、batch=1/4、每条 64 Token，两个 Rank 各 48 层，层输出与 logits 最大误差为 0，两个 Rank 共 8192 个 logits 位置（每 Rank 4096）的 argmax 全部一致，64/64 条独立生成序列与原生完全一致。严格报告和 Token 报告均从同一份四阶段原始数据生成。

原始命令、ABBA 样本、trace、独立 Reference 与验收报告位于 `artifacts/course13_metadata_20261019/`。本轮不引入 Chunk 或异步 stream，不声称通信/计算重叠。

## 2026-10-18：Grouped SwiGLU 双卡优化

本轮把 `EXPERT` 阶段从“每个非空 Expert 单独调用两次 GEMM”改为 course11 风格的 Grouped GEMM：本地 Expert 行稳定打包后，一次计算 `H→2F`，一次计算 `F→H`，中间复用同一个 SwiGLU Kernel。

环境：单机两张空闲 H100 80GB，容器 `vllm033`，CUDA_VISIBLE_DEVICES=4,5；Qwen3-30B-A3B BF16，TP=2、DP=1、EP=2，eager、同步调度、Triton 专家后端。核心 EP 不计 Attention、KV Cache 和请求调度。

| 每 Rank Token 数 | 路径 | 总延迟 (ms) | EXPERT (ms) | 相对 loop |
|---:|---|---:|---:|---:|
| 4 | loop | 3.055 | 2.108 | 1.00× |
| 4 | grouped | 1.312 | 0.417 | 2.33× |
| 128 | loop | 5.779 | 4.878 | 1.00× |
| 128 | grouped | 1.359 | 0.557 | 4.25× |
| 512 | loop | 6.065 | 5.096 | 1.00× |
| 512 | grouped | 1.539 | 0.717 | 3.94× |

同一双卡环境下的整模型请求 `512→1`：原始逐 Expert 路径为 `0.2194 s`，Grouped 路径为 `0.1011 s`，约 `2.17×`。两次运行使用相同模型、prompt、采样参数和 GPU，输出 Token 均为“计算”；表中报告 3 次预热后运行的中位数。

同一环境补测的原生 vLLM 双卡路径为 `0.0537 s`。按这条基线，优化前 Mini-EP 约为原生 `4.09×`，优化后约为原生 `1.88×`；差距缩小，但 Mini-EP 仍比原生慢。

本轮双卡验收完成了单元测试、48 组双卡分布式正确性用例、`loop/grouped` 同输入对照、整模型 `512→1` 输出一致性，以及 `--limit 1 --max-tokens 2` 的真实模型逐层与生成验证。该验证覆盖两个 Rank 的 48 层，层输出最大绝对误差为 0，logits argmax 一致率为 1.0，2/2 条生成序列一致。

四卡 `TP=2 / DP=2 / EP=4` 已在 gpu5 的 GPU 0–3 上完成 Token 级验收。DP=1 使用本地 FP32 Combine；DP=2 默认保留 owner 部分和。本地 Combine 只有 12/14 条完整序列一致，`long_prefill` 和 `uneven` 发生分叉；owner-partial 为 14/14，层输出与 logits 均为零误差。因此 DP=2 的本地 Combine 保留为 `MINIEP_OWNER_PARTIALS=0` 实验选项，不作为默认快速路径。

2026-10-18 的默认路径复测没有设置 `MINIEP_OWNER_PARTIALS`。四个 Rank 均自动选择 owner-partial，严格模式与 Token 模式都通过；四个 Rank 各覆盖 48 层，层输出和 logits 最大误差为 0，独立生成 14/14 条序列一致。

| DP=2 Combine 路径 | 完整序列 | 层输出最大误差 | logits 最大误差 | 结论 |
|---|---:|---:|---:|---|
| 本地 FP32 Combine | 12/14 | 0.5 | 3.0 | 不通过 |
| owner-partial | 14/14 | 0 | 0 | 通过，作为默认 |

返回 A2A 固定使用 FP32。R=4096 的孤立 A2A 微基准中，FP32 返回约 `0.108 ms`，BF16 返回约 `0.079 ms`，但 32 条 prompt × 64 Token 的完整验收中 BF16 和 FP16 都会改变生成 Token 序列，因此 2 字节回传实验代码未保留。

| 返回 dtype | 完整序列一致 | Token 一致率 | 结论 |
|---|---:|---:|---|
| FP32 | 64/64 | 100% | 通过，作为默认 |
| BF16 | 24/64 | 70.99% | 不通过 |
| FP16 | 26/64 | 66.02% | 不通过 |

FP32 的 32×64 验收同时得到：两个 Rank 各 48 层覆盖、层输出最大绝对误差 `0`、logits argmax 一致率 `1.0`、logits 最大绝对误差 `0`。这说明 Token 级门槛本身可以通过，2 字节回传的差异来自额外的 BF16/FP16 舍入。

### P1 尝试结果

- Up+SwiGLU epilogue 融合：单独测量 Up+激活在 R=1024 时约 `0.145 ms`，两段路径约 `0.149 ms`，差异很小；整段 EP 的重复测量落入噪声范围，没有稳定收益，因此未保留，代码仍使用两段路径。
- Expert ID int32：int32 与 int64 的 A2A 都在约 `0.039 ms`，没有可测收益，因此回退 int64，避免额外转换。
- 复用 router_logits：Qwen3 的 `Qwen3MoeSparseMoeBlock.forward` 实际传入 `router_logits=hidden_states`，不是已计算的 gate logits，不能直接复用，已跳过。
- 本地行不发送：最终 Grouped 路径中 `A2A_FORWARD + A2A_RETURN` 约占核心 EP 的 20%，低于 40% 的触发阈值，未进入这一轮；保留为后续通信专项候选。

复现命令、各阶段 JSON 与汇总见 `artifacts/course13_grouped_20261018/`。

### 2026-10-18 四卡阶段分析

环境为 gpu5 的物理 GPU 0–3，Qwen3-30B-A3B BF16、TP=2、DP=2、EP=4。四张卡与其他任务共用，测试前每张卡约有 29.7 GiB 常驻占用，利用率快照为 90%–100%。因此下面的毫秒值只用于比较阶段占比，不代表独占 GPU 的稳态性能。

测试包含 `512→1` prefill 和 `128→32` decode，每个场景先 warmup 1 次，再测量 3 次；每个 Rank 覆盖 48 层。阶段计时使用 CUDA Event，在一次请求结束后统一同步，避免逐层同步改变流水线。

| 阶段 | prefill 中位数 / ms | decode step 中位数 / ms | prefill 阶段占比 |
|---|---:|---:|---:|
| EXPERT | 12.691 | 12.522 | 54.5% |
| EP_REDUCE | 5.198 | 5.082 | 22.3% |
| DISPATCH | 2.633 | 2.550 | 11.3% |
| A2A_FORWARD | 2.594 | 2.557 | 11.1% |
| A2A_RETURN | 0.145 | 0.093 | 0.6% |
| COMBINE | 0.018 | 0.008 | 0.1% |
| ROUTER | 0.017 | 0.016 | 0.1% |

`TP_REASSEMBLE` 在 DP=2 不存在于 Mini-EP Runner 内：vLLM 在 Runner 外完成 TP 行恢复。DP=1 的双卡路径仍保留这个阶段。

同一次运行采集的 Nsight Systems `nvtx_sum` 给出了相近的墙钟排序：

| NVTX 阶段 | 总时间占比 | 中位数 / ms |
|---|---:|---:|
| EXPERT | 49.0% | 12.575 |
| EP_REDUCE | 27.8% | 5.124 |
| A2A_FORWARD | 12.4% | 2.634 |
| DISPATCH | 8.6% | 1.276 |
| A2A_RETURN | 0.5% | 0.118 |
| ROUTER | 0.6% | 0.154 |
| COMBINE | 0.3% | 0.072 |

两套数值的口径不同：CUDA Event 区间会包含同一 CUDA 流上的依赖等待，Nsight NVTX 表格统计的是 Host range；因此 DISPATCH、ROUTER 和 COMBINE 的中位数不会完全相同。

Nsight 的 GPU Kernel 时间组成与墙钟阶段不同：NCCL `SendRecv` 占 54.6%，`AllGather` 占 28.6%，`ReduceScatter` 占 8.3%，FlashInfer all-reduce 占 3.5%；Mini-EP 的 `_grouped_linear_kernel` 只占 1.1%，`_swiglu` 接近 0%。这说明：

- `EXPERT` 的墙钟占比高，但分组 GEMM 本身很小。该阶段主要包含本地排序、计数、offset/tile 规划、`.tolist()` 同步和跨 Rank 等待，属于 Host metadata 与依赖等待问题，不是 GEMM 算力问题。
- `EP_REDUCE` 与两次 A2A 的 NCCL Kernel 占 GPU Kernel 时间的绝大部分。DP=2 的 owner-partial 归约是当前最明确的通信热点；本地 Combine 已验证不满足 Token 级精度，不能直接删除。
- `DISPATCH` 仍有约 11% 的墙钟占比，和 `EXPERT` 一样应由 GPU 侧计数、offset 和布局规划来压缩。
- `A2A_RETURN` 和 `COMBINE` 占比较小，暂不作为第一优化对象。

按这些数据，“本 Rank 行不发送”只影响两次 A2A 的载荷，最多触及约 11.7% 的墙钟；EP=4 下即使本地行全部不计入发送量，粗上限也只有约 3%。它适合在 Host metadata 和 owner-partial 归约处理之后再验证。复现数据见 `artifacts/course13_stage_20261018_v2/` 与 `artifacts/course13_nsys_20261018/`。

### 2026-10-18 P0-1/P0-2 复测

本轮实际尝试了两项优化。A/B 使用同一组共享 GPU、相同模型和 prompt，warmup 1 次、测量 5 次；下表是修正后的最终代码与修改前基线。

| 场景 | 阶段 | 基线 / ms | 最终 / ms | 变化 |
|---|---|---:|---:|---:|
| prefill | EP_REDUCE | 5.200 | 5.167 | -0.65% |
| decode | EP_REDUCE | 5.084 | 5.061 | -0.46% |
| prefill | EXPERT | 12.691 | 12.692 | +0.01% |
| decode | EXPERT | 12.524 | 12.523 | -0.01% |

P0-1 的 Torch GPU count/offset/tile 规划没有稳定收益：与 P0-2 合并运行时 prefill `EXPERT` 从 12.691 ms 变为 12.753 ms，因此没有保留 GPU 规划，最终代码恢复原来的稳定 Host planner。P0-2 保留：owner-partial 现在直接按 source-major 做变长 `all_gatherv`，再调用 `reduce_scatterv`，去掉了 `[max_N,P,H]` 补齐、`all_gather` 列表和 `cat` 中间缓冲。

最终代码的 DP=2 四阶段验收通过：严格模式和 Token 模式均通过，层输出与 logits 最大误差为 0，独立生成 14/14 条序列一致。共享 GPU 上的总阶段中位数变化约为 prefill `-0.20%`、decode `-0.10%`，因此这里不声称端到端吞吐提升；保留 P0-2 的依据是更短的 buffer 路径和精度不退化。复现记录见 `artifacts/course13_p0_opt_20261018/`。

以下为优化前的四卡共享环境记录，保留用于对照。

模型：Qwen3-30B-A3B，BF16；vLLM 0.29.0；TP=2、DP=2、EP=4。
服务器：172.16.240.15，H100 GPU 0、1、4、7，容器 vllm033。

两种实现均使用 eager、同步调度、关闭 prefix caching、Triton 专家计算；原生 EP 使用 vLLM 默认 allgather_reducescatter，Mini-EP 使用本课 A2A 和精度兼容归约。没有启用 validation/shadow/logits 钩子。

每个 DP 两条请求，共四条；输入 token 在两种实现间相同，DP 之间不同。所有运行的输出 token 逐项一致。每个场景、每个独立进程先完整预热一次，再测三次；按 native → miniep → miniep → native 顺序运行，下面报告每种实现六次样本的中位数。模型加载、编译预热、磁盘写入和汇总 collective 不计时。

时间为四 rank 最大端到端墙钟，包含 enqueue、prefill、decode、取回结果和测量边界 CUDA 同步；CUDA Event 同时记录。吞吐为全局生成 token / 中位延迟，TP 副本不重复统计。不是纯 MoE kernel 延迟，也不是饱和服务吞吐。

| 每条请求输入→输出 token | 原生 EP 延迟 (s) | Mini-EP 延迟 (s) | 延迟比 Mini/native | 原生生成 tok/s | Mini-EP 生成 tok/s |
|---|---:|---:|---:|---:|---:|
| 512→1 | 0.1003 | 0.9753 | 9.72× | 39.86 | 4.10 |
| 128→32 | 7.6679 | 26.7983 | 3.49× | 16.69 | 4.78 |
| 512→16 | 4.6969 | 13.3752 | 2.85× | 13.63 | 4.78 |

## 测量波动和适用范围

测试前 GPU 0、1 利用率 100%，GPU 4 为 93%，GPU 7 为 0%；前三张卡与其他任务共享。ABBA 顺序只能部分减小时间漂移，不能消除资源争用。这些数值是共享环境小批量实测，不能外推为独占硬件的性能上限。

| 场景 | 原生 EP 最小～最大 (s) | Mini-EP 最小～最大 (s) |
|---|---:|---:|
| prefill | 0.0778～0.2462 | 0.9225～1.0564 |
| decode | 7.3072～8.7754 | 26.5526～27.0790 |
| mixed | 3.3135～4.9084 | 13.2249～13.5004 |

当前 Mini-EP 的定位是可读的教学实现。在本次配置下，原生 EP 更快。源码中的逐专家 Python 循环、多次 kernel launch、读取路由计数到 CPU，以及用于 BF16 精度对齐的 owner-partial gather/reduce-scatter，都是可能的额外成本；本次没有 Nsight 时间线或逐模块消融，因此不将差距定量归因于其中某一项。没有修改核心实现或放宽精度要求。

## 复现和原始数据

脚本与启动命令见 [README.md](README.md) 的性能对比部分及 [benchmarks/compare_vllm.py](benchmarks/compare_vllm.py)。

仓库原始记录：`artifacts/course13_performance_20261017/`，包含四轮各 rank 的 JSON、完整日志、GPU 采样、模型配置、汇总脚本和 `summary.json`。服务器原始记录：`/SharedData/fuss/miniep-ep4-gpu15-20261016/results/performance/`。

模型结构：48 层、hidden_size=2048、expert_hidden_size=768、128 个专家、Top-8；BF16，EP=4。不使用 chunk pipeline（chunk_size 不适用）。本地报告目录按会话日期命名，GPU 日志保留服务器自身时间。
