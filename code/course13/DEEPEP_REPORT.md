# Course13 DeepEP 对齐、代码整理与性能报告

精度、三轮正式计时和独立阶段采样全部完成。本报告不把整理前计时、短样本或传输字节减少当作最终版本加速证明。

后续核心死代码清理另见 [CORE_CLEANUP.md](CORE_CLEANUP.md)。本报告的完整矩阵和计时仍对应下文标识的冻结版本；不覆盖后续增量结果。

## 结果范围

环境为Python3.12、PyTorch2.13.0+cu130、Triton3.7.1、Transformers5.16.1、FlashInfer0.6.18、CUDA13.0、NCCL2.30.7、driver580.65.06、NVIDIA H100 80GB HBM3。具体GPU UUID、CPU/软件参数和命令保存在每Rank environment.json。

真实 Qwen3-30B-A3B BF16，vLLM 0.29.0，DeepEP 2.0.0+local，Attention TP=2、DP=2、MoE EP=4。对照实际执行 `MoERunner / DeepEPHTAll2AllManager / DeepEPHTPrepareAndFinalize / TritonExperts`，四 Rank 均保存 proof。原生专家计算、DeepEP 算法和 vLLM 安装目录没有修改，没有通信后端回退。

每项均包含 native_check、逐层 compare、teacher-forced 完整 logits、自由生成四阶段；数值、shape、dtype、device、路由 IDs/slot/FP32 权重、实际返回 dtype/字节和四 Rank/48 层/全部请求覆盖都检查。原生三次独立启动逐层 SHA256、完整 logits 和所有 Token 重复一致。候选种子 13 也独立启动三次。

| 配置 | 唯一请求序列 | 唯一生成 Token | 严格零误差 |
|---|---:|---:|---|
| 同 DP 输入，seed13，batch1/4 | 128 | 8192 | 通过 |
| 不同 DP 输入，seed13，启动1 | 128 | 8192 | 通过 |
| 不同 DP 输入，seed13，启动2 | 128 | 8192 | 通过 |
| 不同 DP 输入，seed13，启动3 | 128 | 8192 | 通过 |
| 不均衡、空闲侧、请求先后完成等边界 | 26 | 1426 | 通过 |
| 长 prefill、256-token decode，batch1/4 | 16 | 4096 | 通过 |
| 不同 DP 输入，seed37 | 128 | 8192 | 通过 |
| 不同 DP 输入，seed123 | 128 | 8192 | 通过 |
| 六种性能工作负载，8192 budget/32请求槽 | 92 | 9490 | 通过 |
| 合计，含独立重复启动 | 902 | 64164 | 通过 |

完整语料均为32条 prompt、batch=1/4、64个生成 Token。teacher 检查包括 TP 两侧共128328个采样位置，每个位置覆盖151936词表列；实际保存的原生 logits 为 BF16。没有使用允许误差的 `strict_passed`，同值 NaN/Inf 也不能通过。完整门禁报告见 [原始复核报告](../../artifacts/course13-deepep-final-20261021/final-accuracy/final-accuracy.json)。

请求侧无输入与物理零行分开记录：vLLM 可能给空闲 DP 一行 dummy。额外直接调用真实48层/原生 DeepEP API，四 Rank 各768次对照，其中384次真实零行输入；两侧轮换、对侧2/16行、seed13/37、固定/变长均通过。这是层 API 证明，不把它写成完整调度器的零行执行。

## 最终实现

```text
模型激活 → 原生 Gate/Top-K → 稳定 Rank packing
→ NCCL A2A1：Tokens、全局 IDs、FP32 weights
→ Expert packing / gate-up / 模型 dtype SiLU 后乘 up
→ down FP32 accumulator × FP32 routing weight
→ tokens.dtype → 返回 A2A
→ restore_index 恢复分支，保留模型 dtype 分支舍入
→ owner 内 slot 升序 FP32 累加 → owner 部分和模型 dtype 舍入
→ EP rank 升序 FP32 合并 → 模型 dtype 输出
```

BF16 模型正常回传 BF16；核心 FP32 输入正常回传 FP32。每分支只乘一次权重，Combine 不接收权重参数。Grouped down 默认融合加权及转换；独立 PyTorch Reference、非融合 FP32 中间 tensor 和逐专家 loop 保留。FP16/BF16/FP32 显式 transport 是诊断，完整 FP16 模型仍明确拒绝。

删除了 `integration/reduction.py::reduce_owner_partials`、AGRS owner gather/reduce-scatter、Host/legacy owner-size 开关、owner-partial 输出模式、原生后端 selector 和 source/expert 加权位置切换；旧矩阵/容差报告/重复计时工具由单一 DeepEP 工具替代。历史文档与原始实验保留并标注历史适用范围。

必要的 TP 行分片/重组仍在 `integration/input_layout.py`；Attention、DP/SP 生命周期仍由 vLLM 管理。没有因删除 owner 兼容归约而删除模型自身的通信。

## 与前课的实际代码对应

| 前课 | 当前实体函数 | 必要扩展 |
|---|---|---|
| 1/2 Router、placement、packing | `RouterOutput`、`ExpertParallelTopology`、`pack_by_destination_rank` | 原生选择器保持 tie/slot；直接 Rank packing 组合源端排列，权重同行搬运 |
| 5 NCCL A2A | `exchange_splits`、`equal_split_all_to_all`、`variable_split_all_to_all` | 行为仍是行 split，返回反转 send/recv；小 decode 固定容量 |
| 6 MoE 编排 | `model/moe.py::ep_forward` | 显式六阶段；框架持有权重/Router，故使用无状态 tensor 接口 |
| 9/10 metadata、逆排列 | `dispatch_triton`、`combine_kernel` | restore_index 身份保持；加权移到 down FP32 epilogue，Combine 不重复加权 |
| 11 Grouped GEMM | `GroupedGemmPlan`、`grouped_expert_mlp`、`compute_grouped_local_experts` | GPU plan 免 Host expert counts；SwiGLU、FP32 down 与输入 dtype 返回 |
| 8/12 测量与流水线 | `Stages`、独立 benchmarks | 保留 NVTX/计时；未引入未验证的 Chunk/stream pipeline |

Grouped projection、GPU plan、恢复 kernel 和 tile 配置与整理前算术 AST 一致；保留 `tl.dot(a,b,acc)`，不能为形式一致改回前课的分块乘积再相加。实际函数/布局细节见 [TEACHING](TEACHING.md)。前课源码、README、本地正文均核对；飞书正式1～10课及旧Course13草稿只读阅读，目录未提供11/12课，视频未观看。没有修改远端课件或 course1～12。

## 证据复用与源码身份

复用的仅为有效独立原生 golden：相同权重、推理选项、语料、种子、实际原生类和不可变镜像/原生 Python+扩展指纹，且当前候选重新执行完整 native_check。没有复用旧 Mini-EP 的数值通过结论。新增完整候选矩阵、真实零行、默认回传/字节检查、性能工作负载精度和最终启动回归。

完整矩阵在不可变数值快照 v6 上运行。交付核心仅有两个 wrapper docstring 修正，24个核心执行 AST、签名、默认值、常量和 Kernel 均相同；两套 SHA 及 [审计](../../artifacts/course13-deepep-final-20261021/core-docstring-only-audit.json) 保留。外围验证模式命名整理后的有效环境逐字段相同，且交付入口补验通过。H10042项单元/回归全通过，EP1/2/4每 Rank432例通过，四 Rank各16例失败拒绝检查通过。本地CPU42项中34通过、8项明确CUDA skip。

## 性能状态与方法

最终测量在原先实验完整结束、上述精度门禁全通过后启动。相同 gpu3、物理 GPU0～3、固定镜像、CPU配额8、内存64GiB、模型/输入/调度/dtype；每端独立启动三次，交替顺序，每用例 warmup3、测量5，共15样本。排除加载/JIT/初始化，关闭 shadow、逐层张量落盘和精度探针；逐重复先取最慢 Rank，再报告中位数、p10/p90、原始样本及吞吐。

prefill 从最早 scheduling 到最后 first-token；decode 为 first 到 last-token 间隔除以输出数减1；端到端包含正常入队、模型、输出和边界同步。两者同开 vLLM 请求统计以获得时间戳，其单独开销未消融；额外 Event profiling 单独启动，不混入正式计时。

整理前的六个正式计时任务完整通过并复用。其额外 Mini-EP 阶段采样曾因旧接口不接受 stages 参数失败，原始失败保留，只补跑采样后通过，没有重跑有效正式计时。正式结果及独立采样如下。

## 最终正式计时

| 用例 | DeepEP 端到端/ms | Mini-EP 端到端/ms | 延迟比 M/D | 生成 tok/s D/M | 吞吐比 M/D |
|---|---:|---:|---:|---:|---:|
| decode_b1 | 10400.215 | 11163.723 | 1.073 | 24.615 / 22.931 | 0.932 |
| decode_b4 | 10234.825 | 10814.084 | 1.057 | 100.051 / 94.691 | 0.946 |
| decode_b32 | 10332.622 | 12994.039 | 1.258 | 792.829 / 630.443 | 0.795 |
| prefill_b1_long | 107.175 | 111.570 | 1.041 | 18.661 / 17.926 | 0.961 |
| prefill_b4_short | 83.413 | 103.866 | 1.245 | 95.908 / 77.022 | 0.803 |
| prefill_b4_long | 179.360 | 225.665 | 1.258 | 44.603 / 35.451 | 0.795 |

吞吐统计两个DP的唯一生成Token，不重复计算TP。单输出prefill的生成tok/s不是输入处理吞吐，输入吞吐另列。

| 用例 | prefill/ms D/M | decode每步/ms D/M | 输入 tok/s D/M |
|---|---:|---:|---:|
| decode_b1 | 82.278 / 102.953 | 81.221 / 87.107 | 3111.40 / 2486.57 |
| decode_b4 | 81.054 / 101.501 | 79.929 / 84.349 | 12633.60 / 10088.59 |
| decode_b32 | 95.757 / 126.406 | 80.563 / 101.284 | 85550.32 / 64807.00 |
| prefill_b1_long | 105.475 / 109.937 | — | 38833.74 / 37257.57 |
| prefill_b4_short | 81.665 / 101.643 | — | 12538.96 / 10074.48 |
| prefill_b4_long | 175.475 / 221.298 | — | 93369.40 / 74035.88 |

## 简化前后

| 用例 | 原Mini-EP/ms | 最终Mini-EP/ms | 中位变化 | p10/p90区间重叠 |
|---|---:|---:|---:|---|
| decode_b1 | 11024.473 | 11163.723 | +1.26% | 是 |
| decode_b4 | 11031.622 | 10814.084 | -1.97% | 是 |
| decode_b32 | 13167.130 | 12994.039 | -1.31% | 是 |
| prefill_b1_long | 111.948 | 111.570 | -0.34% | 是 |
| prefill_b4_short | 106.258 | 103.866 | -2.25% | 是 |
| prefill_b4_long | 226.424 | 225.665 | -0.34% | 是 |

此组测量的变化为−2.25%～+1.26%，所有配置区间重叠，未观察到稳定退化，也不足以证明稳定提速。不能据此证明严格零开销。b1长prefill的原生基线从85.191ms变为107.175ms，而Mini-EP从111.948ms变为111.570ms；其延迟比的大幅下降主要伴随原生波动，不能归因于简化。

前后公平性检查逐项确认相同GPU UUID/driver、软件、CUDA/NCCL、CPU配额、权重、模型、完整LLM选项、literal输入和全部自由生成Token。完整中位数、p10/p90、原始分布及归一化变化见 [对照数据](../../artifacts/course13-deepep-final-20261021/final-primary/comparison.json)。


## 阶段观察与瓶颈

额外 profiling 对0/24/47层采样。下表为所选层平均每次MoE调用的CUDA Event经过时间：每轮各阶段分别取最慢Rank，再对5次重复取中位数。各列可能来自不同Rank，不能相加当作同一Rank的端到端，也不是排除等待后的纯Kernel时间。

| 用例/阶段 | Dispatch/ms | A2A1/ms | Expert/ms | A2A2/ms | Combine/ms |
|---|---:|---:|---:|---:|---:|
| decode_b1 | 0.1094 | 0.7633 | 0.2279 | 0.3132 | 0.0280 |
| decode_b4 | 0.1208 | 0.8211 | 0.2694 | 0.3503 | 0.0353 |
| decode_b32 | 0.2710 | 0.6091 | 0.3420 | 0.1450 | 0.0494 |
| prefill_b4_long | 0.4218 | 0.7832 | 1.6346 | 1.1677 | 0.0808 |

小批量decode的主要间隔出现在A2A1和返回通信；batch32还存在较明显的Dispatch间隔。长prefill中Expert和返回通信更突出。阶段时间包含Host提交、依赖与跨Rank等待；没有证据把它们全部归为带宽或GEMM算力瓶颈。Mini-EP仍是显式expanded-branch、独立payload NCCL调用与同步链路，不能把与DeepEP的差距简单等同于返回dtype。

实际BF16返回payload：decode_b1固定布局每Rank每层131072字节（含padding），decode_b4为262144字节；每个原始记录均核对 `return_send_bytes=recv_rows*2048*2`。FP32权重消息及IDs另计。大batch和prefill行数随路由变化，完整布局payload及排除self的估算见 [阶段数据](../../artifacts/course13-deepep-final-20261021/final-primary/profile-summary.json)。这些不是协议层wire bytes。

Native只可靠观测完整MoE区间；未拆分其Dispatch/A2A/Expert/Combine，也未推断其通信字节。采样对Mini-EP端到端中位数的观测扰动约+0.35%～+4.56%；Native长prefill本身存在波动，因此不把profile与primary差异当作纯因果开销。采样数据不混入正式15样本统计，不从其中推断overlap。

## 可复现入口与原始数据

```bash
cd code/course13
python3 -m pip install --no-deps --no-build-isolation -e .
python3 -m unittest discover -s checks -p 'test_*.py'
# 四张GPU，真实BF16模型；本次环境所有运行一致设置 NCCL_NVLS_ENABLE=0。
NCCL_NVLS_ENABLE=0 python3 test/deepep_alignment.py --model "$MODEL_PATH" --output results/deepep-full
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 checks/deepep_empty_rank.py \
  --model "$MODEL_PATH" --output results/deepep-empty
python3 benchmarks/deepep_latency.py --model "$MODEL_PATH" \
  --checkpoint-manifest results/deepep-full/checkpoint.json --output results/deepep-latency \
  --trials 3 --warmup 3 --repeats 5 --profile-stages
```

实际逐任务 argv、exit code、source SHA、输入、四 Rank proof、样本和环境分别保存在如下持久路径；通过现有 qemu-host/h100-jump 路由连接 H100。

- 原先未中断实验：`/experiment/default-exact-v4/deep4-full`。
- 新数值矩阵：gpu5 `/workspace/deepep-final-v6-20261021/seed13`；gpu3 `/experiment/deepep-final-gpu3-v6-20261021`。
- 整理前计时及采样重试：`/experiment/deepep-before-bench-20261021`。
- 最终计时：`/experiment/deepep-final-bench-v6-20261021/results`。
- 冻结模型清单 digest：`9b4fcbad645e5df727e2ace994e87e458adee69d0740d443f3810e0377ed803e`。
- 固定镜像：`sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`。
- 本地归档、原始复核、源码、材料、备份与检查：`artifacts/course13-deepep-final-20261021/`；完整词表原始 .pt 仍在持久 golden 目录，不覆盖旧数据。

## 限制和历史失败

真实数值验收未失败；早期工具链检查曾拒绝缺失插件、Gate占位参数误用，均在当前冻结实现前修正。节点NVLS初始化曾返回CUDA401，后续统一显式NVLS=0；通信仍为真实NCCL/DeepEP，不是后端回退。旧版本和失败日志保留。

结论仅覆盖本报告模型、二进制、H100拓扑与配置。FP16模型、其他模型/硬件/DeepEP版本不在通过范围；Qwen3.5的新DeepEP精度未验证，DP1 shared/routed 联合舍入路径显式拒绝。DP1原生NoDPEP不作DeepEP对照。DeepEP细分阶段和通信字节不能可靠观测时不编造；本环境未采集Nsight Timeline，不作通信计算重叠结论。
