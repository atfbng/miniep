# 返回 A2A 精度实验

本文件记录上一轮显式传输策略实验，原始结果与数据保持不变。当前默认值已改为跟随 `tokens.dtype` 并在专家端先加权；新版本的独立零误差验收见 [EXACT_ALIGNMENT.md](EXACT_ALIGNMENT.md)。以下历史结果不能替代新验收。所有结果都需区分真实 48 层模型与 `--dummy-layers` 合成兼容性检查。

## 路径一：保持原有数值基准

直接把未加权的 Expert down 输出转换成 BF16/FP16，会增加一次路由加权之前的舍入。本实验增加专家端加权：

```text
FP32 down accumulation × FP32 routing weight
→ BF16 transport
→ 原模型 dtype 的分支值
→ 按原 slot 顺序做 owner 部分和
→ 原有 owner 归约
```

每个分支仍只加权一次。前向 A2A 顺序为 Token、Expert ID、可选 FP32 routing weight；三者使用同一组按 EP rank 排列的行 splits。固定容量使用同一 restore_index 打包权重，padding 权重为 0。返回方向互换变长 splits，固定容量保持原 splits。所有 Rank 必须选择相同策略，包括没有来源 Token 的 Rank。

独立加权版保留 FP32 中间结果；融合版把权重乘法放在 down GEMM 写出前，直接写出所选返回 dtype。两者必须对照；BF16 模型、BF16 已加权回传应与相同计算实现的 FP32 回传逐元素一致。FP16 另有范围和双重舍入问题，单独记录，禁止 clipping 掩盖溢出。

核心接口 `ep_forward` 新增 `return_dtype`、`weight_location=source/expert`、`fuse_return_weight`；当时默认为 FP32/source/False；当前默认为 None/expert/None（自动融合）。显式整模型诊断入口对应：

```bash
MINIEP_RETURN_DTYPE=bf16 MINIEP_WEIGHT_LOCATION=expert MINIEP_FUSE_RETURN_WEIGHT=1 \
  python3 test/validate.py --model "$MODEL_PATH" --output results/weighted-bf16 \
  --gpu-memory-utilization 0.5 --kv-cache-mib 512
```

该命令需要两张有足够可用显存的 NVIDIA GPU，以及本课固定版本 vLLM 环境。显存预算应按启动时可用值设置；显式 KV cache 不代表可忽略权重、context 和临时张量。

## 路径二：原生 vLLM A2A

vLLM 0.29.0 在本课 TP=2、DP=1 配置中不会仅凭选择 DeepEP 就启用原生 EP A2A。原生 A2A 对照使用 TP=2、DP=2、EP=4，并记录具体 Prepare/Finalize 类，不能把后端参数字符串作为实际执行证据。

已安装版本的 DeepEP HT 在专家侧先完成加权和局部归约，返回通信要求 BF16。因此它与 Mini-EP 的逐分支回传不仅通信实现不同，还可能有不同的局部求和与跨 Rank 归约顺序。实验分别检查层输出、teacher-forced 全词表 logits 和独立生成序列；原生 A2A 与旧基准的差异不得归因于 transport dtype 一个因素。

优先使用现成原生后端；只有确认兼容性问题无法解决时，才考虑独立、可撤销的 vLLM A2A 适配。原生路径直接调用未修改的 `MoERunner.forward`，不调用 Mini-EP 的 Expert 或 Combine。

## 一次加载权重的整模型矩阵

`test/return_matrix_run.py` 由外围验证插件选择返回策略，默认覆盖 FP32/source、BF16/source、FP16/source、BF16/expert 独立及融合、FP16/expert 融合。每轮使用全新请求，teacher forcing 与 free generation 分开运行，不缓存前一策略的 MoE 输出。基线是同一引擎中的原生 Runner，原生专家与通信代码不被替换。

两张 GPU，旧原生基准：

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_NVLS_ENABLE=0 \
python3 test/return_matrix_run.py --model "$MODEL_PATH" --output results/return-agrs \
  --limit 32 --max-tokens 64 --gpu-memory-utilization 0.5 --kv-cache-mib 512
```

四张 GPU，原生 DeepEP HT 基准：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 \
torchrun --standalone --nproc-per-node=4 test/return_matrix_run.py \
  --model "$MODEL_PATH" --output results/return-deepep --dp-size 2 \
  --native-all2all-backend deepep_high_throughput \
  --limit 32 --max-tokens 64 --gpu-memory-utilization 0.3 --kv-cache-mib 512
```

`--variants` 可选择实验子集。`--dummy-layers 2 --limit 2 --max-tokens 4` 只检查合成权重下的后端兼容性，报告始终标记 `synthetic=true`。DP=2 为每个全局 Rank 写独立目录。完整验收要求四个 Rank 的报告都完整；同一 TP pair 的重复输出不能重复计算成独立请求。

`environment.json` 保存源码哈希、命令、选项和软件版本；`report.json` 按策略记录 layer coverage、误差、logits argmax、完整序列及 Token 一致数。`strict_passed` 同时检查旧有数值门槛和完整生成序列一致性，未完成报告不算通过。不同原生后端的 native 输出还需按相同 prompt/batch 对齐作跨后端比较。

需要逐个隔离引擎时，使用 `benchmarks/return_matrix.py`。它复用同一原生基线的 logits，每个候选单独启动引擎；失败候选也保留 teacher/free 诊断和 strict report。隔离工具只支持 DP=1 的旧原生基准；传入其他 native 后端会明确拒绝，避免把配置字符串误当成实际执行的 A2A。四卡原生 A2A 使用单次加载工具或 `test/dp_validate.py`。

## 独立正确性与计时

```bash
python3 -m unittest discover -s checks -p test_return_transport.py -v
CUDA_VISIBLE_DEVICES=0,1 NCCL_NVLS_ENABLE=0 \
torchrun --standalone --nproc-per-node=2 checks/return_transport.py --output results/return-check-ep2
```

EP=1/4 将进程数和可见 GPU 数同步改成 1/4。NCCL 测试覆盖模型 FP32/BF16、返回 FP32/BF16/FP16、Top-1/2/8、空来源、倾斜路由、loop/grouped、固定/变长、独立/融合加权及 owner 部分和。与独立 Reference 的比较使用实际传输精度对应的容差；同一计算实现下 BF16 加权位置移动及融合的检查仍要求逐元素相同。

计时与精度分开执行；下列命令需要两张 NVIDIA GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 NCCL_NVLS_ENABLE=0 \
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_ep.py \
  --tokens 128 --return-dtype bf16 --weight-location expert --fuse-return-weight \
  --output results/return-benchmark.json
```

旧 FP32/source 对照须显式设置 `--return-dtype fp32 --weight-location source --no-fuse-return-weight`；省略选项现在测试新默认路径。报告按实际 dtype 统计返回字节，并包含额外权重通信。返回字节减少不等于总延迟下降；最终收益需要独立重复测量及代表性 Timeline，不用共享 GPU 的抖动样本推广性能结论。

## 2026-10-20 H100 实测结果

汇总数据与逐次计时样本保存在 [RETURN_TRANSPORT_RESULTS.json](RETURN_TRANSPORT_RESULTS.json)。环境为 vLLM 0.29.0、PyTorch 2.13.0+cu130、Triton 3.7.1；模型为真实 Qwen3-30B-A3B BF16、48 层、128 Experts/Top-8。本轮历史实验时默认实现为 FP32 回传。

### 路径一：双卡、旧原生基准

32 条 prompt，各执行 batch size 1 和 4、每条生成 64 Token，共 64 条序列、4096 个生成 Token。使用 eager、同步调度、关闭 prefix cache，最大 batched tokens 为 2304、KV cache 为 512 MiB。该调度配置与早期 16K 预算不同，失败序列数量不应直接与历史表格作优劣比较。

| 返回策略 | 完整序列一致 | 层输出最大误差 | teacher-forced logits 最大误差 | strict |
|---|---:|---:|---:|---|
| 未加权 BF16 回传 | 25/64 | 0.5 | 12.96875 | 失败 |
| 未加权 FP16 回传 | 24/64 | 0.5 | 13.33203125 | 失败 |
| 专家端先加权、融合 BF16 写出 | **64/64** | **0** | **0** | **通过** |

加权 BF16 融合版在两台 H100 节点分别完成全量验证，结果均为 64/64；两个 TP Rank 各覆盖 48 层，8192 个 logits 位置（含 TP 副本）argmax 全部一致。独立加权 BF16 在真实模型短样本中同样为零误差，并与融合路径完成独立的核心对照。加权 FP16 短样本仍有数值误差，不能推广为 BF16 模型的等价回传方案。

### 路径二：四卡、原生 DeepEP HT

实际日志确认 `DeepEPHTAll2AllManager` 与 `DeepEPHTPrepareAndFinalize`，不是只修改参数名称。使用 TP=2、DP=2、EP=4；每个 DP 输入相同的两条短 prompt，分别按 batch size 1/4 生成 4 Token。以下统计去除了 TP 副本，共 8 条请求序列。该短样本已经出现反例，因此没有把扩大的 32×64 Native A2A 运行当作已通过验收。

先保留旧的 owner 兼容归约（`all_gatherv + reduce_scatterv`）：

| Mini-EP 策略 | 对原生 DeepEP 的序列一致数 | 说明 |
|---|---:|---|
| FP32 回传 | 7/8 | 原有 FP32 也不能直接保证新后端一致 |
| 未加权 BF16 回传 | 7/8 | 不对齐 |
| 未加权 FP16 回传 | 7/8 | 不对齐 |
| 加权 BF16 回传 | 7/8 | 分支舍入已保留，仍有跨 owner 归约差异 |

差异从第一层 MoE 出现，FP32 与加权 BF16 首个最大绝对差均为 0.00390625。另启同拓扑、同请求的纯原生旧后端作为对照，旧原生与 DeepEP 也只有 7/8 条序列一致；相同输入的首步 logits 最大差为 0.8125。不能把这些差异全部归因于返回 dtype。

验证插件增加 `--owner-reduction local_fp32`：只把 Mini-EP 的 owner 合并改成按 Rank 顺序 FP32 累加，生产 Runner 的默认归约保持原状。重新运行真实四卡短样本：

| Mini-EP 策略 | 序列一致数 | 层 / logits 最大误差 | strict |
|---|---:|---:|---|
| FP32 回传 + local FP32 owner 合并 | 8/8 | 0 / 0 | 通过 |
| 专家端加权 BF16 回传 + local FP32 owner 合并 | 8/8 | 0 / 0 | 通过 |
| 未加权 BF16 回传 + local FP32 owner 合并 | 4/8 | 0.5 / 1.875 | 失败 |
| 未加权 FP16 回传 + local FP32 owner 合并 | 4/8 | 0.25 / 2.09375 | 失败 |

两次 DeepEP 对照自身的 Native 输出全部一致，首步 logits 零误差。这个实验定位到了旧兼容归约与新后端的数值边界；它没有把“原生换成 A2A”变成未加权两字节回传的充分条件。原生 Triton down 投影仍在 FP32 累加结果上先加权、再写回模型 dtype。

复现实验时，四卡命令增加 `--owner-reduction local_fp32 --dp-corpus same --limit 2 --max-tokens 4`。`local_fp32` 只存在于验证工具中；其 32×64 验收、非均匀/空 DP 端到端覆盖及生产推广均未完成。默认 `--dp-corpus rotated` 用于后续不同 DP 输入的对照，不能把本轮 same-corpus 短样本当作已验证该模式。

### 核心验证与计时

- H100 单元测试：30/30 通过。本地 CPU：22 项通过、8 项 CUDA 测试明确 skip。
- EP=1/2/4 NCCL：每 Rank 各 720 个记录用例通过，另各有 24 个逐元素相同的 BF16 owner-partial 对照。
- 核心计时：物理 H100 GPU 5/6，预热 3 次、测量 10 次，报告逐次最慢 Rank CUDA Event 时间的中位数。目标卡启动前利用率为 0%，同节点其他卡有任务；以下只是一轮测量，不代表稳态加速。

| 每来源 Token 数 | FP32 TOTAL (ms) | 融合加权 BF16 TOTAL (ms) | FP32 → BF16 返回 A2A (ms) |
|---:|---:|---:|---:|
| 8 | 0.8700 | 1.0030 | 0.0560 → 0.0849 |
| 128 | 1.2761 | 1.2857 | 0.1237 → 0.1090 |
| 512 | 1.2936 | 1.3861 | 0.1338 → 0.1008 |

返回 payload 为原来一半，新增的 FP32 权重通信与打包需计入成本；该轮结果未建立总延迟收益，当时没有切换默认路径。未进行异步/流水线改动，也未产出本轮 Nsight Timeline 或任何 overlap 结论。后续可用既有 `benchmarks/profile_stages.py --cuda-profile --stage-events off`，分别设置/取消三个返回实验环境变量，在相同四卡资源上采集 ROUTER/DISPATCH/A2A_FORWARD/EXPERT/A2A_RETURN/COMBINE 范围。

原始 JSONL、基线 logits 和完整日志保留在 H100 共享目录 `/SharedData/fuss/miniep-ep4-gpu15-20261016/workspace/return-20261020`；本地可审阅归档位于仓库 `artifacts/course13-return-20261020`。模型验收运行不用于性能结论。
