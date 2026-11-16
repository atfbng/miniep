# Course 13 四卡扩展与目录整理验收

完成日期：2026-10-18（Asia/Shanghai；实验从 10 月 16 日开始）。

> 2026-10-19 扩展复测：在 gpu5 的共享 H100 GPU 0–3 上运行 8 类场景（balanced、uneven、双侧空闲、单 Token Decode、宽 prefill、长 decode、长 prefill、2040 Token max-prefill）。四个 Rank 各覆盖 48 层，22/22 条独立生成序列、314/314 个 Token 与原生 EP 一致；层输出与 logits 最大误差为 0，argmax 一致率 100%。

> 2026-10-18 复测：未设置 `MINIEP_OWNER_PARTIALS` 的四阶段验收在 15 的 GPU 0–3 通过。四个 Rank 均自动选择 owner-partial，48 层输出与 logits 误差为 0，独立生成 14/14 一致，严格模式与 Token 模式均通过。DP=1 仍使用本地 FP32 Combine。

## 实现与目录

支持 vLLM 0.29.0 下的 **TP=2、DP=2、EP=4**，同时保留 TP=2、DP=1、EP=2。模型为 Qwen3-30B-A3B BF16，128 Experts 在四个 Rank 上各放置 32 个，Attention 仍使用两卡 TP。

`mini_ep/` 仅包含推理核心、NCCL 通信、Triton Kernel、vLLM 接口及 NVTX。独立 PyTorch 实现在 `reference/`；逐层比较、logits/teacher forcing 和报告在 `validation/`；测试在 `checks/` 与 `test/`；CUDA Event 计时在 `benchmarks/`。生产核心不导入外围目录，边界由静态单元测试检查。代码使用与前面课程相近的小函数和显式布局变换。

vLLM 在四卡模式下已按 TP 切分 MoE 输入，Mini-EP 不重复切分。完整 Token 分发、Expert 计算、返回 A2A 均由 Mini-EP 实现。DP=2 默认保留 owner 部分和并调用原生 NCCL 归约，以保持 BF16 求和顺序；DP=1 使用本地 FP32 Combine。本地 Combine 的四卡对照只有 12/14 条完整序列一致，因此保留为 `MINIEP_OWNER_PARTIALS=0` 实验选项。

## 实际结果

| 项目 | 结果 |
| --- | --- |
| 单元测试 | 16 项通过，包括核心依赖边界、Reference、Triton 和精度回归 |
| 四卡 NCCL TP=2/DP=2/EP=4 | 每 Rank 72 组通过；含 Top-1/2/8、FP32/BF16、不同 H/F、不同 DP 长度及空源 Rank |
| Qwen 四卡逐层比较 | 四个 Rank、48 层，共 18432 次比较，最大绝对误差与相对 L2 均为 0 |
| 同前缀完整 logits | 共 432 次比较，最大绝对误差与相对 L2 均为 0，argmax 一致率 100% |
| 独立贪心生成 | 两个 DP 合计 14/14 个序列、216 个 Token 与原生 EP 一致 |
| 默认路径复测 | 未设置 `MINIEP_OWNER_PARTIALS` 时四个 Rank 自动选择 owner-partial；严格模式与 Token 模式均通过 |
| DP 空闲/提前结束 | DP0/DP1 分别完全空闲、DP0 提前结束、两侧请求数与长度不同均通过 |
| 四卡服务 | 四个 Worker 均确认 TP=2/DP=2/EP=4；健康、普通、流式、聊天和四并发请求通过；正常退出 |
| 双卡回归 | 384 次真实模型层比较误差为 0；双卡核心每 Rank 48 组通过 |
| 基准入口 | 小尺寸 smoke 通过，只验证计时入口可用，不作性能比较 |

四卡验证包含 6 类工作负载：balanced、uneven、idle_dp0、idle_dp1、single_decode、long_prefill。每侧最多两条请求，长输入为 512/768 Token，每条最多生成 16 Token；uneven 中 DP0 只生成 8 Token。logits 的 432 次比较包含 TP 复制结果（Rank 0/1 各 104 次，Rank 2/3 各 112 次）。独立生成统计只计每个 DP 一份输出。

门槛保持为 BF16 层输出 `atol=0.02, rtol=0.02`；logits 相对 L2≤1%、cosine≥0.999、argmax 一致率≥99%。比较的是真实 logits，teacher forcing 仅在记录指标后控制下一 Token。没有修改原生 Expert 计算作为基线。

本次默认路径结果：[严格模式报告](../../artifacts/course13_p0_4gpu_20261018/no-override-gpu15/report-strict.json)、[Token 报告](../../artifacts/course13_p0_4gpu_20261018/no-override-gpu15/report-tokens.json)、[结构测试日志](../../artifacts/course13_p0_4gpu_20261018/structure.log)、[双卡回归日志](../../artifacts/course13_p0_2gpu_20261018-gpu13/)。

早先四卡服务与精度记录：[四卡精度报告](../../artifacts/course13_ep4_20261016/ep4/report.json)、[四卡环境](../../artifacts/course13_ep4_20261016/ep4/environment.json)、[设备与模型文件记录](../../artifacts/course13_ep4_20261016/hardware.json)、[服务验证](../../artifacts/course13_ep4_20261016/service4/requests.json)。

## 运行环境与命令

- 本次四卡复测：`172.16.240.15`，容器 `vllm033`，物理 GPU 0–3；均为 NVIDIA H100 80GB HBM3。
- 测试前四张卡各有约 29.7 GiB 常驻占用，剩余约 47.57 GiB，GPU 利用率快照为 90%–100%。Mini-EP 使用剩余显存与已有任务共用设备，未停止其他任务，因此结果只用于正确性验收。
- 模型每 Rank 权重约 14.96 GiB；本次设置 `gpu_memory_utilization=0.45`、KV Cache 256 MiB、最大上下文 1024、最大批次 Token 2048。早先服务验收使用 GPU 0、1、4、7 和 `gpu_memory_utilization=0.23`。
- vLLM 0.29.0、PyTorch 2.13.0+cu130、Triton 3.7.1、Transformers 5.16.1，Python 3.12。
- 两条路径均使用 eager、同步调度、BF16、原生 Triton MoE 配置；`NCCL_NVLS_ENABLE=0`、`VLLM_MOE_SKIP_PADDING=0`。四卡使用 vLLM 自动启用的 MoE sequence parallel，不启用 DBO/EPLB/量化/推测解码。
- 双卡回归在 `.13` 的空闲 GPU 4、5 上进行。

完整启动命令见 [README](README.md)。本次主要命令为：

```bash
python3 -m unittest discover -s checks -v
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 checks/tp_dp.py
# phase 依次为 compare、native、teacher、free：
for phase in compare native teacher free; do
  NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 test/dp_validate.py \
    --model /model --output /results/ep4 --phase "$phase" \
    --gpu-memory-utilization 0.45 --kv-cache-mib 256 --max-tokens 16
done
python3 test/report_dp.py --output /results/ep4 --acceptance strict
python3 test/report_dp.py --output /results/ep4 --acceptance tokens
NCCL_NVLS_ENABLE=0 python3 test/service_smoke.py --model /model \
  --output /results/service4 --dp-size 2 --gpu-memory-utilization 0.23 --kv-cache-mib 256
```

本次原始运行日志和完整 logits 保留在 `.15` 的 `/SharedData/fuss/miniep-ep4-gpu15-20261016/results/p0-20261018/`；小型报告和日志收集到 `artifacts/course13_p0_4gpu_20261018/`。双卡回归在 `.13` 的空闲 GPU 4、5 上完成，日志收集到 `artifacts/course13_p0_2gpu_20261018-gpu13/`。共用设备上的结果仅用于正确性验收，未做吞吐、overlap 或性能提升结论。

## 交付副本

本次运行副本位于 **172.16.240.15:/SharedData/fuss/miniep-ep4-gpu15-20261016/workspace/p0-20261018/course13**；容器 `vllm033` 内路径为 `/workspace/p0-20261018/course13`。复制包含核心、Reference、验证、测试、基准、依赖与文档；不复制 Python 缓存和旧运行输出。

本课可整体复制独立运行。`python3 tools/check_course_layout.py --courses 13` 检查通过。
