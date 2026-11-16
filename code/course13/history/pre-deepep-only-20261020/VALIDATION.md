> 本文件记录目录重组前的双卡验收。当前四卡支持与目录结构请看 [VALIDATION_EP4.md](VALIDATION_EP4.md) 和 [README](README.md)。
>
> 2026-10-18 更新：默认 Expert 路径已改为 Grouped SwiGLU，DP=1 仍使用本地 Combine。本文的零误差记录属于旧逐 Expert 路径；当前双卡性能与精度复测见 [PERFORMANCE.md](PERFORMANCE.md)。
>
> 当前快速路径已用 32 条 prompt × 64 Token 的 Token 级验收复测：FP32 回传 64/64 完整序列一致，层输出和 logits 均为零误差。BF16/FP16 回传未通过同一长序列门槛，只保留为显式实验选项。

# Course 13 实测验证报告

验证日期：2026-10-16。以下为实际运行结果；完整课程启动说明见 [README](README.md)。

> 2026-10-19 扩展复测：在 gpu3 的空闲 H100 GPU 6、7 上，使用 batch=1/2/4/8、32 条 prompt、每条 64 Token，得到 128/128 条完整序列一致、16,384 个 logits 位置、层与 logits 最大误差均为 0；另用 batch=2/4、每条 128 Token 得到 40/40 条序列一致和 10,240 个 logits 位置，最大误差 0。

## 环境与模型

- 单机两张 NVIDIA H100 80GB HBM3，最终整模型验收使用物理 GPU 5、7（容器内 0、1）。容器名：`vllm032`。
- 模型：`Qwen/Qwen3-30B-A3B`，BF16 非量化；48 层、128 Experts、Top-8，Hidden=2048、Expert Hidden=768。模型由 vLLM 原生加载，每 Rank 持有 64 个完整 Experts。
- TP=2、DP=1、EP=2、PP=1，eager；原生 MoE 后端固定 Triton。Mini-EP 执行真实 NCCL A2A、Triton SwiGLU、返回 A2A 与有序 Combine。
- vLLM 0.29.0、PyTorch 2.13.0+cu130、Triton 3.7.1、Transformers 5.16.1、Python 3.12；驱动 580.65.06、NCCL 2.30.7。
- 镜像：`swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.29.0`；实际镜像 ID：`sha256:082ca6f035279109041ffd3fe0695cb568b29bc580b35c4f297a66a08b216c1b`。
- 服务器 NVLS multicast 初始化报错，实验进程设置 `NCCL_NVLS_ENABLE=0`。两条路径均设置 `VLLM_MOE_SKIP_PADDING=0`，并关闭 cuBLAS BF16 reduced-precision reduction。
- [环境与模型文件记录](../../artifacts/course13_vllm_20261016/acceptance3/environment.json)、[全部 16 个权重分片 SHA-256](../../artifacts/course13_vllm_20261016/acceptance3/weights_sha256.json)。

## 结果

| 验证项目 | 实际结果 |
| --- | --- |
| 单元测试 | 12 项通过，包含 CPU Reference、Triton GEMM/SwiGLU/Combine、空输入和舍入回归 |
| EP=1，Triton 路径 | 48 组用例通过 |
| EP=2，PyTorch 路径 | 每 Rank 48 组用例通过 |
| EP=2，Triton 路径 | 每 Rank 48 组用例通过 |
| 真实模型逐层比较 | 两条短输入和两条长输入，两个 Rank、48 层，共 3840 次比较；最大输入批次 4135 Token；最大绝对误差 0 |
| 固定前缀完整词表 logits | 每 Rank 4096 个位置，合计 8192 次比较；最大绝对误差 0.0，最大相对 L2 误差 0.0，argmax 一致率 100% |
| 独立贪心生成 | 64/64 个完整 Token 序列一致，4096 个生成 Token 一致率 100% |
| OpenAI 兼容服务 | 健康检查、普通 Completion、流式 Completion、Chat Completion、四并发，共 7 个生成请求通过 |
| 服务后端及退出 | 两个 Worker 均记录 `Mini-EP runner: mode=miniep`；SIGINT 后正常退出，exit code 0，无强制 kill |
| 本课独立目录检查 | 语法、本地导入、依赖、路径和来源校验通过 |

模型验收使用固定 seed=13、32 条提示（16 条短输入，16 条约 2048 Token 输入），分别运行 batch=1 和 batch=4，每条生成 64 Token，关闭 thinking。逐层 compare 每条生成 8 Token，并返回原生输出以隔离误差传播。固定前缀阶段在记录真实 logits 后才强制原生 Token；独立生成阶段不强制 Token。

验收门槛保持为：BF16 层输出 `atol=0.02, rtol=0.02`；各位置 logits 相对 L2≤1%、cosine≥0.999，整体 argmax 一致率≥99%。没有通过放宽容差取得通过结果。cosine 的最小值约 0.99999970 来自 FP32 指标计算舍入；实际 logits 逐元素差为零。

[完整精度报告](../../artifacts/course13_vllm_20261016/acceptance3/report.json)包含各层结果、所有生成序列的一致率、两个 Rank 的 Mini-EP 调用次数和分支数量。每个 Rank 在 teacher/free 阶段分别执行了 122880 次 Mini-EP 层调用，覆盖全部 48 层。

## 复现入口

在匹配环境中进入本课目录，设置两张可用 GPU 和模型路径，再执行：

```bash
python3 -m pip install --no-deps --no-build-isolation -e .
python3 -m unittest discover -s checks -v
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=2 checks/distributed.py --backend reference
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=2 checks/distributed.py --backend triton
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=1 checks/distributed.py --backend triton
NCCL_NVLS_ENABLE=0 python3 test/validate.py --model "$MODEL_PATH" --output results/acceptance
NCCL_NVLS_ENABLE=0 python3 test/service_smoke.py --model "$MODEL_PATH" --output results/service
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_ep.py --output results/benchmark.json
```

本次模型挂载路径为 `/model`。最终精度验收复用了已经完成的 native 基线，重新运行 `--phases compare teacher free`；新环境使用上述完整命令即可。测试与服务日志、原生/独立生成输出保存在 [本次实验产物](../../artifacts/course13_vllm_20261016/)。完整 logits 和原始逐层记录保留在远端 `/SharedData/fuss/miniep-course13-20261016/results/acceptance3/`，未复制大体积 Tensor 文件到课程目录。

## Benchmark 与验证范围

合成工作负载：每 Rank 128 Token，H=2048、F=768、128 Experts、Top-8、EP=2、BF16；Triton Expert 和 Combine。warmup 3 次、测量 10 次，以每次最慢 Rank 的 CUDA Event 时间取中位数，完整 EP 路径为 **8.4017 ms**。详细阶段时间与配置见 [benchmark.json](../../artifacts/course13_vllm_20261016/benchmark.json)。该测试不含 Attention、KV Cache 和 TP 输出收集，不能作为 vLLM 服务吞吐，也没有据此声称加速。

本课验证范围为固定 vLLM 版本、BF16、双卡 TP=2/EP=2。EP=4、量化、CUDA Graph、异步通信重叠不属于本次验收；未采集 Nsight Systems Timeline，README 提供后续采集命令。

仓库全量静态检查在原有 `code/course4/test/demo.py` 的来源 SHA-256 校验处失败；本次没有修改该课。course13 的独立检查通过。失败原文保存在 [layout-all.log](../../artifacts/course13_vllm_20261016/layout-all.log)。
