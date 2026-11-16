# Course13：把 Mini-EP 接到真实 Qwen MoE

这是前十二课的 vLLM 接入扩展，允许 Attention TP=2、DP=1/2 与 MoE EP=2/4；基础课程的 Pure EP 不变。唯一精度对照是 **vLLM 0.29.0 的 `deepep_high_throughput`**。正式整模型验收使用真实 Qwen3-30B-A3B BF16、TP=2/DP=2/EP=4。完整结果见 [DEEPEP_REPORT.md](DEEPEP_REPORT.md)，精度范围见 [EXACT_ALIGNMENT.md](EXACT_ALIGNMENT.md)；历史数据适用范围见 [history](history/README.md)。

本轮核心死代码清理、源码等价审计及新增复验见 [CORE_CLEANUP.md](CORE_CLEANUP.md)。历史完整验收保留原版本标识，不作为新版本已重跑全矩阵的声明。

## 沿前课继续读代码

```text
integration/runner.py::route_tokens       框架 Gate/Top-K → RouterOutput
model/moe.py::ep_forward
  distributed/dispatch.py                稳定目的 Rank 打包
  distributed/all_to_all.py              NCCL Tokens/IDs/FP32 weights
  model/expert.py / kernels/grouped_gemm  本地 Expert-major 两次投影
  distributed/all_to_all.py              返回 A2A，默认 tokens.dtype
  kernels/combine_triton.py              逆排列 → owner 部分和 → 合并
integration/input_layout.py              必要的 TP 输入输出适配
```

`RankPackedTensors`、`pack_by_destination_rank`、`ExpertParallelTopology`、`variable_split_all_to_all`、`GroupedGemmPlan`、`compute_grouped_local_experts` 沿用前课的角色和命名。文件实体均在本课，无跨课程导入或符号链接。具体函数、布局及必要差异见 [TEACHING.md](TEACHING.md)。

`mini_ep/` 只包含核心和框架适配；独立 PyTorch Reference、shadow 对照、完整 logits、失败重放、实验编排和计时都在 `reference/`、`validation/`、`checks/`、`test/`、`benchmarks/`。

## 默认数值路径

```text
BF16 输入 → gate/up → BF16 SiLU 再乘 up
→ down FP32 累加 → 乘 FP32 routing weight
→ 转 tokens.dtype → 返回 A2A
→ 分支模型 dtype 舍入 → 每个 owner 按 slot 顺序 FP32 累加
→ owner 部分和转模型 dtype → EP rank 升序 FP32 合并 → 模型 dtype
```

每个分支只加权一次，Combine 接口不接收 routing weights。没有原生 AllGather/ReduceScatter 的 owner 兼容归约或策略切换。必要的 TP 行重组、模型自身的 Attention/DP/SP 通信仍由其所属层处理。

核心支持 FP32/BF16 激活，集成层当前只支持 BF16 模型。返回 dtype 默认由输入推导；`MINIEP_RETURN_DTYPE=fp32/fp16/bf16` 仅为显式传输诊断，不能据此宣称 FP16 模型受支持。Grouped down 默认融合 FP32 加权及转换；核心 `fuse_return_weight=False` 与 PyTorch Reference 保留独立对照。详见 [RETURN_TRANSPORT.md](RETURN_TRANSPORT.md)。

## 安装和启动

需要支持真实 NCCL 的 NVIDIA GPU 环境。整模型 DeepEP 对照需要四张 GPU、已安装的匹配 DeepEP HT 二进制和本地真实权重。固定依赖见 `requirements.txt`；额外 DeepEP 版本/镜像必须随结果记录，不能仅凭后端参数认定已生效。

```bash
cd code/course13
python3 -m pip install --no-deps -e .
# MODEL_PATH 指向真实 Qwen3-30B-A3B 权重目录；四张可见 GPU。
torchrun --standalone --nproc-per-node=4 test/demo.py   --model "$MODEL_PATH" --backend miniep --output results/demo-miniep
```

真实 DeepEP 验收会检查 `DeepEPHTAll2AllManager`、`DeepEPHTPrepareAndFinalize`、`TritonExperts` 和原始 `MoERunner`。不接受 DP=1 的 NoDPEP 路径作为 DeepEP；双卡 Mini-EP 仅保留教学输入分片及 TP 重组能力。Qwen3.5 的历史结果不证明本次 Qwen3 精度结论，当前限制见 [QWEN35.md](QWEN35.md)。

## 独立检查

```bash
python3 -m unittest discover -s checks -p 'test_*.py'
# 两张 GPU；可将进程数换成 1 或 4。NCCL 配置按节点条件设置。
torchrun --standalone --nproc-per-node=2 checks/return_transport.py --output results/core-ep2
# 四张 GPU，全量、独立原生重复运行、逐层/全词表/自由生成检查。
python3 test/deepep_alignment.py --model "$MODEL_PATH" --output results/exact-deep4
```

结果目录必须为新目录，不能覆盖旧实验。`--smoke` 仅检查工具链，不作为全量验收。具体配置、失败保存、重复启动、不同随机种子和缺失证据拒绝条件见 [EXACT_ALIGNMENT.md](EXACT_ALIGNMENT.md)。

## 性能

精度诊断与测量分离。正式测量关闭 shadow、逐层张量落盘和验证日志；warmup 后重复测量，同口径报告最慢 Rank 的 prefill/decode/端到端延迟和吞吐，实际通信 dtype 和字节单独核对。返回字节减少不等于已经加速。命令及证据见 [PERFORMANCE.md](PERFORMANCE.md)。保留六个核心 NVTX 范围；当前同步实现没有 Chunk pipeline 或已证实的通信计算重叠。
