# Course 3：GPU 执行与 GEMM 计时

Python 3.10+ 与 PyTorch；示例设备计时需要一张 CUDA GPU。可用 --device cpu --dtype float32 学习 CPU Reference。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course3/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course3
python3 -m pip install -r requirements.txt
python3 test/demo.py --device cuda --dtype float16
```

## Triton GEMM 与 Expert MLP

`test/demo_triton_gemm.py` 用一张 GPU 对比 PyTorch、固定 Triton 分块和 AutoTune
分块。GEMM 计算 `X[M,K] @ W[N,K].T -> Y[M,N]`；MLP 将上投影、PyTorch GELU
和下投影串起来，返回 `[M,K]`。Triton 内部使用 FP32 累加，输出保持输入 dtype
和行序。输入、权重均为连续行主序张量。Top-K 若已经展开，每行就是一条分支。

```bash
# 在 course3/ 中运行；单 GPU，不使用 torchrun。
python3 test/demo_triton_gemm.py --m 128 --n 2048 --k 1024 \
  --dtype float16 --backend both --operation both \
  --warmup 100 --repeats 100 --output results/triton_gemm.json

# 只看固定分块；非整齐形状展示行、列、K 维的 mask。
python3 test/demo_triton_gemm.py --m 17 --n 97 --k 65 \
  --backend fixed --operation gemm
```

`mini_ep/model/expert.py` 保存独立的 PyTorch Reference；
`mini_ep/kernels/gemm_triton.py` 保存 Kernel、固定封装和调优封装；
`benchmarks/benchmark_triton_gemm.py` 完成数值对照、预热和 CUDA Event 计时。
固定版使用 `16×32×32 / 4 warps / 3 stages`，调优版按形状从 14 组候选配置中选择。
MLP 的上下投影分别调优，启动时打印两份配置。首次编译与调优发生在正式计时前。
JSON 保存输入配置、环境、数值误差、选定参数和每次耗时。

## Proton 分阶段分析

用 Python 启动分析入口，脚本先完成调优、数值检查和预热，再启动 Proton。
报告按实现拆成 `UP / GELU / DOWN`；每段累计 GPU Kernel 时间除以 repeats，
得到平均单次耗时。`bytes` 是张量字节数估算。

```bash
python3 -m pip install -r requirements-profile.txt
python3 benchmarks/profile_mlp.py --m 2048 --n 2048 --k 1024 \
  --backend tuned --dtype float16 --warmup 100 --repeats 100 \
  --output lesson03_mlp_warm
proton-viewer -m time/us lesson03_mlp_warm.hatchet
```

相邻的 `.json` 保存输入、环境、数值对照和上下投影的选定配置。

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```
