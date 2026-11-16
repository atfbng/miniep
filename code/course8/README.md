# Course 8：负载分析与 Profiling

Python 3.10+、支持 CUDA/NCCL 的 PyTorch，以及两张可见 NVIDIA GPU。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course8/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course8
python3 -m pip install -r requirements.txt
torchrun --standalone --nproc-per-node=2 test/demo.py --scenario hotspot --profile torch
```

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```

`--tokens` 表示全组原始 Token 数，`--num-experts` 表示全组 Expert 数。改变进程数时保持全局输入、路由和 dtype 一致。输出默认写入本课当前目录下的 `results/`。

空闲 H100 可能停在约 345 MHz，短 benchmark 会在时钟爬升过程中采样。做稳定测量时加
`--clock-warmup-ms 5000`，让每个 Rank 先执行不计时的矩阵乘预热；报告会记录测量前后的
SM 时钟。`--profile-repeats` 控制 profiled 前向次数，默认 3，做原始耗时统计时可设为 10。
每份报告同时保留该 Rank 的阶段中位数和 Profiler 算子时间。

Trace 由 `--trace-dir` 指定；已有 Nsight SQLite 导出可以用本地 `benchmarks/analyze_overlap.py` 分析。
