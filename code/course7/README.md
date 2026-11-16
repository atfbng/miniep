# Course 7：EP 扩展与固定全局负载

Python 3.10+、支持 CUDA/NCCL 的 PyTorch，以及与 EP 大小对应的 1、2 或 4 张可见 NVIDIA GPU。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course7/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course7
python3 -m pip install -r requirements.txt
torchrun --standalone --nproc-per-node=2 test/demo.py --tokens 1024 --num-experts 8
```

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```

`--tokens` 表示全组原始 Token 数，`--num-experts` 表示全组 Expert 数。改变进程数时保持全局输入、路由和 dtype 一致。输出默认写入本课当前目录下的 `results/`。

## EP=1/2/4 实测实验

完整比较需要四张可见 GPU；`--ep-sizes 1 2` 可只运行单卡和双卡。
两个固定全局负载均采用八个 Expert、Top-2、均匀路由与 FP16：

| 负载 | 全组 Token 数 T | 输入宽度 H | MLP 中间宽度 F | 每 Expert 输入行数 M |
|---|---:|---:|---:|---:|
| small | 1024 | 128 | 256 | 256 |
| large | 8192 | 1024 | 2048 | 2048 |

从当前课目录运行；外层脚本自行启动并等待各组 `torchrun`：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_NVLS_ENABLE=0 \
python3 benchmarks/run_scaling_experiment.py \
  --cases small large --ep-sizes 1 2 4 \
  --trials 3 --warmup 50 --repeats 100 \
  --output-dir results/scaling

python3 -m pip install -r requirements-plot.txt
python3 benchmarks/plot_scaling_experiment.py \
  --input results/scaling/summary.json --output-dir results/scaling/figures
```

每次独立实验先与 Reference 对齐，再测量 100 轮；每轮取最慢 Rank，
一轮实验取这些最大值的中位数，三次独立实验再取中位数。
图中的误差线是三次实验中位数的最小值与最大值。
不同 EP 的配置、全组路由指纹和逐 Expert 计数必须一致。
每对来源/目标的 `send_splits` 与 `recv_splits` 必须匹配。

`raw/<case>/epP/trialI-rankR.json` 保存各 Rank 的原始样本、
阶段样本、发送/接收 splits、设备及软件环境；相邻 `.log` 保存进程日志。
`reports.json` 汇集 Rank 0 报告，`summary.json` 给出延迟、吞吐、加速比、
并行效率和跨 Rank 向量载荷。阶段时间独立测量，不能累加为整层延迟。
2026-10-14 的 H100 实测数据保存在 `results/h100_scaling_20261014/scaling/`。

## 拆解 Dispatch 耗时

```bash
python3 benchmarks/run_scaling_experiment.py --dispatch-breakdown \
  --cases small large --ep-sizes 1 2 4 --trials 3 --warmup 50 --repeats 100
python3 benchmarks/analyze_dispatch.py
python3 benchmarks/plot_dispatch.py
```

`--dispatch-breakdown` 在完整前向中分别测量 Expert 打包、Rank 打包、
计数 AllGather 和接收布局构造，并用 Torch Profiler 统计各区间的
`aten::item` 调用。CUDA Event 区间包含 Host 提交间隔和数据等待；
这些是准备阶段的延迟，不是纯 GPU Kernel 时间。
子阶段增加了细分计时开销；子阶段图用于定位构成，原始路径与对照路径
的性能比较统一采用原有粗粒度 DISPATCH 计时。

分析时每轮选取 DISPATCH 最慢 Rank，四个子阶段全部来自同一 Rank。
对这些配对样本求均值后，再绘制可相加的分解图，剩余区间单独标记。
对照保持 GPU Token、Expert 和两次 A2A 不变，只将小计数矩阵批量搬到 CPU，
复用 Reference 的布局推导，再把 int64 splits 和行索引搬回原 GPU。
对照与原始前向的输出、路由结果、splits 和 Expert 处理行数必须完全一致。
它是独立的诊断对照，不会替换默认前向实现。

本次测量保存在 `results/h100_dispatch_breakdown_20261015/scaling/`，
`dispatch_summary.json` 给出分解与对照数据；`raw/` 保留每个 Rank 的
子阶段 GPU/Host 样本、Profiler 标量读取统计与一致性检查结果。

## 查看图表对应的布局与指标示例

以下示例在 CPU 上调用本课 Router，打印计数与布局数据；性能测量继续使用前面的 NCCL 实验入口。

```bash
python3 test/demo_scaling_layout.py --show placement
python3 test/demo_scaling_layout.py --show routes --token-id 10
python3 test/demo_scaling_layout.py --show strong --single-ms 4 --multi-ms 2.5 --compare-ep 2
python3 test/demo_scaling_layout.py --show weak --tokens-per-rank 32
python3 test/demo_scaling_layout.py --show communication --hidden-size 8 --element-bytes 4
python3 test/demo_scaling_layout.py --show timing
```

它们分别对应输入/Expert 放置、T10 往返、强扩展指标代入、弱扩展批次、跨卡载荷和逐轮最慢 Rank。`--single-ms` 与 `--multi-ms` 是指标例子的输入值。
