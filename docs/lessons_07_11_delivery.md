# 第 7～11 课交付与 H100 验收

五课已写入原有知识库，配套源码 ZIP 位于课程首页。正文与独立源码的 14 个 Python 代码块经 AST 比对一致，H100 导入通过；飞书回读确认代码、标题与六张图示完整。

## 环境与正确性

H100 80GB，4 GPU；PyTorch 2.12.0a0、CUDA 13.2、Triton 3.6.0、NCCL 2.29.7、Python 3.12.3。GPU 按独立容器使用，未停止已有任务。FP32 关闭 TF32；NCCL_NVLS_ENABLE=0 使用普通 NCCL 路径，默认 NVLS 初始化在该服务器报错。

`python -m unittest tests.test_advanced_lessons -v`：7 项通过，GPU 部分含 36 个子用例，覆盖空输入、空 Expert、非连续输入、倾斜路由、边界尺寸和 FP32/FP16/BF16。

`NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=P examples/11_grouped_ep.py`，P=1/2/4：逐项替换 Dispatch、Combine、Grouped Compute 及三项组合，Top-1/Top-2、FP32/BF16，与单 GPU Reference 对齐；每 Rank 20 项，共 140 项通过。

第七课 `examples/07_ep_scaling.py` 在 EP=1/2/4、T=1024、H=128、F=256、E=8、K=2、FP32 上通过。第八课生成两份 PyTorch Rank Trace 和 Nsight Systems .nsys-rep/SQLite；六个阶段 NVTX 标记均存在。

## 算子基准

实际运行 examples/09_triton_dispatch.py、10_triton_combine.py、11_grouped_gemm.py（--kind grouped 与 --kind mlp）。统一 T=257、H=128、F=256、E=8、K=2、FP32，预热 5 次，20 次 CUDA Event 样本，中位数；完整输出见 JSON。

|范围|Reference ms|Triton ms|GPU Kernel 数 Reference/Triton|
|---|---:|---:|---:|
|dispatch|0.281568|0.272592|21/20|
|combine|0.043760|0.029312|4/1|
|grouped|0.189440|0.038608|8/1|
|mlp|0.349808|0.070448|32/3|

Grouped 调度准备约 0.056 ms，计时独立于计算。Grouped GEMM/MLP 数字对应预先准备好的布局、权重与调度；完整 EP 回调还包括打包、权重堆叠与恢复。共享服务器与 4 CPU 容器配额会影响调度延迟，以上结果限定在本次配置；课程正文不将其写成跨配置加速结论。

## 未执行项

未执行面向其他显卡和大规模集群的测试；课程范围内的 H100 正确性、示例、分布式与 Profiling 均已执行。第十二课将按重新读取的目标另行编写验收。
