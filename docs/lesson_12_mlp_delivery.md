# 第十二课 MLP 通算融合复验

正文与配套实现统一使用两次矩阵投影和 GELU，移除了缩放 Expert、`expert_mode` 和 `mode` 分支选择。设备侧 Kernel 在输入就绪后直接调用 `expert_mlp_row()`，随后返回结果；NCCL Reference 与三路线基准使用相同 MLP。结果 JSON 以 `expert="mlp"` 标记计算类型。`input_scale` 是逐轮变化输入的测试元数据，用于检查双缓冲数据更新，与 Expert 计算类型无关；性能基准使用固定输入。

## H100 正确性与代码一致性

在 H100 80GB、CUDA 13.2、PyTorch 2.12.0a0、NVSHMEM 3.6.5、CUTLASS 3.9.2 上重新编译并执行。使用既有独立容器，4 CPU、16 GiB 内存配额，GPU 为共享环境；NVSHMEM 对称堆设置为 64 MiB，同节点传统 P2P 路径。

|检查|结果|
|---|---|
|CUDA 重新编译|成功，CUTLASS 的 double4 弃用提示保存在编译日志|
|NVSHMEM EP=2/4、Top-1/2、均匀/倾斜、空块、六轮双缓冲，固定与变化输入|288 项 PE/轮次检查通过|
|同步 NCCL 与 NVSHMEM 的值、send/recv 行数和 origin|48 项 Rank/配置检查通过|
|同步 NCCL、Chunk NCCL、NVSHMEM，同输入 MLP 对照|24 项 Rank/配置检查通过|
|本地与 H100 三个改动源码文件的 SHA256|全部一致|
|正文 CUDA 摘录与实际编译源码|一致|
|Nsight Systems MLP 路径|每 PE 24 次 exchange_compute_return 和 24 次 combine_returned|

三路线使用每 PE 七个 Token，E=P、H=F=4、FP32，固定输入、路由、权重；NCCL 预热 2 次，NVSHMEM 丢弃第一轮，均取五个样本，逐轮使用最慢 Rank/PE。下面列出 Top-2、倾斜路由的中位数。

|EP|同步 NCCL ms|Chunk NCCL ms|NVSHMEM ms|Kernel 数（Rank 0）Sync/Pipeline/NVSHMEM|
|---|---:|---:|---:|---|
|2|0.248768|1.160800|0.029952|12/42/2|
|4|0.389056|1.711360|0.047616|14/42/2|

计时从已打包输入开始，到原 Token 顺序的加权输出结束，排除路由、打包、计数交换与 Host 结果拷贝。NCCL 的临时 Tensor 分配计入调用，NVSHMEM 使用预分配工作区。NVSHMEM 的两个循环 Kernel 已由 Trace 核对；库初始化、对称分配和阶段外 barrier 独立出现在时间线上。上述小矩阵数据对应设备交接的准备后执行区间。独占 GPU 性能复测未执行，Stream 流水线调度本轮未修改，沿用此前完整前向及重叠分析证据。

## 命令与原始证据

完整命令见附件中的 `h100_validate.sh`，包括 nvcc 编译、EP=2/4 的 MPI 运行、torchrun 对照、三路线基准及 Nsight 采集与 SQLite 导出。主要入口如下：

```bash
mpirun -np 2 ./token_exchange results/fixed2
MINIEP_VARY_ROUNDS=1 MINIEP_DEVICE_TIMESTAMPS=1 mpirun -np 4 ./token_exchange results/varied4
torchrun --standalone --nproc-per-node=4 examples/12_nvshmem_reference.py --prefix results/varied4
torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_fusion.py --prefix results/fixed4 --output-dir results/fusion
```

启动环境变量、库路径、进程超时、完整参数与结果均随附件提供。正文保留必要运行条件，机器状态与验证经历集中在此说明。第 1～11 课及其代码未因本次调整而变更。源码附件同步更新第十二课正文、三个改动实现、运行说明和 SHA256SUMS；此前验收附件保留历史版本证据，本次 MLP 复验附件对应当前实现。
