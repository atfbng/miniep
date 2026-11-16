# 第 12 课 Triton 版本交付说明

本次将课程 12.4～12.8 改为 Triton P2P 的输入 Tile 发送、就绪协议、TMA GEMM、直接回传与缓冲区复用。12.1～12.3 的 NCCL Stream 流水线保留。原 CUDA C++ / NVSHMEM 内容移出课程，既有源码与历史实测继续保存在仓库和原报告中。

## 性能结果

计时覆盖预先打包的输入到原始 Token 顺序 FP32 加权输出；不含路由、打包、行数交换和对称内存初始化。Triton 工作区预分配，原同步 NCCL 基线沿用临时 Tensor 分配。每个路径预热 10 次，正序、逆序各测 30 次，汇总 60 个逐迭代最大 Rank CUDA Event 样本的中位数。单位 ms。

| EP / Expert 数 | N / H / F | 同步 NCCL | Triton 分离 | Triton 融合 | 融合相对同步降低 |
|---|---|---:|---:|---:|---:|
| 2 / 8 | 512 / 2048 / 4096 | 0.394 | 0.192 | 0.221 | 43.9% |
| 4 / 8 | 512 / 2048 / 4096 | 0.363 | 0.223 | 0.246 | 32.3% |
| 4 / 16 | 512 / 2048 / 4096 | 0.431 | 0.213 | 0.250 | 42.0% |
| 2 / 8 | 1024 / 4096 / 14336 | 1.045 | 0.885 | 0.897 | 14.2% |
| 4 / 8 | 1024 / 4096 / 14336 | 1.068 | 0.927 | 1.013 | 5.1% |
| 4 / 16 | 1024 / 4096 / 14336 | 1.186 | 0.982 | 1.017 | 14.2% |
| 4 / 32 | 1024 / 4096 / 14336 | 1.323 | 1.132 | 1.126 | 14.8% |


融合版本在七组配置中均快于原同步 NCCL + Expert Loop 基线；大矩阵降低 5.1%～14.8%。优化分离版在七组中的六组更快，剩余一组差距约 0.5%，不足以判断稳定优势。本实现展示了输入通信与 Up 的设备侧协作，以及 Down 直接回传；当前结果不支持融合普遍优于优化分离版的结论。

EP=4、E=16 的大矩阵中，独立两次 TMA GEMM 与 GELU 为 0.847 ms，独立输入 push 为 0.076 ms，独立输出 push 为 0.073 ms。通信分项不含完整 start/done、Combine 与准备阶段。独立分项的缓存状态、同步边界和整层路径不同，不直接相加解释总耗时。E=8/16 大矩阵复用先前四路测试的输入文件，E=32 与中矩阵使用固定种子生成数据；横向公平性以每行内比较为单位。

## 正确性与时间线

完成 7 组性能配置及 10 组边界配置，共 50 个 Rank 配置。同步、Triton 分离与融合均对齐独立 Reference；最大绝对误差为 0.000251770，使用 atol=0.003、rtol=0.02。覆盖 EP=1/2/4、Top-1/2、空 Expert、单 Token、高度倾斜、非整 Tile、H/F 为 8 的倍数但非 Tile 整数倍、非均匀权重及来源区间跨 Tile 边界。测试包含连续排队执行、输入更新及输出复制，检查值、shape、dtype 和 device。

Nsight Systems 记录 EP=4、E=16 的大矩阵：每种路径包含 12 个逐 Rank NVTX 测量区间（4 Rank × 3 次）。各区间的 Kernel 数见附件 analysis.json；融合路径为主通信/MLP Kernel 与 Combine 两个 Kernel，优化分离路径为七个 Kernel。Nsight 时间线可验证 Kernel 提交边界；单个融合 Kernel 内通信与计算的硬件活动需要进一步的 SM/NVLink 指标。

GPU Metrics 权限探测返回 ERR_NVGPUCTRPERM。内部 exposed communication time 与 overlap ratio 记为 N/A；没有把 Kernel 时间段重叠或设备阶段时间戳换算为已测得的内部重叠率。后续可在具备计数器权限的环境采集 NVLink 与 SM 活动，并与 Tile 发布/消费时间戳联合分析。

## 环境与复现

NVIDIA H100 80GB HBM3，同节点 NVLink；PyTorch 2.12.0a0+0291f960b6.nv26.04.48445190，CUDA 13.2，Triton 3.6.0，NCCL 2.29.7。使用物理 GPU 2、3、6、7；双卡配置使用其中 6、7。容器 CPU 配额 4 核、内存 16 GiB；GPU 上存在其他进程的常驻显存，测量时非独占。各 Rank JSON 记录设备 UUID、原始样本与配置。

**运行基准 · benchmarks/benchmark_triton_peer.py**

```bash
torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_triton_peer.py --tokens 1024 --hidden-size 4096 --expert-hidden-size 14336 --experts 16 --top-k 2 --warmup 10 --repeats 30 --output /results/triton-example
```

完整验收命令见附件 final-sweep.sh、final-edges.sh、final-profile.sh，执行输出见同名 log。脚本中的既有输入路径适用于原测试容器；使用上面的无 prefix 命令可由固定种子重新生成输入。课程 Python 代码块全部通过 AST 语法检查，Triton 摘录由实际测试源码提取；本地未配置 GPU 环境，可执行验证均在 H100 完成。

最终源码 SHA256 见 profile/source.sha256，与课程摘录及本地交付源码一致。JSON 的 compile 字段来自开启阶段时间戳的 PROFILE=True 变体；性能样本使用 PROFILE=False，不以该字段解释 steady-state 寄存器或 spill 开销。

使用的官方资料：[PyTorch Symmetric Memory](https://docs.pytorch.org/docs/main/symmetric_memory.html)、[Triton Persistent Matmul](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)、[NVIDIA PTX 访存代理](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#proxies)。
