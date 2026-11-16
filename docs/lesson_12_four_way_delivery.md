# H100 四路大矩阵与多专家对照

## 四路大矩阵与多专家实测

四条路径已经在 H100 上完成同输入实测：普通同步 NCCL、双 Stream 分块流水线、NVSHMEM 分离执行，以及单 Kernel 通算融合。多专家主测试覆盖 EP=2/4、总计 8/16 个 Expert、每卡 2/4 个 Expert。较大配置使用 H=4096、F=14336、每卡 1024 个原始 Token。

这组六个多专家配置中，普通同步版延迟最低；Stream 版出现了通信与 GEMM 重叠，同时承担分块后的额外计算和调度开销，整体仍慢于普通版。NVSHMEM 分离与单 Kernel 融合的耗时接近，合并 Kernel 没有带来大矩阵上的收益。

### 四条执行路径

|路径|通信与调度|本地 Expert 计算|
|---|---|---|
|普通同步版|整批 Forward NCCL A2A → Expert → Return NCCL A2A → Combine|PyTorch CUDA GEMM，按本地 Expert 循环|
|Stream 分块版|独立通信与计算 Stream，分块异步 A2A，Event 与 Work 连接依赖|与普通版共用相同的 Expert 循环，最后 drain 并 Combine|
|NVSHMEM 分离版|发送、Up＋GELU、Down、回传、Combine 分别启动，共五个 Kernel|CUTLASS 分块 GEMM|
|单 Kernel 融合版|一个 Kernel 内完成发送、Up＋GELU、Down、回传；Combine 独立，共两个 Kernel|与 NVSHMEM 分离版共用相同 CUTLASS 计算、Tile 分配和缓冲区|

普通版与 Stream 版共用 `expert_mlp()`，其输入按来源 Rank 保存；每个本地 Expert 取出自己的行，完成 Up → GELU → Down，再写回相同的接收行位置。Stream 版的通信顺序为 F0、F1、R0、F2、R1……，其中 Fi/Ri 分别表示第 i 个 Chunk 的发送/回传。计算 Stream 等待 Fi 的 Work 完成后消费数据，回传 Stream 等待该 Chunk 的计算 Event。全部 Expert 计算提交后再等待返回并合并，函数结束时将两条 Stream 接回调用方 Stream。每个 Chunk 独占缓冲区，测量边界等待整轮完成后才复用。

NVSHMEM 路径由一个 CTA 推进通信，其余 128 个 CTA 执行 CUTLASS 3.9.2 的 Tensor Core GEMM。Threadblock Tile 为 64×64×32，采用 SM80 兼容路径、FP16 操作数与 FP32 累加；H 和 F 为 8 的倍数。融合主体各阶段通过 Cooperative Grid 同步，输入就绪后执行 Up，Up 完成后执行 Down，随后回传。中间激活保存在全局内存中。分离版调用相同计算组件，保留阶段间 Kernel 边界。

因此，两组共用实现的对照分别衡量 Stream 分块调度、Kernel 合并的影响；跨组结果反映两条完整实现的实际性能，也包含通信方式、数据布局和 GEMM 实现的差异。

### 多专家布局与矩阵形状

令 P 为 EP Size、E 为 Expert 总数，均匀放置时每卡 Q=E/P 个 Expert，配置检查要求 E 能被 P 整除。全局 Expert e 映射到 Rank e//Q，本地 Expert ID 为 e%Q。每卡原始输入为 FP16、CUDA、连续布局的 [N,H]；权重分别为 [Q,F,H] 和 [Q,H,F]，最后两维采用 [out,in] 顺序。

测试使用均匀、等权 Top-2 路由，所有分支都参与返回与加权合并。每个 Expert 的总接收行数 M=N×K/Q；Up 的矩阵为 [M,H] × [H,F]，Down 为 [M,F] × [F,H]。N=1024 时，每卡四个 Expert 对应 M=512，每卡两个 Expert 对应 M=1024。Stream 的 Chunk=256 时，对应每个 Expert、每个 Chunk 的行数分别为 128 和 256。

NCCL 载荷紧凑地按目标 Rank、本地 Expert 排列，接收后为来源 Rank、本地 Expert 顺序。每个 Chunk 的 send/recv split 长度为 P，单位是 Token 行；另有 [P,Q] 的 Expert 行数元数据用于建立本地 Expert 索引。NVSHMEM 使用 FP16 对称缓冲区 [P,Q,C,H]，C=N×K 为每个来源/目标与 Expert 分块的容量，只传输有效行。接收与计算按 [source,local_expert,row,H] 保存，返回按 [expert_owner,local_expert,row,H] 保存。restore 将行还原到原 Token 的 Top-K 分支顺序，再用 FP32 按 1/K 加权求和，输出为 CUDA FP32 [N,H]。

### 大矩阵多专家结果

以下单位为 ms，Stream 主表固定 Chunk=256。各路径每个配置均预热，并收集 60 个有效样本；每轮先取最慢 Rank/PE 的 CUDA Event 延迟，再报告中位数。

|EP|Expert 总数|Expert/卡|N/卡|H|F|每 Expert 行数 M|普通同步|Stream 256|NVSHMEM 分离|单 Kernel 融合|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|2|8|4|512|2048|4096|256|0.4762|1.4678|2.4912|2.5321|
|4|8|2|512|2048|4096|512|0.4444|1.2406|2.7050|2.7160|
|4|16|4|512|2048|4096|256|0.5806|1.5566|3.8175|3.8376|
|2|8|4|1024|4096|14336|512|1.0341|2.2678|14.6049|14.7517|
|4|8|2|1024|4096|14336|1024|1.0752|2.2506|15.8857|15.9633|
|4|16|4|1024|4096|14336|512|1.1790|2.9457|16.7874|16.7925|

H=4096、F=14336、N=1024 的三组大矩阵中，普通同步版为 1.034～1.179 ms，Stream 256 为 2.251～2.946 ms，NVSHMEM 分离为 14.605～16.787 ms，单 Kernel 融合为 14.752～16.793 ms。单 Kernel 融合相对同后端分离的中位数慢约 0.03%～1.01%；两者的波动范围见后表。

### Chunk 大小的影响

Stream 同时测试 128、256、512 三种 Chunk 大小。以下为大矩阵 H=4096、F=14336、N=1024 的中位数，单位 ms。

|EP / Expert 总数 / Expert 每卡|普通同步|Stream 128|Stream 256|Stream 512|
|---|---:|---:|---:|---:|
|2 / 8 / 4|1.0341|3.8818|2.2678|1.4233|
|4 / 8 / 2|1.0752|4.2225|2.2506|1.3570|
|4 / 16 / 4|1.1790|5.6193|2.9457|1.6080|

这组已测试的 Chunk 中，512 的延迟最低，仍比普通同步版高约 26%～38%。Chunk 增大后，每个 Expert 的 GEMM 行数增多，同时减少 A2A、索引操作、GELU 和 Combine 的提交次数。分块规模对计算效率与调度成本的影响，已经超过当前配置中被隐藏的通信时间。

### Timeline 中的重叠

大矩阵 Stream 路径另行使用 Nsight Systems 采集，Chunk=256，每个进程保留三次同步结束的 PIPELINE_PROFILE 区间。通信时间是区间内 NCCL Kernel 的时间并集，计算时间是 GEMM Kernel 的时间并集；交集除以通信时间得到 overlap ratio，通信并集减去交集得到 exposed communication time。GELU、索引和 Combine 独立保留在 Trace 中。

分析器同时识别含 gemm 的 Kernel 名和本次 cuBLAS 使用的 nvjet_ 前缀。回归测试分别验证两种命名的区间交集、进程过滤和区间过滤；实际 Trace 中的 GEMM 数量也与每个 Expert 两次矩阵乘法的调用数核对。

下表在每个进程内对三次采集取平均，再列出各进程的范围。total latency 使用上面的无 Profiler、最慢 Rank 中位数；采集区间指标用于观察重叠关系。

|EP / E / Q|Stream 256 total latency ms|communication ms|GEMM ms|exposed communication ms|overlap ratio|
|---|---:|---:|---:|---:|---:|
|2 / 8 / 4|2.2678|0.14～0.71|1.51～1.55|0.13～0.55|6.58～22.26%|
|4 / 16 / 4|2.9457|0.17～0.86|1.50～1.60|0.16～0.67|5.13～31.72%|

上述多专家 Trace 中，每个进程、每轮出现 32 个 GEMM Kernel、8 个 NCCL Kernel，整条执行共 120 个 Kernel。它们对应四个 Chunk、每卡四个 Expert，每个 Expert 各做一次 Up 和 Down。时间线确认了通信与 GEMM 的重叠；端到端比较同时表明，这组分块没有抵消额外计算、索引和调度的代价。

NVSHMEM 的 EP=4、E=16、N=1024 大矩阵也采集了两个后端的 Timeline。每个 PE 都记录到 10 次融合主体、20 次 Combine、20 次分离通信和20次分离投影，对应融合与分离各 10 轮。完整执行分别为每轮两个与五个 Kernel。PE 0 的分离路径中，两次通信 GPU 时间合计约 7.10 ms，两次投影约 9.70 ms；融合主体约 16.80 ms。减少启动次数对这组毫秒级工作量的影响很小。

### 正确性与测量口径

本次完成 13 个性能配置与 8 个边界配置，共 60 个 Rank/配置检查。覆盖 EP=1/2/4、Top-1/2、每卡 1/2/4 个 Expert、均匀与倾斜路由、空 Expert、N=17 尾块和单 Token Expert。四条路径与独立 FP32 GEMM Reference 对齐；Reference 在 Up、GELU 和 Down 边界显式进行 FP16 舍入，Combine 为 FP32。数值比较使用 atol=0.003、rtol=0.02。

最大绝对误差为 0.00025177。NVSHMEM 融合与分离输出逐元素相同；各 Rank 的 MAX 延迟样本逐项一致。send/recv Rank 行数、Expert 行数、各 Chunk 行数之和、输出 shape/dtype/device 和输入更新后的结果均通过检查。分析器回归测试在本地及 H100 容器中通过，H100 源码 SHA256 与交付源码一致。

计时从已打包输入开始，到原 Token 顺序的 FP32 加权输出结束，包含两次载荷通信、Expert MLP 与 Combine，排除 Router、打包、计数交换、初始化、编译与 Host 拷贝。NCCL 两条路径包含中间 Tensor 分配，NVSHMEM 两条路径复用工作区。GPU 计时使用 CUDA Event，Stream 函数先把两条工作流接回调用方，再记录结束 Event。同步与 barrier 位于测量边界。

NVSHMEM 以融合、分离、分离、融合的顺序分四组，每组预热 10 次、测量 30 次；普通与各 Stream Chunk 按正序、反序分两组，每组同样预热 10 次、测量 30 次，每条路径共 60 个有效样本。Profiler 采集与正式性能测量分开，采集生成的样本单独归档。

运行环境为 H100 80GB、CUDA 13.2、PyTorch 2.12.0a0、NCCL 2.29.7、NVSHMEM 3.6.5、CUTLASS 3.9.2。2 卡使用物理 GPU 6、7，4 卡使用 2、3、6、7，均为同节点 NVLink 连接。容器为 4 CPU、16 GiB 内存配额，多专家 NVSHMEM 对称堆为 2 GiB。GPU 为共享环境，其他进程已有显存分配；运行前后采样利用率为 0%，清单及拓扑随附件保留。

多专家六组主测试使用扩展后的统一源码。每卡一个 Expert 的七组先行四路对照使用先前单 Expert 源码，源码快照及 SHA256 独立保存；其数据在下表单独列出。当前扩展保留 E=P 的默认行为，并重新通过了单卡、单 Expert 正确性检查。

本次没有执行独占 GPU、跨节点、BF16、Grouped GEMM 或按 Chunk 就绪推进的单 Kernel 流水线对照。主表对应当前 Expert Loop、CUTLASS Tile 和设备通信协议。

### 中位数与波动范围

以下为六组多专家主测试的中位数 [P10,P90]，单位 ms。

|EP / E / N / H / F|普通同步|Stream 256|NVSHMEM 分离|单 Kernel 融合|
|---|---:|---:|---:|---:|
|2/8/512/2048/4096|0.4762 [0.3861, 0.5649]|1.4678 [1.4372, 1.5101]|2.4912 [2.4815, 2.4966]|2.5321 [2.5106, 2.5416]|
|4/8/512/2048/4096|0.4444 [0.4121, 0.4773]|1.2406 [1.2016, 1.2987]|2.7050 [2.6719, 2.7184]|2.7160 [2.6818, 2.7376]|
|4/16/512/2048/4096|0.5806 [0.4237, 0.6415]|1.5566 [1.5142, 1.6214]|3.8175 [3.7772, 3.8734]|3.8376 [3.8010, 3.8849]|
|2/8/1024/4096/14336|1.0341 [1.0224, 1.0541]|2.2678 [2.2505, 2.2954]|14.6049 [14.5690, 14.6444]|14.7517 [14.6612, 14.8564]|
|4/8/1024/4096/14336|1.0752 [1.0612, 1.1224]|2.2506 [2.2006, 2.3268]|15.8857 [15.8520, 15.9838]|15.9633 [15.9215, 16.0229]|
|4/16/1024/4096/14336|1.1790 [1.1269, 1.2489]|2.9457 [2.3217, 3.1473]|16.7874 [16.7603, 16.8565]|16.7925 [16.7146, 16.8587]|

### 每卡一个 Expert 的补充对照

下表为先行七组四路对照的中位数，单位 ms。N=128 时 Stream 使用 Chunk=64，其余使用 Chunk=256；原始 JSON 包含全部 Chunk 大小的样本。

|EP / E|N / H / F|普通同步|Stream|NVSHMEM 分离|单 Kernel 融合|
|---|---|---:|---:|---:|---:|
|2 / 2|128 / 512 / 1024|0.2176|0.8125|0.0992|0.0979|
|2 / 2|512 / 2048 / 4096|0.2376|0.8206|1.8883|1.8936|
|4 / 4|512 / 2048 / 4096|0.2728|0.8954|2.1738|2.1754|
|2 / 2|512 / 4096 / 14336|0.4691|0.6284|7.1480|7.2374|
|2 / 2|1024 / 4096 / 14336|0.8622|1.1254|14.3690|14.5503|
|4 / 4|512 / 4096 / 14336|0.5660|0.9118|7.8461|7.8573|
|4 / 4|1024 / 4096 / 14336|0.9912|1.5661|15.6659|15.7557|

### 复现与源码

多专家实现位于 `mini_ep/kernels/nvshmem/large_expert.cu`，最后一个可选参数为 Expert 总数，默认 E=P。四路中的普通版、Stream 版及独立 Reference 位于 `benchmarks/benchmark_stream_fusion.py`。重叠分析与回归测试分别为 `benchmarks/analyze_overlap.py`、`tests/test_overlap_analysis.py`。

附件保留实际执行的 `compile-multi-check.sh`、`multi-sweep.sh`、`extra-checks.sh`、`multi-profile.sh` 及日志，以及单 Expert 的 `sweep.sh`、`profile.sh`。多专家 CLI 依次接收 prefix、N、H、F、K、skew、warmup、repeats、workers、E；MPI 进程数为 EP Size。Python 基准读取同一 prefix，`--chunk-sizes` 指定分块大小。实际命令中的库路径、可见 GPU 和输出目录随执行环境设置。

原始 JSON、Nsight 报告、SQLite、区间交集结果、配置和源码 SHA256 一同交付。输入二进制按源码中的 20261007＋Rank 随机种子生成，附件保留其 SHA256。此前验收记录保持原样，本附件包含本次新增基准与多专家实现。

Stream 依赖语义依据 [PyTorch 2.12 的同步与异步 collective 文档](https://docs.pytorch.org/docs/2.12/distributed.html#synchronous-and-asynchronous-collective-operations)；设备计算和启动依据 [CUTLASS 3.9.2 GEMM](https://github.com/NVIDIA/cutlass/blob/v3.9.2/include/cutlass/gemm/kernel/gemm.h) 与 [NVSHMEM Cooperative Launch](https://docs.nvidia.com/nvshmem/api/latest/api/launch.html)。
