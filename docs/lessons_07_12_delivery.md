# 第 7～12 课交付与 H100 验收

第 7～11 课完成编写、H100 验证与知识库回读后，重新读取修改后的 `docs/计划书.md` 第十二课目标，再完成第十二课的 Stream 和 NVSHMEM 两条路线。六课已放在原知识库，首页目录按第一至十二课排列，源码和原始证据以附件提供。

## 正文与源码

实际阅读风格参考的相关内容：AI Infra Book 第五章的提交、融合与运行时，以及 9.4.1 的 EP 负载；飞书参考的第十三章 Profiler/Stream、第十四章 Triton、第十五章 MoE。正文采用具体问题、布局与小数值例子展开，代码前解释输入，代码后连接数据变化和下一阶段。

保留前六课的知识库节点，新增六课、八张关系图和统一的代码标题。15 个 Python 正文代码块与源码一致并在 H100 导入；两个 CUDA 函数代码块与实际编译源码一致。飞书回读确认标题、代码、公式与图示保留。最终 54 个源码文件的 SHA256 与 H100 实际文件完全一致，附件包含校验清单。

## 运行条件与正确性

H100 80GB，使用 1/2/4 GPU；Python 3.12.3，PyTorch 2.12.0a0+0291f960b6.nv26.04.48445190，CUDA 13.2，Triton 3.6.0，NCCL 2.29.7。设备路线使用 NVSHMEM 3.6.5、CUTLASS 3.9.2、MPI、sm_90。依赖装在本次独立容器和目录中，未修改服务器驱动或停止其他任务。

FP32 关闭 TF32。NCCL 设置 `NCCL_NVLS_ENABLE=0`；NVSHMEM 设置 `NVSHMEM_REMOTE_TRANSPORT=none`、`NVSHMEM_DISABLE_CUDA_VMM=1` 和 `NVSHMEM_DISABLE_NCCL=1`，使用同节点传统 P2P 映射。默认 multicast/VMM 初始化在该环境报 401，采用以上已验证配置。

|课时|实际验收|结果|
|---|---|---|
|7|EP=1/2/4；T=1024，H=128，F=256，E=8，K=2，FP32；完整 NCCL EP 对齐 Reference|通过|
|8|两份 PyTorch Rank Trace；Nsight Systems .nsys-rep/SQLite；六个阶段 NVTX 与 NCCL/GEMM 活动|通过|
|9|稳定 Dispatch 的值与全部 metadata；空输入、空 Expert、非连续输入、FP32/FP16/BF16|通过|
|10|Unpermute、routing weight 与 Top-K；空输入及边界维度，三种 dtype|通过|
|11|Grouped GEMM 与 MLP 对照；完整 EP 逐项替换及三项组合|通过|
|12|Stream EP=1/2/4、Top-1/2、FP32/BF16、末尾短 Chunk；设备 EP=2/4，缩放/MLP、均匀/倾斜、空块、六轮双缓冲|通过|

`python -m unittest tests.test_advanced_lessons -v`：7 项通过，GPU 部分含 36 个子用例。`examples/11_grouped_ep.py` 在 EP=1/2/4 每 Rank 20 项，共 140 项通过。第十二课设备固定输入与逐轮变化输入各完成 288 项 PE/轮次检查，共 576 项；变化输入的同步 NCCL 对照共 48 项，值、send/recv 行数和 origin 均一致。同输入三路线对照在 EP=2/4 各 Rank 的八种配置上通过。

## 算子与系统测量

算子采用 T=257、H=128、F=256、E=8、K=2、FP32，预热 5 次，20 次 CUDA Event 样本，以下为中位数。首次编译在计时前完成，GPU Kernel 数来自独立 Kineto 采样。

|算子范围|Reference ms|Triton ms|Kernel 数 Reference/Triton|
|---|---:|---:|---:|
|dispatch|0.281568|0.272592|21/20|
|combine|0.043760|0.029312|4/1|
|grouped|0.189440|0.038608|8/1|
|mlp|0.349808|0.070448|32/3|

Grouped 调度准备约 0.056 ms，独立计时。表中 GEMM/MLP 使用预先准备的布局、权重与调度；完整 EP 回调还包含打包、权重准备与恢复。

三路线采用每 PE 七个 Token、E=P、H=F=4、FP32，固定输入、权重和路由。执行范围从已打包输入到原 Token 顺序的加权输出；路由、打包、计数、Host 结果拷贝在范围之外。NCCL 调用中的临时分配计入执行，NVSHMEM 复用工作区。JSON 中的通信字节统计来源 Rank 的两次远端激活传输，信号与计数不包含在此字段中。NCCL 预热 2 次，NVSHMEM 丢弃第一轮，各取 5 个样本，在每个重复编号取最慢 Rank/PE。以下展示 Top-2、倾斜路由、MLP 配置。

|EP|同步 NCCL ms|Chunk NCCL ms|NVSHMEM ms|Kernel 数（Rank 0）Sync/Pipeline/NVSHMEM|
|---|---:|---:|---:|---:|
|2|0.240960|1.168800|0.030880|12/42/2|
|4|0.315712|1.472768|0.047552|14/42/2|

NVSHMEM Nsight 捕获每 PE 48 次通信计算 Kernel 和 48 次 Combine，对应八个配置各六轮的两次 launch。采样开关分别捕获同一输入配置，原始 phase_ticks 与两份 Trace 均随附件提供。PE 推进与等待差异影响 Kernel 时长，固定采样开销需要独占时段复测。

Stream 时间线使用 T=4096、H=2048、F=4096、E=8、Top-2、FP32、Chunk=512。两个 Rank 三次范围的 NCCL/GEMM 交集分别约 21.7%～28.1% 和 7.3%～24.6%；暴露通信时间逐范围保存在 lesson12-large-overlap.json。这个 Nsight 采样进程中的整层最慢 Rank 中位数为同步 8.071 ms、Pipeline 17.429 ms。当前配置下，流水线增加了整层延迟；Trace 同时显示通信/计算重叠和较大的 Host 提交间隙。

服务器为共享环境，容器使用 4 CPU 配额。上述性能值限定在记录的输入、软件、工作区和测量范围，独占 GPU 性能复测未执行。设备 Kernel 采用阶段式就绪后消费；较大矩阵、设备侧 Tile 角色流水线、FP16/BF16 协议与跨节点属于扩展范围。

## 复现命令与证据

实际执行的主要命令如下；所有分布式运行均设置超时，并在上述 NCCL/NVSHMEM 环境变量下执行。

```bash
python -m unittest tests.test_advanced_lessons -v
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 examples/11_grouped_ep.py
python examples/09_triton_dispatch.py
python examples/10_triton_combine.py
python examples/11_grouped_gemm.py --kind grouped
python examples/11_grouped_gemm.py --kind mlp
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 examples/12_stream_pipeline.py --tokens 32 --hidden-size 17 --expert-hidden-size 31 --chunk-size 3
nvcc -ccbin mpicxx -std=c++17 -O2 -rdc=true -arch=sm_90 -I"$NVSHMEM_HOME/include" -I"$CUTLASS_HOME/include" $(mpicxx --showme:compile) mini_ep/kernels/nvshmem/token_exchange.cu -L"$NVSHMEM_HOME/lib" -lnvshmem_host -lnvshmem_device -o token_exchange
MINIEP_VARY_ROUNDS=1 MINIEP_DEVICE_TIMESTAMPS=1 mpirun -np 4 ./token_exchange results/varied-nvshmem4
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 examples/12_nvshmem_reference.py --prefix results/varied-nvshmem4
NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_fusion.py --prefix results/final-nvshmem4 --output-dir results/fusion
nsys export --type sqlite --output results/lesson12-large-pipeline.sqlite results/lesson12-large-pipeline.nsys-rep
python benchmarks/analyze_overlap.py results/lesson12-large-pipeline.sqlite --output results/lesson12-large-overlap.json
```

源码修正包括：Histogram 设备输入显式转 int32；FP32 oracle 关闭 TF32；MPI 链接通过 mpicxx Host 编译器；设备协议结果增加计数与恢复索引，双缓冲加入逐轮变化输入对照。依据来自接口语义、编译器诊断与 H100 正确性结果，原始日志独立保存。

附件包括完整源码校验清单、各 Rank JSON、测试日志、两份 Rank Trace、Nsight Systems 报告、对应 SQLite 和重叠分析。第 7～12 课正文、代码和知识库回读均完成；需要补充的资源是独占 GPU 性能复测时段。

## 本次命名与正文复核

本轮将设备算子 `exchange_scale` 改为 `exchange_compute_return`，覆盖发送、Expert 计算与结果返回三项职责；设备 MLP 辅助函数改为 `expert_mlp_row`，等待函数改为 `wait_for_epoch`，缓冲区类型改为 `TokenExchangeBuffers`。对应字段使用 `expert_mode`、`up_weights` 与 `down_weights`；Kernel 局部变量使用 `peer_block_elements`、`signal_base` 和 `row_offset`，区分元素容量、信号起点与行地址。三路线基准中的 Expert 类改为 `ProtocolExpert`。数值计算、通信次序和输出 JSON 的 `mode` 字段保持兼容。

第 7～12 课逐篇通读，缩短知识主题标题，补清通信载荷、Expert 放置示例、稳定分组前缀、低精度乘积与 Tile 边界之间的关系。第九课明确所有 packed metadata 的真实字段名、shape 和 dtype；第十二课修正 Top-1/Top-2 展开行数的表述，补齐流水线代码块的 `combine_reference` 导入，并解释算子的输入、输出与后续 Combine。飞书中部分多行 LaTeX 的换行折叠会连接控制序列与后续符号，已改为显式空格连接的公式。

本轮在 H100 重新编译 CUDA 源码，编译成功；CUTLASS 3.9.2 在 CUDA 13.2 下产生 `double4` 弃用提示，未出现编译错误。实际执行 `review/h100_validate.sh`，完整命令随复验附件提供，结果如下。

|复验范围|结果|
|---|---|
|54 个源码文件的本地/H100 SHA256|全部一致|
|15 个 Python 正文代码块与源码 AST 对照、H100 导入；两个 CUDA 代码块与编译源码对照|通过|
|`tests.test_advanced_lessons`，含 Dispatch、Combine 和 Grouped GEMM GPU 用例|7 项通过|
|正文中的 `run_chunk_pipeline` 函数直接替换实现，EP=1/2/4，FP32/BF16、Top-1/2、尾部短 Chunk|全部通过|
|NVSHMEM EP=2/4，固定输入与逐轮变化输入，两种 Expert、Top-1/2、均匀/倾斜、六轮循环|共 576 项 PE/轮次检查通过|
|同步 NCCL 与 NVSHMEM 的输出、send/recv 行数和 origin 对照|48 项通过|
|同输入同步 NCCL、Chunk NCCL 与 NVSHMEM 三路线基准|全部配置通过|
|更名后的 Nsight Systems Trace|EP=2，每 PE 48 次 `exchange_compute_return` 和 48 次 `combine_returned`，已核对|

本轮数据位于验收附件的 `review/` 目录，包含编译和测试日志、各 Rank JSON、新命名的 `.nsys-rep`、SQLite、源码校验清单及执行脚本。此前 `results/` 目录保留原始测量证据，其中的旧 Kernel 名称对应更名前的采集。上文性能表来自此前测量，本轮复验用于确认命名、代码块和行为一致性。独占 GPU 性能复测仍未执行。本次复验完成后，已停止本次使用的测试容器。

## 十二课图文与自学验收

本轮扩展覆盖第 1～12 课，共增加 14 张图示及相邻的数值推导。前六课分别补充加权回填、恢复索引与字节地址、Linear 投影、Collective 数据去向、变长块边界和单条 Token 的完整往返；后六课补充 EP 规模与通信量、热点发送矩阵、稳定分组的目标位置、融合 Combine 的累加过程、Grouped GEMM 尾部 Tile 与恢复顺序，以及 NVSHMEM 地址与缓冲槽复用。图示紧接相关概念，正文交代输入、推导过程和后续用途。

对照参考书的相关章节复读后，按“具体数据 → 布局或依赖变化 → 实现 → 系统影响”检查十二课。各课沿用已有符号和示例，新增小例子只展开当前难点；标题保持知识主题，机制用连贯段落解释。运行前置条件保留在相关位置，实验命令、验收记录与机器状态仍集中在独立交付说明中。

|验收内容|结果|
|---|---|
|十二课数值、发送矩阵、字节地址、恢复索引及轮次推导|56 项检查通过，见 `review/example-checks.json`|
|14 张新增图示的飞书实际预览|逐张检查文字、箭头、边界与正文，完成一处箭头修正|
|原有图示、图片、附件和链接|回读确认保留；新增图示纳入各课连续编号|
|本轮图文补充前后可执行代码块|十二课全部一致；图文补充未改变可执行代码|
|第 7～12 课代码与当前源码|15 个 Python 摘录 AST 一致，两个 CUDA 摘录一致，命令语法检查通过|
|叙述与自学结构|通读十二课，检查概念依赖、代码前后说明、例子衔接及模板化与防守性表达|

源码附件更新为第 1～12 课完整正文、配套源码与全部本地图片资源；验收附件保留此前原始结果，并增加本轮更名复验及图文检查记录。本轮图文补充使用可独立计算的小规模例子，没有新增性能结论。第一至六课的可执行代码本轮未作修改，未重新执行整套 H100 测试；第七至十二课的 H100 复验结果见上一节。课程阅读无需补充材料；独占 GPU 性能复测仍需另行安排。

## 第十二课 MLP 通算融合复验

正文与配套实现统一使用两次矩阵投影和 GELU，移除了缩放 Expert、`expert_mode` 和 `mode` 分支选择。设备侧 Kernel 在输入就绪后直接调用 `expert_mlp_row()`，随后返回结果；NCCL Reference 与三路线基准使用相同 MLP。结果 JSON 以 `expert="mlp"` 标记计算类型。`input_scale` 是逐轮变化输入的测试元数据，用于检查双缓冲数据更新，与 Expert 计算类型无关；性能基准使用固定输入。

### H100 正确性与代码一致性

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

### 命令与原始证据

完整命令见附件中的 `h100_validate.sh`，包括 nvcc 编译、EP=2/4 的 MPI 运行、torchrun 对照、三路线基准及 Nsight 采集与 SQLite 导出。主要入口如下：

```bash
mpirun -np 2 ./token_exchange results/fixed2
MINIEP_VARY_ROUNDS=1 MINIEP_DEVICE_TIMESTAMPS=1 mpirun -np 4 ./token_exchange results/varied4
torchrun --standalone --nproc-per-node=4 examples/12_nvshmem_reference.py --prefix results/varied4
torchrun --standalone --nproc-per-node=4 benchmarks/benchmark_fusion.py --prefix results/fixed4 --output-dir results/fusion
```

启动环境变量、库路径、进程超时、完整参数与结果均随附件提供。正文保留必要运行条件，机器状态与验证经历集中在此说明。第 1～11 课及其代码未因本次调整而变更。源码附件同步更新第十二课正文、三个改动实现、运行说明和 SHA256SUMS；此前验收附件保留历史版本证据，本次 MLP 复验附件对应当前实现。
