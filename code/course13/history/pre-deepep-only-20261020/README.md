# Course 13：把 Mini-EP 接入 vLLM

本课沿用前面的数据流：Router → Dispatch → NCCL All-to-All → Expert → Return All-to-All → Combine。模型采用 `Qwen/Qwen3-30B-A3B` BF16：48 层、128 Experts、Top-8，Hidden=2048、Expert Hidden=768。vLLM 提供权重加载、Attention、KV Cache、调度和采样；Mini-EP 执行 Token 分发、Expert 计算和返回。

| 模式 | GPU 数 | TP | DP | EP | 每 Rank 的 Experts |
| --- | --- | --- | --- | --- | --- |
| 双卡 | 2 | 2 | 1 | 2 | 64 |
| 四卡 | 4 | 2 | 2 | 4 | 32 |

这是第十三课的 TP × EP 框架接入扩展，前十二课的 Pure EP 实现独立保留。先验证完整数据流与精度，再用第十一课的 Grouped GEMM 思路降低专家阶段延迟。

建议先读 [从前十二课读到 Course13](TEACHING.md)：按前课顺序对照 Rank packing、`origin` 恢复关系、offsets/tile 表和 Combine，再看 Qwen 的 SwiGLU 与 vLLM 的 TP/DP 边界。源码沿用第十一课的 `recv_tokens → packed_tokens → packed_output → result` 布局命名，第九课的 metadata/向量分工仍然适用。

与原生 vLLM 的同配置延迟对照见 [性能报告](PERFORMANCE.md)。报告开头列出优化后的 Mini-EP/native 比值，后面再看各次优化相对自身基线的收益；两种比值不能混用。

## 目录与安装

```text
mini_ep/       核心推理：拓扑、打包、NCCL、Triton、vLLM 适配、NVTX
reference/     独立 PyTorch Router、Expert、Dispatch、Combine、MoE
validation/    原生 EP 对比、logits 采集、teacher forcing、报告
checks/        正确性与回归测试
benchmarks/    独立计时与基准
test/         可启动示例和验证入口
```

`mini_ep` 不导入 `reference`、`validation`、`checks` 或 `test`。Reference 不导入 Triton Kernel。生产插件和验证插件分别注册，只有验证脚本启用后者。整个目录可复制独立运行，不导入其他课程，也不使用跨目录符号链接。

环境固定为 Python 3.12、vLLM 0.29.0、PyTorch 2.13.0+cu130、Triton 3.7.1、Transformers 5.16.1。在匹配的 CUDA/vLLM 环境中进入本目录：

```bash
python3 -m pip install --no-deps --no-build-isolation -e .
export MODEL_PATH=/path/to/Qwen3-30B-A3B
```

依赖版本见 `requirements.txt`。本课使用 eager 和同步调度，关闭量化、EPLB、DBO、推测解码、CUDA Graph 与前缀缓存。若服务器明确报 NVLS multicast 初始化错误，在实验进程设置 `NCCL_NVLS_ENABLE=0`；库中不修改该环境设置。

## 四卡的 Token 在哪里

默认 EP group 的逻辑 Rank 按 DP-major、TP-minor 排列：

| EP rank | DP rank | TP rank | Expert IDs |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0～31 |
| 1 | 0 | 1 | 32～63 |
| 2 | 1 | 0 | 64～95 |
| 3 | 1 | 1 | 96～127 |

DP 0 与 DP 1 处理不同请求，Token 数可以不同；同一 DP 内的两卡共同执行 Attention TP。128 个 Experts 则由四卡共享，而不是每个 DP 再各放一份完整 Expert 集合。

vLLM 0.29 在 TP=2、DP=2 时自动启用 MoE 的行切分：Qwen 在进入 Runner 前，将本 DP 的 `[N_dp,H]` 补齐到 TP 的整数倍，再按 TP rank 切成局部行。Runner 直接处理这些行，不能重复切分。离开 Runner 后，Qwen 在本 TP group 内收集并去除 padding，恢复本 DP 的原始行顺序。

双卡 DP=1 时，vLLM 不做这层行切分，所以 Mini-EP Runner 自己建立唯一源 Token 所有权，并在末尾恢复 TP 复制布局。两条路径都不跨 DP 混合请求身份。

## 核心实现

1. `integration/runner.py` 复用 vLLM 的 gate、Top-K、进程组和已加载权重，检查 TP/DP/EP 成员顺序与 Expert map。Expert 放置公式集中在 `distributed/topology.py`。
2. `distributed/dispatch.py` 把 `[N,K]` 路由展开成 `N*K` 条分支，稳定按目标 Rank 打包。`pack_index`、`restore_index` 保存分支与打包行的双向关系。
3. `distributed/all_to_all.py` 的变长路径交换 int64 行计数，再发送 BF16 Token 与全局 Expert ID；counts 合并为一次 Host 拷贝，split 单位为行。固定容量实验使用等长分段，省去计数交换与 splits 回读。两种路径的接收布局都按来源 Rank 排列。
4. `model/expert.py` 默认把本地 Expert 行稳定打包，再由 `kernels/grouped_swiglu.py` 执行两次分组投影和一次 SwiGLU：`down(silu(gate(x))*up(x))`。稳定排序后，单个 Triton metadata kernel 查找各 Expert 的起止行，生成 offsets 和 tile map，避免 `.tolist()` 与 Host 规划。BF16 使用 BM/BN/BK=128/128/64、8 warps；BK 不变，K 维累加顺序和舍入边界不变。`expert_loop` 与 `kernels/swiglu_triton.py` 保留为逐 Expert 的教学与回退路径；独立 PyTorch Reference 在 `reference/expert.py`，不引入自动调参或异步流水线。
5. Down 在本地用 FP32 累加，在累加结果上乘 FP32 routing weight，然后按 `tokens.dtype` 写出并执行返回 A2A。变长路径互换正向 send/recv splits，固定路径使用同一组等长分段。
6. `kernels/combine_triton.py` 沿第十课思路融合逆排列、路由权重和 slot 顺序累加。DP=1 使用本地 FP32 Combine；DP=2 默认保留 Expert owner 部分和，并通过一次变长 `all_gatherv` 和 `reduce_scatterv` 完成原生顺序归约，不构造最大行数的 padding 或 `cat` 中间缓冲。

DP=2 固定保留原生 owner 部分和归约。早期本地 Combine 实验只有 12/14 条完整序列一致，owner-partial 为 14/14，因此主适配不再提供会破坏精度的 `MINIEP_OWNER_PARTIALS=0` 覆盖；显式传入会报错。相关数值对照保留在外围 Reference、核心基准和历史实验快照中。真实 Token 分发、Expert 计算和返回 A2A 都仍由 Mini-EP 执行。

返回 A2A 默认跟随输入激活 dtype，Grouped GEMM 默认融合加权写出；Combine 消费已加权分支，不再乘权重。核心支持 FP32/BF16 激活，vLLM 接入仍只支持 BF16 模型。显式 FP32/FP16/BF16 传输仅用于诊断，不扩展模型 dtype 支持。两个原生基准的独立零误差验收见 [当前验收说明](EXACT_ALIGNMENT.md)，之前的实验保留在 [返回 A2A 精度实验](RETURN_TRANSPORT.md)。

所有 Rank 都按一致顺序进入 collective。没有请求的 DP 由 vLLM 发起 dummy step，仍需帮助其他 DP 计算本地 Experts。核心库不创建或销毁 vLLM 进程组。

Expert metadata 沿用第九课的 GPU 布局规划思路：`order` 是稳定排序后的接收行索引，kernel 通过 `local_ids[order]` 的二分下界得到 `[L+1]` offsets，再对 `ceil(count/BM)` 做前缀和。已知接收行数为 R，最多分配 `ceil(R/BM)+L-1` 个 tile；多余项标为 `expert=-1`，GEMM 在读取权重前跳过。空 Expert 不生成有效 tile，R=0 直接返回空输出。这样保留原来的有效 tile 顺序、两次投影和 BF16 舍入位置；BF16 tile 为 128/128/64、8 warps，FP32 测试路径保持 16/32/32、4 warps。行数与容量作为 kernel 运行时参数传入，只按二分深度和向量大小分档编译，避免每层不同接收行数触发一次编译。Host planner `reference/grouped_plan.py` 留作独立 metadata 对照，完整数值 Reference 仍位于 `reference/`。

一张 CUDA GPU 可单独测量 metadata（排序在两条路径中均不计时），并导出预热后的 CPU/CUDA trace：

```bash
python3 benchmarks/benchmark_metadata.py --output results/metadata.json \
  --trace-dir results/metadata-traces
```

该微基准不代表整段 EP 或整模型收益。整模型性能使用 `benchmarks/compare_vllm.py`；默认四卡，也支持两卡 `torchrun --nproc-per-node=2 ... --dp-size 1 --gpu-memory-utilization 0.6`。性能结论与真实模型精度验收见 [PERFORMANCE.md](PERFORMANCE.md)。

BF16 有三个要区分的舍入位置：SiLU 后再乘 up、每个加权分支的写回、owner 部分和归约。DP=1 的本地 Combine 使用 FP32 累加；DP=2 默认保留 owner 部分和。独立 Reference 与最小回归测试在外围目录说明这些数值边界。

## 双卡示例

两张 GPU，普通 Python 启动，vLLM 自己创建 Worker：

```bash
export CUDA_VISIBLE_DEVICES=0,1
python3 test/demo.py --model "$MODEL_PATH" --backend miniep --output results/demo
python3 test/validate.py --model "$MODEL_PATH" --output results/ep2
```

`native` 使用原生 EP；`miniep` 使用 Mini-EP；`compare` 逐层比较后返回原生输出，隔离误差传播。默认双卡严格验证保留 32 条提示、batch=1/4、每条 64 Token；扩展验收可传 `--batch-sizes 1 2 4 8`，并用独立的 `--max-tokens 128 --batch-sizes 2 4` 目录测长 decode。每个输出目录只用于一次固定配置。

## 四卡模型验证

四张 GPU，使用 `torchrun` 外部启动器。每个 TP pair 输入相同请求，两个 DP pair 使用不同请求。示例限制上下文和 KV Cache，便于测试阶段控制显存：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
set -e
for phase in compare native teacher free; do
  torchrun --standalone --nproc-per-node=4 test/dp_validate.py \
    --model "$MODEL_PATH" --output results/ep4 --phase "$phase" \
    --gpu-memory-utilization 0.23 --kv-cache-mib 256 --max-tokens 16 \
    --max-model-len 2048
done
python3 test/report_dp.py --output results/ep4
```

四卡单 Rank 权重约 14.96 GiB，还需通信、计算工作区和 KV Cache；显存占用参数不能替代实际可用显存检查。GPU 编号由环境选择，不写在库中。

用例覆盖：两侧均有请求、请求数和长度不同、生成长度不同、一侧完全空闲、单请求 Decode、宽 batch、长 decode、512/768/1024 Token Prefill、2040 Token max-prefill。先验证 48 层输出，再比较同前缀完整词表 logits，最后比较不强制 Token 的独立贪心生成。teacher 阶段先记录真实 logits，之后才强制原生 argmax，防止前缀分歧污染数值比较。

门槛：BF16 层输出 `atol=0.02, rtol=0.02`；同前缀 logits 相对 L2≤1%、cosine≥0.999，argmax 一致率≥99%。报告检查四个 Rank 的覆盖、空闲 DP 是否参与、输出数量与生成长度；缺失结果不能判为通过。完整序列一致率单独报告。

## 单元与分布式测试

```bash
python3 -m unittest discover -s checks -v
# 单机两卡，Reference 与 Triton 对齐：
torchrun --standalone --nproc-per-node=2 checks/distributed.py --backend triton
# 两卡 Token 级快速路径验收（默认 batch=1/4；可显式扩展）：
NCCL_NVLS_ENABLE=0 python3 test/validate.py --model "$MODEL_PATH" --output results/token-fast \
  --limit 32 --max-tokens 64 --batch-sizes 1 2 4 8 --acceptance tokens
# 单机四卡，真正的 NCCL TP=2/DP=2/EP=4，不使用 CPU 通信代替：
torchrun --standalone --nproc-per-node=4 checks/tp_dp.py
```

`strict` 模式继续检查逐层数值和 logits 门槛；`tokens` 模式只把缺失覆盖、非有限值、router 不一致和生成 Token 序列不一致作为硬失败，数值指标仍写入报告。测试包括 Top-1/2/8、不同 H/F、空源 Rank、空 Expert、倾斜路由、FP32/BF16、索引逆变换、DP 不同长度。`checks/test_core_boundary.py` 防止 Reference 或测试依赖重新混入核心包。课程外的静态检查：`python3 tools/check_course_layout.py --courses 13`。

## OpenAI 兼容服务

四卡服务由 vLLM 自己启动两个 DP engine，每个 engine 有两卡 TP；此命令不再套 torchrun：

```bash
python3 test/serve.py --model "$MODEL_PATH" --backend miniep --dp-size 2 \
  --gpu-memory-utilization 0.23 --kv-cache-mib 256 \
  --max-model-len 1024 --max-batched-tokens 2048 --port 8133
# 另一个终端：
python3 test/service_check.py --url http://127.0.0.1:8133 --output results/service.json
```

也可用 `test/service_smoke.py`，传入相同的 `--dp-size 2 --gpu-memory-utilization 0.23 --kv-cache-mib 256`，自动启动、验证并停止临时服务。检查健康、普通/流式 Completion、Chat Completion 和四并发。双卡使用 `--dp-size 1`。

## Benchmark 与复现资料

```bash
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_ep.py \
  --expert-backend grouped \
  --output results/benchmark-grouped.json
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_ep.py \
  --expert-backend loop --output results/benchmark-loop.json
```

该基准测核心 EP，warmup 3 次、测量 10 次，报告每次最慢 Rank 的 CUDA Event 时间及中位数。`--expert-backend` 可在 `grouped` 与 `loop` 间切换。计时代码位于 `benchmarks/timing.py`，不在核心包。它不包含 Attention、KV Cache 或四卡精度适配的额外部分和归约，不代表 vLLM 端到端吞吐。

### 四卡阶段分析

阶段计时只在显式设置 `MINIEP_STAGE_TIMING=1` 时开启；CUDA Event 在一次请求结束后统一结算，不会逐层同步：

```bash
MINIEP_STAGE_TIMING=1 NCCL_NVLS_ENABLE=0 PYTHONPATH=. \
  torchrun --standalone --nproc-per-node=4 benchmarks/profile_stages.py \
  --model "$MODEL_PATH" --output results/stage-profile \
  --warmup 1 --repeats 3
python3 benchmarks/summarize_stages.py \
  --root results/stage-profile --output results/stage-profile/summary.json
```

需要 Nsight 时加上 `--cuda-profile`，并用 `nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop` 包住 `torchrun`。本课不把 Nsight 采集放进默认路径。

为得到干净的 Timeline，prefill 和 decode 分开采集。`--stage-events off` 保留 NVTX、关闭逐阶段 CUDA Event：

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output results/nsys/decode \
  torchrun --standalone --nproc-per-node=4 benchmarks/profile_stages.py \
  --model "$MODEL_PATH" --output results/nsys/decode-data \
  --case decode --stage-events off --cuda-profile --warmup 1 --repeats 1
```

prefill 将 `--case decode` 改为 `--case prefill`。如果 `vllm033` 容器内没有 `nsys`，使用与 `vllm033` 相同的镜像启动临时容器，并把宿主机的 Nsight Systems 目录只读挂载进去；原始 `.nsys-rep` 放在 H100 的 `/SharedData`，不提交仓库。

核心保留 ROUTER、DISPATCH、A2A_FORWARD、EXPERT、A2A_RETURN、COMBINE 的 NVTX 范围；适配层另标注 TP_REASSEMBLE，进入 owner-partial 路径时再标注 EP_REDUCE。未实现异步 overlap。

初始双卡实测见 [VALIDATION.md](VALIDATION.md)；四卡扩展与目录重组结果见 [VALIDATION_EP4.md](VALIDATION_EP4.md)。2026-10-19 的 Profiling、tile A/B 和扩展场景结果见 [PERFORMANCE.md](PERFORMANCE.md)。`SOURCE_MAP.json` 保存本课实际文件的来源与校验值。


### 原生 EP 与 Mini-EP 的端到端性能对比

双卡 TP=2/DP=1/EP=2 或四卡 TP=2/DP=2/EP=4，使用相同 BF16 Qwen 模型、eager 模式、同步调度和
Triton 专家后端。基准不启用 validation 插件；每个场景先预热，再计时。
每个 DP 两条请求，全局分别为两条或四条，测 512→1、128→32、512→16 token。
CUDA Event 和同步边界墙钟分别记录，取所有 Rank 的最大值；吞吐只计算生成
的 token，不重复计算 TP 副本。512→1 包含一次生成，不是纯 MoE kernel 延迟。

```bash
# 在已安装本课的四卡容器内运行；两种后端必须用独立进程。
for run in 1-native 2-miniep 3-miniep 4-native; do
  NCCL_NVLS_ENABLE=0 torchrun --standalone --nproc-per-node=4 \
    benchmarks/compare_vllm.py --model /model --backend "${run#*-}" \
    --warmup 2 --repeats 5 --output "results/performance/$run"
done
# 两卡：进程数改为 2，并增加 --dp-size 1 --gpu-memory-utilization 0.6。
```

模型加载、首次编译、结果序列化不计入计时。保留输出 token 供两种后端复核。
共享 GPU 的结果只代表该次资源争用下的实测，不能替代独占 GPU 的稳态基准。

实测结果见 [PERFORMANCE.md](PERFORMANCE.md)。

## 固定容量与 Host sizes 实验

`MINIEP_FIXED_A2A=off/on/auto` 控制 Token 传输。`off` 使用变长 splits；`on` 使用固定容量；`auto` 仅在所有 EP Rank 的最大来源行数不超过 8 时启用。每个目的 Rank 的容量为 `align_up(max(source_rows) * top_k, 8)`，上限 8192 行；`on` 超限报错，`auto` 回到变长路径。所有 Rank 根据同一份本轮 Host metadata 作决定。

`MINIEP_OWNER_SIZES=legacy/host` 控制 DP=2 owner-partial 的行数来源；`host` 复用框架的 CPU metadata，`legacy` 交换 GPU counts 后回读。当前默认为 `auto`、`host`、`triton`；已完成双卡/四卡完整模型零误差复验与最终 Timeline。可显式设置 `off`、`legacy`、`torch` 对照旧策略。默认返回跟随输入 dtype，padding 不参与 Expert 计算或 Combine。此处此前的性能和 Timeline 数据属于旧 FP32 默认版本，不能用作新默认路径的性能证明。

```bash
# 两卡核心布局对照；不包含 Attention 或调度。
torchrun --standalone --nproc-per-node=2 benchmarks/benchmark_ep.py \
  --tokens 4 --a2a-mode on --fixed-pack-backend triton --warmup 20 --repeats 50 --output results/fixed.json
# 四卡完整请求；先按上文安装插件并设置 MODEL_PATH。
MINIEP_FIXED_A2A=auto MINIEP_OWNER_SIZES=host \
  torchrun --standalone --nproc-per-node=4 benchmarks/compare_vllm.py \
  --model "$MODEL_PATH" --backend miniep --dp-size 2 \
  --warmup 2 --repeats 5 --output results/transport
```

独立 Reference 只使用变长路径，基准显式选择 `--backend reference --a2a-mode off`。固定路径的 `received_branches`、`recv_padding_rows` 在热路径返回 `None`，表示尚未统计；禁止用全组分支总数填写本 Rank 接收数。基准的 `actual_payload_bytes_by_rank` 在测量后按真实路由统计各 Rank 的发送 payload，含 padding 和 self 目的地，不等于链路上的远端字节数。

`MINIEP_FIXED_PACK=torch/triton` 选择固定布局的打包策略。Triton 策略仅在每来源 N*K≤64 时使用稳定前缀 kernel；更大输入保留 PyTorch fallback。核心基准可用 `--fixed-pack-backend triton` 对照；默认采用 `triton` 策略；`torch` 保留为教学和消融对照。返回 metadata 的 `fixed_pack_backend` 记录实际使用的实现，变长路径为 `None`；`fixed_pack_backend_requested` 保留调用者的策略配置，避免把未触发的优化标为已使用。

独立布局 Reference 为 `reference/fixed_dispatch.py`，逐 Token/slot 按目的 Rank 追加，适合手算与 CPU 检查。小批 Triton 路径按第九课的稳定前缀位置写 `combine_index`（主流程中的 `restore_index`），向量 gather 独立执行。固定接收布局的 padding 经稳定 Expert 排序移到前缀，`expert_offsets[0]` 供激活和恢复顺序的 masked load 使用；舍入仍保持 SiLU → BF16 → 乘法 → BF16，Down 保持 FP32 累加，加权后按输入 dtype 写出并返回。详见 TEACHING 的同一组 counts 算例。


## Qwen3.5-35B-A3B 扩展验证

本扩展沿用上面的 Token 往返，仅替换路由 Expert 分支。已完成双卡主语料 128/128、长生成 8/8、四卡 22/22 序列一致验收，逐层与 Logits 最大绝对误差均为 0。最终保留版通过逐层回归与热路径字节码核对；两次未过性能门槛的热路径改写已撤回。详见 [Qwen3.5 接入与验证](QWEN35.md)。

| 项目 | Qwen3-30B-A3B | Qwen3.5-35B-A3B |
|---|---:|---:|
| 层数 | 48 | 40 |
| H / Expert F | 2048 / 768 | 2048 / 512 |
| 路由 Experts / Top-K | 128 / 8 | 256 / 8 |
| Shared Expert | 无 | 1 个，F=512 |
| Attention | 原生 Attention | 原生 Gated DeltaNet / Attention 混合 |

Qwen3.5 使用独立的 `qwen3_next` 工厂入口；模型族从 checkpoint 的 config.json 自动读取。只验证 BF16、文本输入、TP=2、DP=1/2、EP=2/4、eager、同步调度；视觉、量化、MTP 和 fused shared experts 不属于此入口。vLLM 保留 Attention、循环状态、缓存、权重加载和 Shared Expert 的实际 MLP/gate 计算。

Shared Expert 不经过路由 A2A。DP=1 时它的参数按 TP 分片，原生路径先在每 Rank 做 BF16 的 `shared_partial + routed_owner_partial`，再 TP 归约；Mini-EP 必须保留这个舍入边界。DP=2/SP 时每个 Rank 为自己的 Token 行保留完整 Shared Expert 参数，路由结果归约后与该本地结果相加。独立算例见 `reference/shared_expert.py`：256 与 1 的 BF16 加法说明了为什么不能任意移动归约。

使用本地 Qwen3.5 权重，仍从本课目录运行，先对照再测自由生成：

```bash
export MODEL_PATH=/path/to/Qwen3.5-35B-A3B
export CUDA_VISIBLE_DEVICES=0,1
python3 test/demo.py --model "$MODEL_PATH" --backend compare \
  --max-tokens 16 --output results/qwen35-compare
python3 test/validate.py --model "$MODEL_PATH" --limit 32 --max-tokens 64 \
  --batch-sizes 1 2 4 8 --output results/qwen35-ep2
```

四卡沿用上文 `test/dp_validate.py` 的四阶段流程，使用新的 output 目录。验收从 environment.json 的 checkpoint 层数检查覆盖，Qwen3.5 必须覆盖 40 层；缺失层数或输出不能算通过。不要将两个模型的原生 logits 基线、状态或结果目录混用。
