# 默认去重分发与可选 Peer/TMA 传输

去重固定启用，不提供去重开关。通信方式独立选择：

| 配置 | 跨卡传输 | 分发去重 | 目标卡局部归约 | 附加硬件要求 |
|---|---|---|---|---|
| 不设置变量，或 `MINIEP_TRANSPORT=nccl` | NCCL variable-split All-to-All | 是 | 是 | 原 NCCL 路径条件，无需 CUDA P2P/TMA |
| `MINIEP_TRANSPORT=peer` | Peer Memory / TMA | 是 | 是 | 同机 SM90+、CUDA P2P、symmetric memory |

旧值 `peer_dedup` 保留为 `peer` 的兼容别名。它不是另一种算法。3090/4090 等环境使用默认 NCCL 路径即可；模型显存容量、vLLM 版本和框架对照后端仍需满足各自条件。NCCL 可按实际拓扑选择传输，不要求应用直接访问另一张 GPU 的显存。选择不支持的 Peer/TMA 环境会报错，不静默切换实验后端。

```bash
# 默认：NCCL 去重。无需设置开关；也可明确选择 NCCL。
export MINIEP_TRANSPORT=nccl
# 支持 Peer/TMA 的 H100 环境可改用这一传输方式，去重仍然固定启用。
# export MINIEP_TRANSPORT=peer
```

## 相同的数据流，不同的传输

```text
原 Token / Top-K
  → 每个 (来源 rank, Token, 目标 rank) 一份输入 + Expert IDs/路由权重
  → NCCL All-to-All 或 Peer Memory
  → 目标 GPU 卡内展开为 Expert 分支
  → Expert SwiGLU，应用路由权重
  → 目标 GPU 对每个 Token 的本卡分支先求和
  → 每个目标 GPU 只返回一个部分和
  → 来源 GPU 合并各 owner 部分和
```

同一个 Token 在 GPU1 选择四个 Expert 时，分发只传一份输入。四个已加权输出为 `[1,0]`、`[2,0]`、`[3,0]`、`[4,0]` 时，GPU1 先得到 `[10,0]`，只返回这一份。H=2048、BF16 时，远端大向量往返逻辑载荷从32 KiB降到8 KiB，不含路由元数据、信号和链路协议开销。两种传输执行相同数量的 Expert。

## NCCL 实现与布局

- `mini_ep/distributed/dedup.py`：不依赖通信或 Triton 的 PyTorch 去重打包、卡内展开、owner 归约和来源合并，支持 CPU/CUDA。
- `reference/dedup.py`：独立的逐 owner/Token/slot 打包 Reference。
- `mini_ep/model/deduplicated_moe.py`：默认 NCCL 推理链路，复用现有 Grouped GEMM / SwiGLU。
- `mini_ep/integration/runner.py`：框架输入归属与通信选择，两种传输都去重。
- 原 `mini_ep/model/moe.py` 的逐分支 NCCL 实现保留为可独立运行的精度/性能对照，不作为 Runner 默认推理路径。

NCCL 发送缓冲为 destination-major 的唯一向量 `[U,H]`，同一目标内部按原 Token 排列。Expert IDs 和 FP32 权重均为 `[U,K]`，保留原 slot，其他 owner 的 ID 为 `-1`、权重为0。接收缓冲按 source-major 排列，卡内展开成 `[R,H]` Expert 分支；输出先归约回 `[U_received,H]`，再反向 All-to-All。来源用 `[P,N]` restore_index 恢复原 Token 和 owner 关系。

`send_splits`/`recv_splits` 各有P项，单位是唯一 Token-owner 行数，不是 Expert 分支数或字节数；返回方向交换两者。参与者是调用方已有 EP group，global rank、CUDA ordinal 与 EP rank 不混用。所有 rank，包括空来源，按计数交换、输入、ID、权重、部分和返回的同一顺序调用 collective。没有 Token AllGather 或 Tensor Parallel 模拟 EP。

本版 NCCL 去重使用实际变长缓冲，不使用原逐分支的固定容量 padding。`MINIEP_FIXED_A2A`/`MINIEP_FIXED_PACK` 仅属于保留的旧实验；当前 Runner 不读取它们。动态 unique/nonzero 和 Python split-list API 会引入 Host 同步；这是一条可读、可验证的同步实现，不据此声称通信计算重叠或性能提升。

## 精度与归约顺序

Down 先应用 FP32 路由权重，分支输出按原模型/返回 dtype 规则舍入。目标 GPU 严格按原 slot 顺序以 FP32 累加模型 dtype 的分支，再把 owner 部分和舍入到模型 dtype。来源按 owner EP rank 升序以 FP32 加和，最后写回模型 dtype；不重复乘路由权重，不使用无序原子求和。

即使显式指定诊断用的分支 return_dtype，owner 部分和仍用模型 dtype 传输，因为原实现的 owner 舍入边界就是模型 dtype。检查中包含 `[256,1,-256,1] → 2`，避免错误地逐次以 BF16 累加得到1。

## 可选 Peer/TMA 实现

`mini_ep/distributed/peer_dedup.py` 和 `mini_ep/kernels/peer_dedup_triton.py` 保留 [第12课分离执行](https://tvle9mq8jh.feishu.cn/wiki/WCOQw8Oe8iDvQUkl5gjcXS1Onke)的设备信号与 TMA 路径，复用 BF16 TMA Up/SwiGLU 和 weighted grouped Down。传输按 Token-owner 去重，整数 Expert 身份与 FP32 权重仍按分支发送。

每个来源先复制唯一输入，再由同一 CUDA stream 上的 metadata kernel 发布每个目标 Expert tile 的就绪计数。目标 acquire 后在卡内展开，再发布 TMA-ready。generic peer stores 到 TMA async proxy 之间有显式 fence。开始/结束 epoch 信号保护工作区复用和返回读取；所有 rank 必须以一致顺序进入，空来源也参与。只支持工作区创建时的 CUDA stream，销毁前须 drain；缓存按容量桶复用，不能在流之间无保护覆盖。

## 正确性验证

从 `routeep/` 运行。NCCL 检查对照旧分支实现和独立 PyTorch Reference，覆盖 FP32/BF16、Top-1/2/8、空来源/全部为空、倾斜路由、不同隐藏维度、返回 dtype，以及交错 global rank 的 EP 子组。

```bash
PYTHONPATH=. python3 -m unittest discover -s checks -p 'test_dedup.py'
PYTHONPATH=. python3 -m unittest discover -s checks -p 'test_transport.py'
# Gloo/CPU 只验证布局和 collective 语义，不代表 NCCL/GPU 性能。
torchrun --standalone --nproc-per-node=2 checks/nccl_dedup.py --cpu --output results/dedup-cpu
# 两张 GPU，禁用 NCCL P2P，验证应用不依赖 CUDA Peer/TMA。
NCCL_P2P_DISABLE=1 torchrun --standalone --nproc-per-node=2 checks/nccl_dedup.py --output results/dedup-nccl-no-p2p
# 四张 H100，另验证 NCCL 去重与 Peer/TMA 去重逐元素一致。
torchrun --standalone --nproc-per-node=4 checks/nccl_dedup.py --peer-smoke --output results/dedup-nccl-peer
```

本轮结果与原始记录见 [NCCL去重解耦验收](reports/nccl-dedup-20261026/VALIDATION.md)。

## 历史性能证据的适用范围

[H100初测报告](reports/h100-peer-dedup-20261026/REPORT.md)、[扩展报告](reports/h100-extended-20261026/REPORT.md)与[飞书图集](https://tvle9mq8jh.feishu.cn/docx/F3xxdCUQOohBQ2x52cHcWFjKnbK)保留原始源码快照和全部正负场景。报告里的“原 NCCL”“仅 Peer”“peer_dedup”是当时冻结版本，不能把这些数据改名为本次新 NCCL 去重版本的测量结果。

新 NCCL 路径须另行测量才能评价性能；减少逻辑大向量不保证每个 decode/prefill 配置更快。当前新运行的 `peer_suite.py` 对比 NCCL去重、Peer/TMA去重与原生DeepEP HT；复现历史未去重路径应使用报告附带的源码快照或保留的核心对照。
