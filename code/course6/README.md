# Course 6：Mini-EP V1

Python 3.10+、支持 CUDA/NCCL 的 PyTorch，以及两张可见 NVIDIA GPU。

本目录包含本课完整实现和本地依赖副本。可以将整个 `course6/` 复制到任意位置，再从该目录启动；无需安装仓库根目录的包或保留其他课的目录。PyTorch、Triton 等第三方依赖由当前 Python 环境提供。

## 代码与示例

`test/` 保存本课可启动示例；`mini_ep/` 保存本地实现；`benchmarks/` 保存示例使用的基准与分析函数；`checks/` 保存正确性测试。实际存在的子目录按本课需要提供。`SOURCE_MAP.json` 记录代码迁移来源。

## 启动

从仓库根目录进入本课；复制出去时，直接进入复制后的本课目录。使用与驱动匹配的 CUDA 版 PyTorch 环境，再补齐本课第三方依赖。

```bash
cd code/course6
python3 -m pip install -r requirements.txt
torchrun --standalone --nproc-per-node=2 test/demo.py --expected-world-size=2
```

正确性测试也从本课目录启动：

```bash
python3 -m unittest discover -s checks -v
```

## 本课的数据名称

| 变量 | 含义 |
|---|---|
| `dispatch` | 源端的 Dispatch 结果，保存打包输入、Token/slot 索引、恢复索引与路由权重 |
| `rank_pack` | 按目标 Rank 排列的发送布局，保存 `packed_tokens`、`send_splits` 与 `row_origin_index` |
| `tokens_per_expert_all` | AllGather 得到的 `[P,E]` 计数矩阵；行对应来源 Rank，列对应全局 Expert |
| `layout.local_counts` | 当前 Rank 的 `[P,L]` 计数；行对应来源 Rank，列对应本地 Expert |
| `send_splits` / `recv_splits` | 按目标发送、按来源接收的行数，均有 P 项 |
| `layout.recv_offsets` | 接收来源块的起止边界 |
| `layout.rows_by_expert` | 每个本地 Expert 在接收缓冲区中的行索引 |
| `recv_tokens` / `expert_outputs` / `returned` | 收到的输入、按原接收行写回的 Expert 输出、返回源端的结果 |

计数通过 AllGather 交换；输入与 Expert 输出通过两次变长 All-to-All 往返。
