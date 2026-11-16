"""小批量解码场景的 Rank 打包，使用第九课的稳定前缀位置算法。

布局契约仍由 ``distributed.dispatch.pack_by_destination_rank_fixed`` 定义；本文件只把
整数元数据构造与行搬运换成 Triton 实现，All-to-All 的载荷布局和顺序不变。
"""

import torch
import triton
import triton.language as tl
from mini_ep.distributed.dispatch import FixedRankPackedTensors
from mini_ep.distributed.topology import ExpertParallelTopology


@triton.jit
def _fixed_positions_kernel(
    Ids,
    Pack,
    Restore,
    Counts,
    S: tl.constexpr,
    L: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
):
    """作用：为一个目的 Rank 生成固定容量位置和双向索引。

    参数：
        Ids: 展开后的全局专家编号 ``[N*K]``。
        Pack: 固定容量行 -> 原始分支编号的输出表 ``[P*C]``。
        Restore: 原始分支编号 -> 固定容量行的输出表 ``[N*K]``。
        Counts: 每个目的 Rank 的有效行数输出 ``[P]``。
        S: 有效分支数 N*K。
        L: 每个 EP Rank 的本地专家数。
        C: 每个目的 Rank 的固定容量。
        B: 扫描向量的 2 的幂大小。
    """
    # 每个 program 负责一个目的 Rank，扫描同一份展开后的专家编号。
    rank = tl.program_id(0)
    branch = tl.arange(0, B)
    # S 是有效分支数 N*K；超出部分用 -1 填充，且不参与 selected。
    expert = tl.load(Ids + branch, branch < S, other=-1)
    selected = (branch < S) & (expert // L == rank)
    # 与第九课相同：前缀计数减一得到稳定的块内位置，
    # 因此同一目的 Rank 内的分支仍保持原始 token-major/slot-minor 顺序。
    within = tl.cumsum(selected.to(tl.int32)) - 1
    packed_row = rank * C + within
    # Pack 记录固定容量行 -> 原始分支；Restore 记录原始分支 -> 固定容量行。
    tl.store(Pack + packed_row, branch, selected)
    tl.store(Restore + branch, packed_row, selected)
    # Counts 仅用于诊断；正常固定容量通信不读取该计数。
    tl.store(Counts + rank, tl.sum(selected.to(tl.int32)))


@triton.jit
def _fixed_gather_kernel(
    X,
    Ids,
    Weights,
    Pack,
    Y,
    PackedIds,
    PackedWeights,
    H: tl.constexpr,
    TOP_K: tl.constexpr,
    BH: tl.constexpr,
):
    """作用：按 Pack 表搬运 Token 行，并写出对齐的专家编号。

    参数：
        X: 原始 Token ``[N,H]``。
        Ids: 展开后的全局专家编号 ``[N*K]``。
        Pack: 固定容量行 -> 原始分支编号 ``[P*C]``；-1 为 padding。
        Y: 固定容量 Token 输出 ``[P*C,H]``。
        PackedIds: 固定容量专家编号输出 ``[P*C]``。
        H: 隐藏维度。
        TOP_K: Top-K 分支数，用于从分支编号反推源 Token 行。
        BH: 列块大小。
    """
    # 每个 program 处理固定容量缓冲中的一行和一个列块。
    row = tl.program_id(0)
    col = tl.program_id(1) * BH + tl.arange(0, BH)
    branch = tl.load(Pack + row)
    # branch=-1 表示 padding：不读取 Token，列方向写入 0 或跳过。
    valid = branch >= 0
    # 原始分支编号 // TOP_K 即该分支所属的源 Token 行。
    token = tl.maximum(branch, 0) // TOP_K
    value = tl.load(X + token * H + col, valid & (col < H), other=0)
    tl.store(Y + row * H + col, value, col < H)
    # 专家编号只需由第 0 个列块的 program 写一次，避免重复写同一地址。
    if tl.program_id(1) == 0:
        expert = tl.load(Ids + tl.maximum(branch, 0), valid, other=-1)
        tl.store(PackedIds + row, expert)
        weight = tl.load(Weights + tl.maximum(branch, 0), valid, other=0)
        tl.store(PackedWeights + row, weight)


def pack_by_destination_rank_fixed_triton(
    tokens: torch.Tensor,
    ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
    rows_per_rank: int,
) -> FixedRankPackedTensors:
    """作用：用 Triton 把 CUDA 上的 ``[N,H]``/``[N,K]`` 打包为固定容量布局。

    参数：
        tokens: CUDA Token 行 ``[N,H]``，支持 FP16/BF16/FP32。
        ids: 全局专家编号 ``[N,K]``，dtype 为 int32/int64；取值应在
            ``[0,E)``，由外层 EP 接口负责校验。
        topology: 均匀专家放置，提供 EP 大小和本地专家数。
        rows_per_rank: 每个目的 Rank 的固定容量 C，必须满足
            ``C >= max(1,N*K)``。

    返回：
        :class:`FixedRankPackedTensors`，其中 Token、专家编号和 ``restore_index``
        均为连续 Tensor；填充行使用专家编号 ``-1`` 和全零 Token。

    本 Kernel 面向小批量解码，每个来源最多接受 64 个展开分支。每个
    Rank 的 program 扫描同一份很短的 ID 向量，通过稳定前缀计数保证
    每个固定容量位置只有一个写者。Top-K 按 token-major/slot-minor
    顺序展开。

    初始化、位置规划和 gather 都在调用方当前 CUDA Stream 上执行；
    Stream 顺序保证 gather 读取到已经写好的 Pack，无需 Host 回读或
    额外同步。输入允许非连续，函数会先物化为连续布局。
    """
    # 先校验形状、设备和 dtype，确保进入 Kernel 前布局契约成立。
    if tokens.ndim != 2 or ids.ndim != 2 or tokens.shape[0] != ids.shape[0]:
        raise ValueError("expected tokens [N,H] and IDs [N,K] with matching N")
    if ids.shape[1] < 1 or tokens.shape[1] < 1:
        raise ValueError("H and Top-K must be positive")
    if tokens.device.type != "cuda" or ids.device != tokens.device:
        raise ValueError("tokens and IDs must share a CUDA device")
    if ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"IDs must be int32/int64, got {ids.dtype}")
    if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported tokens dtype: {tokens.dtype}")
    if (
        routing_weights.shape != ids.shape
        or routing_weights.dtype != torch.float32
        or routing_weights.device != tokens.device
    ):
        raise TypeError("routing_weights must be FP32 [N,K] on tokens.device")
    branches = ids.numel()
    # B 维按分支数取 2 的幂；超过 64 时回退到通用 PyTorch 打包路径。
    if branches > 64 or rows_per_rank < max(1, branches):
        raise ValueError(
            f"need N*K<=64 and C>=max(1,N*K), got {branches}, {rows_per_rank}"
        )
    x, flat = tokens.contiguous(), ids.contiguous().view(-1)
    rows = topology.ep_size * rows_per_rank
    # -1 初始化 Pack：未被任何分支选中的行保持 padding 标记。
    pack = torch.full((rows,), -1, device=x.device, dtype=torch.int64)
    restore = torch.empty(branches, device=x.device, dtype=torch.int64)
    counts = torch.empty(topology.ep_size, device=x.device, dtype=torch.int64)
    packed_tokens = x.new_empty((rows, x.shape[1]))
    packed_ids = torch.empty_like(pack)
    packed_weights = torch.empty(rows, device=x.device, dtype=torch.float32)
    # 第一步：为每个目的 Rank 生成固定位置与双向索引。
    _fixed_positions_kernel[(topology.ep_size,)](
        flat,
        pack,
        restore,
        counts,
        branches,
        topology.num_local_experts,
        rows_per_rank,
        triton.next_power_of_2(max(1, branches)),
        num_warps=4,
    )
    # 第二步：按 Pack 搬运 Token 行，并同时写出对齐的专家编号。
    _fixed_gather_kernel[(rows, triton.cdiv(x.shape[1], 128))](
        x,
        flat,
        routing_weights.contiguous().view(-1),
        pack,
        packed_tokens,
        packed_ids,
        packed_weights,
        x.shape[1],
        ids.shape[1],
        128,
        num_warps=4,
    )
    return FixedRankPackedTensors(
        packed_tokens, packed_ids, packed_weights, restore, counts, rows_per_rank
    )
