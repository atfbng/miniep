"""Variable-M Expert batches, schedules and independent PyTorch references."""

from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import torch


@dataclass(frozen=True)
class GroupedGemmPlan:
    counts: tuple[int, ...]
    offsets: torch.Tensor
    tile_experts: torch.Tensor
    tile_rows: torch.Tensor
    block_m: int = 16


def create_grouped_plan(
    counts: Sequence[int], device, block_m: int = 16
) -> GroupedGemmPlan:
    """CPU counts [L] -> device int64 offsets [L+1] and tile maps [Q].

    Rows are contiguous Expert-major. Empty Experts generate no tiles. The
    plan is reusable while counts/order are fixed; construction is separate
    from GEMM execution. block_m is the kernel's output-row tile size.
    """
    if not counts or any(
        isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts
    ):
        raise ValueError(
            f"counts must be a nonempty sequence of nonnegative integers, got {counts}"
        )
    if block_m != 16:
        raise ValueError(f"teaching kernel uses block_m=16, got {block_m}")
    offsets = [0]
    experts = []
    rows = []
    for e, m in enumerate(counts):
        offsets.append(offsets[-1] + m)
        for tile in range((m + block_m - 1) // block_m):
            experts.append(e)
            rows.append(tile * block_m)
    tensor = lambda a: torch.tensor(a, dtype=torch.int64, device=device)
    return GroupedGemmPlan(
        tuple(counts), tensor(offsets), tensor(experts), tensor(rows), block_m
    )


def validate_grouped_inputs(
    x: torch.Tensor, weights: torch.Tensor, plan: GroupedGemmPlan
) -> None:
    if x.ndim != 2 or weights.ndim != 3:
        raise ValueError(f"need X [S,K], W [L,N,K], got {x.shape}, {weights.shape}")
    if (
        x.shape[0] != sum(plan.counts)
        or weights.shape[0] != len(plan.counts)
        or x.shape[1] != weights.shape[2]
    ):
        raise ValueError("X/W dimensions must agree with counts and reduction size")
    if x.shape[1] <= 0 or weights.shape[1] <= 0:
        raise ValueError("K and N must be positive")
    if (
        x.device != weights.device
        or x.dtype != weights.dtype
        or plan.offsets.device != x.device
    ):
        raise ValueError("X, W and plan must share device; X and W must share dtype")
    if not x.is_contiguous() or not weights.is_contiguous():
        raise ValueError(
            "grouped GEMM expects contiguous row-major X and Linear-layout W"
        )
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {x.dtype}")


def expert_loop(
    x: torch.Tensor, weights: torch.Tensor, plan: GroupedGemmPlan
) -> torch.Tensor:
    """Independent Reference: [S,K] @ per-Expert [N,K].T -> [S,N]."""
    validate_grouped_inputs(x, weights, plan)
    out = torch.empty((x.shape[0], weights.shape[1]), device=x.device, dtype=x.dtype)
    start = 0
    for e, m in enumerate(plan.counts):
        if m:
            out[start : start + m] = x[start : start + m] @ weights[e].T
        start += m
    return out


def expert_mlp_loop(
    x: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    plan: GroupedGemmPlan,
) -> torch.Tensor:
    """Bias-free GELU Experts: W_up [L,F,H], W_down [L,H,F]."""
    validate_grouped_inputs(x, up_weights, plan)
    if down_weights.shape != (len(plan.counts), x.shape[1], up_weights.shape[1]):
        raise ValueError("down weights must have shape [L,H,F]")
    if (
        down_weights.dtype != x.dtype
        or down_weights.device != x.device
        or not down_weights.is_contiguous()
    ):
        raise ValueError("down weights must share dtype/device and be contiguous")
    out = torch.empty_like(x)
    start = 0
    for e, m in enumerate(plan.counts):
        if m:
            hidden = torch.nn.functional.gelu(x[start : start + m] @ up_weights[e].T)
            out[start : start + m] = hidden @ down_weights[e].T
        start += m
    return out


def pack_local_experts(
    recv_tokens: torch.Tensor, local_expert_index: Sequence[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    """Source-major [R,H] -> Expert-major [R,H] and origin-row permutation.

    local_expert_index lists CUDA/CPU int64 rows per local Expert. The rows
    cover the receive buffer once. counts is a Host tuple used for planning.
    """
    counts = tuple(rows.numel() for rows in local_expert_index)
    origin = torch.cat(local_expert_index)
    packed = recv_tokens.index_select(0, origin).contiguous()
    return packed, origin, counts
