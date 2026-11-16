"""One launch across all variable-M Expert GEMMs, with explicit tile maps."""

from __future__ import annotations
import torch
from mini_ep.kernels.histogram_triton import triton, require_triton_cuda
from mini_ep.kernels.grouped_gemm.reference import (
    validate_grouped_inputs,
    GroupedGemmPlan,
)

if triton is not None:
    import triton.language as tl

    @triton.jit
    def grouped_gemm_kernel(
        X,
        W,
        Y,
        Offsets,
        TileExperts,
        TileRows,
        K: tl.constexpr,
        N: tl.constexpr,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        tile = tl.program_id(0)
        expert = tl.load(TileExperts + tile)
        first = tl.load(Offsets + expert)
        end = tl.load(Offsets + expert + 1)
        rows = first + tl.load(TileRows + tile) + tl.arange(0, BM)
        cols = tl.program_id(1) * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), tl.float32)
        for block in range(tl.cdiv(K, BK)):
            ks = block * BK + tl.arange(0, BK)
            a = tl.load(
                X + rows[:, None] * K + ks[None, :],
                (rows[:, None] < end) & (ks[None, :] < K),
                other=0,
            )
            # W is [L,N,K]; this address expresses its logical transpose.
            b = tl.load(
                W + expert * N * K + cols[None, :] * K + ks[:, None],
                (cols[None, :] < N) & (ks[:, None] < K),
                other=0,
            )
            acc += tl.dot(a, b, input_precision="ieee")
        tl.store(
            Y + rows[:, None] * N + cols[None, :],
            acc,
            (rows[:, None] < end) & (cols[None, :] < N),
        )


def grouped_gemm(
    x: torch.Tensor, weights: torch.Tensor, plan: GroupedGemmPlan
) -> torch.Tensor:
    """Same [S,N] layout and dtype as expert_loop; fixed 16x32x32 tiles."""
    validate_grouped_inputs(x, weights, plan)
    require_triton_cuda(x)
    if plan.tile_experts.device != x.device or plan.tile_rows.device != x.device:
        raise ValueError("tile maps and X must share device")
    out = torch.empty((x.shape[0], weights.shape[1]), device=x.device, dtype=x.dtype)
    q = plan.tile_experts.numel()
    if q:
        grouped_gemm_kernel[(q, triton.cdiv(weights.shape[1], 32))](
            x,
            weights,
            out,
            plan.offsets,
            plan.tile_experts,
            plan.tile_rows,
            x.shape[1],
            weights.shape[1],
            plan.block_m,
            32,
            32,
        )
    return out


def grouped_expert_mlp(
    x: torch.Tensor,
    up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    plan: GroupedGemmPlan,
) -> torch.Tensor:
    """Two grouped projections around the same GELU used by the Reference."""
    hidden = grouped_gemm(x, up_weights, plan)
    activated = torch.nn.functional.gelu(hidden)
    return grouped_gemm(activated.contiguous(), down_weights, plan)


def compute_grouped_local_experts(
    model, recv_tokens: torch.Tensor, layout
) -> torch.Tensor:
    """Pack source-major receive rows, execute grouped MLP, restore receive order.

    Layout indices cover recv_tokens [R,H] once. Up/down weights are stacked
    as [L,F,H]/[L,H,F]. This full teaching path includes packing, schedule and
    weight preparation; the microbenchmark times GEMM with a prepared plan.
    """
    from mini_ep.kernels.grouped_gemm.reference import (
        pack_local_experts,
        create_grouped_plan,
    )

    if any(
        expert.activation_name != "gelu" or expert.use_bias for expert in model.experts
    ):
        raise ValueError("grouped teaching path expects bias-free GELU Experts")
    packed, origin, counts = pack_local_experts(recv_tokens, layout.local_expert_index)
    plan = create_grouped_plan(counts, recv_tokens.device)
    up = torch.stack([expert.up_proj.weight for expert in model.experts]).contiguous()
    down = torch.stack(
        [expert.down_proj.weight for expert in model.experts]
    ).contiguous()
    computed = grouped_expert_mlp(packed, up, down, plan)
    result = torch.empty_like(recv_tokens)
    result.index_copy_(0, origin, computed)
    return result
