"""Token gather in stable Expert order, with Reference-compatible metadata."""

from __future__ import annotations
import torch
from mini_ep.distributed.dispatch import DispatchOutput, _validate_dispatch_inputs
from mini_ep.kernels.histogram_triton import (
    triton,
    require_triton_cuda,
    stable_dispatch_metadata,
)

if triton is not None:
    import triton.language as tl

    @triton.jit
    def dispatch_kernel(
        X,
        Pack,
        Y,
        S: tl.constexpr,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        record = tl.load(Pack + row)
        token = record // TOP_K
        # H is contiguous; the row index gathers from original Token order.
        values = tl.load(X + token * H + cols, cols < H, other=0)
        tl.store(Y + row * H + cols, values, cols < H)


def dispatch_triton(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> DispatchOutput:
    """Same result fields as dispatch_reference; Token layout [T,H] -> [T*K,H].

    CUDA FP32/FP16/BF16 Tokens, int32/int64 IDs, floating routing weights.
    Inputs share a device. Noncontiguous inputs are materialized here; that
    copy belongs to this public call's latency. Metadata remains CUDA int64.
    This inference path supports empty batches and empty Experts.
    """
    _validate_dispatch_inputs(tokens, expert_ids, routing_weights, num_experts)
    require_triton_cuda(tokens)
    if tokens.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported Token dtype {tokens.dtype}")
    x = tokens.contiguous()
    t, h = x.shape
    k = expert_ids.shape[1]
    s = t * k
    counts, offsets, pack, restore = stable_dispatch_metadata(expert_ids, num_experts)
    packed = torch.empty((s, h), device=x.device, dtype=x.dtype)
    if s:
        dispatch_kernel[(s, triton.cdiv(h, 128))](x, pack, packed, s, h, k, 128)
    original_tokens = torch.arange(
        t, device=x.device, dtype=torch.int64
    ).repeat_interleave(k)
    slots = torch.arange(k, device=x.device, dtype=torch.int64).repeat(t)
    return DispatchOutput(
        packed_tokens=packed,
        packed_expert_ids=expert_ids.reshape(-1).to(torch.int64).index_select(0, pack),
        packed_token_indices=original_tokens.index_select(0, pack),
        packed_route_slots=slots.index_select(0, pack),
        packed_routing_weights=routing_weights.reshape(-1).index_select(0, pack),
        tokens_per_expert=counts,
        expert_offsets=offsets,
        restore_index=restore,
        num_tokens=t,
        top_k=k,
    )
