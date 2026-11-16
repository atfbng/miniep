"""Fused inverse permutation, routing multiplication and Top-K reduction."""

from __future__ import annotations
import torch
from mini_ep.distributed.dispatch import DispatchOutput
from mini_ep.kernels.histogram_triton import triton, require_triton_cuda

if triton is not None:
    import triton.language as tl

    @triton.jit
    def combine_kernel(
        Y,
        Restore,
        Weights,
        Out,
        H: tl.constexpr,
        TOP_K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros((BLOCK_H,), tl.float32)
        for slot in tl.static_range(TOP_K):
            packed_row = tl.load(Restore + token * TOP_K + slot)
            value = tl.load(Y + packed_row * H + cols, cols < H, other=0)
            weight = tl.load(Weights + packed_row).to(value.dtype)
            # Round each product to Expert-output dtype, like the Reference.
            product = (value.to(tl.float32) * weight.to(tl.float32)).to(value.dtype)
            acc = acc + product.to(tl.float32)
        tl.store(Out + token * H + cols, acc, cols < H)


def combine_triton(
    packed_outputs: torch.Tensor, dispatch: DispatchOutput
) -> torch.Tensor:
    """Expert-order [T*K,H] -> original-order [T,H], inference only.

    CUDA FP32/FP16/BF16; Top-1/Top-2. dispatch.restore_index is CUDA int64
    [T*K], mapping original Token/slot records to packed rows. The paired
    packed_routing_weights have the same row order as packed_outputs.
    """
    require_triton_cuda(packed_outputs)
    if dispatch.top_k not in (1, 2):
        raise ValueError(f"Top-K must be 1 or 2, got {dispatch.top_k}")
    s = dispatch.num_tokens * dispatch.top_k
    if packed_outputs.ndim != 2 or packed_outputs.shape[0] != s:
        raise ValueError(f"need outputs [{s},H], got {tuple(packed_outputs.shape)}")
    if packed_outputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {packed_outputs.dtype}")
    if dispatch.restore_index.shape != (
        s,
    ) or dispatch.packed_routing_weights.shape != (s,):
        raise ValueError(
            "restore index and weights must contain one entry per assignment"
        )
    if dispatch.restore_index.dtype != torch.int64:
        raise TypeError("restore index must be int64")
    if (
        dispatch.restore_index.device != packed_outputs.device
        or dispatch.packed_routing_weights.device != packed_outputs.device
    ):
        raise ValueError("outputs and metadata must share device")
    if not dispatch.packed_routing_weights.is_floating_point():
        raise TypeError("routing weights must be floating")
    y = packed_outputs.contiguous()
    restore = dispatch.restore_index.contiguous()
    weights = dispatch.packed_routing_weights.contiguous()
    h = y.shape[1]
    out = torch.empty((dispatch.num_tokens, h), device=y.device, dtype=y.dtype)
    if dispatch.num_tokens:
        combine_kernel[(dispatch.num_tokens, triton.cdiv(h, 128))](
            y, restore, weights, out, h, dispatch.top_k, 128, enable_fp_fusion=False
        )
    return out
