"""Lesson-10 inverse lookup with the one DeepEP-compatible reduction order."""

import torch
import triton
import triton.language as tl
from mini_ep.distributed.topology import ExpertParallelTopology


@triton.jit
def combine_kernel(
    Returned,
    Restore,
    Ids,
    Output,
    N,
    H: tl.constexpr,
    K: tl.constexpr,
    P: tl.constexpr,
    LOCAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    token, col = offsets // H, offsets % H
    valid = offsets < N * H
    total = tl.full((BLOCK,), 0.0, tl.float32)
    # Branches are already weighted. Preserve model-dtype branch and owner
    # rounding, then sum owners in ascending EP rank, without NCCL reduction.
    for owner in tl.static_range(P):
        partial = tl.full((BLOCK,), 0.0, tl.float32)
        for slot in tl.static_range(K):
            branch = token * K + slot
            packed = tl.load(Restore + branch, valid, 0)
            expert = tl.load(Ids + branch, valid, 0)
            value = tl.load(Returned + packed * H + col, valid, 0)
            rounded = value.to(Output.dtype.element_ty).to(tl.float32)
            partial = partial + tl.where(expert // LOCAL == owner, rounded, 0.0)
        total = total + partial.to(Output.dtype.element_ty).to(tl.float32)
    tl.store(Output + offsets, total, valid)


def combine_triton(
    returned: torch.Tensor,
    restore_index: torch.Tensor,
    expert_ids: torch.Tensor,
    topology: ExpertParallelTopology,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """Rank-packed weighted [R,H] -> original [N,H], model dtype, same CUDA device.

    R=N*K for variable A2A or P*C for fixed capacity. int64 restore_index [N*K]
    maps token-major/slot-minor records into returned rows. expert_ids [N,K] are
    global int32/int64 IDs. All tensors must share a CUDA device; strided inputs
    are materialized. Routing weights must NOT be multiplied again here.
    """
    if model_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"model dtype must be FP32/BF16, got {model_dtype}")
    if returned.ndim != 2 or expert_ids.ndim != 2 or returned.shape[1] < 1:
        raise ValueError("expected returned [R,H], expert_ids [N,K], H>0")
    n, k = expert_ids.shape
    if k < 1 or restore_index.shape != (n * k,) or restore_index.dtype != torch.int64:
        raise ValueError("restore_index must be int64 [N*K], K>0")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert IDs must be integers")
    if returned.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError("unsupported return transport dtype")
    if not returned.is_cuda or any(
        t.device != returned.device for t in (restore_index, expert_ids)
    ):
        raise ValueError("returned values and metadata must share a CUDA device")
    output = torch.empty(
        (n, returned.shape[1]), device=returned.device, dtype=model_dtype
    )
    if n:
        combine_kernel[(triton.cdiv(n * returned.shape[1], 256),)](
            returned.contiguous(),
            restore_index.contiguous(),
            expert_ids.contiguous(),
            output,
            n,
            returned.shape[1],
            k,
            topology.ep_size,
            topology.num_local_experts,
            256,
            enable_fp_fusion=False,
        )
    return output
