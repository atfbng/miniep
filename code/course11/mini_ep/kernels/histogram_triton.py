"""Stable Dispatch metadata: block histograms and deterministic row positions."""

from __future__ import annotations
import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def histogram_kernel(
        Ids,
        Partial,
        S: tl.constexpr,
        E: tl.constexpr,
        EPAD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block = tl.program_id(0)
        records = block * BLOCK + tl.arange(0, BLOCK)
        ids = tl.load(Ids + records, records < S, other=EPAD).to(tl.int32)
        # Padding IDs lie outside the histogram's bin range.
        counts = tl.histogram(ids, EPAD)
        experts = tl.arange(0, EPAD)
        tl.store(Partial + block * E + experts, counts, experts < E)

    @triton.jit
    def stable_positions_kernel(
        Ids,
        Prefix,
        Offsets,
        Pack,
        Restore,
        S: tl.constexpr,
        E: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        block = tl.program_id(0)
        records = block * BLOCK + tl.arange(0, BLOCK)
        ids = tl.load(Ids + records, records < S, other=-1)
        for expert in tl.static_range(E):
            selected = (records < S) & (ids == expert)
            # Earlier blocks + earlier matching records in this block.
            within = tl.cumsum(selected.to(tl.int32)) - 1
            start = tl.load(Offsets + expert) + tl.load(Prefix + block * E + expert)
            packed_row = start + within
            tl.store(Pack + packed_row, records, selected)
            tl.store(Restore + records, packed_row, selected)


def require_triton_cuda(tensor: torch.Tensor) -> None:
    if triton is None:
        raise RuntimeError("install Triton to run the CUDA kernel path")
    if tensor.device.type != "cuda":
        raise ValueError(f"Triton path requires CUDA, got {tensor.device}")


def stable_dispatch_metadata(
    expert_ids: torch.Tensor, num_experts: int, block: int = 128
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """[T,K] IDs -> counts [E], offsets [E+1], pack/restore [T*K].

    All output metadata is CUDA int64, contiguous. IDs are valid global
    Expert IDs. Packing is stable with respect to flattened Token/slot order.
    Prefix sums use PyTorch CUDA operations; the histogram and position
    assignment use Triton. Complexity is O(S*E) for the position pass.
    """
    require_triton_cuda(expert_ids)
    if (
        num_experts <= 0
        or expert_ids.ndim != 2
        or expert_ids.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError(
            f"need integer [T,K] IDs and E > 0, got {expert_ids.shape}, {expert_ids.dtype}, E={num_experts}"
        )
    if block <= 0 or block & (block - 1):
        raise ValueError(f"block must be power of two, got {block}")
    ids = expert_ids.contiguous().reshape(-1)
    s = ids.numel()
    pack = torch.empty(s, dtype=torch.int64, device=ids.device)
    restore = torch.empty_like(pack)
    if s == 0:
        counts = torch.zeros(num_experts, dtype=torch.int64, device=ids.device)
        return (
            counts,
            torch.zeros(num_experts + 1, dtype=torch.int64, device=ids.device),
            pack,
            restore,
        )
    partial = torch.empty(
        (triton.cdiv(s, block), num_experts), dtype=torch.int64, device=ids.device
    )
    histogram_kernel[(partial.shape[0],)](
        ids, partial, s, num_experts, triton.next_power_of_2(num_experts), block
    )
    counts = partial.sum(0)
    offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
    prefix = (partial.cumsum(0) - partial).contiguous()
    stable_positions_kernel[(partial.shape[0],)](
        ids, prefix, offsets, pack, restore, s, num_experts, block
    )
    return counts, offsets, pack, restore
