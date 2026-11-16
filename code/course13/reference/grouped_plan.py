"""Independent, readable Host planning for stable Expert-major grouped GEMM."""

import torch


def build_grouped_plan_reference(
    counts: list[int], device: torch.device, block_m: int, *, padding_rows: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Counts [L] in stable Expert order -> offsets [L+1] and tile maps.

    Counts and block_m are nonnegative rows / positive rows per tile. Return
    contiguous int64 tensors on the requested CPU/CUDA device. TileExperts
    stores local Expert IDs and TileRows stores Expert-relative row starts.
    Optional nonnegative padding_rows shifts every offset past a leading
    invalid prefix; tile rows remain Expert-relative. Zero-count Experts have
    no tiles. No Top-K dimension is present. This
    Reference uses only Python/PyTorch and does not import optimized kernels.
    """
    if block_m < 1 or not counts or any(count < 0 for count in counts):
        raise ValueError(
            f"expected nonempty nonnegative counts and block_m>0, "
            f"got {counts=}, {block_m=}"
        )
    if not isinstance(padding_rows, int) or padding_rows < 0:
        raise ValueError(
            f"padding_rows must be a nonnegative integer, got {padding_rows!r}"
        )
    expert_offsets = [padding_rows]
    tile_experts: list[int] = []
    tile_rows: list[int] = []
    for expert, count in enumerate(counts):
        expert_offsets.append(expert_offsets[-1] + count)
        for tile in range((count + block_m - 1) // block_m):
            tile_experts.append(expert)
            tile_rows.append(tile * block_m)

    def tensor(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device)

    return tensor(expert_offsets), tensor(tile_experts), tensor(tile_rows)
