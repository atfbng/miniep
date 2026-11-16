"""GPU tile metadata extending the lesson GroupedGemmPlan."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class GroupedGemmPlan:
    """Expert-major plan: int64 contiguous GPU offsets and tile maps.

    offsets [L+1] skip an optional invalid prefix; tile_experts/tile_rows [Q]
    identify an Expert and an Expert-relative row tile. Unused capacity has
    expert=-1. num_rows includes padding; block_m matches the projection tile.
    Unlike Host Reference counts, these fields need no scalar GPU read.
    """

    offsets: torch.Tensor
    tile_experts: torch.Tensor
    tile_rows: torch.Tensor
    block_m: int
    num_rows: int
