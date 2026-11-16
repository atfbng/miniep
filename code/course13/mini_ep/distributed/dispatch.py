"""Stable destination-rank packing of Token/slot records, extending lesson 6."""

from dataclasses import dataclass
import torch
from mini_ep.distributed.topology import ExpertParallelTopology


@dataclass(frozen=True, slots=True)
class RankPackedTensors:
    """Rank-major [N*K,H] rows and aligned identities/FP32 weights.

    pack_index[p] is the original token-major/slot-minor record. Its inverse
    restore_index[b] locates the returned row of record b. send_splits [P]
    are int64 device row counts, not bytes. All fields share the input device.
    Direct rank packing composes the two lesson source permutations, avoiding
    an extra Expert-major copy before communication.
    """

    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_routing_weights: torch.Tensor
    pack_index: torch.Tensor
    restore_index: torch.Tensor
    send_splits: torch.Tensor


@dataclass(frozen=True, slots=True)
class FixedRankPackedTensors:
    """Rank-major [P*C,H] capacity; each destination has a valid prefix.

    restore_index [N*K] maps original records into their fixed slots. Padding
    is zero Token/weight and expert=-1. valid_rows [P] is device diagnostic
    metadata; actual communication sends C rows to each peer without reading it.
    """

    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_routing_weights: torch.Tensor
    restore_index: torch.Tensor
    valid_rows: torch.Tensor
    rows_per_rank: int


def _validate(tokens, expert_ids, routing_weights):
    if (
        tokens.ndim != 2
        or expert_ids.ndim != 2
        or tokens.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError("expected tokens [N,H] and expert_ids [N,K]")
    if tokens.shape[1] < 1 or expert_ids.shape[1] < 1:
        raise ValueError("H and Top-K must be positive")
    if tokens.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise TypeError(f"packing requires floating tokens, got {tokens.dtype}")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be int32/int64")
    if (
        routing_weights.shape != expert_ids.shape
        or routing_weights.dtype != torch.float32
    ):
        raise TypeError("routing_weights must be FP32 [N,K]")
    if expert_ids.device != tokens.device or routing_weights.device != tokens.device:
        raise ValueError("tokens, IDs and weights must share a device")


def pack_by_destination_rank(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
) -> RankPackedTensors:
    """Original [N,H]/[N,K] -> contiguous rank-packed expanded records.

    Stable within each rank: Token then slot. Payload dtype follows inputs;
    global IDs are int64 and weights stay FP32. Strided inputs are allowed.
    This function changes layout only; outer MoE checks ID range/model dtype.
    """
    _validate(tokens, expert_ids, routing_weights)
    flat = expert_ids.reshape(-1).long()
    destinations = topology.destination_rank(flat)
    pack_index = torch.argsort(destinations, stable=True)
    return RankPackedTensors(
        tokens.index_select(0, pack_index // expert_ids.shape[1]).contiguous(),
        flat.index_select(0, pack_index),
        routing_weights.reshape(-1).index_select(0, pack_index),
        pack_index,
        torch.argsort(pack_index),
        torch.bincount(destinations, minlength=topology.ep_size),
    )


def pack_by_destination_rank_fixed(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
    rows_per_rank: int,
) -> FixedRankPackedTensors:
    """Same record identity in [P*C,H], C>=max(1,N*K); zero/-1 padding.

    Counts and permutations stay on device. The independent per-record loop is
    in reference/fixed_dispatch.py; small decode has a matching Triton packer.
    """
    _validate(tokens, expert_ids, routing_weights)
    if rows_per_rank < max(1, expert_ids.numel()):
        raise ValueError(
            f"C={rows_per_rank} must cover max(1,N*K)={max(1,expert_ids.numel())}"
        )
    branches = expert_ids.numel()
    k = expert_ids.shape[1]
    flat = expert_ids.reshape(-1).long()
    destinations = topology.destination_rank(flat)
    counts = torch.bincount(destinations, minlength=topology.ep_size)
    order = torch.argsort(destinations, stable=True)
    sorted_destinations = destinations.index_select(0, order)
    starts = torch.cumsum(counts, dim=0) - counts
    within_sorted = torch.arange(
        branches, device=expert_ids.device
    ) - starts.index_select(0, sorted_destinations)
    within = torch.empty(branches, device=expert_ids.device, dtype=torch.int64)
    within.index_copy_(0, order, within_sorted)
    restore_index = destinations * rows_per_rank + within
    rows = topology.ep_size * rows_per_rank
    packed_tokens = tokens.new_zeros((rows, tokens.shape[1]))
    packed_ids = torch.full((rows,), -1, device=expert_ids.device, dtype=torch.int64)
    packed_weights = routing_weights.new_zeros(rows)
    destination_rows = sorted_destinations * rows_per_rank + within_sorted
    packed_tokens.index_copy_(
        0, destination_rows, tokens.index_select(0, order // k).contiguous()
    )
    packed_ids.index_copy_(0, destination_rows, flat.index_select(0, order))
    packed_weights.index_copy_(0, restore_index, routing_weights.reshape(-1))
    return FixedRankPackedTensors(
        packed_tokens.contiguous(),
        packed_ids.contiguous(),
        packed_weights,
        restore_index.contiguous(),
        counts,
        rows_per_rank,
    )
