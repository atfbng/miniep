"""Independent original-record loop for fixed-capacity destination packing."""

import torch
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.dispatch import FixedRankPackedTensors


def dispatch_fixed_reference(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
    rows_per_rank: int,
) -> FixedRankPackedTensors:
    """[N,H]/[N,K] -> [P*C,H] payload with zero weights/Tokens, expert=-1.

    Host ID reads and individual row copies prioritize clarity, never timing.
    restore_index [N*K] maps original Token/slot records into fixed return slots.
    """
    if (
        tokens.ndim != 2
        or expert_ids.ndim != 2
        or tokens.shape[0] != expert_ids.shape[0]
    ):
        raise ValueError("expected [N,H] and [N,K]")
    if expert_ids.shape[1] < 1 or rows_per_rank < max(1, expert_ids.numel()):
        raise ValueError("invalid K/capacity")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("expert_ids must be int32/int64")
    if (
        routing_weights.shape != expert_ids.shape
        or routing_weights.dtype != torch.float32
    ):
        raise TypeError("routing weights must be FP32 [N,K]")
    if any(x.device != tokens.device for x in (expert_ids, routing_weights)):
        raise ValueError("device mismatch")
    p, k = topology.ep_size, expert_ids.shape[1]
    rows = p * rows_per_rank
    packed = tokens.new_zeros((rows, tokens.shape[1]))
    ids = torch.full((rows,), -1, device=tokens.device, dtype=torch.int64)
    weights = routing_weights.new_zeros(rows)
    restore = torch.empty(expert_ids.numel(), device=tokens.device, dtype=torch.int64)
    counts = [0] * p
    for branch, expert in enumerate(expert_ids.reshape(-1).tolist()):
        if not 0 <= expert < topology.num_experts:
            raise ValueError(f"invalid expert ID {expert}")
        rank = expert // topology.num_local_experts
        row = rank * rows_per_rank + counts[rank]
        packed[row] = tokens[branch // k]
        ids[row] = expert
        weights[row] = routing_weights.reshape(-1)[branch]
        restore[branch] = row
        counts[rank] += 1
    return FixedRankPackedTensors(
        packed,
        ids,
        weights,
        restore,
        torch.tensor(counts, device=tokens.device, dtype=torch.int64),
        rows_per_rank,
    )
