"""Lesson 5: trace variable-row NCCL exchanges with an independent oracle."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_ep.distributed import (
    ExpertParallelTopology, combine_reference, dispatch_reference,
    equal_split_all_to_all, nccl_distributed, variable_split_all_to_all,
)


def exchange_splits(send_splits: list[int], device: torch.device, group=None) -> list[int]:
    """Send one int64 row to each group rank; receive counts in source order."""
    counts = torch.tensor(send_splits, dtype=torch.int64, device=device)
    return equal_split_all_to_all(counts, group=group).cpu().tolist()


def run_case(context, *, top_k: int, case: str, dtype: torch.dtype) -> None:
    """Exchange [assignments,H] rows, scale by expert ID, return and combine.

    Every rank enters count/token/local-ID/return collectives in that order,
    including ranks with zero rows. IDs are int64; payloads preserve dtype and
    device. Scaling writes one output per receive row in the same row order.
    """
    rank, size, device = context.global_rank, context.world_size, context.device
    topology = ExpertParallelTopology(2 * size, size, rank)
    count = 3 if rank == 0 else 2
    if case == "empty_source" and rank == 0 or case == "all_empty":
        count = 0
    tokens = torch.arange(count * 2, device=device, dtype=dtype).reshape(count, 2)
    tokens = tokens + 1 + rank * 10
    if case == "routing" and size == 2:
        ids = torch.tensor([[2, 0], [1, 2], [0, 1]] if rank == 0 else [[3, 0], [2, 1]],
                           device=device, dtype=torch.int64)[:, :top_k]
        weights = torch.tensor([[.25, .75], [.60, .40], [.20, .80]] if rank == 0 else [[.5, .5], [.5, .5]],
                               device=device, dtype=torch.float32)[:, :top_k]
    else:
        ids = (torch.arange(count * top_k, device=device).reshape(count, top_k)
               + rank) % topology.num_experts
        if case == "skew":
            ids = torch.arange(top_k, device=device).expand(count, top_k).contiguous()
        weights = torch.full((count, top_k), 1 / top_k, device=device, dtype=torch.float32)
    weights = weights / weights.sum(dim=1, keepdim=True)
    dispatch = dispatch_reference(tokens, ids, weights, topology.num_experts)
    sends = dispatch.tokens_per_expert.reshape(size, topology.experts_per_rank).sum(1).cpu().tolist()
    receives = exchange_splits(sends, device, context.group)
    recv_tokens = variable_split_all_to_all(dispatch.packed_tokens, sends, receives, group=context.group)
    local_ids = topology.local_expert_id(dispatch.packed_expert_ids).contiguous()
    recv_ids = variable_split_all_to_all(local_ids, sends, receives, group=context.group)
    global_ids = recv_ids + rank * topology.experts_per_rank
    computed = recv_tokens * (global_ids + 1).to(dtype)[:, None]
    returned = variable_split_all_to_all(computed.contiguous(), receives, sends, group=context.group)
    output = combine_reference(returned, dispatch)
    # Independent oracle follows original Token/slot order, without any packing.
    expected = torch.zeros_like(tokens)
    for token in range(count):
        for slot in range(top_k):
            expected[token] += (tokens[token] * (ids[token, slot] + 1).to(dtype)) * weights[token, slot].to(dtype)
    torch.cuda.synchronize(device)
    tolerance = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
    torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(returned, dispatch.packed_tokens * (dispatch.packed_expert_ids + 1).to(dtype)[:, None])
    assert output.shape == tokens.shape and output.dtype == dtype and output.device == device
    assert recv_tokens.shape == (sum(receives), tokens.shape[1])
    assert recv_ids.dtype == torch.int64 and recv_ids.device == device
    assert all(t.is_contiguous() for t in (recv_tokens, recv_ids, returned, output))
    print(f"[rank={rank} local_rank={context.local_rank} ep_size={size} device={device}] "
          f"case={case} top_k={top_k} send_splits={sends} recv_splits={receives} "
          f"received_local_ids={recv_ids.cpu().tolist()} PASS", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    args = parser.parse_args()
    with nccl_distributed(timeout_seconds=args.timeout_seconds) as context:
        if args.expected_world_size is not None and args.expected_world_size != context.world_size:
            raise ValueError(f"expected world size {args.expected_world_size}, got {context.world_size}")
        cases = ("routing", "skew", "empty_source", "all_empty") if args.suite else ("routing",)
        with torch.inference_mode():
            for case in cases:
                for k in (1, 2):
                    run_case(context, top_k=k, case=case, dtype=getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
