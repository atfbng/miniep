"""Compare NVSHMEM files against synchronous NCCL Pure EP, same two-projection GELU MLP."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_ep.distributed import (
    nccl_distributed,
    dispatch_reference,
    pack_by_destination_rank,
    all_gather_expert_counts,
    build_receive_layout,
    variable_split_all_to_all,
    combine_reference,
)
from mini_ep.distributed.topology import ExpertParallelTopology


def reference_exchange(tokens, expert_ids, weights, topology, group):
    """FP32 CUDA [N,4]; metadata int64; return original Token [N,4]."""
    dispatch = dispatch_reference(tokens, expert_ids, weights, topology.num_experts)
    packed = pack_by_destination_rank(dispatch, topology)
    counts = all_gather_expert_counts(dispatch.tokens_per_expert, group=group)
    layout = build_receive_layout(
        topology, counts, local_tokens_per_expert=dispatch.tokens_per_expert
    )
    send, recv = packed.send_splits.tolist(), layout.recv_splits.tolist()
    received = variable_split_all_to_all(packed.packed_tokens, send, recv, group=group)
    w = torch.arange(1, 17, device=tokens.device, dtype=torch.float32).reshape(4, 4) * (
        0.01 * (topology.ep_rank + 1)
    )
    computed = torch.nn.functional.gelu(received @ w.T) @ w.T
    returned = variable_split_all_to_all(computed, recv, send, group=group)
    expert_major = torch.empty_like(returned)
    expert_major.index_copy_(0, packed.row_origin_index, returned)
    return combine_reference(expert_major, dispatch), packed, layout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    with nccl_distributed(timeout_seconds=120) as ctx, torch.no_grad():
        cases = json.load(open(a.prefix + f"-rank{ctx.global_rank}.json"))
        topology = ExpertParallelTopology(
            ctx.world_size, ctx.world_size, ctx.global_rank
        )
        x = (
            torch.arange(
                ctx.global_rank * 28 + 1,
                ctx.global_rank * 28 + 29,
                device=ctx.device,
                dtype=torch.float32,
            )
            * 0.01
        ).reshape(7, 4)
        for case in cases:
            if case.get("expert") != "mlp":
                raise ValueError("Expected NVSHMEM results with expert=mlp")
            k = case["top_k"]
            q = torch.arange(
                ctx.global_rank * 7 * k,
                (ctx.global_rank + 1) * 7 * k,
                device=ctx.device,
            )
            ids = (q % ctx.world_size).reshape(7, k)
            if case["scenario"]:
                ids[:, 0] = 0
                if k == 2:
                    ids[:, 1] = ctx.world_size - 1
            weights = torch.full((7, k), 1 / k, device=ctx.device, dtype=torch.float32)
            expected, packed, layout = reference_exchange(
                x * case.get("input_scale", 1),
                ids,
                weights,
                topology,
                ctx.group,
            )
            assert case["send_rows"] == packed.send_splits.tolist()
            assert case["recv_rows"] == layout.recv_splits.tolist()
            assert (
                case["origin"]
                == (
                    packed.packed_token_indices * k + packed.packed_route_slots
                ).tolist()
            )
            actual = torch.tensor(
                case["values"], device=ctx.device, dtype=torch.float32
            ).reshape(7, 4)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
        print(
            "NCCL/NVSHMEM",
            ctx.world_size,
            ctx.global_rank,
            len(cases),
            "PASS",
            flush=True,
        )


if __name__ == "__main__":
    main()
