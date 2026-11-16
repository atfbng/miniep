#!/usr/bin/env python3
"""Lesson 6: Mini-EP V1 on 2 ranks, validated against the single-GPU reference.

Run with two ranks on two GPUs::

    torchrun --standalone --nproc-per-node=2 test/demo.py \
        --expected-world-size=2 --timeout-seconds=120

Every rank builds the *same* single-GPU reference on the CPU from a fixed
seed, keeps only its own expert shard, and runs
Router -> Dispatch -> All-to-All #1 -> local Experts -> All-to-All #2 ->
Unpermute -> Combine on its own slice of the token batch.  The last step of the
example gathers every rank's output back into the full batch and compares it
with the single-GPU reference on rank 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import torch.distributed as dist

# Keep the course example runnable before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_ep.distributed import DistributedContext, nccl_distributed  # noqa: E402
from mini_ep.model import SingleGPUMoE, distribute_single_gpu_moe  # noqa: E402


# Gate rows are scaled by these factors in the ``skewed`` scenario.  The first
# expert attracts most assignments, so later experts can end up idle and the
# send/recv splits become visibly imbalanced.
SKEWED_GATE_SCALE = (8.0, 4.0, 0.05, 0.05)

# Distributed and single-GPU runs execute the same operations on different
# devices, so FP16/BF16 need a looser tolerance than FP32.  The values are the
# comparison budgets used by this example, not claims about accuracy.
COMPARISON_TOLERANCES = {
    "float32": (1e-4, 1e-5),
    "float16": (1e-2, 2e-3),
    "bfloat16": (2e-2, 8e-3),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--expert-hidden-size", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--dtype",
        choices=tuple(COMPARISON_TOLERANCES),
        default="float32",
    )
    parser.add_argument(
        "--scenario",
        choices=("random", "skewed"),
        default="random",
        help="routing regime: uniform random gate, or a strongly tilted gate",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_reference(
    args: argparse.Namespace,
) -> tuple[SingleGPUMoE, torch.Tensor]:
    """Build one identical reference model and token batch on every rank.

    Returns:
        The FP32 single-GPU model and the full token batch, both on the CPU.
        Fixed seeds make every rank derive the same weights and the same
        Tokens; the EP run then only differs in *where* the experts live.
    """

    torch.manual_seed(args.seed)
    model = SingleGPUMoE(
        hidden_size=args.hidden_size,
        expert_hidden_size=args.expert_hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    if args.scenario == "skewed":
        if len(SKEWED_GATE_SCALE) != args.num_experts:
            raise ValueError(
                "the skewed scenario is defined for "
                f"{len(SKEWED_GATE_SCALE)} experts, got {args.num_experts}"
            )
        with torch.no_grad():
            for expert_id, scale in enumerate(SKEWED_GATE_SCALE):
                model.router.gate.weight[expert_id].mul_(scale)

    generator = torch.Generator().manual_seed(args.seed + 1)
    tokens = torch.randn(
        args.tokens,
        args.hidden_size,
        generator=generator,
    )
    return model, tokens


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def describe_comparison(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype_name: str,
) -> tuple[bool, str]:
    """Compare a distributed result with the single-GPU reference."""

    rtol, atol = COMPARISON_TOLERANCES[dtype_name]
    max_abs_diff = (actual.float() - expected.float()).abs().max().item()
    summary = (
        f"max_abs_diff={max_abs_diff:.3e} rtol={rtol:g} atol={atol:g}"
    )
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        first_line = str(error).strip().splitlines()[0]
        return False, f"{summary} {first_line}"
    return True, summary


def run_lesson(context: DistributedContext, args: argparse.Namespace) -> None:
    """Execute Mini-EP V1 on this rank and verify it against the reference."""

    ep_size = context.world_size
    ep_rank = context.global_rank
    device = context.device
    group = context.group

    if args.tokens % ep_size != 0:
        raise ValueError(
            "the token batch must split evenly over the EP group so that the "
            f"final comparison can all-gather it, got tokens={args.tokens} and "
            f"ep_size={ep_size}"
        )
    tokens_per_rank = args.tokens // ep_size
    dtype = resolve_dtype(args.dtype)

    reference, all_tokens = build_reference(args)
    local_slice = slice(ep_rank * tokens_per_rank, (ep_rank + 1) * tokens_per_rank)
    local_tokens = all_tokens[local_slice].to(device=device, dtype=dtype)

    ep_model = distribute_single_gpu_moe(
        reference,
        ep_size=ep_size,
        ep_rank=ep_rank,
    ).to(device=device, dtype=dtype)

    # ``no_grad`` rather than ``inference_mode``: the collectives write into
    # preallocated buffers, and inference tensors reject such in-place updates
    # outside InferenceMode.
    with torch.no_grad():
        result = ep_model(local_tokens, group=group)

    failures: list[str] = []

    # Metadata invariants that hold on every rank.
    num_assignments = result.tokens_per_expert.sum().item()
    if num_assignments != tokens_per_rank * args.top_k:
        failures.append(
            "tokens_per_expert must sum to local_tokens * top_k, got "
            f"{num_assignments} and {tokens_per_rank * args.top_k}"
        )
    if int(result.send_splits.sum().item()) != num_assignments:
        failures.append("send_splits must sum to the number of assignments")
    if int(result.local_expert_rows.sum().item()) != int(
        result.recv_splits.sum().item()
    ):
        failures.append(
            "local experts must process every received row exactly once"
        )
    if result.hidden_states.device != device:
        failures.append(
            f"hidden_states device must be {device}, got {result.hidden_states.device}"
        )
    if result.hidden_states.dtype != dtype:
        failures.append(
            f"hidden_states dtype must be {dtype}, got {result.hidden_states.dtype}"
        )

    # Cross-rank invariant: rank r must receive exactly what the other ranks
    # send to r.  Gathering send_splits gives every rank the full matrix.
    send_matrix = torch.zeros((ep_size, ep_size), dtype=torch.int64, device=device)
    send_matrix[ep_rank] = result.send_splits
    dist.all_reduce(send_matrix, op=dist.ReduceOp.SUM, group=group)
    expected_recv = send_matrix[:, ep_rank].contiguous()
    if not torch.equal(expected_recv, result.recv_splits):
        failures.append(
            "recv_splits must equal the send_splits that peers aimed at this "
            f"rank, got {result.recv_splits.tolist()} and {expected_recv.tolist()}"
        )

    # Verification-only gather: rebuild the full batch output from all ranks.
    full_output = torch.empty(
        (args.tokens, args.hidden_size), dtype=dtype, device=device
    )
    dist.all_gather_into_tensor(
        full_output, result.hidden_states.contiguous(), group=group
    )

    comparison = "not compared on this rank"
    if ep_rank == 0:
        reference_model = reference.to(device=device, dtype=dtype)
        with torch.no_grad():
            expected = reference_model(
                all_tokens.to(device=device, dtype=dtype)
            ).hidden_states
        passed, comparison = describe_comparison(full_output, expected, args.dtype)
        if not passed:
            failures.append(f"distributed EP != single-GPU reference: {comparison}")

    # Make every rank agree on the outcome before any rank exits, so a failure
    # never leaves a peer waiting inside a collective.
    failure_flag = torch.tensor(
        [1 if failures else 0], dtype=torch.int64, device=device
    )
    dist.all_reduce(failure_flag, op=dist.ReduceOp.MAX, group=group)
    failed = bool(failure_flag.item())

    head = (
        f"[global_rank={ep_rank} local_rank={context.local_rank} "
        f"world_size={ep_size} device={device} ep_rank={ep_rank} "
        f"ep_size={ep_size} scenario={args.scenario} dtype={args.dtype} "
        f"top_k={args.top_k} tokens={args.tokens}]"
    )
    print(
        f"{head} local_expert_ids={list(ep_model.topology.local_expert_ids)} "
        f"expert_ids={result.expert_ids.cpu().tolist()} "
        f"tokens_per_expert_all="
        f"{result.tokens_per_expert_all.cpu().tolist()} "
        f"send_splits={result.send_splits.cpu().tolist()} "
        f"recv_splits={result.recv_splits.cpu().tolist()} "
        f"local_expert_rows={result.local_expert_rows.cpu().tolist()}",
        flush=True,
    )
    print(f"{head} reference_check={comparison}", flush=True)
    for failure in failures:
        print(f"{head} FAIL {failure}", flush=True)
    print(f"{head} {'FAIL' if failed else 'PASS'}", flush=True)

    if failed:
        raise RuntimeError(
            f"Mini-EP V1 verification failed on rank {ep_rank}: {failures}"
        )


def main() -> None:
    args = parse_args()
    if args.expected_world_size is not None and args.expected_world_size <= 0:
        raise ValueError(
            "--expected-world-size must be positive, got "
            f"{args.expected_world_size}"
        )

    with nccl_distributed(timeout_seconds=args.timeout_seconds) as context:
        if (
            args.expected_world_size is not None
            and context.world_size != args.expected_world_size
        ):
            raise RuntimeError(
                "torchrun world size mismatch: expected "
                f"{args.expected_world_size}, got {context.world_size}"
            )
        run_lesson(context, args)


if __name__ == "__main__":
    main()
