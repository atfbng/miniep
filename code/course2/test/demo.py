#!/usr/bin/env python3
"""Lesson 2: inspect Router decisions and expert-major Token packing."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

# Keep the course example runnable before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_ep.distributed import combine_reference, dispatch_reference  # noqa: E402
from mini_ep.model import TopKRouter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, choices=(1, 2), default=2)
    parser.add_argument("--capacity-factor", type=float, default=1.0)
    parser.add_argument("--max-packed-rows", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses one CUDA GPU when available, otherwise CPU",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_tokens", "hidden_size", "num_experts", "max_packed_rows"):
        value = getattr(args, name)
        if value <= 0:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be positive, got {value}")
    if args.top_k > args.num_experts:
        raise ValueError(
            f"--top-k ({args.top_k}) cannot exceed --num-experts ({args.num_experts})"
        )
    if not math.isfinite(args.capacity_factor) or args.capacity_factor <= 0:
        raise ValueError(
            f"--capacity-factor must be positive, got {args.capacity_factor}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    router = TopKRouter(
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    ).to(device=device, dtype=dtype)
    tokens = torch.randn(
        args.num_tokens,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        routing = router(tokens)
        dispatched = dispatch_reference(
            tokens,
            routing.expert_ids,
            routing.routing_weights,
            args.num_experts,
        )
        # Identity experts make the layout round trip easy to verify: because
        # each Token's Top-K weights sum to one, Combine must recover tokens.
        restored_tokens = combine_reference(dispatched.packed_tokens, dispatched)

    print(
        f"device={device}, dtype={dtype}, tokens={tuple(tokens.shape)}, "
        f"experts={args.num_experts}, top_k={args.top_k}"
    )
    print("\nToken -> Expert routing (original Token order):")
    for token_id in range(args.num_tokens):
        routes = []
        for slot in range(args.top_k):
            expert_id = int(routing.expert_ids[token_id, slot].item())
            weight = float(routing.routing_weights[token_id, slot].item())
            routes.append(f"slot{slot}: E{expert_id} (weight={weight:.4f})")
        print(f"  T{token_id:02d} -> " + ", ".join(routes))

    counts = [int(value) for value in dispatched.tokens_per_expert.cpu().tolist()]
    offsets = [int(value) for value in dispatched.expert_offsets.cpu().tolist()]
    total_assignments = dispatched.num_assignments
    ideal_average = total_assignments / args.num_experts
    imbalance_ratio = max(counts) / ideal_average
    illustrative_capacity = math.ceil(
        args.capacity_factor * total_assignments / args.num_experts
    )

    print(f"\nAssignments per expert: {counts}")
    print(f"Expert offsets: {offsets}")
    print(
        "Load summary: "
        f"assignments={total_assignments}, ideal_average={ideal_average:.2f}, "
        f"max/average={imbalance_ratio:.3f}"
    )
    print(
        "Illustrative capacity only (not enforced): "
        f"factor={args.capacity_factor:.2f}, capacity={illustrative_capacity}"
    )

    shown_rows = min(dispatched.num_assignments, args.max_packed_rows)
    print("\nPacked rows (expert-major order):")
    for packed_row in range(shown_rows):
        expert_id = int(dispatched.packed_expert_ids[packed_row].item())
        token_id = int(dispatched.packed_token_indices[packed_row].item())
        route_slot = int(dispatched.packed_route_slots[packed_row].item())
        weight = float(dispatched.packed_routing_weights[packed_row].item())
        print(
            f"  P{packed_row:02d}: E{expert_id} <- T{token_id:02d}/slot{route_slot} "
            f"(weight={weight:.4f})"
        )
    if shown_rows < dispatched.num_assignments:
        print(f"  ... {dispatched.num_assignments - shown_rows} packed rows omitted")

    max_error = float((restored_tokens.float() - tokens.float()).abs().max().item())
    print(f"\nPacked shape: {tuple(dispatched.packed_tokens.shape)}")
    print(f"Combined shape: {tuple(restored_tokens.shape)}")
    print(f"Identity-expert round-trip max error: {max_error:.6g}")


if __name__ == "__main__":
    main()
