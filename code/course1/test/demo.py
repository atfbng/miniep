#!/usr/bin/env python3
"""Lesson 1: inspect Top-K routing in a minimal single-GPU MoE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Keep the course example runnable before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_ep.model import SingleGPUMoE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--expert-hidden-size", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=7)
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


def main() -> None:
    args = parse_args()
    if args.num_tokens <= 0:
        raise ValueError(f"--num-tokens must be positive, got {args.num_tokens}")

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = SingleGPUMoE(
        hidden_size=args.hidden_size,
        expert_hidden_size=args.expert_hidden_size,
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
        result = model(tokens)

    print(
        f"device={device}, dtype={dtype}, tokens={tuple(tokens.shape)}, "
        f"experts={args.num_experts}, top_k={args.top_k}"
    )
    print("\nToken routing (original token order):")
    for token_id in range(args.num_tokens):
        routes = []
        for slot in range(args.top_k):
            expert_id = int(result.expert_ids[token_id, slot].item())
            weight = float(result.routing_weights[token_id, slot].item())
            routes.append(f"E{expert_id} (weight={weight:.4f})")
        print(f"  T{token_id:02d} -> " + ", ".join(routes))

    counts = [int(value) for value in result.tokens_per_expert.cpu().tolist()]
    print(f"\nAssignments per expert: {counts}")
    print(f"Output shape: {tuple(result.hidden_states.shape)}")
    print(
        "Routing-weight sums: "
        f"{result.routing_weights.sum(dim=-1).float().cpu().tolist()}"
    )


if __name__ == "__main__":
    main()
