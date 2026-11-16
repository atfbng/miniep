"""Reproduce the H100 comparison between dispatch_reference and dispatch_triton."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_kernels import run_case  # noqa: E402


CASES = [
    (257, 128, 8, 2, torch.float32),
    (1024, 128, 8, 2, torch.float32),
    (4096, 128, 8, 2, torch.float32),
    (16384, 128, 8, 2, torch.float32),
    (4096, 512, 8, 2, torch.float32),
    (4096, 128, 16, 2, torch.float32),
    (4096, 128, 8, 4, torch.float32),
    (4096, 128, 8, 2, torch.float16),
]


def warm_clocks(seconds: float) -> None:
    """Raise the GPU clock before collecting steady-state samples."""

    device = torch.device("cuda")
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        a @ b
    torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--clock-warmup", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/dispatch_compare.json",
        help="JSON output path, relative to the current directory",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    warm_clocks(args.clock_warmup)

    reports = []
    print(
        "| T | H | E | K | dtype | reference (ms) | Triton (ms) | speedup | correctness |"
    )
    print("|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for tokens, hidden, experts, top_k, dtype in CASES:
        report = run_case(
            "dispatch",
            tokens=tokens,
            hidden_size=hidden,
            num_experts=experts,
            top_k=top_k,
            dtype=dtype,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        )
        ref = report["reference_median_ms"]
        tri = report["triton_median_ms"]
        dtype_name = str(dtype).removeprefix("torch.")
        print(
            f"| {tokens} | {hidden} | {experts} | {top_k} | {dtype_name} | "
            f"{ref:.4f} | {tri:.4f} | {ref / tri:.2f}x | {report['correctness']} |",
            flush=True,
        )
        reports.append(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": reports[0]["environment"],
        "method": {
            "clock_warmup_s": args.clock_warmup,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
            "timing": "CUDA Events; median of repeats after warmup",
        },
        "reports": reports,
    }
    output.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
