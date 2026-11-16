"""Correctness-first microbenchmarks for lessons 9–11."""

from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import torch
from mini_ep.distributed.dispatch import dispatch_reference, combine_reference
from mini_ep.profiler.timer import measure_latency
from mini_ep.kernels.dispatch_triton import dispatch_triton
from mini_ep.kernels.combine_triton import combine_triton
from mini_ep.kernels.grouped_gemm.reference import (
    create_grouped_plan,
    expert_loop,
    expert_mlp_loop,
)
from mini_ep.kernels.grouped_gemm.triton_impl import grouped_gemm, grouped_expert_mlp


def kernel_counts(operation, device):
    """Count device kernel activities for one warmed call via Kineto events."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        operation()
        torch.cuda.synchronize(device)
    return sum(
        event.device_type == torch.autograd.DeviceType.CUDA
        and "memcpy" not in event.name.lower()
        and "memset" not in event.name.lower()
        for event in prof.events()
    )


def run_case(
    kind,
    tokens=257,
    hidden_size=128,
    num_experts=8,
    top_k=2,
    dtype=torch.float32,
    device="cuda",
    warmup=5,
    repeats=20,
    seed=0,
):
    """Return value/shape/dtype/device checks plus comparable steady-state timings.

    Dispatch measures the whole public call, including metadata construction.
    Combine receives prepared Expert-order outputs. Grouped GEMM/MLP receive
    an identical prebuilt schedule; its construction latency is also reported.
    """
    device = torch.device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    if (
        tokens < 0
        or hidden_size <= 0
        or num_experts <= 0
        or not 1 <= top_k <= num_experts
    ):
        raise ValueError("need T >= 0, H/E > 0, and 1 <= K <= E")
    torch.manual_seed(seed)
    x = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
    ids = torch.stack(
        [
            (torch.arange(tokens, device=device) + slot) % num_experts
            for slot in range(top_k)
        ],
        1,
    )
    weights = torch.rand(tokens, top_k, device=device, dtype=torch.float32)
    weights = weights / weights.sum(1, keepdim=True)
    dispatch = dispatch_reference(x, ids, weights, num_experts)
    prep = None
    flops = None
    if kind == "dispatch":
        baseline = lambda: dispatch_reference(x, ids, weights, num_experts)
        optimized = lambda: dispatch_triton(x, ids, weights, num_experts)
        a, b = baseline(), optimized()
        for name in vars(a):
            av, bv = getattr(a, name), getattr(b, name)
            if isinstance(av, torch.Tensor):
                torch.testing.assert_close(av, bv, rtol=0, atol=0)
            else:
                assert av == bv
    elif kind == "combine":
        y = torch.randn_like(dispatch.packed_tokens)
        baseline = lambda: combine_reference(y, dispatch)
        optimized = lambda: combine_triton(y, dispatch)
    elif kind in ("grouped", "mlp"):
        counts = dispatch.tokens_per_expert.cpu().tolist()
        plan = create_grouped_plan(counts, device)
        prep = measure_latency(
            lambda: create_grouped_plan(counts, device).offsets,
            device=device,
            warmup=warmup,
            repeats=repeats,
        ).median_ms
        packed = dispatch.packed_tokens
        f = hidden_size * 2
        up = torch.randn(num_experts, f, hidden_size, device=device, dtype=dtype) * 0.1
        down = (
            torch.randn(num_experts, hidden_size, f, device=device, dtype=dtype) * 0.1
        )
        if kind == "grouped":
            baseline = lambda: expert_loop(packed, up, plan)
            optimized = lambda: grouped_gemm(packed, up, plan)
            flops = 2 * packed.shape[0] * hidden_size * f
        else:
            baseline = lambda: expert_mlp_loop(packed, up, down, plan)
            optimized = lambda: grouped_expert_mlp(packed, up, down, plan)
            flops = 4 * packed.shape[0] * hidden_size * f
    else:
        raise ValueError(f"unknown kind {kind}")
    if kind != "dispatch":
        tolerance = {
            "torch.float32": (1e-4, 1e-5),
            "torch.float16": (1e-2, 2e-3),
            "torch.bfloat16": (2e-2, 8e-3),
        }[str(dtype)]
        torch.testing.assert_close(
            optimized(), baseline(), rtol=tolerance[0], atol=tolerance[1]
        )
    # First optimized calls above trigger JIT before measured warmup/samples.
    measured_baseline = (
        (lambda: baseline().packed_tokens) if kind == "dispatch" else baseline
    )
    measured_optimized = (
        (lambda: optimized().packed_tokens) if kind == "dispatch" else optimized
    )
    a = measure_latency(
        measured_baseline, device=device, warmup=warmup, repeats=repeats
    )
    b = measure_latency(
        measured_optimized, device=device, warmup=warmup, repeats=repeats
    )
    kernels = {
        "reference": kernel_counts(baseline, device),
        "triton": kernel_counts(optimized, device),
    }
    report = {
        "kind": kind,
        "config": {
            "tokens": tokens,
            "batch_size": 1,
            "seq_len": tokens,
            "hidden_size": hidden_size,
            "expert_hidden_size": hidden_size * 2,
            "num_experts": num_experts,
            "top_k": top_k,
            "dtype": str(dtype),
            "ep_size": 1,
            "chunk_size": None,
            "warmup": warmup,
            "repeats": repeats,
            "seed": seed,
        },
        "correctness": "PASS",
        "reference_median_ms": a.median_ms,
        "triton_median_ms": b.median_ms,
        "reference_samples_ms": a.samples_ms,
        "triton_samples_ms": b.samples_ms,
        "cuda_kernel_count": kernels,
        "plan_prepare_median_ms": prep,
        "tokens_per_expert": dispatch.tokens_per_expert.cpu().tolist(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "TF32": False,
        },
    }
    if flops:
        report["gemm_flops"] = flops
        report["triton_gemm_tflops"] = flops / (b.median_ms * 1e9)
    return report


def main(default_kind=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kind",
        choices=["dispatch", "combine", "grouped", "mlp"],
        default=default_kind or "dispatch",
    )
    p.add_argument("--tokens", type=int, default=257)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--output", default="results/kernel.json")
    args = p.parse_args()
    with torch.no_grad():
        report = run_case(
            args.kind,
            args.tokens,
            args.hidden_size,
            args.num_experts,
            args.top_k,
            getattr(torch, args.dtype),
            warmup=args.warmup,
            repeats=args.repeats,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
