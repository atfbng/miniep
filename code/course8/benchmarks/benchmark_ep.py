"""Lessons 7–8: fixed global batches, per-iteration max-rank latency, traces."""

from __future__ import annotations
import argparse, json, platform, statistics, os, time
from pathlib import Path
import torch
import torch.distributed as dist
from mini_ep.distributed import nccl_distributed
from mini_ep.model.moe import distribute_single_gpu_moe
from mini_ep.profiler.stages import StageRecorder, run_ep_stages
from mini_ep.utils.course_workload import make_workload, load_metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=1024, help="global original Tokens")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--expert-hidden-size", type=int, default=256)
    p.add_argument("--num-experts", type=int, default=8)
    p.add_argument("--top-k", type=int, choices=[1, 2], default=2)
    p.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    p.add_argument("--scenario", choices=["uniform", "hotspot"], default="uniform")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument(
        "--clock-warmup-ms",
        type=int,
        default=0,
        help="unmeasured GPU burn-in to move idle H100s out of the low-clock state",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--profile", choices=["none", "torch", "nvtx"], default="none")
    p.add_argument("--profile-repeats", type=int, default=3)
    p.add_argument("--trace-dir", default="results/ep-traces")
    p.add_argument("--output", default="results/ep.json")
    return p.parse_args()


def max_rank_samples(samples, device, group):
    """Reduce matching repetition slots after timing; no metric collective inside a sample."""
    tensor = torch.tensor(samples, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
    return tensor.cpu().tolist()


def measure_forward(model, tokens, group, warmup, repeats):
    """Current-stream end-to-end CUDA samples; inputs already reside on device."""
    if warmup < 0 or repeats <= 0:
        raise ValueError(f"need warmup >= 0, repeats > 0; got {warmup}, {repeats}")
    for _ in range(warmup):
        model(tokens, group=group)
    torch.cuda.synchronize(tokens.device)
    dist.barrier(group=group)
    pairs = []
    for _ in range(repeats):
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        model(tokens, group=group)
        ended.record()
        pairs.append((started, ended))
    torch.cuda.synchronize(tokens.device)
    return max_rank_samples([a.elapsed_time(b) for a, b in pairs], tokens.device, group)


def warm_up_clocks(device, duration_ms):
    """Keep both EP ranks' devices busy before timing.

    Idle H100s can sit at a few hundred MHz and ramp during a short benchmark,
    which makes the first measured iterations much slower.  This unmeasured
    burn-in is not part of any reported latency.
    """
    if duration_ms < 0:
        raise ValueError(f"clock_warmup_ms must be non-negative, got {duration_ms}")
    if duration_ms == 0:
        return
    left = torch.randn(4096, 4096, device=device)
    right = torch.randn(4096, 4096, device=device)
    started = time.perf_counter()
    product = None
    while (time.perf_counter() - started) * 1000 < duration_ms:
        product = left @ right
    torch.cuda.synchronize(device)
    # Keep the result alive through the final synchronization.
    if product is not None:
        del product
    del left, right


def current_clock_mhz(device):
    """Return the current SM clock when the PyTorch build exposes it."""
    try:
        return torch.cuda.clock_rate(device)
    except (AttributeError, RuntimeError):
        return None


def profiler_key_averages(prof):
    """Serialize PyTorch Profiler's per-operator device/CPU timings in JSON."""
    rows = []
    for item in prof.key_averages():
        self_device_us = getattr(item, "self_device_time_total", None)
        if self_device_us is None:
            self_device_us = getattr(item, "self_cuda_time_total", 0)
        device_us = getattr(item, "device_time_total", None)
        if device_us is None:
            device_us = getattr(item, "cuda_time_total", 0)
        rows.append(
            {
                "key": item.key,
                "count": item.count,
                "self_device_time_total_us": float(self_device_us),
                "device_time_total_us": float(device_us),
                "self_cpu_time_total_us": float(item.self_cpu_time_total),
                "cpu_time_total_us": float(item.cpu_time_total),
            }
        )
    return sorted(
        rows,
        key=lambda row: row["self_device_time_total_us"],
        reverse=True,
    )


def run_benchmark(context, args):
    torch.backends.cuda.matmul.allow_tf32 = False
    p = context.world_size
    r = context.global_rank
    device = context.device
    group = context.group
    if args.tokens % p or args.num_experts % p:
        raise ValueError(
            f"T={args.tokens} and E={args.num_experts} must divide evenly by P={p}"
        )
    if (
        args.warmup < 0
        or args.repeats <= 0
        or args.clock_warmup_ms < 0
        or args.profile_repeats <= 0
    ):
        raise ValueError(
            "need warmup >= 0, repeats > 0, clock_warmup_ms >= 0, "
            "and profile_repeats > 0"
        )
    dtype = getattr(torch, args.dtype)
    reference, global_tokens = make_workload(
        args.tokens,
        args.hidden_size,
        args.expert_hidden_size,
        args.num_experts,
        args.top_k,
        args.scenario,
        args.seed,
    )
    model = distribute_single_gpu_moe(reference, ep_size=p, ep_rank=r).to(
        device=device, dtype=dtype
    )
    t = args.tokens // p
    tokens = (
        global_tokens[r * t : (r + 1) * t].to(device=device, dtype=dtype).contiguous()
    )
    # Correctness is outside the measured forward interval.
    expected_model = reference.to(device=device, dtype=dtype)
    expected = expected_model(tokens).hidden_states
    result = model(tokens, group=group)
    tolerances = {
        "float32": (1e-4, 1e-5),
        "float16": (1e-2, 2e-3),
        "bfloat16": (2e-2, 8e-3),
    }
    torch.testing.assert_close(
        result.hidden_states,
        expected,
        rtol=tolerances[args.dtype][0],
        atol=tolerances[args.dtype][1],
    )
    assert result.send_splits.sum().item() == tokens.shape[0] * args.top_k
    assert result.local_expert_rows.sum().item() == result.recv_splits.sum().item()
    warm_up_clocks(device, args.clock_warmup_ms)
    clock_before_measure_mhz = current_clock_mhz(device)
    samples = measure_forward(model, tokens, group, args.warmup, args.repeats)
    clock_after_measure_mhz = current_clock_mhz(device)
    recorder = StageRecorder(device, timing=True)
    for _ in range(args.repeats):
        staged = run_ep_stages(model, tokens, recorder, group=group)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(staged.hidden_states, result.hidden_states)
    stage_samples = recorder.samples_ms()
    stages = {
        name: statistics.median(max_rank_samples(v, device, group))
        for name, v in stage_samples.items()
    }
    stage_rank_medians = {
        name: statistics.median(v) for name, v in stage_samples.items()
    }
    metrics = load_metrics(result.tokens_per_expert_all.cpu(), p)
    # Off-diagonal assignment rows count payload sent across Rank boundaries once.
    counts = result.tokens_per_expert_all.reshape(p, p, -1).sum(-1)
    remote_rows = (counts.sum() - counts.diag().sum()).item()
    latency = statistics.median(samples)
    report = {
        "config": vars(args)
        | {"ep_size": p, "batch_size": 1, "seq_len": args.tokens, "chunk_size": None},
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(device),
            "python": platform.python_version(),
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "default"),
            "clock_before_measure_mhz": clock_before_measure_mhz,
            "clock_after_measure_mhz": clock_after_measure_mhz,
        },
        "correctness": "PASS",
        "rank": r,
        "max_rank_samples_ms": samples,
        "total_latency_ms": latency,
        "tokens_per_second": args.tokens * 1000 / latency,
        "stage_max_rank_median_ms": stages,
        "stage_rank_median_ms": stage_rank_medians,
        "communication_bytes": int(
            2 * remote_rows * args.hidden_size * tokens.element_size()
        ),
        "communication_unit": "two A2A activation payloads, cross-rank send bytes, excludes counts",
        **metrics,
    }
    profile_rows = None
    if args.profile != "none":
        trace = Path(args.trace_dir)
        trace.mkdir(parents=True, exist_ok=True)
        labels = StageRecorder(device, timing=False)
        if args.profile == "torch":
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
            ) as prof:
                for _ in range(args.profile_repeats):
                    run_ep_stages(model, tokens, labels, group=group)
            prof.export_chrome_trace(str(trace / f"rank{r}.json"))
            profile_rows = profiler_key_averages(prof)
        else:
            for _ in range(args.profile_repeats):
                run_ep_stages(model, tokens, labels, group=group)
        torch.cuda.synchronize(device)
    if profile_rows is not None:
        report["profile_key_averages"] = profile_rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_name(output.stem + f"-rank{r}" + output.suffix).write_text(
        json.dumps(report, indent=2)
    )
    if r == 0:
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    return report


def main():
    args = parse_args()
    with nccl_distributed(
        timeout_seconds=args.timeout_seconds
    ) as context, torch.no_grad():
        run_benchmark(context, args)


if __name__ == "__main__":
    main()
