"""Lessons 7–8: fixed global batches, per-iteration max-rank latency, traces."""

from __future__ import annotations
import argparse, json, platform, statistics, os, hashlib
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--timeout-seconds", type=int, default=120)
    p.add_argument("--dispatch-breakdown", action="store_true", help="measure Dispatch sub-stages and CPU-metadata control")
    p.add_argument("--profile", choices=["none", "torch", "nvtx"], default="none")
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
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("need warmup >= 0 and repeats > 0")
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
    assert result.hidden_states.shape == tokens.shape
    assert result.hidden_states.dtype == dtype
    assert result.hidden_states.device == device
    torch.testing.assert_close(
        result.hidden_states,
        expected,
        rtol=tolerances[args.dtype][0],
        atol=tolerances[args.dtype][1],
    )
    assert result.send_splits.sum().item() == tokens.shape[0] * args.top_k
    assert result.local_expert_rows.sum().item() == result.recv_splits.sum().item()
    samples = measure_forward(model, tokens, group, args.warmup, args.repeats)
    # 阶段路径也预热；正式样本只记录预热之后的执行。
    warm_recorder = StageRecorder(device, timing=False)
    for _ in range(args.warmup):
        run_ep_stages(model, tokens, warm_recorder, group=group)
    torch.cuda.synchronize(device)
    dist.barrier(group=group)
    recorder = StageRecorder(device, timing=True)
    for _ in range(args.repeats):
        staged = run_ep_stages(model, tokens, recorder, group=group)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(staged.hidden_states, result.hidden_states)
    local_stage_samples = recorder.samples_ms()
    stage_samples = {
        name: max_rank_samples(v, device, group)
        for name, v in local_stage_samples.items()
    }
    stages = {name: statistics.median(v) for name, v in stage_samples.items()}
    metrics = load_metrics(result.tokens_per_expert_all.cpu(), p)
    # Off-diagonal assignment rows count payload sent across Rank boundaries once.
    counts = result.tokens_per_expert_all.reshape(p, p, -1).sum(-1)
    remote_rows = (counts.sum() - counts.diag().sum()).item()
    latency = statistics.median(samples)
    # 路由指纹覆盖全组逐 Token/slot ID 与权重，核对强扩展的工作相同。
    with torch.no_grad():
        global_routing = reference.router(global_tokens.to(device=device, dtype=dtype))
    route_bytes = global_routing.expert_ids.cpu().contiguous().numpy().tobytes()
    weight_bytes = global_routing.routing_weights.cpu().float().contiguous().numpy().tobytes()
    report = {
        "config": vars(args)
        | {"ep_size": p, "batch_size": 1, "seq_len": args.tokens, "chunk_size": None},
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": torch.cuda.nccl.version(),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(device), "uuid", "unavailable")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "default"),
            "python": platform.python_version(),
            "NCCL_NVLS_ENABLE": os.environ.get("NCCL_NVLS_ENABLE", "default"),
        },
        "correctness": "PASS",
        "rank": r,
        "max_rank_samples_ms": samples,
        "measurement": "median of per-repetition max-rank CUDA Event latency; model forward, inputs on GPU",
        "routing_sha256": hashlib.sha256(route_bytes + weight_bytes).hexdigest(),
        "send_splits": result.send_splits.cpu().tolist(),
        "recv_splits": result.recv_splits.cpu().tolist(),
        "local_expert_ids": list(model.topology.local_expert_ids),
        "stage_local_samples_ms": local_stage_samples,
        "stage_max_rank_samples_ms": stage_samples,
        "total_latency_ms": latency,
        "tokens_per_second": args.tokens * 1000 / latency,
        "stage_max_rank_median_ms": stages,
        "communication_bytes": int(
            2 * remote_rows * args.hidden_size * tokens.element_size()
        ),
        "communication_unit": "two A2A activation payloads, cross-rank send bytes, excludes counts",
        **metrics,
    }
    if args.dispatch_breakdown:
        from benchmarks.dispatch_breakdown import run_dispatch_breakdown
        report["dispatch_breakdown"] = run_dispatch_breakdown(
            model, tokens, group, result, args.warmup, args.repeats
        )
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
                for _ in range(3):
                    run_ep_stages(model, tokens, labels, group=group)
            prof.export_chrome_trace(str(trace / f"rank{r}.json"))
        else:
            for _ in range(3):
                run_ep_stages(model, tokens, labels, group=group)
        torch.cuda.synchronize(device)
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
