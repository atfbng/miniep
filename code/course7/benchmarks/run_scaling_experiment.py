"""Repeat EP=1/2/4 on two fixed global workloads and collect auditable reports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys

COURSE_ROOT = Path(__file__).resolve().parents[1]
WORKLOADS = {
    "small": {"tokens": 1024, "hidden_size": 128, "expert_hidden_size": 256, "dtype": "float16"},
    "large": {"tokens": 8192, "hidden_size": 1024, "expert_hidden_size": 2048, "dtype": "float16"},
}


def summarize(reports: list[dict]) -> dict:
    """Summarize trial medians; enforce identical global routes within each case.

    Reports contain max-rank CUDA samples, global Expert counts, and row-based
    send/receive splits. The median/min/max below refer to independent trials,
    not ranks or individual repetition slots. Speedup uses that case's EP=1.
    """
    if not reports:
        raise ValueError("no experiment reports")
    groups: dict[tuple[str, int], list[dict]] = {}
    cases: dict[str, tuple] = {}
    seen = set()
    for report in reports:
        case, ep, trial = report["case"], report["config"]["ep_size"], report["trial"]
        key = (case, ep, trial)
        if key in seen:
            raise ValueError(f"duplicate experiment report: {key}")
        seen.add(key)
        if report["correctness"] != "PASS":
            raise ValueError(f"reference check failed: {key}")
        config = report["config"]
        identity = tuple(config[k] for k in ("tokens", "hidden_size", "expert_hidden_size", "num_experts", "top_k", "dtype", "seed", "scenario")) + (report["routing_sha256"], tuple(report["tokens_per_expert"]))
        if case in cases and cases[case] != identity:
            raise ValueError(f"global input/configuration/routing changed within case {case}")
        cases[case] = identity
        if len(report["max_rank_samples_ms"]) != config["repeats"]:
            raise ValueError(f"incorrect repetition count: {key}")
        if sum(report["rows_per_rank"]) != config["tokens"] * config["top_k"]:
            raise ValueError(f"Expert rows do not cover all branches: {key}")
        groups.setdefault((case, ep), []).append(report)
    rows = []
    for (case, ep), trials in sorted(groups.items()):
        values = [r["total_latency_ms"] for r in trials]
        if any(v <= 0 for v in values):
            raise ValueError(f"nonpositive latency: {case}, EP={ep}")
        row = {
            "case": case, "ep_size": ep, "trials": len(trials),
            "config": trials[0]["config"],
            "latency_ms": statistics.median(values),
            "trial_min_ms": min(values), "trial_max_ms": max(values),
            "tokens_per_second": trials[0]["config"]["tokens"] * 1000 / statistics.median(values),
            "communication_bytes": trials[0]["communication_bytes"],
            "tokens_per_expert": trials[0]["tokens_per_expert"],
            "rows_per_rank": trials[0]["rows_per_rank"],
            "stages_ms": {name: statistics.median([r["stage_max_rank_median_ms"][name] for r in trials]) for name in trials[0]["stage_max_rank_median_ms"]},
        }
        rows.append(row)
    baselines = {r["case"]: r["latency_ms"] for r in rows if r["ep_size"] == 1}
    for row in rows:
        if row["case"] not in baselines:
            raise ValueError(f"missing EP=1 baseline for {row['case']}")
        row["speedup"] = baselines[row["case"]] / row["latency_ms"]
        row["parallel_efficiency"] = row["speedup"] / row["ep_size"]
    return {"statistic": "median of independent trial medians; each trial uses median of repetition-wise max Rank CUDA latency", "rows": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--ep-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--dispatch-breakdown", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--job-timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=Path("results/scaling"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trials <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("need trials/repeats > 0 and warmup >= 0")
    if len(set(args.ep_sizes)) != len(args.ep_sizes) or 1 not in args.ep_sizes:
        raise ValueError("EP sizes must be unique and include the EP=1 baseline")
    if any(ep not in (1, 2, 4) for ep in args.ep_sizes):
        raise ValueError("this experiment uses EP=1/2/4")
    if args.job_timeout <= args.timeout_seconds or args.timeout_seconds <= 0:
        raise ValueError("need job timeout > collective timeout > 0")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < max(args.ep_sizes):
        raise RuntimeError(f"need {max(args.ep_sizes)} visible CUDA GPUs")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for trial in range(args.trials):
        # Rotate EP order between trials to reduce order-dependent drift.
        order = args.ep_sizes[trial % len(args.ep_sizes):] + args.ep_sizes[:trial % len(args.ep_sizes)]
        for case in args.cases:
            shape = WORKLOADS[case]
            for ep in order:
                target = out / "raw" / case / f"ep{ep}" / f"trial{trial}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={ep}", "--max-restarts=0", "test/demo.py"]
                for key, value in (shape | {"num_experts": 8, "top_k": 2, "scenario": "uniform", "seed": args.seed, "warmup": args.warmup, "repeats": args.repeats, "timeout_seconds": args.timeout_seconds, "output": target}).items():
                    cmd += ["--" + key.replace("_", "-"), str(value)]
                if args.dispatch_breakdown:
                    cmd += ["--dispatch-breakdown"]
                env = dict(os.environ)
                env.setdefault("OMP_NUM_THREADS", "1")
                env.setdefault("NCCL_NVLS_ENABLE", "0")
                print(f"{case}: EP={ep}, trial={trial + 1}/{args.trials}", flush=True)
                with target.with_suffix(".log").open("w") as log:
                    job = subprocess.Popen(cmd, cwd=COURSE_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = job.wait(timeout=args.job_timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(job.pid, signal.SIGTERM)
                        try:
                            job.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(job.pid, signal.SIGKILL)
                            job.wait()
                        raise RuntimeError(f"job timed out; rank logs: {target.with_suffix('.log')}")
                if code:
                    raise RuntimeError(f"job failed with exit {code}; rank logs: {target.with_suffix('.log')}")
                rank_reports = [json.loads(target.with_name(target.stem + f"-rank{rank}.json").read_text()) for rank in range(ep)]
                for rank, record in enumerate(rank_reports):
                    if record["rank"] != rank or record["correctness"] != "PASS":
                        raise RuntimeError(f"rank validation failed: {target}, rank={rank}")
                    if record["max_rank_samples_ms"] != rank_reports[0]["max_rank_samples_ms"]:
                        raise RuntimeError("max-rank sample reduction differs between ranks")
                # Each source/destination pair must agree on row-based splits.
                for source in range(ep):
                    for destination in range(ep):
                        if rank_reports[source]["send_splits"][destination] != rank_reports[destination]["recv_splits"][source]:
                            raise RuntimeError(f"mismatched split at {source}->{destination}")
                record = rank_reports[0] | {"case": case, "trial": trial, "report_path": str(target.relative_to(out))}
                reports.append(record)
                (out / "reports.json").write_text(json.dumps(reports, indent=2))
                cases_seen = {r["case"] for r in reports}
                if all(any(r["case"] == c and r["config"]["ep_size"] == 1 for r in reports) for c in cases_seen):
                    (out / "summary.json").write_text(json.dumps(summarize(reports), indent=2))
    (out / "summary.json").write_text(json.dumps(summarize(reports), indent=2))
    print(f"Saved {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
