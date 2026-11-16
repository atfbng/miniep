"""为原生 DeepEP LL/HT 各启动全新引擎做对照，不执行 RouteEP。"""

import argparse
from collections import defaultdict
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.deepep_native_backend import BACKENDS
from benchmarks.deepep_report import assess_job, load
from benchmarks.deepep_metrics import distribution
from benchmarks.deepep_workloads import PRESETS


def summarize_native_trials(jobs: list[dict]) -> dict:
    errors = []
    grouped = defaultdict(list)
    per_trial = defaultdict(dict)
    for job in jobs:
        root = Path(job["path"])
        check = assess_job(root, job.get("world_size", 4))
        env = load(root / "rank0/environment.json", {})
        if (
            not check["passed"]
            or check["backend"] != "native"
            or check["profile"] != "off"
        ):
            errors.append(dict(job=job["name"], detail=check))
            continue
        selected = env["options"]["all2all_backend"]
        if selected != job["all2all_backend"]:
            errors.append(dict(job=job["name"], error="requested backend mismatch"))
        per_trial[job["trial"]][selected] = root
        for row in load(root / "rank0/samples.json", []):
            grouped[(row["case"], selected)].append(row)
    cases = {}
    for (case, backend), rows in grouped.items():
        cases.setdefault(case, {})[backend] = dict(
            e2e_ms=distribution([q["max_rank_ms"][0] for q in rows]),
            prefill_ms=distribution([q["max_rank_ms"][2] for q in rows]),
            decode_step_ms=(
                distribution([q["max_rank_ms"][4] for q in rows])
                if rows[0]["max_rank_ms"][4]
                else None
            ),
            generated_tokens_per_second=distribution(
                [q["generated_tokens_per_second"] for q in rows]
            ),
            prefill_tokens_per_second=distribution(
                [q["prefill_tokens_per_second"] for q in rows]
            ),
        )
    expected_backends = {q["all2all_backend"] for q in jobs}
    if len(per_trial) < 3 or len({q["path"] for q in jobs}) != len(jobs):
        errors.append(
            dict(error="fewer than three independent trials or duplicated jobs")
        )
    for trial, pair in per_trial.items():
        if set(pair) != expected_backends:
            errors.append(dict(trial=trial, error="missing requested backend"))
    token_differences = []
    stability = {}
    for trial, pair in per_trial.items():
        for backend, root in pair.items():
            world_size = load(root / "rank0/environment.json").get("world_size", 4)
            for rank in range(world_size):
                tokens = [
                    (q["case"], q["repetition"], q["token_ids"])
                    for q in load(root / f"rank{rank}/samples.json")
                ]
                key = (backend, rank)
                if key in stability and stability[key] != tokens:
                    errors.append(
                        dict(
                            trial=trial,
                            rank=rank,
                            backend=backend,
                            error="native restart token instability",
                        )
                    )
                stability[key] = tokens
        if len(pair) != 2:
            continue
        left, right = [pair[k] for k in BACKENDS]
        world_size = load(left / "rank0/environment.json").get("world_size", 4)
        for rank in range(world_size):
            a, b = left / f"rank{rank}", right / f"rank{rank}"
            x, y = load(a / "environment.json"), load(b / "environment.json")
            if load(a / "manifest.json") != load(b / "manifest.json"):
                errors.append(
                    dict(trial=trial, rank=rank, error="literal inputs differ")
                )
            for field in (
                "versions",
                "source_sha256",
                "checkpoint_digest",
                "model_parameters",
                "cuda_version",
                "nccl_version",
                "cpu_quota",
                "warmup",
                "repeats",
                "profile",
            ):
                if x.get(field) != y.get(field):
                    errors.append(
                        dict(trial=trial, rank=rank, error=f"{field} differs")
                    )
            opts = lambda env: {
                k: v for k, v in env["options"].items() if k != "all2all_backend"
            }
            if opts(x) != opts(y):
                errors.append(
                    dict(trial=trial, rank=rank, error="inference options differ")
                )

            def devices(env):
                rows = csv.DictReader(io.StringIO(env["gpu_before"]))
                return sorted(
                    (
                        q[next(k for k in q if k.strip() == "uuid")].strip(),
                        q[next(k for k in q if k.strip() == "driver_version")].strip(),
                    )
                    for q in rows
                )

            if devices(x) != devices(y):
                errors.append(
                    dict(trial=trial, rank=rank, error="GPU UUID/driver differs")
                )
            first = load(a / "samples.json")
            second = load(b / "samples.json")
            if [(q["case"], q["repetition"], q["token_ids"]) for q in first] != [
                (q["case"], q["repetition"], q["token_ids"]) for q in second
            ]:
                token_differences.append(
                    dict(
                        trial=trial,
                        rank=rank,
                        scope="free generation HT versus LL; not a zero-error accuracy claim",
                    )
                )
    for case, values in cases.items():
        ht, ll = values.get("deepep_high_throughput"), values.get("deepep_low_latency")
        if ht and ll:
            values["ll_over_ht"] = dict(
                e2e_latency=ll["e2e_ms"]["median"] / ht["e2e_ms"]["median"],
                prefill_latency=ll["prefill_ms"]["median"] / ht["prefill_ms"]["median"],
                decode_latency=(
                    ll["decode_step_ms"]["median"] / ht["decode_step_ms"]["median"]
                    if ll["decode_step_ms"]
                    else None
                ),
                generated_throughput=ll["generated_tokens_per_second"]["median"]
                / ht["generated_tokens_per_second"]["median"],
            )
    return dict(
        passed=bool(cases) and not errors,
        errors=errors,
        cases=cases,
        ht_ll_free_token_differences=token_differences,
        miniep_executed=False,
        precision_scope="native backend identity and within-backend repeatability; no per-layer/logits HT/LL or RouteEP accuracy alignment",
        statistic="fresh engines/backend; warmups and repeats recorded per job; median/p10/p90 of per-repetition maximum-rank latency",
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-manifest", required=True)
    p.add_argument("--tp-size", type=int, choices=[1, 2], default=2)
    p.add_argument(
        "--backends",
        nargs="+",
        choices=list(BACKENDS),
        default=["deepep_low_latency", "deepep_high_throughput"],
    )
    p.add_argument(
        "--cases",
        nargs="+",
        choices=[q[0] for q in PRESETS],
        default=[q[0] for q in PRESETS],
    )
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--job-timeout", type=int, default=7200)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--nvshmem-qp-depth", type=int)
    p.add_argument("--enable-chunked-prefill", action="store_true")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--kv-cache-mib", type=int, default=512)
    p.add_argument("--release-unused-cache-before-buffer-init", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if (
        a.trials < 3
        or a.warmup < 1
        or a.repeats < 3
        or len(set(a.backends)) != len(a.backends)
        or a.max_num_batched_tokens < (32 if a.enable_chunked_prefill else 2304)
    ):
        p.error(
            "trials>=3, warmup>=1, repeats>=3, token budget>=2304, distinct backends required"
        )
    required = 2 * (a.max_num_batched_tokens + 1)
    depth = a.nvshmem_qp_depth or (1 << (required - 1).bit_length())
    if depth < required:
        p.error(f"NVSHMEM queue depth must be >= {required}")
    course = Path(__file__).resolve().parents[1]
    plan = [
        dict(
            name=f"trial{trial}-{backend}",
            trial=trial,
            all2all_backend=backend,
            world_size=a.tp_size * 2,
        )
        for trial in range(a.trials)
        for backend in (a.backends if trial % 2 == 0 else list(reversed(a.backends)))
    ]
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="running", miniep_executed=False, plan=plan, jobs=[], started=time.time()
    )

    def save():
        tmp = root / "suite.tmp"
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(root / "suite.json")

    save()
    for job in plan:
        output = root / job["name"]
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={a.tp_size * 2}",
            "--monitor_interval=1",
            str(course / "benchmarks/deepep_latency_run.py"),
            "--backend",
            "native",
            "--tp-size",
            str(a.tp_size),
            "--all2all-backend",
            job["all2all_backend"],
            "--label",
            job["all2all_backend"],
            "--model",
            str(Path(a.model).resolve()),
            "--checkpoint-manifest",
            str(Path(a.checkpoint_manifest).resolve()),
            "--output",
            str(output),
            "--warmup",
            str(a.warmup),
            "--repeats",
            str(a.repeats),
            "--max-num-batched-tokens",
            str(a.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(a.gpu_memory_utilization),
            "--kv-cache-mib",
            str(a.kv_cache_mib),
            "--cases",
            *a.cases,
        ]
        if a.enable_chunked_prefill:
            cmd.append("--enable-chunked-prefill")
        if a.release_unused_cache_before_buffer_init:
            cmd.append("--release-unused-cache-before-buffer-init")
        record = dict(
            job,
            path=str(output),
            command=cmd,
            started=time.time(),
            nvshmem_qp_depth=depth,
            nic_handler="gpu",
        )
        state["jobs"].append(record)
        save()
        with (root / (job["name"] + ".log")).open("w") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=course,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **os.environ,
                    "NCCL_NVLS_ENABLE": "0",
                    "MINIEP_STAGE_TIMING": "0",
                    "OMP_NUM_THREADS": "1",
                    "NVSHMEM_QP_DEPTH": str(depth),
                    "NVSHMEM_IBGDA_NIC_HANDLER": "gpu",
                },
            )
            record["pid"] = proc.pid
            save()
            try:
                status = proc.wait(timeout=a.job_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status = 124
        record.update(exit_code=status, finished=time.time())
        output.mkdir(exist_ok=True)
        check = assess_job(output, a.tp_size * 2)
        (output / "acceptance.json").write_text(json.dumps(check, indent=2))
        record["passed"] = status == 0 and check["passed"]
        save()
        if not record["passed"]:
            state.update(status="failed", failed_job=job["name"])
            save()
            return 1
    result = summarize_native_trials(state["jobs"])
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    state.update(
        status="passed" if result["passed"] else "failed", finished=time.time()
    )
    save()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
