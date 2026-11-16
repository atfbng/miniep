"""Paired fresh-engine RouteEP / native DeepEP latency measurements."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.deepep_report import assess_job, summarize_trials
from benchmarks.deepep_workloads import PRESETS


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-manifest", required=True)
    p.add_argument("--label", default="final")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument(
        "--cases",
        nargs="+",
        choices=[q[0] for q in PRESETS],
        default=[q[0] for q in PRESETS],
    )
    p.add_argument("--profile-stages", action="store_true")
    p.add_argument("--job-timeout", type=int, default=7200)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--kv-cache-mib", type=int, default=512)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if a.trials < 3 or a.warmup < 1 or a.repeats < 3:
        p.error("trials>=3, warmup>=1, repeats>=3 required")
    course = Path(__file__).resolve().parents[1]
    plan = []
    for trial in range(a.trials):
        order = ("native", "miniep") if trial % 2 == 0 else ("miniep", "native")
        for backend in order:
            plan.append(
                dict(
                    name=f"trial{trial}-{backend}",
                    trial=trial,
                    backend=backend,
                    profile="off",
                )
            )
    if a.profile_stages:
        for backend in ("native", "miniep"):
            plan.append(
                dict(
                    name=f"profile-{backend}",
                    trial=0,
                    backend=backend,
                    profile="events",
                )
            )
    if a.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    root = Path(a.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    journal = dict(
        status="running", label=a.label, plan=plan, jobs=[], started=time.time()
    )

    def save():
        tmp = root / "suite.tmp"
        tmp.write_text(json.dumps(journal, indent=2))
        tmp.replace(root / "suite.json")

    save()
    for item in plan:
        output = root / item["name"]
        command = [
            "torchrun",
            "--standalone",
            "--nproc-per-node=4",
            "--monitor_interval=1",
            str(course / "benchmarks/deepep_latency_run.py"),
            "--model",
            str(Path(a.model).resolve()),
            "--output",
            str(output),
            "--checkpoint-manifest",
            str(Path(a.checkpoint_manifest).resolve()),
            "--backend",
            item["backend"],
            "--label",
            a.label,
            "--warmup",
            str(a.warmup),
            "--repeats",
            str(a.repeats),
            "--profile",
            item["profile"],
            "--gpu-memory-utilization",
            str(a.gpu_memory_utilization),
            "--kv-cache-mib",
            str(a.kv_cache_mib),
            "--cases",
            *a.cases,
        ]
        record = dict(item, path=str(output), command=command, started=time.time())
        journal["jobs"].append(record)
        save()
        with (root / (item["name"] + ".log")).open("w") as log:
            proc = subprocess.Popen(
                command,
                cwd=course,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["pid"] = proc.pid
            save()
            try:
                status = proc.wait(timeout=a.job_timeout)
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status = 124
        record.update(exit_code=status, finished=time.time())
        output.mkdir(exist_ok=True)
        proof = assess_job(output)
        (output / "acceptance.json").write_text(json.dumps(proof, indent=2))
        record["passed"] = status == 0 and proof["passed"]
        save()
        if not record["passed"]:
            journal.update(status="failed", failed_job=item["name"])
            save()
            return 1
    result = summarize_trials(journal["jobs"])
    (root / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    journal.update(
        status="passed" if result["passed"] else "failed", finished=time.time()
    )
    save()
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
