"""Stage-gated exact DeepEP acceptance on four H100 GPUs.

Run from a provisioned vLLM 0.29.0 course environment. The default return transport follows model activations. Every job owns a fresh four-rank process group; timeout/failure tears
it down before the suite can proceed. Results are accuracy evidence, not timings.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validation.alignment_report import summarize_run


def checkpoint_manifest(model: Path) -> dict:
    files = {}
    for path in sorted(model.iterdir()):
        if not path.is_file():
            continue
        sha = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha.update(block)
        stat = path.stat()
        files[path.name] = dict(
            bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha.hexdigest()
        )
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("real local checkpoint required")
    return dict(
        files=files,
        digest=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    )


def compare_native(left: Path, right: Path, output: Path) -> dict:
    """Compare independent native runs on identical prefixes, including vocab tensors."""
    import torch
    from validation.exact import exact_tensor_metrics

    torch.set_num_threads(1)
    errors = []
    first = None
    positions = 0
    for rank in range(
        json.loads((left / "rank0/environment.json").read_text())["world_size"]
    ):
        a = left / f"rank{rank}"
        b = right / f"rank{rank}"
        for name in ("manifest.json", "native/outputs.json", "native-moe.jsonl"):
            if (a / name).read_bytes() != (b / name).read_bytes():
                errors.append(f"{name} rank={rank}")
        af = {p.name for p in (a / "baseline_logits").glob("*.pt")}
        bf = {p.name for p in (b / "baseline_logits").glob("*.pt")}
        if af != bf:
            errors.append(f"baseline tensor coverage rank={rank}")
        for name in sorted(af & bf):
            x = torch.load(
                a / "baseline_logits" / name, map_location="cpu", weights_only=True
            )
            y = torch.load(
                b / "baseline_logits" / name, map_location="cpu", weights_only=True
            )
            stats = exact_tensor_metrics(x, y)
            positions += x.shape[0]
            if not stats["exact_equal"]:
                errors.append(f"logits rank={rank} file={name}")
                if first is None:
                    first = dict(rank=rank, file=name, **stats)
    result = dict(
        passed=not errors,
        errors=errors,
        first_difference=first,
        logits_positions_including_tp=positions,
    )
    output.write_text(json.dumps(result, indent=2))
    return result


def build_jobs() -> list[dict]:
    """Decision-complete order. Each candidate references an independent native job."""
    jobs = []

    def job(name, kind, reference=None, **kw):
        jobs.append(
            dict(
                name=name,
                kind=kind,
                reference=reference,
                suite="corpus",
                corpus="rotated",
                seed=13,
                limit=32,
                steps=64,
                **kw,
            )
        )

    for repeat in range(1, 4):
        job(f"native-rotated-13-r{repeat}", "native")
    jobs.append(
        dict(
            name="native-same-13",
            kind="native",
            reference=None,
            suite="corpus",
            corpus="same",
            seed=13,
            limit=32,
            steps=64,
        )
    )
    for corpus in ("same", "rotated"):
        jobname = f"candidate-{corpus}-13-r1"
        jobs.append(
            dict(
                name=jobname,
                kind="candidate",
                reference=(
                    "native-same-13" if corpus == "same" else "native-rotated-13-r1"
                ),
                suite="corpus",
                corpus=corpus,
                seed=13,
                limit=32,
                steps=64,
            )
        )
    for suite, steps in (("boundaries", 64), ("long", 256)):
        jobs.append(
            dict(
                name=f"native-{suite}-13",
                kind="native",
                reference=None,
                suite=suite,
                corpus="rotated",
                seed=13,
                limit=32,
                steps=steps,
            )
        )
        jobs.append(
            dict(
                name=f"candidate-{suite}-13",
                kind="candidate",
                reference=f"native-{suite}-13",
                suite=suite,
                corpus="rotated",
                seed=13,
                limit=32,
                steps=steps,
            )
        )
    for repeat in (2, 3):
        job(
            f"candidate-rotated-13-r{repeat}",
            "candidate",
            f"native-rotated-13-r{repeat}",
        )
    for seed in (37, 123):
        jobs.append(
            dict(
                name=f"native-rotated-{seed}",
                kind="native",
                reference=None,
                suite="corpus",
                corpus="rotated",
                seed=seed,
                limit=32,
                steps=64,
            )
        )
        jobs.append(
            dict(
                name=f"candidate-rotated-{seed}",
                kind="candidate",
                reference=f"native-rotated-{seed}",
                suite="corpus",
                corpus="rotated",
                seed=seed,
                limit=32,
                steps=64,
            )
        )
    return jobs


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["deepep_high_throughput"],
        default="deepep_high_throughput",
    )
    p.add_argument("--world-size", type=int, choices=[4], default=4)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="limited harness check; never full acceptance",
    )
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--kv-cache-mib", type=int, default=512)
    p.add_argument(
        "--reuse-completed",
        help="reuse completed independent native jobs from an earlier suite; keep all old data",
    )
    p.add_argument(
        "--checkpoint-manifest",
        help="reuse previously hashed immutable checkpoint; verify size/mtime",
    )
    p.add_argument("--job-timeout", type=int, default=7200)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--jobs",
        nargs="+",
        help="run named jobs plus native prerequisites; never label this a full-suite pass",
    )
    p.add_argument(
        "--audit-native-reuse",
        action="store_true",
        help="audit unchanged stock runtime before reusing native-only golden from a different Mini-EP version",
    )
    args = p.parse_args()
    os.environ["NCCL_NVLS_ENABLE"] = "0"
    jobs = build_jobs()
    full_plan = jobs.copy()
    if args.jobs:
        names = set(args.jobs)
        unknown = names - {q["name"] for q in jobs}
        if unknown:
            p.error(f"unknown jobs: {sorted(unknown)}")
        names.update(
            q["reference"] for q in jobs if q["name"] in names and q["reference"]
        )
        if any(name.endswith(("13-r2", "13-r3")) for name in names):
            names.add("native-rotated-13-r1")
        jobs = [q for q in jobs if q["name"] in names]
    if args.backend == "deepep_high_throughput" and args.world_size != 4:
        p.error("DeepEP suite requires four GPUs")
    if args.smoke:
        jobs = jobs[:1] + [dict(jobs[5], reference=jobs[0]["name"])]
        for job in jobs:
            job.update(limit=2, steps=4)

    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return 0
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    course = Path(__file__).resolve().parents[1]
    journal = dict(
        backend=args.backend,
        world_size=args.world_size,
        smoke=args.smoke,
        status="hashing_checkpoint",
        jobs=[],
        plan=jobs,
        full_plan=full_plan,
        selected_jobs=args.jobs,
        started=time.time(),
    )

    def save():
        temp = root / "suite.tmp"
        temp.write_text(json.dumps(journal, indent=2))
        temp.replace(root / "suite.json")

    save()
    frozen = root / "checkpoint.json"
    if args.checkpoint_manifest:
        checkpoint = json.loads(Path(args.checkpoint_manifest).read_text())
        for name, info in checkpoint["files"].items():
            stat = (Path(args.model) / name).stat()
            if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
                raise RuntimeError(f"checkpoint changed since SHA256 capture: {name}")
    else:
        checkpoint = checkpoint_manifest(Path(args.model))
    frozen.write_text(json.dumps(checkpoint, indent=2))
    runtime = None
    audits = {}
    if args.audit_native_reuse:
        from validation.native_reuse import capture_runtime

        runtime = capture_runtime()
        (root / "native-runtime.json").write_text(json.dumps(runtime, indent=2))
    resolved = {}
    old = Path(args.reuse_completed).resolve() if args.reuse_completed else None
    for job in jobs:
        reused = None
        if old and job["kind"] == "native":
            from validation.alignment_report import load

            path = old / job["name"]
            if not path.exists():
                old_jobs = load(old / "suite.json", {}).get("jobs", [])
                alias = next(
                    (
                        q.get("reused_from")
                        for q in old_jobs
                        if q.get("name") == job["name"]
                    ),
                    None,
                )
                if alias:
                    path = Path(alias)
            proof = load(path / "acceptance.json", {})
            prior = load(path / "rank0/environment.json", {})
            spec = prior.get("spec", {})
            matches = all(
                spec.get(k) == job[v]
                for k, v in [
                    ("seed", "seed"),
                    ("suite", "suite"),
                    ("corpus", "corpus"),
                    ("limit", "limit"),
                    ("steps", "steps"),
                ]
            )
            core = spec.get("source_sha256", {})
            core_names = {name for name in core if name.startswith("mini_ep/")}
            current_core = {
                str(path.relative_to(course))
                for path in (course / "mini_ep").rglob("*.py")
            }
            core_ok = (
                core_names == current_core
                and bool(core_names)
                and all(
                    hashlib.sha256((course / name).read_bytes()).hexdigest() == sha
                    for name, sha in core.items()
                    if name.startswith("mini_ep/")
                )
            )
            if not core_ok and runtime is not None and matches and proof.get("passed"):
                from validation.native_reuse import audit_native

                audits[job["name"]] = audit_native(path, runtime)
                core_ok = True
            options_ok = (
                spec.get("options", {}).get("gpu_memory_utilization")
                == args.gpu_memory_utilization
                and spec.get("options", {}).get("kv_cache_memory_bytes")
                == args.kv_cache_mib * 1024 * 1024
            )
            if (
                proof.get("passed")
                and options_ok
                and matches
                and core_ok
                and spec.get("checkpoint") == checkpoint["digest"]
                and prior.get("backend") == args.backend
                and prior.get("world_size") == args.world_size
            ):
                # Re-evaluate all raw coverage with the current, stricter reader.
                check = summarize_run(path, world_size=args.world_size)
                if check["passed"]:
                    reused = path
        if reused:
            resolved[job["name"]] = reused
            journal["jobs"].append(
                dict(
                    name=job["name"], reused_from=str(reused), exit_code=0, passed=True
                )
            )
            if job["name"] in ("native-rotated-13-r2", "native-rotated-13-r3"):
                equality = compare_native(
                    resolved["native-rotated-13-r1"],
                    reused,
                    root / (job["name"] + "-repeat.json"),
                )
                if not equality["passed"]:
                    journal.update(
                        status="native_reference_unstable", failed_job=job["name"]
                    )
                    save()
                    return 1
            save()
            continue
        command = [
            "torchrun",
            "--standalone",
            f"--nproc-per-node={args.world_size}",
            "--monitor_interval=1",
            str(course / "test/deepep_alignment_run.py"),
            "--backend",
            args.backend,
            "--world-size",
            str(args.world_size),
            "--model",
            str(Path(args.model).resolve()),
            "--output",
            str(root / job["name"]),
            "--kind",
            job["kind"],
            "--checkpoint-manifest",
            str(frozen),
            "--suite",
            job["suite"],
            "--corpus",
            job["corpus"],
            "--seed",
            str(job["seed"]),
            "--limit",
            str(job["limit"]),
            "--max-tokens",
            str(job["steps"]),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
            "--kv-cache-mib",
            str(args.kv_cache_mib),
        ]
        reference = resolved[job["reference"]] if job["reference"] else None
        resolved[job["name"]] = root / job["name"]
        if reference:
            command += ["--reference", str(reference)]
        if job["kind"] == "candidate" and job["reference"] in audits:
            audit_path = root / (job["reference"] + "-reuse-audit.json")
            audit_path.write_text(json.dumps(audits[job["reference"]], indent=2))
            command += [
                "--native-reference-audit",
                str(audit_path),
                "--native-runtime-manifest",
                str(root / "native-runtime.json"),
            ]
        item = dict(name=job["name"], command=command, start=time.time())
        journal["jobs"].append(item)
        journal["status"] = "running"
        save()
        with (root / (job["name"] + ".log")).open("w") as log:
            proc = subprocess.Popen(
                command,
                cwd=course,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            item["pid"] = proc.pid
            save()
            try:
                status = proc.wait(timeout=args.job_timeout)
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                status = 124
        item.update(exit_code=status, end=time.time())
        (root / job["name"]).mkdir(exist_ok=True)
        result = summarize_run(root / job["name"], reference, args.world_size)
        (root / job["name"] / "acceptance.json").write_text(
            json.dumps(result, indent=2)
        )
        item["passed"] = status == 0 and result["passed"]
        if not item["passed"]:
            journal["status"] = "failed"
            journal["failed_job"] = job["name"]
            save()
            return 1
        if job["name"] in ("native-rotated-13-r2", "native-rotated-13-r3"):
            equality = compare_native(
                resolved["native-rotated-13-r1"],
                root / job["name"],
                root / (job["name"] + "-repeat.json"),
            )
            if not equality["passed"]:
                journal["status"] = "native_reference_unstable"
                journal["failed_job"] = job["name"]
                save()
                return 1
        save()
    journal["status"] = (
        "smoke_passed"
        if args.smoke
        else "selected_jobs_passed" if args.jobs else "passed"
    )
    journal["finished"] = time.time()
    save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
