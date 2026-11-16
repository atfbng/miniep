"""Reject incomplete/mismatched timing jobs and compare common workloads."""

from collections import defaultdict
import json
import math
from pathlib import Path
from benchmarks.deepep_metrics import distribution


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def assess_job(root: Path) -> dict:
    errors = []
    environments = []
    manifests = []
    samples = []
    for rank in range(4):
        folder = root / f"rank{rank}"
        env = load(folder / "environment.json", {})
        environments.append(env)
        manifest = load(folder / "manifest.json", [])
        manifests.append(manifest)
        data = load(folder / "samples.json", [])
        samples.append(data)
        completed = load(folder / "completed.json", {})
        proof = load(folder / "backend.json", {})
        if not env or env.get("rank") != rank:
            errors.append(f"missing/wrong rank {rank}")
        if (
            not env.get("checkpoint_digest")
            or not env.get("source_sha256")
            or not env.get("versions")
        ):
            errors.append(f"missing model/code/software evidence rank {rank}")
        if not manifest:
            errors.append(f"missing manifest rank {rank}")
        expected = {
            (q["name"], i) for q in manifest for i in range(env.get("repeats", 0))
        }
        actual = {(q["case"], q["repetition"]) for q in data}
        if actual != expected or len(data) != len(actual):
            errors.append(f"incomplete samples rank {rank}")
        if completed.get("cases") != [q["name"] for q in manifest] or completed.get(
            "samples"
        ) != len(data):
            errors.append(f"incomplete job rank {rank}")
        if (proof.get("tp_size"), proof.get("dp_size"), proof.get("ep_size")) != (
            2,
            2,
            4,
        ):
            errors.append(f"wrong actual topology rank {rank}")
        if (
            env.get("profile") == "off"
            and proof.get("startup_wrapper_removed") is not True
        ):
            errors.append(f"benchmark wrapper retained rank {rank}")
        if env.get("backend") == "native":
            if any(
                proof.get(k) != v
                for k, v in dict(
                    manager="DeepEPHTAll2AllManager",
                    prepare_finalize="DeepEPHTPrepareAndFinalize",
                    experts="TritonExperts",
                    runner="MoERunner",
                ).items()
            ):
                errors.append(f"native backend fallback rank {rank}")
        elif env.get("backend") == "miniep":
            transport = load(folder / "transport.json", {})
            meta = transport.get("metadata", {})
            calls = transport.get("collectives", [])
            if (
                transport.get("input_dtype") != "torch.bfloat16"
                or meta.get("return_dtype") != "torch.bfloat16"
                or len(calls) < 4
                or calls[-1].get("dtype") != "torch.bfloat16"
                or calls[-1].get("output_dtype") != "torch.bfloat16"
                or calls[-1].get("numel", 0) * calls[-1].get("element_size", 0)
                != meta.get("return_send_bytes")
                or transport.get("startup_wrapper_removed") is not True
            ):
                errors.append(f"actual default transport not proved rank {rank}")
        else:
            errors.append(f"unknown backend rank {rank}")
        by_name = {q["name"]: q for q in manifest}
        baseline_tokens = {}
        for q in data:
            case = by_name.get(q["case"])
            if case is None:
                continue
            vals = q.get("max_rank_ms", [])
            local = q.get("local_ms", [])
            if (
                len(vals) != 5
                or len(local) != 5
                or any(not math.isfinite(x) or x < 0 for x in vals + local)
            ):
                errors.append(f"invalid timing rank {rank}")
                continue
            if any(x <= 0 for x in vals[:3]):
                errors.append(f"nonpositive timing rank {rank}")
            if (case["steps"] > 1 and any(x <= 0 for x in vals[3:])) or (
                case["steps"] == 1 and any(vals[3:])
            ):
                errors.append(f"decode boundary mismatch rank {rank}")
            if q.get("unique_output_tokens") != 2 * case["batch_size"] * case["steps"]:
                errors.append(f"duplicate/missing output accounting rank {rank}")
            if q.get("unique_input_tokens") != 2 * case["batch_size"] * case["seq_len"]:
                errors.append(f"duplicate/missing prefill accounting rank {rank}")
            tokens = q.get("token_ids", [])
            if len(tokens) != case["batch_size"] or any(
                len(x) != case["steps"] for x in tokens
            ):
                errors.append(f"incomplete generation rank {rank}")
            if q["case"] in baseline_tokens and baseline_tokens[q["case"]] != tokens:
                errors.append(f"nonrepeatable generation rank {rank}")
            baseline_tokens[q["case"]] = tokens
            if env.get("profile") == "off" and q.get("profiling") is not None:
                errors.append(f"profiling overhead in primary samples rank {rank}")
    if any(
        e.get("options") != environments[0].get("options")
        or e.get("source_sha256") != environments[0].get("source_sha256")
        for e in environments[1:]
    ):
        errors.append("rank configurations/source snapshots differ")
    for a, b in ((0, 1), (2, 3)):
        if manifests[a] != manifests[b]:
            errors.append(f"TP manifests differ {a}/{b}")
        if [(q["case"], q["repetition"], q["token_ids"]) for q in samples[a]] != [
            (q["case"], q["repetition"], q["token_ids"]) for q in samples[b]
        ]:
            errors.append(f"TP outputs differ {a}/{b}")
    if not errors:
        maps = [{(q["case"], q["repetition"]): q for q in rows} for rows in samples]
        for key, first in maps[0].items():
            per_rank = [d[key] for d in maps]
            expected = [max(q["local_ms"][i] for q in per_rank) for i in range(5)]
            if any(q["max_rank_ms"] != expected for q in per_rank):
                errors.append("per-repetition slowest rank not used")
    return dict(
        passed=not errors,
        errors=sorted(set(errors)),
        backend=environments[0].get("backend"),
        profile=environments[0].get("profile"),
        cases=len(manifests[0]),
        repetitions=environments[0].get("repeats"),
        ranks=4,
    )


def summarize_trials(jobs: list[dict]) -> dict:
    """Primary statistics: median per-trial maximum rank, retaining raw samples."""
    grouped = defaultdict(list)
    errors = []
    for job in jobs:
        root = Path(job["path"])
        proof = assess_job(root)
        if not proof["passed"]:
            errors.append(dict(path=str(root), errors=proof["errors"]))
            continue
        if proof["profile"] != "off":
            continue
        for q in load(root / "rank0/samples.json", []):
            grouped[(job["backend"], q["case"])].append(q)
    result = {}
    for (backend, case), rows in grouped.items():
        result.setdefault(case, {})[backend] = dict(
            e2e_ms=distribution([q["max_rank_ms"][0] for q in rows]),
            cuda_interval_ms=distribution([q["max_rank_ms"][1] for q in rows]),
            prefill_batch_ms=distribution([q["max_rank_ms"][2] for q in rows]),
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
    for case, values in result.items():
        if not all(k in values for k in ("native", "miniep")):
            errors.append(dict(case=case, error="missing paired backend"))
            continue
        native = values["native"]
        mini = values["miniep"]
        values["miniep_over_deepep"] = dict(
            e2e_latency=mini["e2e_ms"]["median"] / native["e2e_ms"]["median"],
            prefill_latency=mini["prefill_batch_ms"]["median"]
            / native["prefill_batch_ms"]["median"],
            generated_throughput=mini["generated_tokens_per_second"]["median"]
            / native["generated_tokens_per_second"]["median"],
            prefill_throughput=mini["prefill_tokens_per_second"]["median"]
            / native["prefill_tokens_per_second"]["median"],
            decode_latency=(
                mini["decode_step_ms"]["median"] / native["decode_step_ms"]["median"]
                if mini["decode_step_ms"]
                else None
            ),
        )
    # Literal manifests and free output tokens must match between paired jobs.
    by_trial = defaultdict(dict)
    for job in jobs:
        if job.get("profile", "off") == "off":
            by_trial[job["trial"]][job["backend"]] = Path(job["path"])
    for trial, pair in by_trial.items():
        if set(pair) != {"native", "miniep"}:
            errors.append(dict(trial=trial, error="incomplete pair"))
            continue
        for rank in range(4):
            left = pair["native"] / f"rank{rank}"
            right = pair["miniep"] / f"rank{rank}"
            if load(left / "manifest.json") != load(right / "manifest.json"):
                errors.append(dict(trial=trial, rank=rank, error="input mismatch"))
            for key in ("options", "versions", "checkpoint_digest", "model_parameters"):
                if load(left / "environment.json", {}).get(key) != load(
                    right / "environment.json", {}
                ).get(key):
                    errors.append(dict(trial=trial, rank=rank, error=f"{key} mismatch"))
            a = [
                (x["case"], x["repetition"], x["token_ids"])
                for x in load(left / "samples.json", [])
            ]
            b = [
                (x["case"], x["repetition"], x["token_ids"])
                for x in load(right / "samples.json", [])
            ]
            if a != b:
                errors.append(dict(trial=trial, rank=rank, error="output mismatch"))
    return dict(
        passed=bool(result) and not errors,
        errors=errors,
        cases=result,
        statistic="median of per-repetition maximum-rank latencies; p10/p90 and raw samples retained",
        native_detailed_stages="not observed; no fabricated native stage split",
        overlap="not claimed",
    )
