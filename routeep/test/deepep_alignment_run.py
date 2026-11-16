"""一个隔离的四 Rank 精确验证 job，由 deepep_alignment.py 启动。

native job 保持原始 MoERunner 类不变。候选 job 先在 native 前缀上逐层比较
MoE，再分别执行 teacher forcing 和自由生成。所有精度探针都在 mini_ep 之外，
不会进入性能结论。
"""

import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_ep.integration.config import engine_options, model_metadata
from validation.config import configure
from validation.alignment_cases import build_cases
from validation.alignment_report import rows, summarize_rank, references_match
from validation.model_info import validated_model_shape


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        choices=["deepep_high_throughput"],
        default="deepep_high_throughput",
    )
    p.add_argument("--world-size", type=int, choices=[4], default=4)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["native", "candidate"], required=True)
    p.add_argument("--reference")
    p.add_argument("--checkpoint-manifest", required=True)
    p.add_argument(
        "--suite",
        choices=["corpus", "boundaries", "long", "performance"],
        default="corpus",
    )
    p.add_argument("--corpus", choices=["same", "rotated"], default="rotated")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--kv-cache-mib", type=int, default=512)
    p.add_argument("--native-reference-audit")
    p.add_argument("--native-runtime-manifest")
    p.add_argument("--transport", choices=["nccl", "peer"], default="nccl",
                   help="candidate 的 RouteEP 传输（MINIEP_TRANSPORT）；两种都固定去重")
    args = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != args.world_size:
        p.error("torchrun WORLD_SIZE differs from --world-size")
    if not 1 <= args.limit <= 32 or not 1 <= args.max_tokens <= 256:
        p.error("limit=1..32 and max-tokens=1..256 required")
    if args.kind == "candidate" and not args.reference:
        p.error("candidate requires independent --reference")
    rank = int(os.environ["RANK"])
    root = Path(args.output).resolve() / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=False)
    if os.environ.get("MINIEP_STAGE_TIMING", "0") != "0":
        p.error(
            "accuracy checks require MINIEP_STAGE_TIMING=0; use the separate benchmark profiler"
        )
    configure("deepep_native" if args.kind == "native" else "deepep_exact", str(root))
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ.update(MINIEP_RETURN_DTYPE="auto", MINIEP_EXACT_BACKEND=args.backend,
                      MINIEP_TRANSPORT=args.transport)
    checkpoint = json.loads(Path(args.checkpoint_manifest).read_text())
    for name, info in checkpoint["files"].items():
        stat = (Path(args.model) / name).stat()
        if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"frozen checkpoint changed: {name}")
    options = engine_options(args.model, dp_size=args.world_size // 2, external=True)
    options.update(
        all2all_backend=args.backend,
        seed=args.seed,
        enable_dbo=False,
        enable_eplb=False,
        quantization=None,
        max_model_len=2304,
        max_num_batched_tokens=8192 if args.suite == "performance" else 2304,
        max_num_seqs=32 if args.suite == "performance" else 4,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024 * 1024,
        safetensors_load_strategy="prefetch",
        safetensors_prefetch_num_threads=2,
        distributed_timeout_seconds=180,
        cpu_distributed_timeout_seconds=180,
    )
    course = Path(__file__).resolve().parents[1]
    sources = {
        str(f.relative_to(course)): hashlib.sha256(f.read_bytes()).hexdigest()
        for folder in ("mini_ep", "validation", "test", "benchmarks")
        for f in (course / folder).rglob("*.py")
    }
    versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "torch", "triton", "transformers")
    }
    if versions["vllm"] != "0.29.0":
        raise RuntimeError(f"expected vLLM 0.29.0, got {versions}")
    spec = dict(
        options=options,
        seed=args.seed,
        suite=args.suite,
        corpus=args.corpus,
        limit=args.limit,
        steps=args.max_tokens,
        source_sha256=sources,
        versions=versions,
        checkpoint=checkpoint["digest"],
    )
    metadata = model_metadata(args.model)
    shape = validated_model_shape(args.model)
    env = dict(
        kind=args.kind,
        world_size=args.world_size,
        backend=args.backend,
        transport=args.transport,
        rank=rank,
        local_rank=int(os.environ["LOCAL_RANK"]),
        signature=digest(spec),
        spec=spec,
        synthetic=False,
        model_family=shape["model_family"],
        layers=metadata["num_hidden_layers"],
        hidden_size=shape["hidden_size"],
        model_parameters=shape,
        command=sys.argv,
        hostname=socket.gethostname(),
        gpu_snapshot=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu",
                "--format=csv",
            ],
            text=True,
        ),
    )
    if args.native_reference_audit:
        from validation.native_reuse import verify_runtime

        runtime = json.loads(Path(args.native_runtime_manifest).read_text())
        verify_runtime(runtime)
        audit = json.loads(Path(args.native_reference_audit).read_text())
        if audit["reference"] != str(Path(args.reference).resolve()):
            raise RuntimeError("reference audit path differs")
        env.update(
            native_reference_audit=audit, native_runtime_digest=runtime["digest"]
        )
    (root / "environment.json").write_text(json.dumps(env, indent=2))
    from vllm import LLM, SamplingParams

    llm = LLM(**options)
    manifests = build_cases(
        llm.get_tokenizer(),
        args.suite,
        rank // 2,
        args.corpus,
        args.limit,
        args.max_tokens,
        dp_size=args.world_size // 2,
    )
    (root / "manifest.json").write_text(json.dumps(manifests, indent=2))
    reference = (
        Path(args.reference).resolve() / f"rank{rank}" if args.reference else None
    )
    if reference:
        other = json.loads((reference / "environment.json").read_text())
        if not references_match(env, other):
            raise RuntimeError("independent reference signature differs")
        if json.loads((reference / "manifest.json").read_text()) != manifests:
            raise RuntimeError("reference requests differ")
    control = root / "control.json"
    completed = {}
    phases = (
        ("native",)
        if args.kind == "native"
        else ("native_check", "compare", "teacher", "free")
    )
    for phase in phases:
        completed[phase] = []
        outputs = []
        folder = root / phase
        folder.mkdir()
        for case in manifests:
            state = dict(
                key=case["name"],
                batch_key=case["name"],
                phase=phase,
                acceptance="exact",
                baseline_dir=str(
                    (reference if reference else root) / "baseline_logits"
                ),
                matrix_mode=(
                    "native"
                    if phase in ("native", "native_check")
                    else "compare" if phase == "compare" else "miniep"
                ),
                skip_logits=not case["prompts"],
                logical_requests=len(case["prompts"]),
                owner_reduction="local_fp32",
            )
            from validation.plugin import set_control, flush_records

            set_control(state)
            temp = control.with_suffix(".tmp")
            temp.write_text(json.dumps(state))
            temp.replace(control)
            if case["prompts"]:
                sampling = [
                    SamplingParams(
                        temperature=0, seed=args.seed, max_tokens=n, ignore_eos=True
                    )
                    for n in case.get(
                        "request_steps", [case["steps"]] * len(case["prompts"])
                    )
                ]
                llm.enqueue(case["prompts"], sampling, use_tqdm=False)
            generated = llm.wait_for_completion(use_tqdm=False)
            if args.kind == "native":
                from validation.deepep_backend import flush_native_observer

                flush_native_observer()
            flush_records()
            if len(generated) != len(case["prompts"]):
                raise RuntimeError("request count differs")
            for i, (request, result) in enumerate(
                zip(case["prompts"], generated, strict=True)
            ):
                if result.prompt_token_ids != request["prompt_token_ids"]:
                    raise RuntimeError("request output order differs")
                outputs.append(
                    dict(
                        case=case["name"],
                        index=i,
                        token_ids=result.outputs[0].token_ids,
                    )
                )
            completed[phase].append(case["name"])
            (folder / "outputs.json").write_text(json.dumps(outputs))
            (root / "completed.json").write_text(json.dumps(completed))
            print(f'COMPLETE rank={rank} phase={phase} case={case["name"]}', flush=True)
        set_control({})
        flush_records()
        control.unlink()
    if args.kind == "native":
        coverage = Counter()
        for q in rows(root / f"native-logits-rank{rank}.jsonl"):
            if not q["finite"]:
                raise RuntimeError("nonfinite native logits")
            file = (
                root
                / "baseline_logits"
                / f'{q["key"]}-rank{rank}-step{q["step"]:03d}.pt'
            )
            if not file.exists():
                raise RuntimeError(f"missing full-vocabulary baseline {file}")
            coverage[q["key"]] += q["rows"]
        (root / "native-logits.json").write_text(
            json.dumps({q["name"]: coverage[q["name"]] for q in manifests})
        )
    report = summarize_rank(root, reference)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print("EXACT_RESULT", json.dumps(report), flush=True)
    del llm
    gc.collect()
    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

    destroy_model_parallel()
    destroy_distributed_environment()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        from validation.plugin import flush_records

        flush_records()
        traceback.print_exc()
        root = os.environ.get("MINIEP_REPORT_DIR")
        if root:
            (Path(root) / "driver-failure.json").write_text(
                json.dumps(dict(error=str(error), type=type(error).__name__), indent=2)
            )
        sys.stdout.flush()
        sys.stderr.flush()
        # 让 torchrun 尽快看到失败 Rank；不要在进程组销毁阶段等待仍在
        # 进入下一次 native/mini 集合通信的其它 Rank。
        os._exit(1)
