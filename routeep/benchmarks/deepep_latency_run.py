"""One fresh TP1/DP2 or TP2/DP2 timing job; no accuracy probes.

Scope is offline token IDs -> generated tokens, including normal scheduler,
attention, router, EP and detokenization. Model loading, JIT, input construction,
warmup, boundary collectives and result serialization are outside measurements.
"""

import argparse
from datetime import timedelta
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_ep.integration.config import configure, engine_options, model_metadata
from mini_ep.integration.transport import get_transport
from benchmarks.deepep_workloads import build_workloads, PRESETS
from benchmarks.deepep_metrics import request_timing, batch_timing
from validation.model_info import validated_model_shape


def gpu_snapshot() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,uuid,driver_version,memory.used,utilization.gpu,clocks.sm,power.draw",
            "--format=csv",
        ],
        text=True,
    )


def clock_warmup(seconds: float) -> None:
    import torch

    values = torch.ones((2048, 2048), device="cuda", dtype=torch.bfloat16)
    until = time.perf_counter() + seconds
    while time.perf_counter() < until:
        result = values @ values
        # Outside the measured model path; bound the queued warmup work.
        torch.cuda.synchronize()
    del values


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--workload-spec", help="JSON extended workloads; exact IDs persisted in manifest")
    p.add_argument("--max-model-len", type=int, default=2304)
    p.add_argument("--max-num-seqs", type=int, default=32)
    p.add_argument("--output", required=True)
    p.add_argument("--checkpoint-manifest", required=True)
    p.add_argument("--backend", choices=["native", "miniep"], required=True)
    p.add_argument("--safetensors-load-strategy", choices=["lazy", "prefetch"], default="prefetch")
    p.add_argument("--transport", type=lambda value: "peer" if value == "peer_dedup" else value,
                   choices=["auto", "nccl", "peer"],
                   help="RouteEP defaults to MINIEP_TRANSPORT or auto (peer on SM90+ single-host P2P, else nccl); both always deduplicate")
    p.add_argument("--owner-fixed", choices=["auto", "off"], default="auto",
                   help="NCCL dedup fixed-capacity A2A policy (MINIEP_OWNER_FIXED); auto is the runner default")
    p.add_argument("--shared-stream", choices=["auto", "off"], default="off",
                   help="Qwen3.5 shared expert side stream (MINIEP_SHARED_STREAM); auto only for TP=1")
    p.add_argument("--tp-size", type=int, choices=[1, 2], default=2)
    p.add_argument(
        "--all2all-backend",
        choices=["deepep_high_throughput", "deepep_low_latency"],
        default="deepep_high_throughput",
        help="LL is a native-only diagnostic; RouteEP remains HT-aligned",
    )
    p.add_argument(
        "--label", required=True, help="snapshot label, e.g. before/final/native"
    )
    p.add_argument(
        "--cases",
        nargs="+",
        default=None,
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--clock-warmup-seconds", type=float, default=5.0)
    p.add_argument("--profile", choices=["off", "events"], default="off")
    p.add_argument("--cuda-profile-case", choices=[q[0] for q in PRESETS])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.30)
    p.add_argument("--kv-cache-mib", type=int, default=512)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--enable-chunked-prefill", action="store_true")
    p.add_argument("--release-unused-cache-before-buffer-init", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    if a.transport is None:
        a.transport = get_transport() if a.backend == "miniep" else "nccl"
    if a.backend != "native" and a.all2all_backend != "deepep_high_throughput":
        p.error("LL benchmarking is native-only; do not change RouteEP's target")
    if a.backend != "native" and a.enable_chunked_prefill:
        p.error("chunked-prefill experiments here are native-only")
    if a.backend != "native" and a.release_unused_cache_before_buffer_init:
        p.error("buffer initialization memory experiments are native-only")
    if a.max_num_batched_tokens < 32:
        p.error("token budget must cover max_num_seqs=32")
    if a.max_model_len < 1 or a.max_num_seqs < 1:
        p.error("model length and sequence capacity must be positive")
    if not a.enable_chunked_prefill and a.max_num_batched_tokens < a.max_model_len:
        p.error(
            f"unchunked prefill requires max-num-batched-tokens >= {a.max_model_len}"
        )
    if a.all2all_backend == "deepep_low_latency":
        depth = int(os.environ.get("NVSHMEM_QP_DEPTH", "1024"))
        required = 2 * (a.max_num_batched_tokens + 1)
        if depth < required:
            p.error(
                f"NVSHMEM_QP_DEPTH={depth} must be >= {required} for this LL capacity"
            )
    if a.transport == "peer" and a.backend != "miniep":
        p.error("peer transport requires --backend miniep")
    world_size = a.tp_size * 2
    if int(os.environ.get("WORLD_SIZE", "0")) != world_size:
        p.error(
            "use torchrun with four ranks"
            if world_size == 4
            else "use torchrun with two ranks"
        )
    if a.warmup < 1 or a.repeats < 3 or a.clock_warmup_seconds < 0:
        p.error("warmup>=1, repeats>=3, clock warmup>=0 required")
    rank = int(os.environ["RANK"])
    root = Path(a.output).resolve() / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=False)
    # Exercise actual production defaults, clearing inherited experiment policies.
    for key in list(os.environ):
        if key.startswith("MINIEP_"):
            os.environ.pop(key)
    configure(a.backend)
    os.environ["MINIEP_TRANSPORT"] = a.transport
    # 外部 MINIEP_* 已清除；固定容量策略只能通过显式参数选择，并记录在环境快照中。
    os.environ["MINIEP_OWNER_FIXED"] = a.owner_fixed
    os.environ["MINIEP_SHARED_STREAM"] = a.shared_stream
    os.environ.update(
        VLLM_PLUGINS="",
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        NCCL_NVLS_ENABLE="0",
        MINIEP_BENCH_DIR=str(root),
        MINIEP_BENCH_PROFILE=a.profile,
        MINIEP_BENCH_ALL2ALL_BACKEND=a.all2all_backend,
        MINIEP_BENCH_TP_SIZE=str(a.tp_size),
    )
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    checkpoint = json.loads(Path(a.checkpoint_manifest).read_text())
    for name, info in checkpoint["files"].items():
        stat = (Path(a.model) / name).stat()
        if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"validated checkpoint changed: {name}")
    expected = validated_model_shape(a.model)
    metadata = model_metadata(a.model)
    selected = sorted(
        {0, metadata["num_hidden_layers"] // 2, metadata["num_hidden_layers"] - 1}
    )
    os.environ["MINIEP_BENCH_LAYERS"] = ",".join(map(str, selected))
    os.environ["MINIEP_BENCH_MODEL_LAYERS"] = str(metadata["num_hidden_layers"])
    options = engine_options(a.model, dp_size=2, external=True, tp_size=a.tp_size)
    options.update(
        all2all_backend=a.all2all_backend,
        tensor_parallel_size=a.tp_size,
        seed=a.seed,
        max_model_len=a.max_model_len,
        max_num_batched_tokens=a.max_num_batched_tokens,
        enable_chunked_prefill=a.enable_chunked_prefill,
        max_num_seqs=a.max_num_seqs,
        enable_dbo=False,
        enable_eplb=False,
        quantization=None,
        gpu_memory_utilization=a.gpu_memory_utilization,
        kv_cache_memory_bytes=a.kv_cache_mib * 1024 * 1024,
        safetensors_load_strategy=a.safetensors_load_strategy,
        safetensors_prefetch_num_threads=2,
        distributed_timeout_seconds=180,
        cpu_distributed_timeout_seconds=180,
        disable_log_stats=False,
    )
    course = Path(__file__).resolve().parents[1]
    sources = {
        str(f.relative_to(course)): hashlib.sha256(f.read_bytes()).hexdigest()
        for folder in ("mini_ep", "benchmarks", "validation")
        for f in (course / folder).rglob("*.py")
    }
    environment = dict(
        options=options,
        backend=a.backend,
        label=a.label,
        transport=a.transport,
        rank=rank,
        world_size=world_size,
        profile=a.profile,
        release_unused_cache_before_buffer_init=a.release_unused_cache_before_buffer_init,
        scope="offline token IDs to full generated/detokenized outputs",
        warmup=a.warmup,
        repeats=a.repeats,
        clock_warmup_seconds=a.clock_warmup_seconds,
        checkpoint_digest=checkpoint["digest"],
        model_parameters=expected,
        cpu_load=os.getloadavg(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_quota=(
            Path("/sys/fs/cgroup/cpu.max").read_text()
            if Path("/sys/fs/cgroup/cpu.max").exists()
            else None
        ),
        cuda_version=torch.version.cuda,
        nccl_version=torch.cuda.nccl.version(),
        debug_environment={
            k: os.environ.get(k)
            for k in (
                "CUDA_LAUNCH_BLOCKING",
                "NCCL_DEBUG",
                "TORCH_DISTRIBUTED_DEBUG",
                "VLLM_TORCH_PROFILER_DIR",
            )
        },
        nvshmem_environment={
            k: v for k, v in os.environ.items() if k.startswith("NVSHMEM_")
        },
        source_sha256=sources,
        hostname=socket.gethostname(),
        gpu_before=gpu_snapshot(),
        command=sys.argv,
        versions={
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton", "transformers", "deep_ep")
        },
    )
    (root / "environment.json").write_text(json.dumps(environment, indent=2))
    from benchmarks import deepep_plugin as observer

    observer.register()
    if a.release_unused_cache_before_buffer_init:
        from benchmarks.deepep_native_backend import install_buffer_init_memory_probe

        install_buffer_init_memory_probe()
    from vllm import LLM, SamplingParams
    from vllm.distributed import (
        get_dp_group,
        get_tp_group,
        destroy_model_parallel,
        destroy_distributed_environment,
    )

    llm = LLM(**options)
    if a.backend == "native":
        from benchmarks.deepep_native_backend import verify_native_layers

        llm.collective_rpc(verify_native_layers, timeout=180)
    dp_rank = get_dp_group().rank_in_group
    tp_rank = get_tp_group().rank_in_group
    if a.workload_spec:
        from benchmarks.extended_workloads import build_extended_workloads

        available = build_extended_workloads(llm.get_tokenizer(), dp_rank, a.workload_spec)
    else:
        available = build_workloads(llm.get_tokenizer(), dp_rank)
    cases = [q for q in available if a.cases is None or q["name"] in a.cases]
    if not cases or (a.cases is not None and set(a.cases) - {q["name"] for q in cases}):
        raise ValueError(f"unknown or empty workload selection: {a.cases}")
    for case in cases:
        lengths = [len(q["prompt_token_ids"]) for q in case["prompts"]]
        # 开启 chunked prefill 时，vLLM 会把超出预算的 prefill 拆成多步执行，
        # 所以 Token 总数只在关闭 chunked prefill 时需要 <= 调度预算。
        over_budget = not a.enable_chunked_prefill and sum(lengths) > a.max_num_batched_tokens
        if (len(lengths) > a.max_num_seqs or over_budget
            or max(lengths) + case["steps"] > a.max_model_len):
            raise ValueError(f"case exceeds declared engine capacity: {case['name']}")
    (root / "manifest.json").write_text(json.dumps(cases, indent=2))
    results = []

    def execute(case):
        llm.enqueue(
            case["prompts"],
            SamplingParams(
                temperature=0, seed=a.seed, max_tokens=case["steps"], ignore_eos=True
            ),
            use_tqdm=False,
        )
        return llm.wait_for_completion(use_tqdm=False)

    for case in cases:
        warmup_steps = 0
        for _ in range(a.warmup):
            if a.profile == "events":
                observer.start_warmup_count()
            execute(case)
            if a.profile == "events":
                warmup_steps = max(
                    warmup_steps,
                    observer.end_warmup_count(metadata["num_hidden_layers"]),
                )
        clock_warmup(a.clock_warmup_seconds)
        before = gpu_snapshot()
        cpu_before = (
            Path("/sys/fs/cgroup/cpu.stat").read_text()
            if Path("/sys/fs/cgroup/cpu.stat").exists()
            else None
        )
        if a.cuda_profile_case == case["name"]:
            torch.cuda.profiler.start()
        for repetition in range(a.repeats):
            if a.profile == "events":
                observer.begin_case(
                    case["name"],
                    max(
                        case["steps"] + len(case["prompts"]) + 8,
                        2 * warmup_steps + 8,
                    ),
                    metadata["num_hidden_layers"],
                    len(selected) if a.backend == "miniep" else 0,
                )
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            # Allocate timing handles before ROI as well.
            begin.record()
            end.record()
            torch.cuda.synchronize()
            started = time.perf_counter()
            begin.record()
            generated = execute(case)
            end.record()
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - started) * 1000
            cuda_ms = begin.elapsed_time(end)
            profile = observer.finish_case() if a.profile == "events" else None
            if len(generated) != len(case["prompts"]):
                raise RuntimeError("incomplete benchmark batch")
            requests = []
            tokens = []
            for prompt, result in zip(case["prompts"], generated, strict=True):
                if result.prompt_token_ids != prompt["prompt_token_ids"]:
                    raise RuntimeError("request order/input changed")
                output = result.outputs[0].token_ids
                if len(output) != case["steps"]:
                    raise RuntimeError("incomplete generated sequence")
                if result.num_cached_tokens not in (None, 0):
                    raise RuntimeError("prefix cache unexpectedly used")
                requests.append(request_timing(result.metrics, len(output)))
                tokens.append(output)
            intervals = batch_timing(requests)
            local = [
                elapsed_ms,
                cuda_ms,
                intervals["prefill_batch_ms"],
                intervals["decode_batch_ms"] or 0.0,
                intervals["decode_step_ms"] or 0.0,
            ]
            maximum = torch.tensor(local, device="cuda", dtype=torch.float64)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            counts = torch.tensor(
                [
                    sum(map(len, tokens)),
                    sum(len(x["prompt_token_ids"]) for x in case["prompts"]),
                ],
                device="cuda",
                dtype=torch.int64,
            )
            if tp_rank:
                counts.zero_()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            maximum = maximum.cpu().tolist()
            out_count, in_count = counts.cpu().tolist()
            record = dict(
                case=case["name"],
                repetition=repetition,
                local_ms=local,
                max_rank_ms=maximum,
                unique_output_tokens=out_count,
                unique_input_tokens=in_count,
                generated_tokens_per_second=out_count / (maximum[0] / 1000),
                prefill_tokens_per_second=in_count / (maximum[2] / 1000),
                token_ids=tokens,
                request_metrics=requests,
                profiling=profile,
            )
            results.append(record)
            (root / "samples.json").write_text(
                json.dumps(results, indent=2, allow_nan=False)
            )
            print(
                f'MEASURE rank={rank} backend={a.backend} case={case["name"]} repeat={repetition} max_e2e_ms={maximum[0]:.3f}',
                flush=True,
            )
        if a.cuda_profile_case == case["name"]:
            torch.cuda.profiler.stop()
        (root / (case["name"] + "-gpu.json")).write_text(
            json.dumps(
                dict(
                    before=before,
                    after=gpu_snapshot(),
                    cpu_before=cpu_before,
                    cpu_after=(
                        Path("/sys/fs/cgroup/cpu.stat").read_text()
                        if Path("/sys/fs/cgroup/cpu.stat").exists()
                        else None
                    ),
                ),
                indent=2,
            )
        )
    if not (root / "backend.json").exists():
        raise RuntimeError("actual backend proof missing")
    if a.backend == "miniep" and not (root / "transport.json").exists():
        raise RuntimeError("actual RouteEP transport proof missing")
    (root / "completed.json").write_text(
        json.dumps(dict(cases=[x["name"] for x in cases], samples=len(results)))
    )
    del llm
    gc.collect()
    if a.transport in ("peer", "auto"):
        # auto 可能解析成 peer；未创建过工作区时清理是空操作。
        from mini_ep.distributed.peer_dedup import clear_deduplicated_peer_workspaces

        clear_deduplicated_peer_workspaces()
    destroy_model_parallel()
    destroy_distributed_environment()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
