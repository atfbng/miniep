"""Real zero-row DeepEP/Mini-EP layer API checks using all 48 loaded Qwen layers.

This supplements end-to-end idle-request cases, where vLLM may supply dummy
rows. It does not claim the full scheduler executes a physically empty tensor.
Run four ranks. Native MoERunner and kernels are unchanged; a separate Mini-EP
Runner shares the same weights under a distinct diagnostic registry name.
"""

import argparse
from datetime import timedelta
import gc
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def exercise(worker):
    worker = getattr(worker, "worker", worker)
    import torch
    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.distributed import get_dp_group, get_ep_group
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from mini_ep.integration.runner import MiniEPRunner
    from validation.deepep_backend import verify_backend
    from validation.exact import exact_tensor_flag, exact_tensor_metrics

    cfg = worker.vllm_config
    rank = dist.get_rank()
    dp = get_dp_group().rank_in_group
    group = get_ep_group().device_group
    root = Path(os.environ["MINIEP_REPORT_DIR"])
    native = [m for m in worker.get_model().modules() if type(m) is MoERunner]
    if len(native) != 48:
        raise RuntimeError(f"expected 48 original MoERunners, got {len(native)}")
    records = []
    with set_current_vllm_config(cfg), torch.inference_mode():
        minis = [
            MiniEPRunner(
                layer_name=m.layer_name + ".empty_rank_probe",
                moe_config=m.moe_config,
                router=m.router,
                routed_experts=m.routed_experts,
                gate=m.gate,
            )
            for m in native
        ]
        for seed in (13, 37):
            for counts in ((0, 2), (2, 0), (0, 16), (16, 0)):
                for mode in ("auto", "off"):
                    n = counts[dp] // 2
                    for layer, (original, mini) in enumerate(
                        zip(native, minis, strict=True)
                    ):
                        torch.manual_seed(seed + rank * 100 + layer)
                        hidden = (
                            torch.randn(n, 2048, device="cuda", dtype=torch.bfloat16)
                            * 0.2
                        )
                        mini.fixed_a2a_mode = mode
                        routes = []
                        select = original.router.select_experts

                        def capture(*args, **kwargs):
                            result = select(*args, **kwargs)
                            routes.append(tuple(x.detach().clone() for x in result))
                            return result

                        with set_forward_context(
                            None,
                            cfg,
                            num_tokens=counts[dp],
                            num_tokens_across_dp=torch.tensor(
                                counts, dtype=torch.int32
                            ),
                        ):
                            original.router.select_experts = capture
                            try:
                                expected = original(
                                    hidden.clone(), hidden.clone()
                                ).clone()
                            finally:
                                original.router.select_experts = select
                            verify_backend(original)
                            original.router.select_experts = capture
                            split = len(routes)
                            try:
                                actual, meta = mini.forward_ep(
                                    hidden, router_logits=hidden
                                )
                            finally:
                                original.router.select_experts = select
                        a, b = routes[:split], routes[split:]
                        route_ok = torch.ones((), device="cuda", dtype=torch.bool)
                        if n:
                            if len(a) != 1 or len(b) != 1:
                                route_ok.fill_(False)
                            else:
                                for x, y in zip(a[0], b[0], strict=True):
                                    route_ok &= exact_tensor_flag(x, y)
                        else:
                            # No branch exists; validate any actual router outputs
                            # rather than inventing a nonempty routing execution.
                            for group_routes in (a, b):
                                for output in group_routes:
                                    if len(output) != 2 or any(
                                        x.shape != (0, 8) for x in output
                                    ):
                                        route_ok.fill_(False)
                        value_ok = exact_tensor_flag(actual, expected)
                        meta_ok = (
                            actual.shape == (n, 2048)
                            and actual.dtype == hidden.dtype
                            and actual.device == hidden.device
                            and meta["return_buffer_dtype"] == str(hidden.dtype)
                            and meta["return_dtype"] == str(hidden.dtype)
                            and meta["return_send_bytes"]
                            == meta["recv_rows"] * 2048 * hidden.element_size()
                            and meta["source_tokens"] == n
                        )
                        bad = (~(value_ok & route_ok)).to(torch.int32)
                        if not meta_ok:
                            bad.fill_(1)
                        dist.all_reduce(bad, op=dist.ReduceOp.MAX, group=group)
                        failed = bool(bad.item())
                        record = dict(
                            seed=seed,
                            counts=list(counts),
                            fixed_mode=mode,
                            layer=layer,
                            framework_rows=n,
                            router_exact=bool(route_ok.item()),
                            exact_equal=not failed,
                            native_router_calls=len(a),
                            mini_router_calls=len(b),
                            **meta,
                        )
                        records.append(record)
                        if failed:
                            failure = root / "first-failure"
                            failure.mkdir(exist_ok=True)
                            torch.save(
                                dict(
                                    input=hidden.cpu(),
                                    native=expected.cpu(),
                                    mini=actual.cpu(),
                                    native_routes=[[v.cpu() for v in r] for r in a],
                                    mini_routes=[[v.cpu() for v in r] for r in b],
                                ),
                                failure / "tensors.pt",
                            )
                            (failure / "summary.json").write_text(
                                json.dumps(
                                    dict(
                                        record=record,
                                        metrics=exact_tensor_metrics(actual, expected),
                                    ),
                                    indent=2,
                                    allow_nan=False,
                                )
                            )
                            ack = torch.ones(1, device="cuda")
                            dist.all_reduce(ack, group=group)
                            ack.item()
                            raise RuntimeError(
                                f"empty-row native/Mini-EP mismatch rank={rank} layer={layer}"
                            )
    (root / "layers.json").write_text(json.dumps(records, indent=2, allow_nan=False))
    result = dict(
        passed=True,
        rank=rank,
        layers=48,
        cases=len(records),
        physical_empty_cases=sum(q["framework_rows"] == 0 for q in records),
        scope="direct real-weight 48-layer API; independent from scheduler dummy padding",
    )
    (root / "completed.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        p.error("use torchrun --nproc-per-node=4")
    import torch
    import torch.distributed as dist
    from mini_ep.integration.config import configure, engine_options

    rank = int(os.environ["RANK"])
    root = Path(a.output).resolve() / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=False)
    configure("native")
    os.environ.update(
        VLLM_ENABLE_V1_MULTIPROCESSING="0",
        MINIEP_EXACT_BACKEND="deepep_high_throughput",
        MINIEP_REPORT_DIR=str(root),
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    from vllm import LLM

    options = engine_options(a.model, dp_size=2, external=True)
    options.update(
        max_model_len=2304,
        max_num_batched_tokens=2304,
        max_num_seqs=4,
        gpu_memory_utilization=0.3,
        kv_cache_memory_bytes=512 * 1024 * 1024,
        safetensors_load_strategy="prefetch",
        safetensors_prefetch_num_threads=2,
        distributed_timeout_seconds=180,
        cpu_distributed_timeout_seconds=180,
    )
    (root / "environment.json").write_text(
        json.dumps(dict(options=options, rank=rank, command=sys.argv), indent=2)
    )
    llm = LLM(**options)
    result = llm.collective_rpc(exercise, timeout=600)
    print(json.dumps(result), flush=True)
    del llm
    gc.collect()
    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
