"""Native HT/LL identity checks for experiments; no inference implementation."""

BACKENDS = {
    "deepep_high_throughput": (
        "DeepEPHTAll2AllManager",
        "DeepEPHTPrepareAndFinalize",
        "TritonExperts",
    ),
    "deepep_low_latency": (
        "DeepEPLLAll2AllManager",
        "DeepEPLLPrepareAndFinalize",
        "BatchedTritonExperts",
    ),
}


def validate_native_backend(
    proof: dict, backend: str, expected_topology: tuple[int, int, int] = (2, 2, 4)
) -> None:
    """Reject fallback, non-native runners, changed topology or LL FP8 dispatch."""
    if backend not in BACKENDS:
        raise ValueError(f"unsupported native benchmark backend {backend!r}")
    actual = tuple(proof.get(k) for k in ("manager", "prepare_finalize", "experts"))
    if (
        actual != BACKENDS[backend]
        or proof.get("runner") != "MoERunner"
        or tuple(proof.get(k) for k in ("tp_size", "dp_size", "ep_size"))
        != expected_topology
    ):
        raise RuntimeError(f"native {backend} fallback or wrong topology: {proof}")
    if backend == "deepep_low_latency" and (
        proof.get("use_fp8_dispatch") is not False
        or proof.get("input_dtype") != "torch.bfloat16"
    ):
        raise RuntimeError(f"LL experiment requires actual BF16 dispatch: {proof}")


def install_buffer_init_memory_probe() -> None:
    """Once at startup, release this process's unused cache before DeepEP allocates.

    No live tensor or other process allocation is freed. Restore the original
    constructor before warmup/timing; expert computation is never intercepted.
    """
    import inspect
    import json
    import os
    from pathlib import Path
    import deep_ep
    import torch

    original = deep_ep.Buffer.__init__

    def memory():
        free, total = torch.cuda.mem_get_info()
        return dict(
            free=free,
            total=total,
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
        )

    def initialize(self, *args, **kwargs):
        values = inspect.signature(original).bind(self, *args, **kwargs).arguments
        info = dict(
            initialization_only=True,
            scope="current process unused PyTorch cache only",
            before=memory(),
            parameters={
                k: v for k, v in values.items() if k not in ("self", "group", "comm")
            },
        )
        torch.cuda.empty_cache()
        info["after_cache_release"] = memory()
        path = Path(os.environ["MINIEP_BENCH_DIR"]) / "buffer-init-memory.json"
        path.write_text(json.dumps(info, indent=2))
        try:
            original(self, *args, **kwargs)
        finally:
            deep_ep.Buffer.__init__ = original
        info.update(
            completed=True,
            init_hook_removed=deep_ep.Buffer.__init__ is original,
            after_initialization=memory(),
        )
        path.write_text(json.dumps(info, indent=2))

    deep_ep.Buffer.__init__ = initialize


def verify_native_layers(worker) -> dict:
    """Outside timing: inspect every original native runner on each worker."""
    import json
    import os
    from pathlib import Path
    from vllm.distributed import get_ep_group, get_tp_group, get_dp_group
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    worker = getattr(worker, "worker", worker)
    group = get_ep_group()
    backend = os.environ["MINIEP_BENCH_ALL2ALL_BACKEND"]
    tp = int(os.environ.get("MINIEP_BENCH_TP_SIZE", "2"))
    expected_layers = int(os.environ["MINIEP_BENCH_MODEL_LAYERS"])
    runners = [m for m in worker.get_model().modules() if isinstance(m, MoERunner)]
    if len(runners) != expected_layers:
        raise RuntimeError(f"expected {expected_layers} original MoE runners, got {len(runners)}")
    records = []
    for runner in runners:
        kernel = runner._quant_method.moe_kernel
        proof = dict(
            layer=runner.layer_name,
            manager=type(group.device_communicator.all2all_manager).__name__,
            prepare_finalize=type(kernel.prepare_finalize).__name__,
            experts=type(kernel.fused_experts).__name__,
            runner=type(runner).__name__,
            tp_size=get_tp_group().world_size,
            dp_size=get_dp_group().world_size,
            ep_size=group.world_size,
            use_fp8_dispatch=getattr(kernel.prepare_finalize, "use_fp8_dispatch", None),
            input_dtype=str(worker.vllm_config.model_config.dtype),
            weight_dtype=str(runner.routed_experts.w13_weight.dtype),
            max_tokens_per_rank=getattr(
                kernel.prepare_finalize, "max_tokens_per_rank", None
            ),
            nvshmem_qp_depth=getattr(
                getattr(kernel.prepare_finalize, "buffer", None),
                "nvshmem_qp_depth",
                None,
            ),
            num_rdma_bytes=getattr(
                getattr(kernel.prepare_finalize, "buffer", None), "num_rdma_bytes", None
            ),
        )
        validate_native_backend(proof, backend, (tp, 2, tp * 2))
        if proof["weight_dtype"] != "torch.bfloat16":
            raise RuntimeError(f"native experiment requires BF16 weights: {proof}")
        records.append(proof)
    if len({q["layer"] for q in records}) != expected_layers:
        raise RuntimeError("duplicate native layer identities")
    result = dict(backend=backend, layers=records, native_only=True)
    (Path(os.environ["MINIEP_BENCH_DIR"]) / "model_backend.json").write_text(
        json.dumps(result, indent=2)
    )
    return dict(native_only=True, verified_layers=len(records), backend=backend)
