"""只用于 Benchmark 的启动探针和可选事件采样，位于 mini_ep 之外。

正常测量会在 warmup 结束前恢复所有启动包装；可选 profiling 会在 case 边界
同步后保留事件。这里不做张量克隆、影子前向、逐层文件 IO 或热路径同步。
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import torch
from mini_ep.integration.transport import get_transport

_STATE = dict(
    active=None, events=[], stages=[], traffic=[], pool=[], pool_index=0,
    counting_warmup=False, warmup_calls=0,
)


def start_warmup_count() -> None:
    """开始统计一次未计时的 workload 中实际发生了多少次 MoE 调用。"""
    _STATE.update(counting_warmup=True, warmup_calls=0)


def end_warmup_count(model_layers: int) -> int:
    """返回 warmup 期间观察到的完整模型步数，并在此之后才分配事件。"""
    calls = _STATE["warmup_calls"]
    _STATE["counting_warmup"] = False
    if calls < model_layers or calls % model_layers:
        raise RuntimeError(
            f"warmup MoE coverage={calls} does not contain whole {model_layers}-layer steps"
        )
    return calls // model_layers


def begin_case(
    name: str, max_model_steps: int, model_layers: int, selected_layers: int
) -> None:
    """在测量区间开始前初始化事件句柄，并由当前 case 持有这些事件。

    事件会先在预热阶段 record 一次并同步，避免把首次 CUDA Event 分配算进 ROI。
    """
    if _STATE["events"] or _STATE["stages"]:
        raise RuntimeError("drain the previous profiling case before reuse")
    count = 2 * max_model_steps * (model_layers + 8 * selected_layers)
    pool = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    # CUDA Event 首次使用时才真正构造；先 record 一次，把分配开销留在 ROI 之外。
    for event in pool:
        event.record()
    torch.cuda.synchronize()
    _STATE.update(active=name, pool=pool, pool_index=0)


def _event_pair():
    position = _STATE["pool_index"]
    if position + 2 > len(_STATE["pool"]):
        raise RuntimeError(
            f"profiling event bound exhausted at {position + 2}/{len(_STATE['pool'])}; "
            "no timing result accepted"
        )
    _STATE["pool_index"] = position + 2
    return _STATE["pool"][position : position + 2]


class _StageSamples:
    """与核心 Stages 使用相同阶段协议，但事件在测量前已预分配。"""

    timings = True

    def __init__(self):
        self.events = {}

    @contextmanager
    def stage(self, name):
        start, end = _event_pair()
        torch.cuda.nvtx.range_push(name)
        start.record()
        try:
            yield
        finally:
            end.record()
            torch.cuda.nvtx.range_pop()
            self.events.setdefault(name, []).append((start, end))


def finish_case() -> dict:
    """调用方已在测量边界同步一次，此时事件时间可以安全读取。"""
    result = dict(
        moe=[
            dict(layer=layer, milliseconds=start.elapsed_time(end))
            for layer, start, end in _STATE["events"]
        ],
        stages=[
            dict(layer=layer, stage=name, milliseconds=start.elapsed_time(end))
            for layer, name, start, end in _STATE["stages"]
        ],
        traffic=_STATE["traffic"],
    )
    _STATE.update(active=None, events=[], stages=[], traffic=[], pool=[], pool_index=0)
    return result


def _write(name: str, data: dict) -> None:
    root = Path(os.environ["MINIEP_BENCH_DIR"])
    (root / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def _native_proof(runner, input_tensor) -> dict:
    from vllm.distributed import get_ep_group, get_tp_group, get_dp_group

    group = get_ep_group()
    manager = group.device_communicator.all2all_manager
    kernel = runner._quant_method.moe_kernel
    proof = dict(
        manager=type(manager).__name__,
        prepare_finalize=type(kernel.prepare_finalize).__name__,
        experts=type(kernel.fused_experts).__name__,
        runner=type(runner).__name__,
        ep_size=group.world_size,
        tp_size=get_tp_group().world_size,
        dp_size=get_dp_group().world_size,
        input_dtype=str(input_tensor.dtype),
        use_fp8_dispatch=getattr(kernel.prepare_finalize, "use_fp8_dispatch", None),
    )
    from benchmarks.deepep_native_backend import validate_native_backend

    backend = os.environ.get("MINIEP_BENCH_ALL2ALL_BACKEND", "deepep_high_throughput")
    tp = int(os.environ.get("MINIEP_BENCH_TP_SIZE", "2"))
    validate_native_backend(proof, backend, (tp, 2, tp * 2))
    proof["all2all_backend"] = backend
    return proof


def _install_forward_proof(cls, *, native: bool, profile: bool) -> None:
    original = cls.forward
    proved = False

    def forward(self, *args, **kwargs):
        nonlocal proved
        active = profile and _STATE["active"] is not None
        if profile and _STATE["counting_warmup"]:
            _STATE["warmup_calls"] += 1
        if active:
            start, end = _event_pair()
            start.record()
        result = original(self, *args, **kwargs)
        if active:
            end.record()
            _STATE["events"].append((self.layer_name, start, end))
        if not proved:
            if native:
                proof = _native_proof(
                    self, args[0] if args else kwargs["hidden_states"]
                )
            else:
                from vllm.distributed import get_ep_group, get_tp_group, get_dp_group

                proof = dict(
                    runner=type(self).__name__,
                    ep_size=get_ep_group().world_size,
                    tp_size=get_tp_group().world_size,
                    dp_size=get_dp_group().world_size,
                    transport=("Triton peer memory (deduplicated)" if self.transport == "peer" else "torch.distributed NCCL all_to_all_single (deduplicated)"),
                )
                tp = int(os.environ.get("MINIEP_BENCH_TP_SIZE", "2"))
                if (proof["tp_size"], proof["dp_size"], proof["ep_size"]) != (tp, 2, tp * 2):
                    raise RuntimeError(
                        f"benchmark RouteEP requires TP{tp}/DP2/EP{tp * 2}: {proof}"
                    )
            if not profile:
                cls.forward = original
            proof["startup_wrapper_removed"] = not profile and cls.forward is original
            _write("backend.json", proof)
            proved = True
        return result

    cls.forward = forward


def _install_core_proof() -> None:
    import mini_ep.integration.runner as integration
    import torch.distributed as dist

    original = integration.ep_forward

    def forward(*args, **kwargs):
        calls = []
        collective = dist.all_to_all_single

        def trace(*values, **options):
            data = options.get("input", values[1] if len(values) > 1 else None)
            output = options.get("output", values[0] if values else None)
            calls.append(
                dict(
                    input_shape=list(data.shape),
                    output_shape=list(output.shape),
                    dtype=str(data.dtype),
                    output_dtype=str(output.dtype),
                    element_size=data.element_size(),
                    numel=data.numel(),
                    device=str(data.device),
                    send_splits=options.get("input_split_sizes"),
                    recv_splits=options.get("output_split_sizes"),
                )
            )
            return collective(*values, **options)

        dist.all_to_all_single = trace
        try:
            result, meta = original(*args, **kwargs)
        finally:
            dist.all_to_all_single = collective
        tensor = args[0]
        # 变长去重：计数+3 次分发+返回=5 次；固定容量字节缓冲：1 次分发+返回=2 次。
        minimum = 2 if meta.get("layout") == "unique-token-owner-fixed" else 4
        if len(calls) < minimum or calls[-1]["dtype"] != str(tensor.dtype):
            raise RuntimeError(f"default return transport was not exercised: {calls}")
        integration.ep_forward = original
        _write(
            "transport.json",
            dict(
                input_dtype=str(tensor.dtype),
                collectives=calls,
                source_file=original.__code__.co_filename,
                startup_wrapper_removed=integration.ep_forward is original,
                metadata={k: v for k, v in meta.items() if not k.startswith("_")},
            ),
        )
        return result, meta

    integration.ep_forward = forward


def _install_stage_samples(cls) -> None:
    selected = {int(x) for x in os.environ["MINIEP_BENCH_LAYERS"].split(",")}
    original = cls.forward_ep

    def forward(self, *args, **kwargs):
        match = re.search(r"\.layers\.(\d+)\.", self.layer_name)
        active = (
            _STATE["active"] is not None
            and match is not None
            and int(match[1]) in selected
        )
        if not active:
            return original(self, *args, **kwargs)
        stages = _StageSamples()
        output, meta = original(self, *args, stages=stages, **kwargs)
        for name, pairs in stages.events.items():
            _STATE["stages"].extend(
                (self.layer_name, name, start, end) for start, end in pairs
            )
        _STATE["traffic"].append(
            dict(
                layer=self.layer_name,
                **{k: v for k, v in meta.items() if not k.startswith("_")},
            )
        )
        return output, meta

    cls.forward_ep = forward


def register() -> None:
    if not os.environ.get("MINIEP_BENCH_DIR"):
        raise RuntimeError("benchmark plugin requires an isolated output directory")
    if os.environ.get("MINIEP_VALIDATION_MODE") or os.environ.get("MINIEP_REPORT_DIR"):
        raise RuntimeError(
            "accuracy probes must be disabled for performance measurement"
        )
    from mini_ep.integration.plugin import prepare_runtime, register as register_core

    prepare_runtime()
    backend = os.environ["MINIEP_BACKEND"]
    profile = os.environ.get("MINIEP_BENCH_PROFILE") == "events"
    if backend == "native":
        from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

        _install_forward_proof(MoERunner, native=True, profile=profile)
    elif backend == "miniep":
        register_core()
        from mini_ep.integration.runner import MiniEPRunner

        # auto 在 Runner 构造时才确定实际传输；两个入口都装探针，只有真正被调用的那个写 transport.json。
        requested = get_transport()
        if requested == "peer":
            _install_peer_proof()
        elif requested == "auto":
            # 不支持 Peer 的环境里 auto 会解析成 nccl；探针导入失败时只跳过 Peer 探针。
            try:
                _install_peer_proof()
            except ImportError:
                pass
        if requested in ("nccl", "auto"):
            _install_core_proof()
        _install_forward_proof(MiniEPRunner, native=False, profile=profile)
        if profile:
            _install_stage_samples(MiniEPRunner)
    else:
        raise ValueError(f"unsupported benchmark backend {backend!r}")


def _install_peer_proof() -> None:
    """只在启动时核对真实 Peer 去重路径，计时前移除包装，避免探针进入热路径。"""
    import mini_ep.distributed.peer_dedup as peer
    name = "deduplicated_peer_forward"
    expected_transport = "peer"
    original = getattr(peer, name)

    def forward(*args, **kwargs):
        result, meta = original(*args, **kwargs)
        if (meta.get("transport") != expected_transport or not meta.get("deduplicated")
            or meta.get("expert_backend") != "peer_tma_swiglu"):
            raise RuntimeError(f"peer core was not exercised: {meta}")
        # 首次调用确认后恢复原入口；后续预热和稳态计时不再写文件或做影子计算。
        setattr(peer, name, original)
        _write("transport.json", dict(
            transport="Triton peer memory (deduplicated)", input_dtype=str(args[0].dtype),
            source_file=original.__code__.co_filename,
            startup_wrapper_removed=getattr(peer, name) is original, metadata=meta,
        ))
        return result, meta

    setattr(peer, name, forward)
