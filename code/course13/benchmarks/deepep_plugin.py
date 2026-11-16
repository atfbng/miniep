"""Benchmark-only startup proofs and optional event sampling, outside mini_ep.

Normal measurements restore every startup wrapper before warmup ends. Optional
profiling retains events until the driver has synchronized at the case boundary;
no tensor cloning, shadow forward, per-layer file IO or hot-path synchronize.
"""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import torch

_STATE = dict(active=None, events=[], stages=[], traffic=[], pool=[], pool_index=0)


def begin_case(
    name: str, max_model_steps: int, model_layers: int, selected_layers: int
) -> None:
    """Initialize event handles before the measured interval, retaining ownership."""
    if _STATE["events"] or _STATE["stages"]:
        raise RuntimeError("drain the previous profiling case before reuse")
    count = 2 * max_model_steps * (model_layers + 8 * selected_layers)
    pool = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
    # Event construction is lazy; record once here to keep allocation out of ROI.
    for event in pool:
        event.record()
    torch.cuda.synchronize()
    _STATE.update(active=name, pool=pool, pool_index=0)


def _event_pair():
    position = _STATE["pool_index"]
    if position + 2 > len(_STATE["pool"]):
        raise RuntimeError("profiling event bound exhausted; no timing result accepted")
    _STATE["pool_index"] = position + 2
    return _STATE["pool"][position : position + 2]


class _StageSamples:
    """Same stage protocol as core Stages, with preallocated timing events."""

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
    """Caller synchronized once at measurement boundary; events are now ready."""
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


def _native_proof(runner) -> dict:
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
    )
    if (
        proof["manager"] != "DeepEPHTAll2AllManager"
        or proof["prepare_finalize"] != "DeepEPHTPrepareAndFinalize"
        or proof["experts"] != "TritonExperts"
        or proof["runner"] != "MoERunner"
        or (proof["tp_size"], proof["dp_size"], proof["ep_size"]) != (2, 2, 4)
    ):
        raise RuntimeError(
            f"benchmark native backend fallback or wrong topology: {proof}"
        )
    return proof


def _install_forward_proof(cls, *, native: bool, profile: bool) -> None:
    original = cls.forward
    proved = False

    def forward(self, *args, **kwargs):
        nonlocal proved
        active = profile and _STATE["active"] is not None
        if active:
            start, end = _event_pair()
            start.record()
        result = original(self, *args, **kwargs)
        if active:
            end.record()
            _STATE["events"].append((self.layer_name, start, end))
        if not proved:
            if native:
                proof = _native_proof(self)
            else:
                from vllm.distributed import get_ep_group, get_tp_group, get_dp_group

                proof = dict(
                    runner=type(self).__name__,
                    ep_size=get_ep_group().world_size,
                    tp_size=get_tp_group().world_size,
                    dp_size=get_dp_group().world_size,
                    transport="torch.distributed NCCL all_to_all_single",
                )
                if (proof["tp_size"], proof["dp_size"], proof["ep_size"]) != (2, 2, 4):
                    raise RuntimeError(
                        f"benchmark Mini-EP requires TP2/DP2/EP4: {proof}"
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
        if len(calls) < 4 or calls[-1]["dtype"] != str(tensor.dtype):
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

        _install_core_proof()
        _install_forward_proof(MiniEPRunner, native=False, profile=profile)
        if profile:
            _install_stage_samples(MiniEPRunner)
    else:
        raise ValueError(f"unsupported benchmark backend {backend!r}")
