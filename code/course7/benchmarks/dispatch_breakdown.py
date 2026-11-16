"""Nested Dispatch timings and a CPU-metadata control; no changes to payload A2A."""
from __future__ import annotations
from contextlib import contextmanager
import time
import torch
import torch.distributed as dist
from mini_ep.distributed import build_receive_layout
from mini_ep.profiler.stages import StageRecorder, run_ep_stages

PARTS = ("EXPERT_PACK", "RANK_PACK", "COUNT_ALLGATHER", "RECEIVE_LAYOUT")


class DispatchRecorder(StageRecorder):
    """CUDA intervals include Host submission gaps; Host times include GPU waits."""
    def __init__(self, device):
        super().__init__(device, timing=True)
        self.host_samples_ms = {name: [] for name in PARTS}

    @contextmanager
    def stage(self, name):
        with torch.profiler.record_function("DISPATCH_DETAIL/" + name):
            start = time.perf_counter()
            with super().stage(name):
                yield
            self.host_samples_ms[name].append((time.perf_counter() - start) * 1000)


def cpu_metadata_receive_layout(topology, tokens_per_expert_all, *, local_tokens_per_expert=None):
    """Control: copy small [P,E] counts to CPU and reuse the original layout math.

    Each row s contains source s's Expert counts (not Token vectors). Returned
    splits, offsets, [P,L] local counts and each expert's receive-row indices
    remain contiguous int64 tensors on the original CUDA device. All validation
    and source-major/Expert-major ordering are preserved. This deliberately
    simple benchmark control is separate from the course's GPU Reference.
    """
    device = tokens_per_expert_all.device
    local = None if local_tokens_per_expert is None else local_tokens_per_expert.cpu()
    layout = build_receive_layout(topology, tokens_per_expert_all.cpu(), local_tokens_per_expert=local)
    return type(layout)(
        recv_splits=layout.recv_splits.to(device),
        recv_offsets=layout.recv_offsets.to(device),
        per_source_local_expert_counts=layout.per_source_local_expert_counts.to(device),
        local_expert_index=tuple(rows.to(device) for rows in layout.local_expert_index),
    )


def measure_dispatch(model, tokens, group, warmup, repeats, *, detailed=False, host_metadata=False):
    """Run the complete synchronous forward; all reductions occur after samples."""
    kwargs = {"receive_layout_impl": cpu_metadata_receive_layout} if host_metadata else {}
    for _ in range(warmup):
        run_ep_stages(model, tokens, StageRecorder(tokens.device), group=group, **kwargs)
    torch.cuda.synchronize(tokens.device)
    dist.barrier(group=group)
    outer = StageRecorder(tokens.device, timing=True)
    inner = DispatchRecorder(tokens.device) if detailed else None
    for _ in range(repeats):
        result = run_ep_stages(model, tokens, outer, group=group, dispatch_recorder=inner, **kwargs)
    torch.cuda.synchronize(tokens.device)
    values = outer.samples_ms()
    output = {"dispatch_ms": values["DISPATCH"]}
    if inner is not None:
        output["parts_ms"] = inner.samples_ms()
        output["host_parts_ms"] = inner.host_samples_ms
    return result, output


def profile_scalar_reads(model, tokens, group, *, steps=3):
    """Attribute CPU aten::item / aten::equal calls to the four Dispatch scopes.

    Timings here describe a separate instrumented profiling run, not benchmark
    samples. cpu_time_total for aten::item includes its nested scalar read;
    it must not be added to _local_scalar_dense time again.
    """
    detail = DispatchRecorder(tokens.device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(steps):
            run_ep_stages(model, tokens, StageRecorder(tokens.device), group=group, dispatch_recorder=detail)
        torch.cuda.synchronize(tokens.device)
    result = {name: {"aten_item_calls_per_forward": 0, "aten_item_cpu_us_per_forward": 0.0, "aten_equal_calls_per_forward": 0, "aten_equal_cpu_us_per_forward": 0.0} for name in PARTS}
    for event in profiler.events():
        if event.name not in ("aten::item", "aten::equal"):
            continue
        parent = event.cpu_parent
        while parent is not None and not parent.name.startswith("DISPATCH_DETAIL/"):
            parent = parent.cpu_parent
        if parent is None:
            continue
        part = parent.name.split("/", 1)[1]
        prefix = "aten_item" if event.name == "aten::item" else "aten_equal"
        result[part][prefix + "_calls_per_forward"] += 1 / steps
        result[part][prefix + "_cpu_us_per_forward"] += event.cpu_time_total / steps
    return result


def run_dispatch_breakdown(model, tokens, group, expected, warmup, repeats):
    """Save local raw samples for original, nested, and CPU-metadata paths."""
    outputs = {}
    for label, detailed, host_metadata in [("control", False, False), ("detailed", True, False), ("cpu_metadata", False, True)]:
        actual, values = measure_dispatch(model, tokens, group, warmup, repeats, detailed=detailed, host_metadata=host_metadata)
        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=0)
        for field in ("expert_ids", "routing_weights", "tokens_per_expert_all", "send_splits", "recv_splits", "local_expert_rows"):
            torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
        outputs[label] = values
    outputs["scalar_reads"] = profile_scalar_reads(model, tokens, group)
    outputs["correctness"] = "PASS"
    return outputs
