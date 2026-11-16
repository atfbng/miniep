"""Assert the executed native backend, including independent native-only runs."""

import json
import os
from functools import wraps
from pathlib import Path


_PENDING_NATIVE = []


def flush_native_observer() -> None:
    """Drain pinned copies after a model step; no per-layer host GPU wait.

    Each copy is queued on its producer stream before workspace reuse, with an
    event retaining the host buffer until ready. Only CPU values are hashed.
    """
    import hashlib
    import torch
    from validation.plugin import append_record

    pending = list(_PENDING_NATIVE)
    _PENDING_NATIVE.clear()
    for host, event, record in pending:
        event.synchronize()
        if not torch.isfinite(host).all().item():
            root = Path(os.environ["MINIEP_REPORT_DIR"]) / "first-failure"
            root.mkdir(exist_ok=True)
            (root / "summary.json").write_text(
                json.dumps(dict(kind="nonfinite-native-moe", **record))
            )
            torch.save(host, root / "native-output.pt")
            raise RuntimeError(f"nonfinite independent native MoE output: {record}")
        record["output_sha256"] = hashlib.sha256(
            host.view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        append_record("native-moe.jsonl", record)


def verify_backend(runner) -> dict:
    from vllm.distributed import get_ep_group

    group = get_ep_group()
    manager = group.device_communicator.all2all_manager
    kernel = runner._quant_method.moe_kernel
    proof = dict(
        manager=type(manager).__name__,
        prepare_finalize=type(kernel.prepare_finalize).__name__,
        experts=type(kernel.fused_experts).__name__,
        runner=type(runner).__name__,
        ep_size=group.world_size,
    )
    backend = os.environ["MINIEP_EXACT_BACKEND"]
    if backend != "deepep_high_throughput":
        raise RuntimeError(f"unsupported reference backend {backend}")
    expected = ("DeepEPHTAll2AllManager", "DeepEPHTPrepareAndFinalize")
    proof["backend"] = backend
    if (
        (proof["manager"], proof["prepare_finalize"]) != expected
        or proof["experts"] != "TritonExperts"
        or proof["ep_size"] != int(os.environ["WORLD_SIZE"])
    ):
        raise RuntimeError(
            f"actual native backend differs from frozen reference: {proof}"
        )
    root = Path(os.environ["MINIEP_REPORT_DIR"])
    path = root / "backend.json"
    if not path.exists():
        path.write_text(json.dumps(proof, indent=2))
    return proof


def install_native_observer() -> None:
    """Observe the original MoERunner without installing a replacement Runner."""
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
    from validation.plugin import read_control, append_record

    original = MoERunner.forward

    @wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        result = original(self, hidden_states, *args, **kwargs)
        if type(self) is not MoERunner:
            raise RuntimeError(
                f"independent native baseline replaced Runner: {type(self)}"
            )
        verify_backend(self)
        control = read_control()
        if control:
            import torch

            # Source and copy share the producer stream. Subsequent workspace
            # reuse is ordered after this copy; the event owns host readiness.
            host = torch.empty_like(
                result,
                device="cpu",
                pin_memory=True,
                memory_format=torch.contiguous_format,
            )
            host.copy_(result.detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(result.device))
            _PENDING_NATIVE.append(
                (
                    host,
                    event,
                    dict(
                        key=control["key"],
                        layer=self.layer_name,
                        framework_rows=hidden_states.shape[0],
                        logical_requests=control["logical_requests"],
                    ),
                )
            )
        return result

    MoERunner.forward = forward
