"""Per-call validation metadata; no numerical policies in the Runner adapter."""

import torch.distributed as dist
from validation.plugin import read_control, append_record


def record_call(runner, record: dict) -> None:
    """Keep diagnostics outside inference; report the actual shared/DP mode."""
    runner._validation_calls = getattr(runner, "_validation_calls", 0) + 1
    record.update(
        layer=runner.layer_name, rank=dist.get_rank(), call=runner._validation_calls
    )
    control = read_control()
    if control:
        record["key"] = control["key"]
        events = record.pop("_stage_events", None)
        if events:
            raise ValueError(
                "accuracy stage timing is disabled; use benchmarks/deepep_latency.py --profile-stages"
            )
        append_record(f"moe-rank{dist.get_rank()}.jsonl", record)
