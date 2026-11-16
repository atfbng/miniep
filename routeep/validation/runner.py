"""每次 MoE 调用的验证元数据；Runner 适配层不包含数值策略。"""

import torch.distributed as dist
from validation.plugin import read_control, append_record


def record_call(runner, record: dict) -> None:
    """把诊断信息留在推理路径之外，并记录真实的 shared/DP 模式。"""
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
