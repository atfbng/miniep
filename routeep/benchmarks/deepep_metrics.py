"""定义原生 DeepEP 与 RouteEP 共用的计时指标。"""

import math
import statistics


def request_timing(metrics, output_tokens: int) -> dict:
    """读取 vLLM 0.29 engine-core 的单调时钟，不混用墙上时钟。

    批量 Prefill 从首次调度到最后一个首 Token；Decode 从首个首 Token 到最后一个
    Token（要求同一批次生成长度相同）。端到端时间另行测量，不能用这些区间相加代替。
    """
    if metrics is None:
        raise RuntimeError("vLLM request metrics unavailable; enable log_stats")
    values = {
        name: float(getattr(metrics, name))
        for name in ("scheduled_ts", "first_token_ts", "last_token_ts")
    }
    if (
        not all(math.isfinite(x) for x in values.values())
        or not 0
        < values["scheduled_ts"]
        <= values["first_token_ts"]
        <= values["last_token_ts"]
    ):
        raise RuntimeError(f"invalid engine-core timestamps: {values}")
    if getattr(metrics, "is_corrupted", False):
        raise RuntimeError("vLLM marked request logits corrupted")
    if output_tokens < 1 or int(metrics.num_generation_tokens) != output_tokens:
        raise RuntimeError("generated-token metrics differ from actual output")
    return dict(**values, output_tokens=output_tokens)


def batch_timing(requests: list[dict]) -> dict:
    if not requests:
        raise ValueError("a measured DP batch must have requests")
    lengths = {q["output_tokens"] for q in requests}
    if len(lengths) != 1:
        raise ValueError("decode timing requires equal output lengths within the batch")
    steps = next(iter(lengths))
    first = min(q["first_token_ts"] for q in requests)
    prefill = max(q["first_token_ts"] for q in requests) - min(
        q["scheduled_ts"] for q in requests
    )
    decode = max(q["last_token_ts"] for q in requests) - first if steps > 1 else None
    return dict(
        prefill_batch_ms=prefill * 1000,
        decode_batch_ms=decode * 1000 if decode is not None else None,
        decode_step_ms=decode * 1000 / (steps - 1) if decode is not None else None,
    )


def distribution(samples: list[float]) -> dict:
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(samples)

    def percentile(p):
        position = p * (len(ordered) - 1)
        lo = int(position)
        fraction = position - lo
        return (
            ordered[lo] * (1 - fraction)
            + ordered[min(lo + 1, len(ordered) - 1)] * fraction
        )

    return dict(
        median=statistics.median(samples),
        minimum=min(samples),
        maximum=max(samples),
        p10=percentile(0.1),
        p90=percentile(0.9),
        samples=len(samples),
    )
