"""验收用的精确、有限性 Tensor 比较，不提供容差回退。

形状、dtype、device、每个元素和 NaN/Inf 都必须完全符合预期；失败时记录
首个差异位置，方便定位。
"""

import math
import torch


def exact_tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """比较 shape/dtype/device 完全一致的 Tensor；NaN/Inf 一律失败。

    有符号零与 torch.equal 一样视为相等。返回首个差异坐标和值，且保证可 JSON
    序列化；输入可以为空，也可以非连续。
    """
    contract = (
        actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and actual.device == expected.device
    )
    if not contract:
        return dict(
            exact_equal=False,
            finite=False,
            contract_match=False,
            actual_shape=list(actual.shape),
            expected_shape=list(expected.shape),
            actual_dtype=str(actual.dtype),
            expected_dtype=str(expected.dtype),
            actual_device=str(actual.device),
            expected_device=str(expected.device),
            max_abs=None,
            mismatch_elements=None,
            first_difference=None,
        )
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    different = (
        (actual != expected) | ~torch.isfinite(actual) | ~torch.isfinite(expected)
    )
    mismatches = int(different.sum().item())
    first = None
    if mismatches:
        index = tuple(int(v) for v in different.nonzero()[0].tolist())

        def value(x):
            number = x[index].item()
            return (
                number
                if not isinstance(number, float) or math.isfinite(number)
                else str(number)
            )

        first = dict(index=list(index), actual=value(actual), expected=value(expected))
    return dict(
        exact_equal=finite and mismatches == 0,
        finite=finite,
        contract_match=True,
        mismatch_elements=mismatches,
        max_abs=(
            float((actual.float() - expected.float()).abs().max().item())
            if actual.numel() and finite
            else (0.0 if finite else None)
        ),
        first_difference=first,
    )


def exact_logits_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """在 teacher forcing 之前比较完整 [positions,vocabulary] logits。"""
    result = exact_tensor_metrics(actual, expected)
    result.update(
        rows=actual.shape[0],
        argmax_matches=(
            int((actual.argmax(-1) == expected.argmax(-1)).sum().item())
            if result["contract_match"] and result["finite"] and actual.shape[0]
            else 0
        ),
    )
    result["passed"] = result["exact_equal"]
    return result


def exact_tensor_flag(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """返回设备端 bool 标量：要求 shape/dtype/device/值完全相等且有限。

    该函数不做 Host 读取、不使用容差，用于协调分布式失败。昂贵的坐标和数值统计
    只在失败后收集。
    """
    if (
        actual.shape != expected.shape
        or actual.dtype != expected.dtype
        or actual.device != expected.device
    ):
        return torch.zeros((), device=actual.device, dtype=torch.bool)
    same = actual == expected
    if actual.is_floating_point() or actual.is_complex():
        same = same & torch.isfinite(actual) & torch.isfinite(expected)
    return same.all()
