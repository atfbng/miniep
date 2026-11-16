"""Exact, finite tensor comparisons for acceptance (no tolerance fallback)."""

import math
import torch


def exact_tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    """Compare equal-shape/dtype/device tensors; NaN/Inf always fail.

    Signed zero compares equal as in torch.equal. First differing coordinates
    and values are JSON-safe; inputs may be empty and need not be contiguous.
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
    """Compare complete [positions,vocabulary] logits before teacher forcing."""
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
    """Device bool scalar: exact shape/dtype/device/value and finite contract.

    No host read or tolerance. Used to decide a coordinated distributed failure;
    expensive coordinate/value statistics are collected only after a failure.
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
