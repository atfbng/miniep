"""Independent softmax/Top-K reference, with FP32 normalized routing weights."""

import torch


def router_reference(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return global expert IDs [N,K] int64 and weights [N,K] FP32 on input device.

    Input [N,E] is token-major floating point; noncontiguous tensors are allowed.
    Exact ties may choose a different valid expert from a fused router.
    """
    if (
        logits.ndim != 2
        or not logits.is_floating_point()
        or not 1 <= top_k <= logits.shape[1]
    ):
        raise ValueError(
            f"expected floating logits [N,E], 1<=K<=E; shape={logits.shape}, K={top_k}"
        )
    weights, ids = torch.softmax(logits.float(), dim=-1).topk(top_k, dim=-1)
    return ids, weights / weights.sum(dim=-1, keepdim=True)


def routing_matches_reference(
    logits: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
) -> bool:
    """Verify valid Top-K sets and FP32 normalized weights, allowing exact ties.

    Native CUDA and torch.topk need not break equal-logit ties in the same order.
    Every selected score must reach the K-th score, with K distinct valid IDs.
    """
    ref_ids, _ = router_reference(logits, ids.shape[1])
    scores = torch.softmax(logits.float(), dim=-1)
    selected = scores.gather(1, ids.long())
    cutoff = scores.gather(1, ref_ids).min(-1, keepdim=True).values
    distinct = (ids.sort(-1).values[:, 1:] != ids.sort(-1).values[:, :-1]).all()
    expected = selected / selected.sum(-1, keepdim=True)
    return bool(
        distinct
        and (selected >= cutoff).all()
        and torch.allclose(weights, expected, atol=1e-6, rtol=1e-5)
    )
