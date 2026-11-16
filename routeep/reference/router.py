"""独立 softmax/Top-K Reference，路由权重为 FP32 归一化结果。"""

import torch


def router_reference(
    logits: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回输入设备上的全局专家编号 [N,K] int64 和权重 [N,K] FP32。

    输入 [N,E] 是 token-major 浮点张量，允许非连续。完全相同的 logit 可能让
    融合 Router 选择另一个同样合法的专家，因此测试不应假设 tie 的专家编号相同。
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
    """校验 Top-K 集合和 FP32 归一化权重，允许完全相同的 logit tie。

    原生 CUDA 和 torch.topk 对相同 logit 的打破顺序可能不同。每个被选中的分数
    必须达到第 K 名分数，且 K 个专家编号互不重复、都在合法范围内。
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
