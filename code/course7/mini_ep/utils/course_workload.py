"""Fixed global batches and controlled routes for lessons 7–8."""

from __future__ import annotations
import torch
from mini_ep.model.moe import SingleGPUMoE


def make_workload(
    tokens: int,
    hidden_size: int,
    expert_hidden_size: int,
    num_experts: int,
    top_k: int,
    scenario: str,
    seed: int = 0,
) -> tuple[SingleGPUMoE, torch.Tensor]:
    """Return a CPU FP32 model and original-order [T,H] batch.

    The first E input features encode gate scores. The gate selects these
    features with an identity projection, so every EP size sees the same
    expert IDs and weights. Top-K IDs are distinct within each Token.
    """
    if tokens <= 0 or hidden_size < num_experts:
        raise ValueError(
            f"need tokens > 0 and H >= E, got T={tokens}, H={hidden_size}, E={num_experts}"
        )
    if scenario not in ("uniform", "hotspot"):
        raise ValueError(f"unknown scenario {scenario!r}")
    torch.manual_seed(seed)
    model = SingleGPUMoE(hidden_size, expert_hidden_size, num_experts, top_k)
    generator = torch.Generator().manual_seed(seed + 1)
    x = torch.randn(tokens, hidden_size, generator=generator)
    scores = torch.full((tokens, num_experts), -10.0)
    for t in range(tokens):
        if scenario == "uniform":
            chosen = [(t * top_k + j) % num_experts for j in range(top_k)]
        elif top_k == 1:
            chosen = (
                [0 if t < int(tokens * 0.6) else 1 + t % (num_experts - 1)]
                if num_experts > 1
                else [0]
            )
        else:
            chosen = [0] + [1 + (t + j) % (num_experts - 1) for j in range(top_k - 1)]
        for slot, expert in enumerate(chosen):
            scores[t, expert] = 10.0 - slot
    x[:, :num_experts] = scores
    with torch.no_grad():
        model.router.gate.weight.zero_()
        model.router.gate.weight[:, :num_experts].copy_(torch.eye(num_experts))
    return model, x


def load_metrics(counts: torch.Tensor, ep_size: int) -> dict[str, object]:
    """CPU counts [E] or [P,E] -> global Expert and Rank row statistics."""
    if counts.ndim not in (1, 2) or counts.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"need integer counts [E] or [P,E], got {counts.shape}/{counts.dtype}"
        )
    if ep_size <= 0 or counts.shape[-1] % ep_size:
        raise ValueError(f"E={counts.shape[-1]} must be divisible by P={ep_size}")
    global_counts = counts.sum(0) if counts.ndim == 2 else counts
    rank_counts = global_counts.reshape(ep_size, -1).sum(1)

    def ratio(v):
        mean = v.float().mean().item()
        return v.max().item() / mean if mean else 0.0

    return {
        "tokens_per_expert": global_counts.tolist(),
        "rows_per_rank": rank_counts.tolist(),
        "expert_imbalance": ratio(global_counts),
        "rank_imbalance": ratio(rank_counts),
    }
