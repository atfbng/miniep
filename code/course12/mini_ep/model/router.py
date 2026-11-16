"""Top-K token router for the single-GPU MoE lesson."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class RouterOutput:
    """Routing metadata in original token order.

    Attributes:
        logits: Unnormalized gate scores, shape ``[num_tokens, num_experts]``.
        expert_ids: Selected global expert IDs, shape ``[num_tokens, top_k]``.
        routing_weights: FP32 selected probabilities renormalized to sum to one
            for each token, shape ``[num_tokens, top_k]``.
        tokens_per_expert: Assignment counts, shape ``[num_experts]``. For
            Top-2 routing its sum is twice ``num_tokens``.
    """

    logits: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    tokens_per_expert: torch.Tensor


class TopKRouter(nn.Module):
    """Route every token to ``top_k`` experts using a learned linear gate."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if not 1 <= top_k <= num_experts:
            raise ValueError(
                f"top_k must be in [1, num_experts={num_experts}], got {top_k}"
            )

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, tokens: torch.Tensor) -> RouterOutput:
        """Compute routing decisions for ``[num_tokens, hidden_size]`` tokens."""
        if tokens.ndim != 2 or tokens.shape[1] != self.hidden_size:
            raise ValueError(
                "tokens must have shape [num_tokens, hidden_size] with "
                f"hidden_size={self.hidden_size}, got {tuple(tokens.shape)}"
            )

        logits = self.gate(tokens)

        # Keep routing probabilities in FP32 for stable Top-K normalization.
        # Expert compute can still use FP16/BF16; Combine casts each weight to
        # the corresponding expert output dtype before multiplication.
        probabilities = F.softmax(logits.float(), dim=-1)
        routing_weights, expert_ids = torch.topk(
            probabilities, k=self.top_k, dim=-1
        )
        routing_weights = routing_weights / routing_weights.sum(
            dim=-1, keepdim=True
        )
        tokens_per_expert = torch.bincount(
            expert_ids.reshape(-1), minlength=self.num_experts
        )

        return RouterOutput(
            logits=logits,
            expert_ids=expert_ids,
            routing_weights=routing_weights,
            tokens_per_expert=tokens_per_expert,
        )
