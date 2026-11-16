"""Expert MLP used by the single-GPU MoE reference implementation."""

from __future__ import annotations

import torch
from torch import nn


class ExpertMLP(nn.Module):
    """A small feed-forward network owned by one expert.

    Input and output both use ``[num_tokens, hidden_size]`` layout. The module
    does not know anything about routing: it only processes the tokens already
    assigned to this expert.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        *,
        activation: str = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if expert_hidden_size <= 0:
            raise ValueError(
                "expert_hidden_size must be positive, "
                f"got {expert_hidden_size}"
            )

        activations: dict[str, nn.Module] = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
        }
        if activation not in activations:
            supported = ", ".join(sorted(activations))
            raise ValueError(
                f"unsupported activation {activation!r}; choose one of: {supported}"
            )

        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        # Keep the constructor arguments that shape this MLP.  Distributing a
        # single-GPU model to several ranks rebuilds each local expert from
        # them, so the EP shard cannot silently pick a different activation or
        # bias setting than the reference it is compared against.
        self.activation_name = activation
        self.use_bias = bias
        self.up_proj = nn.Linear(hidden_size, expert_hidden_size, bias=bias)
        self.activation = activations[activation]
        self.down_proj = nn.Linear(expert_hidden_size, hidden_size, bias=bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Transform expert-local tokens without changing their order."""
        if tokens.ndim != 2 or tokens.shape[1] != self.hidden_size:
            raise ValueError(
                "tokens must have shape [num_tokens, hidden_size] with "
                f"hidden_size={self.hidden_size}, got {tuple(tokens.shape)}"
            )
        return self.down_proj(self.activation(self.up_proj(tokens)))
