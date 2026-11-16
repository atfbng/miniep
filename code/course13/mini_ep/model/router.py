"""The lesson RouterOutput contract used by the framework adapter.

Routing algorithms stay in the independent Reference or in the native vLLM
selector. This shared data object introduces no extra GPU computation.
"""

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class RouterOutput:
    """Source-local [N,K] global IDs and FP32 weights, on the input device.

    logits are [N,E] model-dtype gate outputs. Expert counts are planned on GPU
    after dispatch, rather than attached to the source-local routing result.
    """

    logits: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
