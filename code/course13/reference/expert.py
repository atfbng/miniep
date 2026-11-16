"""SwiGLU reference with explicit storage/accumulation precision."""

import torch
import torch.nn.functional as F


def swiglu_reference(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """Expert rows [M,H], gate/up [2F,H], down [H,F] -> FP32 [M,H].

    Weights and x have the same FP32/BF16 dtype/device and contiguous row-major
    layout. Gate precedes up. BF16 GEMM1, SiLU, and gate/up multiplication are rounded separately;
    GEMM2 retains FP32 accumulation so routing can precede its BF16 store.
    No routing weights are applied here. CPU is supported for reference tests.
    """
    gate_up = F.linear(x, w13)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = F.silu(gate) * up
    if x.is_cuda and x.dtype == torch.bfloat16:
        return torch.mm(activated, w2.t(), out_dtype=torch.float32)
    return F.linear(activated.float(), w2.float())


def expert_loop(
    tokens: torch.Tensor, local_ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """[R,H] received rows -> FP32 [R,H], preserving receive-buffer row order.

    w13 [L,2F,H], w2 [L,H,F]; ids int64 [R] in [0,L). Sort once, read counts
    once, then run only nonempty experts. No process-group ownership here.
    """
    compute = swiglu_reference
    order = torch.argsort(local_ids, stable=True)
    ordered = tokens.index_select(0, order)
    counts = torch.bincount(local_ids, minlength=w13.shape[0]).tolist()
    output = torch.empty(tokens.shape, dtype=torch.float32, device=tokens.device)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            result = compute(ordered[start : start + count], w13[expert], w2[expert])
            output.index_copy_(0, order[start : start + count], result)
        start += count
    return output
