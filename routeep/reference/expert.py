"""SwiGLU Reference，明确写出存储和累加的精度边界。"""

import torch
import torch.nn.functional as F


def swiglu_reference(
    x: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """单个专家：[M,H] 行、gate/up [2F,H]、down [H,F] -> FP32 [M,H]。

    输入和权重使用相同的 FP32/BF16 dtype/device，连续行主序。前 F 行是 gate，
    后 F 行是 up。GEMM1、SiLU、gate/up 乘法分别按模型 dtype 舍入；GEMM2 保留
    FP32 累加，让路由权重可以在其 BF16 存储前应用。本函数不乘路由权重。

    CPU 可运行，用于 Reference 测试。
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
    """[R,H] 接收行 -> FP32 [R,H]，保持接收缓冲的行顺序。

    w13 [L,2F,H]，w2 [L,H,F]；ids 是 int32/int64 [R]，取值 [0,L)。只排序一次、
    读一次计数，然后只计算非空专家。本函数不负责进程组或专家归属。
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
