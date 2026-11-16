"""可读的 PyTorch SwiGLU 本地专家实现，可在 CPU/CUDA 上运行，不依赖 Triton。

用途：作为 Grouped/Triton 路径的语义 Reference。它按专家循环处理行，优先
保证易读和精度边界清晰，不用于性能比较。
"""
import torch
import torch.nn.functional as F


def compute_local_experts_torch(
    tokens: torch.Tensor, local_ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor,
) -> torch.Tensor:
    """对本地专家执行 SwiGLU：[R,H] 分支 -> 保持接收顺序的 FP32 [R,H]。

    参数：
        tokens: 已按来源展开的分支行 [R,H]，每一行只属于一个本地专家。
        local_ids: 每行的本地专家编号 [R]，取值 [0,L)。
        w13: 本地 gate/up 权重 [L,2F,H]。
        w2: 本地 down 权重 [L,H,F]。

    精度：Gate/up、SiLU 和乘法分别按模型 dtype 舍入；Down 在 FP32 中累加。
    本函数不应用路由权重，也不发起通信。
    """
    result = torch.empty(tokens.shape, dtype=torch.float32, device=tokens.device)
    for expert in range(w13.shape[0]):
        rows = torch.where(local_ids == expert)[0]
        if rows.numel():
            gate, up = F.linear(tokens.index_select(0, rows), w13[expert]).chunk(2, -1)
            # 不把中间步骤统统提升为 FP32，保留 BF16 模型每步存储的舍入语义。
            activated = F.silu(gate) * up
            result.index_copy_(0, rows, F.linear(activated.float(), w2[expert].float()))
    return result
