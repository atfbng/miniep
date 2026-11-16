"""执行本 Rank 专家上的 Triton SwiGLU，并保持接收缓冲的行顺序。

这里的输入已经完成 Dispatch 和第一次 All-to-All，每一行代表一个
Top-K 分支，且该分支的专家位于当前 Rank。模块只负责本地专家计算，
不选择路由，也不负责通信或进程组管理；Grouped 路径可按调用方提供的
逐分支权重，在 down 投影写出前加权。
"""

import torch


def compute_local_experts_loop(
    tokens: torch.Tensor, local_ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """作用：逐专家执行 SwiGLU：[R,H] 接收行 -> FP32 [R,H] 接收行顺序。

    参数：
        tokens: 第一次 All-to-All 收到的行，形状 ``[R,H]``；每一行是
            一个已经展开的 Top-K 分支，不是原始 Token。
        local_ids: 每行对应的本地专家编号，形状 ``[R]``，取值 ``[0,L)``。
        w13: 本地 gate/up 权重，形状 ``[L,2F,H]``。
        w2: 本地 down 权重，形状 ``[L,H,F]``。

    返回：
        与 ``tokens`` 同形状的 FP32 结果，行顺序与接收缓冲一致，便于
        第二次 All-to-All 继续使用相同的分段。

    实现只排序一次、统计一次计数，然后跳过空专家。本函数不创建也不
    销毁进程组；专家是否属于当前 Rank 由调用方通过 ``local_ids`` 保证。
    """
    from mini_ep.kernels.swiglu_triton import swiglu_triton

    # 与第十一课相同：source-major 接收行 -> Expert-major 连续段。
    # 稳定排序保证同一专家内的行仍保持接收顺序。
    # origin[p] 表示打包行 p 来自接收缓冲的哪一行。
    origin = torch.argsort(local_ids, stable=True)
    packed_tokens = tokens.index_select(0, origin)
    # -1 是固定 A2A 的 padding 标记；+1 后它落入 bucket 0，
    # 因此 counts[0] 就是 padding 行数，后续循环从 counts[1] 开始。
    # .tolist() 会把计数拷到 CPU，这是教学版 Python 循环的同步点。
    shifted = local_ids.to(torch.int64) + 1
    counts = torch.bincount(shifted, minlength=w13.shape[0] + 1).tolist()
    result = torch.empty(tokens.shape, dtype=torch.float32, device=tokens.device)
    # 打包后的布局为 [padding | expert 0 | expert 1 | ...]，start 指向当前段首行。
    start = counts[0]
    for expert, count in enumerate(counts[1:]):
        if count:
            expert_output = swiglu_triton(
                packed_tokens[start : start + count], w13[expert], w2[expert]
            )
            # 写回排序前的位置，确保返回 A2A 的分段仍然对应原来的发送者。
            result.index_copy_(0, origin[start : start + count], expert_output)
        start += count
    # 固定 A2A 的 padding 行没有有效专家，显式写 0，避免未初始化内存
    # 在返回 All-to-All 中传播。
    padding = counts[0]
    if padding:
        result.index_fill_(0, origin[:padding], 0.0)
    return result


# Same local packing / grouped projections / receive-order restoration as lesson 11.
from mini_ep.kernels.grouped_gemm.triton_impl import compute_grouped_local_experts
