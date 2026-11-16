"""处理 vLLM 的 TP 输入切分和输出拼回，不涉及专家 Dispatch。

DP=1 时多个 TP Rank 收到相同的输入行，需要确定“哪一行由哪个来源 Rank 负责”；
DP=2 时每行已经属于当前来源，只需按区间取出。专家 Token 的交换由后面的
All-to-All 负责，本文件不改变专家归属。
"""

import torch
import torch.distributed as dist


def token_range(tokens: int, rank: int, size: int) -> tuple[int, int]:
    """计算 TP 组内某个 Rank 的源 Token 半开区间 [start,end)。

    参数：
        tokens: 源 Token 总数 N，必须非负。
        rank: TP 组内编号，满足 0 <= rank < size。
        size: TP 组大小，必须为正。

    返回：
        (start, end)，且 start <= end。切分只由 TP 规则决定，与 DP/EP Rank 无关。

    本函数不发起通信。DP=1 时调用方用它把复制的 TP 输入行分配给唯一来源 Rank，
    避免同一 Token 被重复计算。
    """
    if tokens < 0 or size < 1 or not 0 <= rank < size:
        raise ValueError(
            f"expected N>=0, size>0, 0<=rank<size; got {tokens}, {rank}, {size}"
        )
    # 左闭右开区间；即使 N 不能整除 size，也保证每行恰好有一个发送者。
    return tokens * rank // size, tokens * (rank + 1) // size


def gather_tp_rows(
    local: torch.Tensor, total: int, group: dist.ProcessGroup
) -> torch.Tensor:
    """把 TP 各 Rank 互不重叠的 Token 分片拼回完整 [N,H]。

    参数：
        local: 当前 TP Rank 负责的局部行 [n_local,H]。
        total: 拼接后的总行数 N。
        group: TP 进程组，组内 Rank 数决定切分方式。

    返回：
        按全局 Token 顺序拼接的 [N,H]。

    实现细节：all_gather 需要等长输入，所以各 Rank 先补零到相同容量；收集后
    再按 token_range 的确定性区间去掉 padding。这里处理的是 TP 输出布局，
    不是专家 Dispatch。
    """
    size = dist.get_world_size(group)
    if not total:
        return local.new_empty((0, local.shape[1]))
    # TP 边界只拼回互不重叠的 Token 行；专家分发已经由前面的 A2A 完成。
    # 不等长分片先补到相同长度，收集后再按 token_range 的规则去掉 padding。
    capacity = (total + size - 1) // size
    padded = local.new_zeros((capacity, local.shape[1]))
    # 局部行始终是完整结果的前 n_local 行，其余位置补零。
    padded[: local.shape[0]].copy_(local)
    gathered = [torch.empty_like(padded) for _ in range(size)]
    dist.all_gather(gathered, padded, group=group)
    # 每个来源 Rank r 的有效长度由同一个半开区间公式确定，无需额外元数据。
    return torch.cat(
        [
            x[: total * (r + 1) // size - total * r // size]
            for r, x in enumerate(gathered)
        ],
        dim=0,
    )
