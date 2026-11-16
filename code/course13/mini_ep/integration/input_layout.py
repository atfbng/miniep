"""Framework TP input partitions and output reassembly; no expert dispatch."""

import torch
import torch.distributed as dist


def token_range(tokens: int, rank: int, size: int) -> tuple[int, int]:
    """作用：计算 TP 组内某个 Rank 的源 Token 左闭右开区间。

    参数：
        tokens: 源 Token 总数 N，必须非负。
        rank: TP 组内 Rank 编号，必须满足 ``0 <= rank < size``。
        size: TP 组大小，必须为正数。

    返回：
        ``(start, end)``，且 ``start <= end``。该区间只由 TP 组内的
        切分规则决定，与 DP/EP Rank 无关。

    本函数不发起通信；调用方用它把 DP=1 时复制的 TP 输入行划分给
    唯一的来源 Rank，避免同一 Token 被重复计算。
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
    """作用：把 TP 各 Rank 互不重叠的 Token 分片拼回完整 ``[N,H]``。

    参数：
        local: 当前 TP Rank 负责的局部行 ``[n_local,H]``。
        total: 拼接后的总 Token 行数 N。
        group: TP 进程组；组内 Rank 数决定切分方式。

    返回：
        按全局 Token 顺序拼接的 ``[N,H]`` 张量。

    这里处理的是 TP 输出布局，不是专家 Dispatch；专家 Token 已经在前面的
    All-to-All 中完成交换。为了让 ``all_gather`` 接收等长缓冲，各 Rank 的
    局部行会先补零到相同容量，收集后再按 :func:`token_range` 的确定性
    区间去掉 padding。
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
