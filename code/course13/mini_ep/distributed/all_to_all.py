"""同步 NCCL 集合通信封装；所有 EP Rank 必须以完全相同的顺序调用。"""

import torch
import torch.distributed as dist

# 固定容量路径按 8 行对齐，便于小规模解码批次获得规整的发送分段。
FIXED_A2A_ALIGN = 8
# 固定容量路径的保守上限，避免异常配置一次分配过大的缓冲。
FIXED_A2A_MAX_ROWS = 8192
# auto 模式只为每个来源最多 8 个 Token 的短解码批次启用固定容量。
FIXED_A2A_DECODE_TOKENS = 8


def fixed_rows_per_rank(mode: str, sizes: list[int], top_k: int) -> int | None:
    """作用：选择固定容量 All-to-All 中每个目的 Rank 的行数。

    参数：
        mode: 固定容量策略：

            - ``"off"``：始终返回 ``None``，使用变长 A2A；
            - ``"auto"``：仅对短解码批次启用固定容量；
            - ``"on"``：强制启用，容量非法时直接报错而不静默回退。
        sizes: Host 侧各来源 EP Rank 的 Token 行数，按 EP rank 顺序排列；
            必须是非空、非负的 Python 整数列表。
        top_k: 每个 Token 的 Top-K 分支数，必须为正。

    返回：
        每个目的 Rank 的固定容量 C，已向上按 8 行对齐；不启用固定路径
        时返回 ``None``。每个来源 Rank 最多发送 ``max(sizes)*top_k`` 行，
        因此 C 取该值即可覆盖所有目的 Rank。

    该函数只读取 Host 侧元数据，不进行 GPU 同步或集合通信。
    """
    # 策略名称必须在进入分布式路径前校验，避免各 Rank 行为不一致。
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"fixed A2A mode must be auto/on/off, got {mode!r}")
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if not sizes or any(not isinstance(n, int) or n < 0 for n in sizes):
        raise ValueError(
            f"sizes must be nonempty nonnegative Host integers, got {sizes}"
        )
    maximum = max(sizes)
    # 没有 Token 或明确关闭时，不分配固定容量缓冲。
    if maximum == 0 or mode == "off":
        return None
    # auto 只覆盖短解码批次；预填充或较大批次继续使用变长路径。
    if mode == "auto" and maximum > FIXED_A2A_DECODE_TOKENS:
        return None
    # 最坏情况下，每个来源 Token 的 K 个分支都发往同一个目的 Rank。
    rows = maximum * top_k
    # 向上对齐到 FIXED_A2A_ALIGN，保持发送块大小规整。
    rows = (rows + FIXED_A2A_ALIGN - 1) // FIXED_A2A_ALIGN * FIXED_A2A_ALIGN
    if rows > FIXED_A2A_MAX_ROWS:
        # 强制模式下容量越界应显式失败，不能悄悄退回另一条通信布局。
        if mode == "on":
            raise ValueError(f"fixed A2A capacity={rows} exceeds {FIXED_A2A_MAX_ROWS}")
        return None
    return rows


def exchange_splits(
    counts: torch.Tensor, group: dist.ProcessGroup
) -> tuple[list[int], list[int]]:
    """作用：交换各 EP Rank 的行数，返回发送和接收的 split 列表。

    参数：
        counts: int64 设备 Tensor ``[P]``；``counts[r]`` 表示本 Rank 发往
            EP Rank r 的行数。
        group: 已创建的 EP 进程组，所有成员必须同时调用本函数。

    返回：
        ``(send, recv)`` 两个 Python 整数列表，长度均为 P：
        ``send[r]`` 是发往 Rank r 的行数，``recv[r]`` 是从 Rank r 收到的
        行数。单位都是 dim-0 行数，不是字节或元素数。

    每个 Rank 先用一次 All-to-All 交换计数，再把计数拷回 CPU。Host 拷贝
    是 ``all_to_all_single`` 的 split-list API 所必需的；因此这里明确是
    同步路径，不声称已实现通信/计算重叠。
    """
    # counts[r] 是发给 rank r 的行数；交换后 received[r] 是从 rank r 接收的行数。
    received = torch.empty_like(counts)
    dist.all_to_all_single(received, counts, group=group)
    # 两个方向合并成一次 Host 拷贝；后续 A2A 仍使用同一组 Python split。
    send, recv = torch.stack((counts, received), dim=0).tolist()
    return send, recv


def variable_split_all_to_all(
    x: torch.Tensor, send: list[int], recv: list[int], group: dist.ProcessGroup
) -> torch.Tensor:
    """作用：按 Rank 分段执行变长 All-to-All，交换 dim-0 行。

    参数：
        x: 输入张量，形状 ``[sum(send), ...]``；第 0 维按目的 Rank 连续
            分段，每段大小为 ``send[r]``。非第 0 维布局保持不变。
        send: 长度 P 的发送行数列表，``send[r]`` 对应发往 Rank r 的行数。
        recv: 长度 P 的接收行数列表，``recv[r]`` 对应从 Rank r 收到的行数。
        group: 已创建的 EP 进程组。

    返回：
        ``[sum(recv), ...]`` 的新张量。第 0 维按来源 Rank 排列，来源 s
        的段位于 ``sum(recv[:s])`` 之后。dtype、设备和尾部维度保持不变。

    所有 Rank 必须传入一致的 split 列表并同时进入集合通信。某个 split
    为 0 时仍必须参与调用，不能用提前返回替代。
    """
    # 输入按目的 rank 分段，输出按来源 rank 分段；split 单位是行，不是字节。
    out = x.new_empty((sum(recv), *x.shape[1:]))
    dist.all_to_all_single(
        out,
        x.contiguous(),
        output_split_sizes=recv,
        input_split_sizes=send,
        group=group,
    )
    return out


def equal_split_all_to_all(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """作用：执行等长分段的 All-to-All，不读取或交换行数。

    参数：
        x: 输入张量 ``[P*C, ...]``；每个目的 Rank 对应连续的 C 行。
        group: 已创建的 EP 进程组，组大小必须能整除 ``x.shape[0]``。

    返回：
        与输入同形状的新张量。输出按来源 Rank 排列：来源 s 的行位于
        ``[s*C, (s+1)*C)``。dtype、设备和尾部维度保持不变。

    每个 Rank 向每个目的 Rank 正好发送 C 行。填充行也是传输载荷的一部分，
    接收后必须由专家元数据（``expert_id=-1``）屏蔽或跳过。由于所有 Rank
    使用相同长度，本路径无需计数交换和 Host 回读。
    """
    size = dist.get_world_size(group)
    if size < 1 or x.shape[0] % size:
        raise ValueError(
            f"fixed A2A needs a positive EP size dividing rows; "
            f"rows={x.shape[0]}, EP size={size}"
        )
    out = x.new_empty((x.shape[0], *x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), group=group)
    return out
