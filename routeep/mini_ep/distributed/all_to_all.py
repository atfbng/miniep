"""对 torch.distributed All-to-All 的薄封装，专门交换“行分段”的张量。

所有函数都是集合通信：同一 EP 组内的 Rank 必须以相同顺序调用。split 的单位
始终是第 0 维的行数，不是字节数或元素数。
"""

import torch
import torch.distributed as dist

# 固定容量路径按 8 行对齐，便于小规模解码批次获得规整的发送分段。
FIXED_A2A_ALIGN = 8
# 固定容量路径的保守上限，避免异常配置一次分配过大的缓冲。
FIXED_A2A_MAX_ROWS = 8192
# auto 模式只为每个来源最多 8 个 Token 的短解码批次启用固定容量。
FIXED_A2A_DECODE_TOKENS = 8


def fixed_rows_per_rank(mode: str, sizes: list[int], top_k: int) -> int | None:
    """选择固定容量 All-to-All 中每个目的 Rank 的行数。

    参数：
        mode: 固定容量策略：
            - "off"：关闭，返回 None；
            - "auto"：仅短解码批次启用；
            - "on"：强制启用，容量非法时直接报错。
        sizes: Host 侧每个来源 Rank 的 Token 行数，按 EP rank 排列。
        top_k: 每个 Token 的 Top-K 分支数。

    返回：
        每个目的 Rank 的固定容量 C，已向上按 8 行对齐；不启用时返回 None。
        最坏情况下所有分支都发往同一个 Rank，因此 C 取 max(sizes)*top_k。

    本函数只读 Host 元数据，不同步 GPU，也不发起集合通信。
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
    """交换各 EP Rank 的行数，得到发送和接收 split。

    参数：
        counts: 设备端 int64 [P]，counts[r] 表示本 Rank 发往 EP Rank r 的行数。
        group: 已创建的 EP 进程组，所有成员必须同时调用。

    返回：
        (send, recv)，两个长度为 P 的 Python 列表。send[r] 是发往 Rank r 的行数，
        recv[r] 是从 Rank r 收到的行数。

    实现先用一次 All-to-All 交换计数，再把计数拷回 Host，因为
    all_to_all_single 的 split 参数需要 Python 列表。这一步有 Host 同步，
    属于同步路径，不能据此声称已实现通信/计算重叠。
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
    """按 Rank 分段执行变长 All-to-All，交换第 0 维的行。

    参数：
        x: 输入 [sum(send), ...]，第 0 维按目的 Rank 连续分段。
        send: 长度 P，send[r] 是发往 Rank r 的行数。
        recv: 长度 P，recv[r] 是从 Rank r 收到的行数。
        group: 已创建的 EP 进程组。

    返回：
        新张量 [sum(recv), ...]。输出第 0 维按来源 Rank 排列，来源 s 的段从
        sum(recv[:s]) 开始；dtype、设备和其余维度不变。

    所有 Rank 必须传入一致的 split 并同时调用。某个 split 为 0 也不能提前返回，
    否则各 Rank 的集合通信顺序会不一致。
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
    """执行等长分段的 All-to-All，不交换计数。

    参数：
        x: [P*C, ...]，每个目的 Rank 对应连续的 C 行。
        group: 已创建的 EP 进程组，组大小必须整除第 0 维。

    返回：
        与输入同形状的新张量。输出按来源 Rank 排列：来源 s 的行位于
        [s*C, (s+1)*C)。dtype、设备和其余维度不变。

    固定容量路径的填充行也会被传输，接收方必须用 expert_id=-1 识别并跳过。
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


# 去重固定容量只覆盖 decode/短 prefill；更大的批次 padding 浪费和单 program 扫描都不划算。
FIXED_OWNER_MAX_TOKENS = 256


def fixed_owner_capacity(mode: str, sizes: list[int]) -> int | None:
    """选择去重固定容量 All-to-All 的每目标行数 C。

    参数：
        mode: "auto"（默认，max(sizes)<=FIXED_OWNER_MAX_TOKENS 时启用）或 "off"。
        sizes: Host 侧各来源 EP Rank 的 Token 行数，所有 Rank 必须得到同一列表。

    返回：
        C = max(sizes)；不启用或全部为空时返回 None（走变长路径）。
        去重后同一 Token 对同一目标最多一行，所以 C 不需要乘 Top-K。
    本函数只读 Host 元数据，不同步 GPU，也不发起集合通信。
    """
    if mode not in ("auto", "off"):
        raise ValueError(f"MINIEP_OWNER_FIXED must be auto/off, got {mode!r}")
    if not sizes or any(not isinstance(n, int) or n < 0 for n in sizes):
        raise ValueError(f"sizes must be nonempty nonnegative Host integers, got {sizes}")
    maximum = max(sizes)
    if mode == "off" or maximum == 0 or maximum > FIXED_OWNER_MAX_TOKENS:
        return None
    return maximum
