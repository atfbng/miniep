"""与具体通信后端无关的“Token/目标卡去重”布局和有序归约。

解决的问题：一个 Token 的多个 Top-K 分支如果落在同一个目标 Rank，逐分支发送
会重复传输同一份 [H] 向量。本文件让每个 (Token, 目标 Rank) 只发送一份输入，
Expert 分支在接收卡内部再展开。

本文件只使用 PyTorch，可在 CPU/CUDA 上运行，不导入 Triton、Peer 显存或进程组。
专家编号是全局编号，采用均匀放置 E=P*L。
"""
from dataclasses import dataclass

import torch

from mini_ep.distributed.dispatch import _validate
from mini_ep.distributed.topology import ExpertParallelTopology


@dataclass(frozen=True)
class TokenOwnerPack:
    """去重后的发送布局：每个 (Token, 目标卡) 只保留一个向量。

    字段含义（N 为 Token 数，K 为 Top-K，H 为隐藏维，P 为 EP 大小）：
        packed_tokens: [U,H]，U 是实际存在的 (Token, owner) 对数。
        packed_expert_ids: [U,K]，int64；保留原 K 个 slot，不属于该 owner 的
            slot 写 -1。
        packed_routing_weights: [U,K]，FP32；不属于该 owner 的 slot 写 0。
        send_splits: [P]，每个目标 Rank 收到的唯一向量行数。
        restore_index: [P,N]，(owner, 原 Token) -> 返回行号；不存在时为 -1。

    所有字段连续并与输入同设备，向量 dtype 跟随输入。
    """
    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_routing_weights: torch.Tensor
    send_splits: torch.Tensor
    restore_index: torch.Tensor


def pack_token_owners(
    tokens: torch.Tensor, expert_ids: torch.Tensor,
    routing_weights: torch.Tensor, topology: ExpertParallelTopology,
) -> TokenOwnerPack:
    """把 [N,H] Token 和 [N,K] 路由打包成“每个 Token/目标卡一份”的输入。

    返回 TokenOwnerPack。支持非连续输入。实现先用 unique 得到唯一的
    (owner, token) 键，再按 owner、原 Token 排序，因此每个目的 Rank 的行连续。

    注意：CUDA 上的动态 unique 可能触发 Host 同步，所以本路径不声称异步流水线。
    专家编号范围和权重有限性/非负性由外层模型检查。
    """
    _validate(tokens, expert_ids, routing_weights)
    n, k = expert_ids.shape
    base = max(n, 1)
    destinations = topology.destination_rank(expert_ids)
    token_rows = torch.arange(n, device=tokens.device)[:, None].expand(n, k)
    # 把 (owner, token) 编成整数键；排序去重后自然得到目标卡优先的连续分段。
    keys = torch.unique((destinations * base + token_rows).reshape(-1), sorted=True)
    owners, source_tokens = keys // base, keys % base
    selected_ids = expert_ids.long().index_select(0, source_tokens)
    # 向量只留一份，但保留原 K 个 slot 身份；目标卡据此展开自己负责的分支。
    belongs = topology.destination_rank(selected_ids) == owners[:, None]
    packed_ids = torch.where(belongs, selected_ids, -1).contiguous()
    packed_weights = torch.where(
        belongs, routing_weights.index_select(0, source_tokens), 0.0,
    ).contiguous()
    # 来源卡回传时按 [owner, 原 token] 查行；-1 表示本 Token 没选中该卡。
    restore = torch.full((topology.ep_size, n), -1, dtype=torch.int64, device=tokens.device)
    restore[owners, source_tokens] = torch.arange(keys.numel(), device=tokens.device)
    return TokenOwnerPack(
        tokens.index_select(0, source_tokens).contiguous(), packed_ids, packed_weights,
        torch.bincount(owners, minlength=topology.ep_size), restore,
    )


def pack_token_owners_fixed(
    tokens: torch.Tensor, expert_ids: torch.Tensor,
    routing_weights: torch.Tensor, topology: ExpertParallelTopology, capacity: int,
) -> TokenOwnerPack:
    """固定容量版去重打包：每个目标卡固定 C 行，不需要交换计数或 Host 同步。

    与 pack_token_owners 的唯一区别是行位置：
        packed_tokens: [P*C,H]。目标 r 的段为 [r*C, (r+1)*C)，段内先是选中 r 的
            Token（原 Token 顺序，与变长布局相同），其后是 padding；padding 行内容
            未初始化，接收方只能通过 ID=-1 识别并跳过。
        packed_expert_ids: [P*C,K] int64，padding 行全为 -1。
        packed_routing_weights: [P*C,K] FP32，padding 行全为 0。
        send_splits: [P] 设备端有效行数，仅供诊断；通信使用等长分段。
        restore_index: [P,N]，(owner, Token) -> 返回缓冲行号，等于 r*C+段内位置。

    capacity 必须不小于 EP 组内所有来源的最大 Token 数（同一 Token 对同一目标
    最多一行），由调用方从 Host 元数据得到，所有 Rank 必须相同。
    本函数是 PyTorch Reference，用 index_copy_ 把不存在的 (owner, Token) 写入
    一个丢弃行；CUDA 默认路径使用 kernels/dedup_triton.py 中的等价 Kernel。
    """
    _validate(tokens, expert_ids, routing_weights)
    n, k = expert_ids.shape
    p, h = topology.ep_size, tokens.shape[1]
    if capacity < n:
        raise ValueError(f"fixed owner capacity={capacity} must be >= local tokens={n}")
    device = tokens.device
    owners = torch.arange(p, device=device)
    # slot_owned[r,t,s]：Token t 的第 s 个分支属于目标 r。
    slot_owned = topology.destination_rank(expert_ids).unsqueeze(0) == owners[:, None, None]
    present = slot_owned.any(-1)
    # 段内位置是对原 Token 顺序的前缀计数，因此有效行顺序与变长布局一致。
    rows = owners[:, None] * capacity + present.long().cumsum(1) - 1
    restore = torch.where(present, rows, -1)
    discard = p * capacity
    target = torch.where(present, rows, discard).reshape(-1)
    packed_tokens = tokens.new_empty((discard + 1, h))
    packed_tokens.index_copy_(0, target, tokens.unsqueeze(0).expand(p, n, h).reshape(-1, h))
    packed_ids = torch.full((discard + 1, k), -1, dtype=torch.int64, device=device)
    packed_ids.index_copy_(0, target, torch.where(
        slot_owned, expert_ids.long().unsqueeze(0), -1).reshape(-1, k))
    packed_weights = torch.zeros((discard + 1, k), dtype=torch.float32, device=device)
    packed_weights.index_copy_(0, target, torch.where(
        slot_owned, routing_weights.unsqueeze(0), 0.0).reshape(-1, k))
    # 丢弃行可能被多次写入（重复索引），切掉后不影响结果。
    packed_ids[discard:] = -1
    return TokenOwnerPack(
        packed_tokens[:discard].contiguous(), packed_ids[:discard].contiguous(),
        packed_weights[:discard].contiguous(), present.sum(1), restore,
    )


def expand_owner_branches(
    tokens: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """在接收卡内把唯一向量展开成 Expert 分支。

    输入是 source-major 的 [U,H]/[U,K]（只含本 Rank 收到的 owner 数据）。
    返回：
        [R,H] 连续分支向量、[R] int64 全局专家编号、[R] FP32 权重，
        以及 [U,K] int64 lookup（每个 slot 对应的分支行号，无效 slot 为 -1）。

    展开顺序是“先唯一行、再原 slot”。这里只复制本卡数据，不增加跨卡大向量传输。
    """
    valid = expert_ids >= 0
    # 二维 nonzero 按行再按 slot 排序，保持原 slot 顺序以复现浮点归约次序。
    pair_rows, slots = torch.where(valid)
    lookup = torch.full_like(expert_ids, -1, dtype=torch.int64)
    lookup[pair_rows, slots] = torch.arange(pair_rows.numel(), device=tokens.device)
    return (
        tokens.index_select(0, pair_rows).contiguous(),
        expert_ids[pair_rows, slots].long().contiguous(),
        routing_weights[pair_rows, slots].contiguous(), lookup,
    )


def reduce_owner_branches(
    weighted_branches: torch.Tensor, lookup: torch.Tensor, *, model_dtype: torch.dtype,
) -> torch.Tensor:
    """在回传前把已加权的 [R,H] 分支合并成 [U,H] 部分和。

    精度规则（必须与 Reference 一致）：
        1. 每个分支先舍入到模型 dtype（例如 BF16）；
        2. 按原 slot 顺序用 FP32 累加；
        3. 每个 owner 的部分和最后只舍入一次。

    这样保留原来源卡 Combine 的舍入边界，也兼容显式指定的分支 return_dtype。
    不能重复乘路由权重，也不能改成无序浮点原子求和。
    """
    total = torch.zeros(
        (lookup.shape[0], weighted_branches.shape[1]),
        dtype=torch.float32, device=weighted_branches.device,
    )
    if weighted_branches.shape[0]:
        # 即便分支传输 dtype 更高，原 Combine 也先落到模型 dtype，再做 owner 合并。
        branches = weighted_branches.to(model_dtype)
        for slot in range(lookup.shape[1]):
            rows = lookup[:, slot]
            # -1 不能用于索引；先安全取第0行，再用 mask 置零，空分支由外层跳过。
            values = branches.index_select(0, rows.clamp_min(0)).float()
            total = total + torch.where(rows[:, None] >= 0, values, 0.0)
    return total.to(model_dtype)


def combine_owner_partials(returned: torch.Tensor, restore_index: torch.Tensor) -> torch.Tensor:
    """把返回的唯一向量 [U,H] 按 owner 升序合并，恢复原始 [N,H]。

    参数：
        returned: 各 owner 的部分和，已经加权并完成 owner 舍入。
        restore_index: [P,N]，由来源卡打包时生成；(owner, token) -> 返回行。

    缺失的 owner 贡献 0，全部为空也有效。实现按 owner 升序做 FP32 累加，
    最后转回输入 dtype，加法顺序本身也是精度约定。
    """
    total = torch.zeros(
        (restore_index.shape[1], returned.shape[1]),
        dtype=torch.float32, device=returned.device,
    )
    if returned.shape[0]:
        # 不用 index_add_/原子累加：owner 升序的 FP32 加法顺序也是精度约定。
        for owner in range(restore_index.shape[0]):
            rows = restore_index[owner]
            values = returned.index_select(0, rows.clamp_min(0)).float()
            total = total + torch.where(rows[:, None] >= 0, values, 0.0)
    return total.to(returned.dtype)
