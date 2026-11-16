"""逐 owner/Token/slot 的独立循环打包 Reference，用于 CPU/CUDA 精度检查。

它不复用生产版的 unique/argsort 实现，而是按三个显式循环构造唯一向量、
保留原 slot 编号并生成 restore 表，便于独立核对生产布局是否漏行或错位。
"""
import torch
from mini_ep.distributed.dedup import TokenOwnerPack
from mini_ep.distributed.topology import ExpertParallelTopology


def pack_token_owners_reference(tokens, ids, weights, topology: ExpertParallelTopology):
    """为每个实际使用的 (Token, owner) 生成一份向量。

    参数：
        tokens: 原始 Token [N,H]。
        ids: 全局专家编号 [N,K]。
        weights: FP32 路由权重 [N,K]。
        topology: 均匀专家放置。

    返回：
        TokenOwnerPack。返回行按 owner、再按 Token 排列；每个向量的 K 个 slot
        保留原位置，不属于该 owner 的 slot 写 -1/0。空输入返回空格布局。
    """
    rows, packed_ids, packed_weights, counts = [], [], [], []
    n, k = ids.shape
    restore = torch.full((topology.ep_size, n), -1, device=tokens.device, dtype=torch.long)
    # 有意保留三层可读循环，独立核对向量版 unique/掩码/恢复索引的语义。
    for owner in range(topology.ep_size):
        count = 0
        for token in range(n):
            slots = [s for s in range(k) if int(ids[token, s]) // topology.num_local_experts == owner]
            if slots:
                restore[owner, token] = len(rows)
                rows.append(tokens[token])
                one_ids = torch.full((k,), -1, device=tokens.device, dtype=torch.long)
                one_weights = weights.new_zeros(k)
                for slot in slots:
                    one_ids[slot] = ids[token, slot]
                    one_weights[slot] = weights[token, slot]
                packed_ids.append(one_ids);packed_weights.append(one_weights);count += 1
        counts.append(count)
    return TokenOwnerPack(
        torch.stack(rows) if rows else tokens.new_empty((0, tokens.shape[1])),
        torch.stack(packed_ids) if rows else ids.new_empty((0, k), dtype=torch.long),
        torch.stack(packed_weights) if rows else weights.new_empty((0, k)),
        torch.tensor(counts, device=tokens.device), restore,
    )
