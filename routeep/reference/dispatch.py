"""可独立阅读的 Dispatch Reference：Rank 打包和 DeepEP 舍入边界。

实现故意使用逐记录循环和显式排序，不使用优化 Kernel。它用于在 CPU 或 GPU
上核对生产布局和数值行为，不参与性能结论。
"""

import torch
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.dispatch import RankPackedTensors


def dispatch_reference(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    topology: ExpertParallelTopology,
) -> RankPackedTensors:
    """原始 [N,H]/[N,K] -> 稳定的目的 Rank 展开行。

    每个分支按目标 Rank 排序，同一目标内保持 token-major/slot-minor 顺序。
    返回打包张量、发送计数和恢复索引；实现只使用可读的 PyTorch 操作。
    """
    if (
        tokens.ndim != 2
        or expert_ids.ndim != 2
        or tokens.shape[0] != expert_ids.shape[0]
        or expert_ids.shape[1] < 1
    ):
        raise ValueError("expected Tokens [N,H] and IDs [N,K], K>0")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("IDs must be int32/int64")
    if (
        routing_weights.shape != expert_ids.shape
        or routing_weights.dtype != torch.float32
    ):
        raise TypeError("weights must be FP32 [N,K]")
    if tokens.device != expert_ids.device or tokens.device != routing_weights.device:
        raise ValueError("device mismatch")
    if expert_ids.numel() and (
        expert_ids.min().item() < 0 or expert_ids.max().item() >= topology.num_experts
    ):
        raise ValueError("Expert ID outside topology")
    flat = expert_ids.reshape(-1).long()
    destinations = topology.destination_rank(flat)
    groups = [torch.where(destinations == rank)[0] for rank in range(topology.ep_size)]
    order = torch.cat(groups)
    inverse = torch.empty_like(order)
    inverse.index_copy_(0, order, torch.arange(order.numel(), device=order.device))
    return RankPackedTensors(
        tokens.index_select(0, order // expert_ids.shape[1]).contiguous(),
        flat.index_select(0, order),
        routing_weights.reshape(-1).index_select(0, order),
        order,
        inverse,
        torch.tensor(
            [x.numel() for x in groups], device=tokens.device, dtype=torch.int64
        ),
    )


def owner_partials_reference(
    returned: torch.Tensor,
    restore_index: torch.Tensor,
    expert_ids: torch.Tensor,
    topology: ExpertParallelTopology,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """已加权 Rank-major 行 -> 模型 dtype 的 [P,N,H] owner 贡献。

    这是独立诊断实现，用来暴露必须保留的舍入边界：每个分支先舍入到模型 dtype，
    同一 owner 的 slot 按升序用 FP32 累加，最后再舍入一次。它不是生产归约的
    替代实现。
    """
    n, k = expert_ids.shape
    branches = (
        returned.index_select(0, restore_index)
        .reshape(n, k, returned.shape[1])
        .to(model_dtype)
    )
    owners = topology.destination_rank(expert_ids)
    partials = []
    for owner in range(topology.ep_size):
        partial = torch.zeros(
            (n, returned.shape[1]), device=returned.device, dtype=torch.float32
        )
        for slot in range(k):
            partial = partial + torch.where(
                owners[:, slot, None] == owner, branches[:, slot].float(), 0.0
            )
        partials.append(partial.to(model_dtype))
    return torch.stack(partials)


def combine_reference(
    returned: torch.Tensor,
    restore_index: torch.Tensor,
    expert_ids: torch.Tensor,
    topology: ExpertParallelTopology,
    *,
    model_dtype: torch.dtype,
) -> torch.Tensor:
    """已加权 Rank-major [R,H] -> 原始 [N,H]，不再次加权。

    Owner 部分和保留模型 dtype 的舍入；最终按 EP Rank 升序做顺序 FP32 加法，
    最后只做一次模型 dtype 存储。
    """
    if model_dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"model dtype must be FP32/BF16, got {model_dtype}")
    parts = owner_partials_reference(
        returned, restore_index, expert_ids, topology, model_dtype=model_dtype
    )
    total = torch.zeros_like(parts[0], dtype=torch.float32)
    for owner in range(topology.ep_size):
        total = total + parts[owner].float()
    return total.to(model_dtype)
