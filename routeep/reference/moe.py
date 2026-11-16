"""独立 PyTorch MoE Reference：专家计算、同步 NCCL EP 和显式舍入。"""

import torch
import torch.distributed as dist
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.all_to_all import exchange_splits, variable_split_all_to_all
from mini_ep.model.contracts import validate_inputs
from mini_ep.profiler.stages import Stages
from reference.dispatch import dispatch_reference, combine_reference
from reference.expert import expert_loop


def moe_reference(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    *,
    ep_size: int = 1,
) -> torch.Tensor:
    """单设备持有全部权重，再模拟选定的 EP Rank 归属。

    该入口用于小规模正确性检查，不启动通信。
    """
    validate_inputs(
        tokens,
        expert_ids,
        routing_weights,
        gate_up_weights,
        down_weights,
        ExpertParallelTopology(gate_up_weights.shape[0], 1, 0),
    )
    n, k = expert_ids.shape
    expanded = (
        tokens[:, None, :].expand(n, k, tokens.shape[1]).reshape(n * k, tokens.shape[1])
    )
    unweighted = expert_loop(
        expanded, expert_ids.reshape(-1).long(), gate_up_weights, down_weights
    )
    weighted = (unweighted * routing_weights.reshape(-1, 1)).to(tokens.dtype)
    return combine_reference(
        weighted,
        torch.arange(n * k, device=tokens.device),
        expert_ids,
        ExpertParallelTopology(gate_up_weights.shape[0], ep_size, 0),
        model_dtype=tokens.dtype,
    )


def ep_forward_reference(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    stages: Stages | None = None,
    validate: bool = True,
    return_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, dict]:
    """原始 [N,H] -> 原始 [N,H]；独立计算和 Rank 布局。

    模型 dtype 为 FP32/BF16；down 产生 FP32，路由权重在 FP32 中相乘，之后才做
    返回 dtype 转换。所有 Rank 以相同顺序进入计数、Token、全局 ID、FP32 权重和
    返回集合通信，零行来源也参与。split 是第 0 维行数，返回方向交换 send/recv。
    本实现不导入任何优化 Kernel。
    """
    if tokens.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(
            f"only FP32/BF16 model activations supported, got {tokens.dtype}"
        )
    size = dist.get_world_size(group)
    topology = ExpertParallelTopology(
        gate_up_weights.shape[0] * size, size, dist.get_rank(group)
    )
    if validate:
        validate_inputs(
            tokens, expert_ids, routing_weights, gate_up_weights, down_weights, topology
        )
    return_dtype = tokens.dtype if return_dtype is None else return_dtype
    if return_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError("unsupported transport dtype")
    stages = stages or Stages()
    with stages.stage("DISPATCH"):
        rank_pack = dispatch_reference(tokens, expert_ids, routing_weights, topology)
    with stages.stage("A2A_FORWARD"):
        send, recv = exchange_splits(rank_pack.send_splits, group)
        received = variable_split_all_to_all(rank_pack.packed_tokens, send, recv, group)
        ids = variable_split_all_to_all(rank_pack.packed_expert_ids, send, recv, group)
        weights = variable_split_all_to_all(
            rank_pack.packed_routing_weights, send, recv, group
        )
    with stages.stage("EXPERT"):
        down = expert_loop(
            received, topology.local_expert_id(ids), gate_up_weights, down_weights
        )
        outputs = (down * weights[:, None]).to(return_dtype)
    with stages.stage("A2A_RETURN"):
        returned = variable_split_all_to_all(outputs, recv, send, group)
    with stages.stage("COMBINE"):
        result = combine_reference(
            returned,
            rank_pack.restore_index,
            expert_ids,
            topology,
            model_dtype=tokens.dtype,
        )
    return result, dict(
        send_splits=send,
        recv_splits=recv,
        source_tokens=tokens.shape[0],
        model_dtype=str(tokens.dtype),
        return_dtype=str(outputs.dtype),
        preweighted=True,
        send_rows=rank_pack.packed_tokens.shape[0],
        recv_rows=received.shape[0],
        return_send_bytes=outputs.numel() * outputs.element_size(),
        weight_send_bytes=rank_pack.packed_routing_weights.numel() * 4,
    )
