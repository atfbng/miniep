"""显式的逐分支 EP 链路：打包 -> 第一次 A2A -> Expert -> 返回 A2A -> Combine。

Router 的 ID 和权重由框架适配层提供。本模块不感知 vLLM 生命周期、TP/DP
输入切分或原生兼容归约，只实现可以独立阅读和测试的教学版专家并行。
"""

import torch
import torch.distributed as dist
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.dispatch import (
    pack_by_destination_rank,
    pack_by_destination_rank_fixed,
)
from mini_ep.distributed.all_to_all import (
    exchange_splits,
    variable_split_all_to_all,
    equal_split_all_to_all,
)
from mini_ep.model.contracts import validate_inputs
from mini_ep.model.expert import (
    compute_local_experts_loop,
    compute_grouped_local_experts,
)
from mini_ep.kernels.combine_triton import combine_triton
from mini_ep.profiler.stages import Stages


def ep_forward(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_weights: torch.Tensor,
    down_weights: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    stages: Stages | None = None,
    validate: bool = True,
    expert_backend: str = "grouped",
    fixed_rows_per_rank: int | None = None,
    source_rows: list[int] | None = None,
    fixed_pack_backend: str = "triton",
    return_dtype: torch.dtype | None = None,
    fuse_return_weight: bool | None = None,
) -> tuple[torch.Tensor, dict]:
    """逐分支 EP 前向：原始 [N,H] 输入，返回原始顺序 [N,H]。

    输入/输出：
        tokens: 当前来源 Rank 的 Token [N,H]，支持 FP32/BF16。
        expert_ids/routing_weights: 未展开的 [N,K] 全局专家编号和 FP32 权重。
        gate_up_weights/down_weights: 本地权重 [L,2F,H]/[L,H,F]，dtype 与输入一致。
        group: 调用方持有的 EP 进程组；Rank 是组内编号，不是 CUDA 序号。

    专家放置：
        L=E/P，目标 Rank = expert_id//L，本地编号 = expert_id%L。

    布局与通信：
        默认使用变长 A2A：发送缓冲按目标 Rank 打包成 [N*K,H]，接收缓冲按来源
        Rank 排列成 [R,H]；split 是 P 个行数，返回方向交换 send/recv。
        固定容量路径使用 [P*C,H]，填充行 Token=0、权重=0、expert_id=-1，并复用
        同一个 restore_index；因为每个 Rank 都发送 C 行，不需要读取 GPU 计数。

    精度：
        Down 在 FP32 中累加；Grouped 路径可在 down 写出前融合一次路由权重，
        fuse_return_weight=False 则保留 FP32 张量并在外部单独相乘。Combine 先按
        模型 dtype 舍入分支/owner，再按 owner Rank 升序做 FP32 最终合并。

    所有 Rank 必须使用相同的选项，并按“计数（变长）、Token、ID、权重、返回”
    的顺序进入集合通信，包括零行来源。
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
    requested_dtype = return_dtype
    return_dtype = tokens.dtype if return_dtype is None else return_dtype
    if return_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"return transport must be FP32/BF16/FP16, got {return_dtype}")
    if expert_backend not in ("grouped", "loop"):
        raise ValueError(f"unknown expert backend {expert_backend!r}")
    if fuse_return_weight is None:
        fuse_return_weight = expert_backend == "grouped"
    if fuse_return_weight and expert_backend != "grouped":
        raise ValueError("fused down weighting requires grouped GEMM")
    if fixed_pack_backend not in ("torch", "triton"):
        raise ValueError("fixed pack backend must be torch/triton")
    if source_rows is not None:
        if (
            len(source_rows) != size
            or any(n < 0 for n in source_rows)
            or source_rows[topology.ep_rank] != tokens.shape[0]
        ):
            raise ValueError(
                f"source_rows={source_rows} must match EP size={size} and local N={tokens.shape[0]}"
            )
    fixed = fixed_rows_per_rank is not None
    if fixed and fixed_rows_per_rank < 1:
        raise ValueError("fixed capacity must be positive")
    stages = stages or Stages()
    fixed_pack_used = None
    with stages.stage("DISPATCH"):
        if fixed:
            pack = pack_by_destination_rank_fixed
            fixed_pack_used = "torch"
            if fixed_pack_backend == "triton" and expert_ids.numel() <= 64:
                from mini_ep.kernels.dispatch_triton import (
                    pack_by_destination_rank_fixed_triton,
                )

                pack = pack_by_destination_rank_fixed_triton
                fixed_pack_used = "triton"
            rank_pack = pack(
                tokens, expert_ids, routing_weights, topology, fixed_rows_per_rank
            )
        else:
            rank_pack = pack_by_destination_rank(
                tokens, expert_ids, routing_weights, topology
            )
    with stages.stage("A2A_FORWARD"):
        if fixed:
            send_splits = recv_splits = [fixed_rows_per_rank] * size
            recv_tokens = equal_split_all_to_all(rank_pack.packed_tokens, group)
            recv_ids = equal_split_all_to_all(rank_pack.packed_expert_ids, group)
            recv_weights = equal_split_all_to_all(
                rank_pack.packed_routing_weights, group
            )
        else:
            send_splits, recv_splits = exchange_splits(rank_pack.send_splits, group)
            recv_tokens = variable_split_all_to_all(
                rank_pack.packed_tokens, send_splits, recv_splits, group
            )
            recv_ids = variable_split_all_to_all(
                rank_pack.packed_expert_ids, send_splits, recv_splits, group
            )
            recv_weights = variable_split_all_to_all(
                rank_pack.packed_routing_weights, send_splits, recv_splits, group
            )
    with stages.stage("EXPERT"):
        local_ids = topology.local_expert_id(recv_ids, padding=fixed)
        if expert_backend == "grouped":
            expert_outputs = compute_grouped_local_experts(
                recv_tokens,
                local_ids,
                gate_up_weights,
                down_weights,
                padding=fixed,
                branch_weights=recv_weights if fuse_return_weight else None,
                output_dtype=return_dtype if fuse_return_weight else torch.float32,
            )
        else:
            expert_outputs = compute_local_experts_loop(
                recv_tokens, local_ids, gate_up_weights, down_weights
            )
        if not fuse_return_weight:
            expert_outputs = (expert_outputs * recv_weights[:, None]).to(return_dtype)
    with stages.stage("A2A_RETURN"):
        returned = (
            equal_split_all_to_all(expert_outputs, group)
            if fixed
            else variable_split_all_to_all(
                expert_outputs, recv_splits, send_splits, group
            )
        )
    with stages.stage("COMBINE"):
        result = combine_triton(
            returned,
            rank_pack.restore_index,
            expert_ids,
            topology,
            model_dtype=tokens.dtype,
        )
    send_rows = rank_pack.packed_tokens.shape[0]
    recv_rows = recv_tokens.shape[0]
    return result, dict(
        model_dtype=str(tokens.dtype),
        return_dtype=str(expert_outputs.dtype),
        return_buffer_dtype=str(returned.dtype),
        return_dtype_requested=(
            "auto" if requested_dtype is None else str(requested_dtype)
        ),
        preweighted=True,
        fuse_return_weight=fuse_return_weight,
        owner_reduction="local_fp32",
        return_send_bytes=expert_outputs.numel() * expert_outputs.element_size(),
        weight_send_bytes=rank_pack.packed_routing_weights.numel() * 4,
        send_splits=send_splits,
        recv_splits=recv_splits,
        send_rows=send_rows,
        recv_rows=recv_rows,
        source_tokens=tokens.shape[0],
        branches=expert_ids.numel(),
        expert_backend=expert_backend,
        received_branches=None if fixed else recv_rows,
        layout="fixed" if fixed else "variable",
        fixed_pack_backend=fixed_pack_used,
        fixed_rows_per_rank=fixed_rows_per_rank,
        send_padding_rows=send_rows - expert_ids.numel() if fixed else 0,
        recv_padding_rows=None if fixed else 0,
    )
