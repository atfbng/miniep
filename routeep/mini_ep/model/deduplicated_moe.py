"""默认 EP 推理链：唯一 Token/目标卡 NCCL All-to-All -> 卡内展开 -> 目标卡归约。

与 model/moe.py 的逐分支实现相比，这里先把同一个 Token 发往同一个目标卡的
多个分支合并成一份大向量，减少跨卡重复传输。Expert 编号和 FP32 路由权重
仍按分支携带，目标卡再在本地展开给 Expert 计算。

本模块不依赖 Peer/TMA；默认 CUDA 计算使用 Grouped GEMM，PyTorch 路径只用于
CPU 语义检查。
"""
import torch
import torch.distributed as dist

from mini_ep.distributed.all_to_all import (
    equal_split_all_to_all, exchange_splits, variable_split_all_to_all,
)
from mini_ep.distributed.dedup import (
    pack_token_owners, pack_token_owners_fixed, expand_owner_branches,
    reduce_owner_branches, combine_owner_partials,
)
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.model.contracts import validate_inputs
from mini_ep.profiler.stages import Stages


def ep_forward(
    tokens: torch.Tensor, expert_ids: torch.Tensor, routing_weights: torch.Tensor,
    gate_up_weights: torch.Tensor, down_weights: torch.Tensor, group: dist.ProcessGroup,
    *, stages: Stages | None = None, validate: bool = True,
    expert_backend: str = "grouped", return_dtype: torch.dtype | None = None,
    fixed_capacity: int | None = None,
) -> tuple[torch.Tensor, dict]:
    """执行去重后的 EP 前向：原始 [N,H]/[N,K] -> 原始顺序 [N,H]。

    输入/输出：
        tokens: 当前来源 Rank 的 Token [N,H]，支持 FP32/BF16。
        expert_ids/routing_weights: 未展开的 [N,K] 全局专家编号和 FP32 权重。
        gate_up_weights/down_weights: 本地权重 [L,2F,H]/[L,H,F]。
        group: 调用方持有的 EP 进程组；Rank 是组内编号，不是 CUDA 序号。

    每一步的数据布局：
        1. DISPATCH 把 (Token, 目标卡) 去重成 [U,H] 唯一向量；ID/权重仍是 [U,K]，
           不属于该 owner 的 slot 写 -1/0。
        2. 第一次 A2A 交换唯一向量、ID 和权重；接收端按来源 Rank 连续排列。
        3. EXPERT 在接收卡内展开成 [R,H] 分支，再执行本地 Expert。
        4. 目标卡按原 slot 顺序把同一 owner 的分支用 FP32 合并成 [U,H] 部分和，
           舍入一次后回传。
        5. COMBINE 按 owner 升序合并部分和，恢复原始 Token 顺序。

    通信约定：
        所有 Rank 严格按“计数交换 -> 输入 A2A -> ID A2A -> 权重 A2A -> 返回 A2A”
        的顺序参与，空来源也不能提前返回。split 是 P 个唯一行数；返回方向交换
        send/recv。部分和用模型 dtype 传输，以保留原舍入边界。

    固定容量模式（fixed_capacity=C）：
        每个目标卡固定 C 行，发送缓冲 [P*C,H]，padding 行 ID 全为 -1、权重为 0。
        不交换计数，三次分发和一次返回都用等长 all_to_all_single；C 必须不小于
        EP 组内最大来源 Token 数，且所有 Rank 相同（调用方从 Host 元数据得到）。
        接收端 padding 行不参与 Expert 计算，返回值为 0，来源端 restore 不引用它们。
        该模式下 Dispatch/Expert/Combine 都不读回 GPU 计数，整层没有 Host 同步。
        CUDA 且 N<=FIXED_OWNER_TRITON_MAX_TOKENS 时，激活、int32 ID、FP32 权重
        打进同一个 uint8 行缓冲 [P*C, H*elem+8K]，分发只发起一次 A2A。

    精度与限制：
        分支先舍入到 return_dtype，再转模型 dtype；owner 归约和最终合并都使用
        顺序 FP32 加法。生产路径为 NCCL/CUDA；PyTorch+Gloo/CPU 只验证语义。
        动态 unique/nonzero 和 Python split 列表会带来 Host 同步，因此这里不
        声称已实现通信/计算重叠。
    """
    if tokens.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"only FP32/BF16 model activations supported, got {tokens.dtype}")
    topology = ExpertParallelTopology(
        gate_up_weights.shape[0] * dist.get_world_size(group),
        dist.get_world_size(group), dist.get_rank(group),
    )
    if validate:
        validate_inputs(tokens, expert_ids, routing_weights, gate_up_weights, down_weights, topology)
    requested_dtype = return_dtype
    return_dtype = tokens.dtype if return_dtype is None else return_dtype
    if return_dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise ValueError(f"return transport must be FP32/BF16/FP16, got {return_dtype}")
    if expert_backend not in ("grouped", "grouped_unfused", "loop", "torch"):
        raise ValueError(f"unknown expert backend {expert_backend!r}")
    stages = stages or Stages()
    fixed = fixed_capacity is not None
    if fixed and expert_backend != "grouped":
        raise ValueError("fixed-capacity owner A2A requires expert_backend='grouped'")
    # CUDA 短批次：一个 Kernel 打出“激活|int32 ID|FP32 权重”的字节行，分发只需一次 A2A。
    fused_bytes = False
    if fixed and tokens.is_cuda:
        from mini_ep.kernels.dedup_triton import FIXED_OWNER_TRITON_MAX_TOKENS
        fused_bytes = tokens.shape[0] <= FIXED_OWNER_TRITON_MAX_TOKENS
    with stages.stage("DISPATCH"):
        if fused_bytes:
            from mini_ep.kernels.dedup_triton import (
                pack_token_owners_fixed_triton, split_owner_rows,
            )

            send_buffer, restore_index, _ = pack_token_owners_fixed_triton(
                tokens.contiguous(), expert_ids, routing_weights,
                topology.num_local_experts, topology.ep_size, fixed_capacity)
        elif fixed:
            packed = pack_token_owners_fixed(
                tokens, expert_ids, routing_weights, topology, fixed_capacity)
            restore_index = packed.restore_index
        else:
            packed = pack_token_owners(tokens, expert_ids, routing_weights, topology)
            restore_index = packed.restore_index
    with stages.stage("A2A_FORWARD"):
        if fused_bytes:
            send = recv = None
            # 单次等长 A2A；接收后按字节偏移取视图，不额外拷贝。
            received_buffer = equal_split_all_to_all(send_buffer, group)
            received, ids, weights = split_owner_rows(
                received_buffer, tokens.shape[1], expert_ids.shape[1], tokens.dtype)
        elif fixed:
            # 等长分段：发送/接收都是 P 段各 C 行，不需要计数交换。
            send = recv = None
            received = equal_split_all_to_all(packed.packed_tokens, group)
            ids = equal_split_all_to_all(packed.packed_expert_ids, group)
            weights = equal_split_all_to_all(packed.packed_routing_weights, group)
        else:
            # split 单位已从 Expert 分支行变为唯一 Token/owner 行，三个发送缓冲共用它。
            # 空来源不能提前返回，否则各 rank 会进入不同的 collective 顺序。
            send, recv = exchange_splits(packed.send_splits, group)
            received = variable_split_all_to_all(packed.packed_tokens, send, recv, group)
            ids = variable_split_all_to_all(packed.packed_expert_ids, send, recv, group)
            weights = variable_split_all_to_all(packed.packed_routing_weights, send, recv, group)
    with stages.stage("EXPERT"):
        if expert_backend == "grouped":
            # 默认 CUDA 路径：唯一行直接 gather 成 Expert-major，Grouped GEMM 后
            # 一个 Kernel 完成 slot 顺序的 owner 归约；与下面的分步实现逐元素一致。
            from mini_ep.kernels.dedup_triton import compute_owner_partials_triton

            owner_partials = compute_owner_partials_triton(
                received, ids, weights, gate_up_weights, down_weights,
                return_dtype=return_dtype,
            )
            # 精确分支数需要读回 GPU 计数；融合路径为避免 Host 同步不报告。
            received_branches = None
        else:
            expanded, branch_ids, branch_weights, lookup = expand_owner_branches(received, ids, weights)
            local_ids = topology.local_expert_id(branch_ids)
            # 只有接收卡内展开成分支，Expert 计算量与原实现一致。
            if expert_backend == "grouped_unfused":
                from mini_ep.model.expert import compute_grouped_local_experts

                branches = compute_grouped_local_experts(
                    expanded, local_ids, gate_up_weights, down_weights,
                    branch_weights=branch_weights, output_dtype=return_dtype,
                )
            else:
                if expert_backend == "torch":
                    from mini_ep.model.expert_torch import compute_local_experts_torch as compute
                else:
                    from mini_ep.model.expert import compute_local_experts_loop as compute
                branches = (compute(expanded, local_ids, gate_up_weights, down_weights)
                            * branch_weights[:, None]).to(return_dtype)
            # 权重已应用；在通信前完成 owner 边界的顺序归约和模型 dtype 舍入。
            owner_partials = reduce_owner_branches(branches, lookup, model_dtype=tokens.dtype)
            received_branches = expanded.shape[0]
    with stages.stage("A2A_RETURN"):
        # 回传按来源分段，故 send/recv 与分发方向相反；每个 owner 只回一份部分和。
        if fixed:
            returned = equal_split_all_to_all(owner_partials, group)
        else:
            returned = variable_split_all_to_all(owner_partials, recv, send, group)
    with stages.stage("COMBINE"):
        if returned.is_cuda and expert_backend == "grouped":
            from mini_ep.kernels.dedup_triton import combine_owner_partials_triton

            result = combine_owner_partials_triton(returned, restore_index)
        else:
            result = combine_owner_partials(returned, restore_index)
    return result, dict(
        transport="nccl", deduplicated=True, layout="unique-token-owner-fixed" if fixed else "unique-token-owner",
        model_dtype=str(tokens.dtype), return_dtype=str(return_dtype),
        return_buffer_dtype=str(returned.dtype),
        return_dtype_requested="auto" if requested_dtype is None else str(requested_dtype),
        preweighted=True, fuse_return_weight=expert_backend in ("grouped", "grouped_unfused"),
        owner_reduction="destination_fp32", send_splits=send, recv_splits=recv,
        send_rows=received.shape[0] if fixed else packed.packed_tokens.shape[0],
        recv_rows=received.shape[0],
        source_tokens=tokens.shape[0], branches=expert_ids.numel(),
        received_branches=received_branches, expert_backend=expert_backend,
        input_send_bytes=received.shape[0] * tokens.shape[1] * tokens.element_size(),
        return_send_bytes=owner_partials.numel() * owner_partials.element_size(),
        weight_send_bytes=received.shape[0] * expert_ids.shape[1] * 4,
        id_send_bytes=received.shape[0] * expert_ids.shape[1] * (4 if fused_bytes else 8),
        forward_collectives=1 if fused_bytes else 3 if fixed else 4,
        fixed_pack_backend="triton" if fused_bytes else "torch" if fixed else None, fixed_rows_per_rank=fixed_capacity,
        # 固定容量模式的精确 padding 行数需要读回 GPU 计数，这里不报告。
        send_padding_rows=None if fixed else 0, recv_padding_rows=None if fixed else 0,
    )
