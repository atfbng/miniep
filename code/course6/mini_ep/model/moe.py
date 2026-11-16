"""Readable single-GPU MoE reference implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.distributed as dist
from torch import nn

from mini_ep.distributed.all_to_all import (
    all_gather_expert_counts,
    variable_split_all_to_all,
)
from mini_ep.distributed.dispatch import (
    DispatchOutput,
    combine_reference,
    dispatch_reference,
    pack_by_destination_rank,
)
from mini_ep.distributed.topology import (
    ExpertParallelTopology,
    build_receive_layout,
)
from mini_ep.model.expert import ExpertMLP
from mini_ep.model.router import RouterOutput, TopKRouter
from mini_ep.profiler.nvtx import nvtx_range


@dataclass
class ExpertAssignment:
    """One expert's view of the tokens dispatched to it.

    ``token_indices`` identifies original token positions and ``route_slots``
    identifies the Top-K slot. ``tokens`` preserves that assignment order.
    """

    expert_id: int
    token_indices: torch.Tensor
    route_slots: torch.Tensor
    tokens: torch.Tensor
    routing_weights: torch.Tensor


@dataclass
class MoEOutput:
    """MoE result plus routing metadata used for teaching and inspection."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    tokens_per_expert: torch.Tensor


def dispatch_tokens(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> list[ExpertAssignment]:
    """Group original-order tokens into one logical batch per expert.

    Args:
        tokens: Original token order, shape ``[num_tokens, hidden_size]``.
        expert_ids: Expert IDs, shape ``[num_tokens, top_k]``, integer dtype.
        routing_weights: Combine weights with the same shape as ``expert_ids``.
        num_experts: Total number of local experts.

    Returns:
        A list of length ``num_experts``. Empty experts receive tensors whose
        first dimension is zero. This is a readable PyTorch reference; it does
        not physically construct the rank-packed buffer used by later lessons.
    """
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be 2-D, got shape {tuple(tokens.shape)}")
    if expert_ids.ndim != 2:
        raise ValueError(
            f"expert_ids must be 2-D, got shape {tuple(expert_ids.shape)}"
        )
    if expert_ids.shape != routing_weights.shape:
        raise ValueError(
            "expert_ids and routing_weights must have the same shape, got "
            f"{tuple(expert_ids.shape)} and {tuple(routing_weights.shape)}"
        )
    if expert_ids.shape[0] != tokens.shape[0]:
        raise ValueError(
            "routing metadata must have one row per token, got "
            f"{expert_ids.shape[0]} rows for {tokens.shape[0]} tokens"
        )
    if expert_ids.device != tokens.device or routing_weights.device != tokens.device:
        raise ValueError("tokens and routing metadata must be on the same device")
    if expert_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"expert_ids must use an integer dtype, got {expert_ids.dtype}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if expert_ids.numel() > 0:
        smallest_id = int(expert_ids.min().item())
        largest_id = int(expert_ids.max().item())
        if smallest_id < 0 or largest_id >= num_experts:
            raise ValueError(
                f"expert IDs must be in [0, {num_experts}), got range "
                f"[{smallest_id}, {largest_id}]"
            )

    assignments: list[ExpertAssignment] = []
    for expert_id in range(num_experts):
        # expert_ids 的两个维度分别是“原始 Token 位置”和“Top-K 槽位”。
        # 同一个 Token 在 Top-K 路由中可能出现在多个 Expert 的 assignment 中。
        token_indices, route_slots = torch.where(expert_ids == expert_id)
        assignments.append(
            ExpertAssignment(
                expert_id=expert_id,
                token_indices=token_indices,
                route_slots=route_slots,
                # 按当前 Expert 的 assignment 顺序收集 Token；空 Expert 会自然
                # 得到形状为 [0, hidden_size] 的张量，无需单独分支处理。
                tokens=tokens.index_select(0, token_indices),
                routing_weights=routing_weights[token_indices, route_slots],
            )
        )
    return assignments


def combine_expert_outputs(
    original_tokens: torch.Tensor,
    assignments: Sequence[ExpertAssignment],
    expert_outputs: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Restore original token order and sum weighted Top-K expert outputs."""
    if len(assignments) != len(expert_outputs):
        raise ValueError(
            "assignments and expert_outputs must have equal length, got "
            f"{len(assignments)} and {len(expert_outputs)}"
        )

    # Combine 的目标布局恢复为原始 Token 顺序。先清零，随后累加各条
    # Top-K 路由分支的加权 Expert 输出。
    combined = torch.zeros_like(original_tokens)
    for assignment, expert_output in zip(assignments, expert_outputs, strict=True):
        if expert_output.shape != assignment.tokens.shape:
            raise ValueError(
                f"expert {assignment.expert_id} output shape must be "
                f"{tuple(assignment.tokens.shape)}, got {tuple(expert_output.shape)}"
            )
        if expert_output.device != original_tokens.device:
            raise ValueError("expert outputs and original tokens must share a device")
        # Router 权重通常保留为 FP32；乘法前转换到 Expert 输出 dtype，避免
        # 混合精度输入把结果意外提升为 FP32。
        weighted_output = expert_output * assignment.routing_weights.to(
            expert_output.dtype
        ).unsqueeze(-1)
        # 遍历不同 assignment 时，同一 Token 位置会因 Top-K 路由被多次命中，
        # 因此必须用 index_add_ 累加，不能用普通索引赋值覆盖先前的分支结果。
        combined.index_add_(0, assignment.token_indices, weighted_output)
    return combined


class SingleGPUMoE(nn.Module):
    """Pure PyTorch MoE reference with explicit Dispatch and Combine steps."""

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        activation: str = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size,
                    expert_hidden_size,
                    activation=activation,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, tokens: torch.Tensor) -> MoEOutput:
        """Run Router → Dispatch → Expert MLP → weighted Combine."""
        # Router 输出仍按原始 Token 顺序排列：每行包含该 Token 的 Top-K
        # Expert ID 及对应的归一化权重。
        routing: RouterOutput = self.router(tokens)

        # Dispatch 仅在逻辑上按 Expert 分组；此单 GPU Reference 不构造后续
        # 分布式实现所需的 rank-packed 通信缓冲区。
        assignments = dispatch_tokens(
            tokens,
            routing.expert_ids,
            routing.routing_weights,
            self.num_experts,
        )
        # 每个 Expert 只处理分配给自己的局部 Token 批次，并保持批次内顺序。
        expert_outputs = [
            expert(assignment.tokens)
            for expert, assignment in zip(self.experts, assignments, strict=True)
        ]

        # 根据 token_indices 恢复原始 Token 顺序，并完成 routing weight 加权
        # 与 Top-K 分支求和。
        hidden_states = combine_expert_outputs(
            tokens, assignments, expert_outputs
        )

        return MoEOutput(
            hidden_states=hidden_states,
            router_logits=routing.logits,
            expert_ids=routing.expert_ids,
            routing_weights=routing.routing_weights,
            tokens_per_expert=routing.tokens_per_expert,
        )


@dataclass(frozen=True)
class ExpertParallelForwardResult:
    """Distributed EP forward result for one rank's local Tokens.

    Attributes:
        hidden_states: MoE output in the original order of this rank's local
            Tokens, shape ``[num_local_tokens, hidden_size]``.  It already
            includes the Top-K routing weights.
        router_logits: Gate scores of this rank's local Tokens, shape
            ``[num_local_tokens, num_experts]``.
        expert_ids: Global expert IDs of this rank's local Tokens, shape
            ``[num_local_tokens, top_k]``.  These are *global* IDs: a Token
            routed to an expert owned by another rank keeps that global ID.
        routing_weights: Top-K weights matching ``expert_ids``, shape
            ``[num_local_tokens, top_k]``, FP32.
        tokens_per_expert: Assignment counts of this rank's local Tokens, shape
            ``[num_experts]``.  Rows, not unique Tokens.
        tokens_per_expert_all: Counts gathered from every EP rank, shape
            ``[ep_size, num_experts]``, ``int64``.  Row ``s`` describes source
            rank ``s`` and is the metadata All-to-All #1 is sized from.
        send_splits: Rows this rank sends to each destination rank, shape
            ``[ep_size]``, ``int64``, in ascending destination-rank order.
        recv_splits: Rows this rank receives from each source rank, shape
            ``[ep_size]``, ``int64``, in ascending source-rank order.
        local_expert_rows: Rows each *local* expert processed, shape
            ``[num_local_experts]``, ``int64``, in local expert order.  A zero
            entry is an idle local expert and is expected under skewed routing.
    """

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    tokens_per_expert: torch.Tensor
    tokens_per_expert_all: torch.Tensor
    send_splits: torch.Tensor
    recv_splits: torch.Tensor
    local_expert_rows: torch.Tensor


class ExpertParallelMoE(nn.Module):
    """Pure Expert Parallel MoE built from the single-GPU reference pieces.

    Each rank runs the same module but owns only
    ``experts_per_rank = num_experts // ep_size`` expert MLPs, indexed by
    ``local_expert_id``.  A forward pass executes, in this order:

    1. ``ROUTER``: gate and Top-K on this rank's local Tokens;
    2. ``DISPATCH``: Top-K expansion, expert-major packing, rank packing, and
       the expert-count exchange that sizes the receive buffer;
    3. ``A2A_FORWARD``: variable-size All-to-All #1 sends every assignment to
       the rank that owns its expert;
    4. ``EXPERT``: each local expert processes its gathered rows;
    5. ``A2A_RETURN``: All-to-All #2 returns every row to the rank it came
       from, using the same splits in the opposite direction;
    6. ``COMBINE``: unpermute back to expert-major order, then apply the
       Top-K routing weights and sum.

    The module is an inference path.  ``torch.distributed`` collectives do not
    record autograd history, so calling ``backward()`` on
    ``hidden_states`` is not supported.  Wrap the call in ``torch.no_grad()``
    rather than ``torch.inference_mode()``: the collectives write into
    preallocated buffers, and inference Tensors reject those in-place updates
    on CPU backends such as gloo.
    """

    def __init__(
        self,
        hidden_size: int,
        expert_hidden_size: int,
        num_experts: int,
        top_k: int,
        *,
        ep_size: int,
        ep_rank: int,
        activation: str = "gelu",
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.topology = ExpertParallelTopology(
            num_experts=num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
        # The router is replicated: every rank holds the same gate weights and
        # scores only its own Tokens.
        self.router = TopKRouter(hidden_size, num_experts, top_k)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size,
                    expert_hidden_size,
                    activation=activation,
                    bias=bias,
                )
                for _ in range(self.topology.num_local_experts)
            ]
        )

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        group: dist.ProcessGroup | None = None,
    ) -> ExpertParallelForwardResult:
        """Run one EP forward pass on this rank's local Tokens.

        Args:
            tokens: This rank's local Tokens in original order, shape
                ``[num_local_tokens, hidden_size]``.  Different ranks hold
                different Tokens; the collective itself redistributes them.
            group: EP Process Group.  ``None`` selects the default Process
                Group, which lessons 4 to 6 use as the EP group.  ``ep_size``
                passed to the constructor must match the group's world size,
                and ``ep_rank`` must match this process's rank inside it.

        Returns:
            An :class:`ExpertParallelForwardResult` for this rank's local
            Tokens.
        """

        if tokens.ndim != 2 or tokens.shape[1] != self.hidden_size:
            raise ValueError(
                "tokens must have shape [num_local_tokens, hidden_size] with "
                f"hidden_size={self.hidden_size}, got {tuple(tokens.shape)}"
            )
        if tokens.device != self.router.gate.weight.device:
            raise ValueError(
                "tokens and expert weights must share a device, got "
                f"{tokens.device} and {self.router.gate.weight.device}"
            )

        with nvtx_range("ROUTER"):
            routing: RouterOutput = self.router(tokens)

        with nvtx_range("DISPATCH"):
            dispatch: DispatchOutput = dispatch_reference(
                tokens,
                routing.expert_ids,
                routing.routing_weights,
                self.num_experts,
            )
            rank_pack = pack_by_destination_rank(dispatch, self.topology)
            tokens_per_expert_all = all_gather_expert_counts(
                routing.tokens_per_expert,
                group=group,
            )
            layout = build_receive_layout(
                self.topology,
                tokens_per_expert_all,
                local_tokens_per_expert=routing.tokens_per_expert,
            )

        # 行数元数据必须在 Host 侧交给 collective 入口；这里显式取出，避免在
        # 被测路径中隐含一次未说明的设备同步。
        send_splits = rank_pack.send_splits.tolist()
        recv_splits = layout.recv_splits.tolist()

        with nvtx_range("A2A_FORWARD"):
            recv_tokens = variable_split_all_to_all(
                rank_pack.packed_tokens,
                send_splits,
                recv_splits,
                group=group,
            )

        with nvtx_range("EXPERT"):
            expert_outputs = torch.empty_like(recv_tokens)
            local_expert_rows = []
            for local_expert_id, rows in enumerate(layout.rows_by_expert):
                local_tokens = recv_tokens.index_select(0, rows)
                expert_outputs.index_copy_(
                    0,
                    rows,
                    self.experts[local_expert_id](local_tokens),
                )
                local_expert_rows.append(int(rows.numel()))
            if sum(local_expert_rows) != recv_tokens.shape[0]:
                raise RuntimeError(
                    "receive layout must cover every received row exactly once, "
                    f"got {sum(local_expert_rows)} rows for a buffer with "
                    f"{recv_tokens.shape[0]} rows"
                )

        with nvtx_range("A2A_RETURN"):
            returned = variable_split_all_to_all(
                expert_outputs,
                recv_splits,
                send_splits,
                group=group,
            )

        with nvtx_range("COMBINE"):
            # All-to-All #2 returns rows in exactly the rank-packed layout of
            # All-to-All #1, so Unpermute only has to undo the rank packing.
            expert_major_outputs = torch.empty_like(returned)
            expert_major_outputs.index_copy_(
                0,
                rank_pack.row_origin_index,
                returned,
            )
            hidden_states = combine_reference(expert_major_outputs, dispatch)

        return ExpertParallelForwardResult(
            hidden_states=hidden_states,
            router_logits=routing.logits,
            expert_ids=routing.expert_ids,
            routing_weights=routing.routing_weights,
            tokens_per_expert=routing.tokens_per_expert,
            tokens_per_expert_all=tokens_per_expert_all,
            send_splits=rank_pack.send_splits,
            recv_splits=layout.recv_splits,
            local_expert_rows=torch.tensor(
                local_expert_rows, dtype=torch.int64, device=tokens.device
            ),
        )


def distribute_single_gpu_moe(
    reference: SingleGPUMoE,
    *,
    ep_size: int,
    ep_rank: int,
) -> ExpertParallelMoE:
    """Slice a single-GPU reference model into this rank's EP shard.

    The router is replicated on every rank, and rank ``ep_rank`` receives the
    expert MLPs whose global IDs are
    ``[ep_rank * experts_per_rank, (ep_rank + 1) * experts_per_rank)``.  Weight
    values are copied from ``reference``, so a distributed run can be compared
    against the exact same single-GPU model.

    Args:
        reference: Model that owns all experts, typically built on the CPU with
            a fixed seed so that every rank derives identical weights.
        ep_size: Number of EP ranks that will run the shard.
        ep_rank: Rank that owns the returned shard.

    Returns:
        An :class:`ExpertParallelMoE` on the same device and dtype as
        ``reference``.
    """

    if not reference.experts:
        raise ValueError("reference model must own at least one expert")

    template = reference.experts[0]
    ep_model = ExpertParallelMoE(
        hidden_size=reference.hidden_size,
        expert_hidden_size=template.expert_hidden_size,
        num_experts=reference.num_experts,
        top_k=reference.router.top_k,
        ep_size=ep_size,
        ep_rank=ep_rank,
        activation=template.activation_name,
        bias=template.use_bias,
    )
    ep_model.router.load_state_dict(reference.router.state_dict())
    for local_expert_id, global_expert_id in enumerate(
        ep_model.topology.local_expert_ids
    ):
        ep_model.experts[local_expert_id].load_state_dict(
            reference.experts[global_expert_id].state_dict()
        )
    return ep_model.to(
        device=reference.router.gate.weight.device,
        dtype=reference.router.gate.weight.dtype,
    )
