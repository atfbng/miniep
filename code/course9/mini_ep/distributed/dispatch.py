"""Pure PyTorch token Dispatch and Combine reference implementation.

This module does not perform distributed communication.  It introduces the
contiguous, expert-major layout that later Expert Parallel lessons will pack
again by destination rank before ``all_to_all_single``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_ep.distributed.topology import ExpertParallelTopology


_INTEGER_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class DispatchOutput:
    """Tokens and metadata after stable expert-major packing.

    ``num_assignments = num_tokens * top_k``.  Every Tensor is on the same
    device as the input ``tokens``.

    Attributes:
        packed_tokens: Token rows in expert-major order, shape
            ``[num_assignments, hidden_size]``.  Top-K has been expanded, so
            one original Token appears ``top_k`` times.
        packed_expert_ids: Global expert ID for each packed row, shape
            ``[num_assignments]``.  IDs are non-decreasing.
        packed_token_indices: Original Token row for each packed row, shape
            ``[num_assignments]``.
        packed_route_slots: Original Top-K slot for each packed row, shape
            ``[num_assignments]``.
        packed_routing_weights: Combine weight for each packed row, shape
            ``[num_assignments]``.
        tokens_per_expert: Assignment count for each expert, shape
            ``[num_experts]``.  Counts are assignments, not unique Tokens.
        expert_offsets: Exclusive prefix sum of ``tokens_per_expert``, shape
            ``[num_experts + 1]``.  Expert ``e`` owns packed rows
            ``[expert_offsets[e], expert_offsets[e + 1])``.
        restore_index: Inverse packing permutation, shape
            ``[num_assignments]``.  ``packed.index_select(0, restore_index)``
            restores token-major, then route-slot-major assignment order.
        num_tokens: Number of original Token rows.
        top_k: Number of selected experts per Token.
    """

    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_token_indices: torch.Tensor
    packed_route_slots: torch.Tensor
    packed_routing_weights: torch.Tensor
    tokens_per_expert: torch.Tensor
    expert_offsets: torch.Tensor
    restore_index: torch.Tensor
    num_tokens: int
    top_k: int

    @property
    def num_assignments(self) -> int:
        """Return the number of expanded Top-K assignments."""

        return self.num_tokens * self.top_k


def _validate_dispatch_inputs(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> None:
    if tokens.ndim != 2:
        raise ValueError(
            "tokens must have shape [num_tokens, hidden_size], got "
            f"{tuple(tokens.shape)}"
        )
    if tokens.shape[1] <= 0:
        raise ValueError(f"hidden_size must be positive, got {tokens.shape[1]}")
    if not tokens.is_floating_point():
        raise TypeError(
            f"tokens must use a floating-point dtype, got {tokens.dtype}"
        )
    if expert_ids.ndim != 2:
        raise ValueError(
            "expert_ids must have shape [num_tokens, top_k], got "
            f"{tuple(expert_ids.shape)}"
        )
    if expert_ids.shape != routing_weights.shape:
        raise ValueError(
            "expert_ids and routing_weights must have the same shape, got "
            f"{tuple(expert_ids.shape)} and {tuple(routing_weights.shape)}"
        )
    if expert_ids.shape[0] != tokens.shape[0]:
        raise ValueError(
            "routing metadata must have one row per Token, got "
            f"{expert_ids.shape[0]} rows for {tokens.shape[0]} Tokens"
        )
    if expert_ids.shape[1] <= 0:
        raise ValueError(f"top_k must be positive, got {expert_ids.shape[1]}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if expert_ids.shape[1] > num_experts:
        raise ValueError(
            f"top_k ({expert_ids.shape[1]}) cannot exceed num_experts "
            f"({num_experts})"
        )
    if expert_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"expert_ids must use int32 or int64, got {expert_ids.dtype}")
    if not routing_weights.is_floating_point():
        raise TypeError(
            "routing_weights must use a floating-point dtype, got "
            f"{routing_weights.dtype}"
        )
    if (
        expert_ids.device != tokens.device
        or routing_weights.device != tokens.device
    ):
        raise ValueError("tokens and routing metadata must be on the same device")
    if expert_ids.numel() > 0:
        smallest_id = int(expert_ids.min().item())
        largest_id = int(expert_ids.max().item())
        if smallest_id < 0 or largest_id >= num_experts:
            raise ValueError(
                f"expert IDs must be in [0, {num_experts}), got range "
                f"[{smallest_id}, {largest_id}]"
            )


def dispatch_reference(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> DispatchOutput:
    """Expand Top-K assignments and pack Token rows by expert.

    Args:
        tokens: Original Token order, shape ``[num_tokens, hidden_size]``.
            Any floating dtype, device, stride, and contiguous state accepted
            by ``torch.index_select`` are supported.
        expert_ids: Global expert IDs in original Token order, shape
            ``[num_tokens, top_k]``, dtype int32 or int64.
        routing_weights: Weight paired with every ``expert_ids`` entry, shape
            ``[num_tokens, top_k]``, floating dtype.
        num_experts: Total expert count.  IDs must be in
            ``[0, num_experts)``.

    Returns:
        A :class:`DispatchOutput` whose ``packed_tokens`` is contiguous and in
        stable expert-major order.  Assignments routed to the same expert keep
        their original token-major, route-slot-major relative order.

    This is an allocation-based teaching Reference.  It deliberately does not
    implement capacity dropping, rank packing, NCCL communication, or a fused
    Triton/CUDA kernel.
    """

    _validate_dispatch_inputs(tokens, expert_ids, routing_weights, num_experts)

    num_tokens, hidden_size = tokens.shape
    top_k = expert_ids.shape[1]
    device = tokens.device

    # Flattening defines the canonical assignment order:
    # (Token 0, slot 0), (Token 0, slot 1), (Token 1, slot 0), ...
    flat_expert_ids = expert_ids.reshape(-1).to(torch.int64)
    flat_routing_weights = routing_weights.reshape(-1)
    flat_token_indices = torch.arange(
        num_tokens, device=device, dtype=torch.int64
    ).repeat_interleave(top_k)
    flat_route_slots = torch.arange(
        top_k, device=device, dtype=torch.int64
    ).repeat(num_tokens)

    # Stable sorting groups assignments by Expert without scrambling their
    # relative Token/slot order inside one Expert batch.
    pack_index = torch.argsort(flat_expert_ids, stable=True)
    restore_index = torch.argsort(pack_index)

    packed_token_indices = flat_token_indices.index_select(0, pack_index)
    packed_tokens = tokens.index_select(0, packed_token_indices).contiguous()
    packed_expert_ids = flat_expert_ids.index_select(0, pack_index)
    packed_route_slots = flat_route_slots.index_select(0, pack_index)
    packed_routing_weights = flat_routing_weights.index_select(0, pack_index)

    tokens_per_expert = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_offsets = torch.empty(
        num_experts + 1, device=device, dtype=torch.int64
    )
    expert_offsets[0] = 0
    torch.cumsum(tokens_per_expert, dim=0, out=expert_offsets[1:])

    # Keep the empty-input shape explicit for readers and future kernel tests.
    packed_tokens = packed_tokens.reshape(-1, hidden_size)

    return DispatchOutput(
        packed_tokens=packed_tokens,
        packed_expert_ids=packed_expert_ids,
        packed_token_indices=packed_token_indices,
        packed_route_slots=packed_route_slots,
        packed_routing_weights=packed_routing_weights,
        tokens_per_expert=tokens_per_expert,
        expert_offsets=expert_offsets,
        restore_index=restore_index,
        num_tokens=num_tokens,
        top_k=top_k,
    )


def combine_reference(
    packed_expert_outputs: torch.Tensor,
    dispatch: DispatchOutput,
) -> torch.Tensor:
    """Restore Token order and sum weighted Top-K expert outputs.

    Args:
        packed_expert_outputs: Expert results in exactly the expert-major row
            order described by ``dispatch``, shape
            ``[num_assignments, hidden_size]``.  The Tensor may be strided.
        dispatch: Metadata returned by :func:`dispatch_reference`.

    Returns:
        Combined outputs in original Token order, shape
        ``[num_tokens, hidden_size]``.  The output dtype and device match
        ``packed_expert_outputs``.  Routing weights are cast to the expert
        output dtype immediately before multiplication.
    """

    if packed_expert_outputs.ndim != 2:
        raise ValueError(
            "packed_expert_outputs must have shape "
            "[num_assignments, hidden_size], got "
            f"{tuple(packed_expert_outputs.shape)}"
        )
    if packed_expert_outputs.shape != dispatch.packed_tokens.shape:
        raise ValueError(
            "packed_expert_outputs must match packed_tokens shape, got "
            f"{tuple(packed_expert_outputs.shape)} and "
            f"{tuple(dispatch.packed_tokens.shape)}"
        )
    if packed_expert_outputs.device != dispatch.packed_tokens.device:
        raise ValueError(
            "packed_expert_outputs and Dispatch metadata must share a device"
        )

    hidden_size = packed_expert_outputs.shape[1]
    token_major_outputs = packed_expert_outputs.index_select(
        0, dispatch.restore_index
    ).reshape(dispatch.num_tokens, dispatch.top_k, hidden_size)
    token_major_weights = dispatch.packed_routing_weights.index_select(
        0, dispatch.restore_index
    ).reshape(dispatch.num_tokens, dispatch.top_k, 1)

    return (
        token_major_outputs
        * token_major_weights.to(dtype=packed_expert_outputs.dtype)
    ).sum(dim=1)


@dataclass(frozen=True)
class RankPackedTensors:
    """Rank-packed rows that a single All-to-All #1 call sends.

    ``num_send_rows = num_assignments`` for a complete exchange: every packed
    row belongs to exactly one expert and therefore to exactly one destination
    rank.  All Tensors are on the device of the source ``DispatchOutput``.

    Attributes:
        packed_tokens: Token rows in rank-packed order, shape
            ``[num_send_rows, hidden_size]``, contiguous.  Rows are grouped by
            destination rank in ascending order; inside one destination group
            they keep the expert-major order produced by
            :func:`dispatch_reference`.
        packed_expert_ids: Global expert ID of each sent row, shape
            ``[num_send_rows]``.
        packed_token_indices: Original Token row of each sent row, shape
            ``[num_send_rows]``.  It is not part of the exchanged payload;
            the origin rank keeps it locally.
        packed_route_slots: Original Top-K slot of each sent row, shape
            ``[num_send_rows]``.
        packed_routing_weights: Combine weight of each sent row, shape
            ``[num_send_rows]``.  Also kept on the origin rank.
        send_splits: Rows sent to each destination rank, shape ``[ep_size]``
            ``int64``.  This is the ``input_split_sizes`` argument of
            All-to-All #1.
        send_offsets: Exclusive prefix sum of ``send_splits``, shape
            ``[ep_size + 1]``.
        row_origin_index: Position of each sent row in the expert-major order
            of the source ``DispatchOutput``, shape ``[num_send_rows]``.  It is
            a permutation used by Unpermute after All-to-All #2 has returned
            the rows in exactly this layout.
    """

    packed_tokens: torch.Tensor
    packed_expert_ids: torch.Tensor
    packed_token_indices: torch.Tensor
    packed_route_slots: torch.Tensor
    packed_routing_weights: torch.Tensor
    send_splits: torch.Tensor
    send_offsets: torch.Tensor
    row_origin_index: torch.Tensor

    @property
    def num_send_rows(self) -> int:
        """Return how many rows this rank sends in All-to-All #1."""

        return int(self.packed_tokens.shape[0])


def pack_by_destination_rank(
    dispatch: DispatchOutput,
    topology: ExpertParallelTopology,
) -> RankPackedTensors:
    """Reorder expert-major rows into rank-packed rows.

    Args:
        dispatch: Output of :func:`dispatch_reference`.  ``packed_tokens`` is
            in expert-major order, so ``packed_expert_ids`` is non-decreasing.
        topology: Uniform placement that maps every expert ID to a rank.

    Returns:
        A :class:`RankPackedTensors` whose ``packed_tokens`` can be handed to
        :func:`mini_ep.distributed.variable_split_all_to_all` together with
        ``send_splits``.

    With uniform placement ``destination_rank = expert_id // experts_per_rank``
    is non-decreasing in ``expert_id``, so a stable sort by destination rank
    keeps the input order inside every group and ``row_origin_index`` is the
    identity permutation.  The sort is still applied explicitly: it keeps the
    function correct for any placement strategy that maps experts to ranks in a
    different order, and it makes the packing step visible in the profile.

    Raises:
        ValueError: If the Dispatch metadata was produced for a different
            expert count than the topology describes.
    """

    if dispatch.tokens_per_expert.numel() != topology.num_experts:
        raise ValueError(
            "Dispatch metadata and topology disagree on the expert count: "
            f"{dispatch.tokens_per_expert.numel()} vs {topology.num_experts}"
        )

    destination_rank = topology.destination_rank(dispatch.packed_expert_ids)
    row_origin_index = torch.argsort(destination_rank, stable=True).contiguous()

    send_splits = torch.zeros(
        topology.ep_size,
        dtype=torch.int64,
        device=destination_rank.device,
    )
    send_splits.scatter_add_(
        0,
        destination_rank,
        torch.ones_like(destination_rank, dtype=torch.int64),
    )
    send_splits = send_splits.contiguous()

    send_offsets = torch.zeros(
        topology.ep_size + 1,
        dtype=torch.int64,
        device=destination_rank.device,
    )
    torch.cumsum(send_splits, dim=0, out=send_offsets[1:])

    return RankPackedTensors(
        packed_tokens=dispatch.packed_tokens.index_select(
            0, row_origin_index
        ).contiguous(),
        packed_expert_ids=dispatch.packed_expert_ids.index_select(
            0, row_origin_index
        ),
        packed_token_indices=dispatch.packed_token_indices.index_select(
            0, row_origin_index
        ),
        packed_route_slots=dispatch.packed_route_slots.index_select(
            0, row_origin_index
        ),
        packed_routing_weights=dispatch.packed_routing_weights.index_select(
            0, row_origin_index
        ),
        send_splits=send_splits,
        send_offsets=send_offsets,
        row_origin_index=row_origin_index,
    )
