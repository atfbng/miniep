"""Expert placement and rank-layout math for Mini-EP Expert Parallel lessons.

Lesson 6 uses **uniform placement**: every EP rank owns
``experts_per_rank = num_experts // ep_size`` consecutive experts.  The two
formulas that connect an expert ID with a rank are kept here so that Dispatch,
All-to-All and the MoE orchestration never repeat them::

    destination_rank = expert_id // experts_per_rank
    local_expert_id  = expert_id %  experts_per_rank

The module is import-safe: it performs no communication and allocates no large
Tensor.  Rank-layout helpers only reshape and index the counts that the caller
already exchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mini_ep.distributed.groups import DistributedContext


_INTEGER_DTYPES = (torch.int32, torch.int64)


@dataclass(frozen=True)
class ExpertParallelTopology:
    """Uniform Expert-to-Rank placement inside one EP group.

    Attributes:
        num_experts: Total number of experts in the model, counted over the
            whole EP group.
        ep_size: Number of ranks that participate in EP collectives.
        ep_rank: This process's index inside the EP group.  Lessons 4 to 6 use
            the default Process Group as the EP group, so ``ep_rank`` equals
            the global rank and ``ep_size`` equals the world size.

    Raises:
        ValueError: If a count is non-positive, if ``ep_rank`` is outside the
            group, or if uniform placement is impossible because
            ``num_experts % ep_size != 0``.
    """

    num_experts: int
    ep_size: int
    ep_rank: int

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError(
                f"num_experts must be positive, got {self.num_experts}"
            )
        if self.ep_size <= 0:
            raise ValueError(f"ep_size must be positive, got {self.ep_size}")
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                "uniform expert placement requires num_experts % ep_size == 0, "
                f"got num_experts={self.num_experts} and ep_size={self.ep_size}"
            )
        if not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}"
            )

    @property
    def experts_per_rank(self) -> int:
        """Return how many experts each EP rank owns."""

        return self.num_experts // self.ep_size

    @property
    def num_local_experts(self) -> int:
        """Return the number of experts this rank owns."""

        return self.experts_per_rank

    @property
    def local_expert_ids(self) -> tuple[int, ...]:
        """Return the global expert IDs owned by this rank, in local order."""

        first = self.ep_rank * self.experts_per_rank
        return tuple(range(first, first + self.experts_per_rank))

    def global_expert_id(self, local_expert_id: int) -> int:
        """Map this rank's local expert index to its global expert ID."""

        if not 0 <= local_expert_id < self.experts_per_rank:
            raise ValueError(
                "local_expert_id must be in "
                f"[0, {self.experts_per_rank}), got {local_expert_id}"
            )
        return self.ep_rank * self.experts_per_rank + local_expert_id

    def destination_rank(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Return the destination rank of every global expert ID.

        Args:
            expert_ids: Integer Tensor of global expert IDs, any shape.

        Returns:
            An ``int64`` Tensor with the same shape, on the same device.  Each
            element is in ``[0, ep_size)``.  Callers are expected to have
            validated the IDs against ``num_experts`` already.
        """

        _require_integer(expert_ids, "expert_ids")
        return torch.div(
            expert_ids.to(torch.int64),
            self.experts_per_rank,
            rounding_mode="floor",
        )

    def local_expert_id(self, expert_ids: torch.Tensor) -> torch.Tensor:
        """Return this rank's local expert index for global expert IDs.

        The result is well defined for every ID; an ID that belongs to another
        rank simply produces the local index the same formula would use, which
        is why callers filter by :meth:`destination_rank` first.
        """

        _require_integer(expert_ids, "expert_ids")
        return expert_ids.to(torch.int64) % self.experts_per_rank

    @classmethod
    def from_context(
        cls,
        context: DistributedContext,
        num_experts: int,
    ) -> ExpertParallelTopology:
        """Build the lesson-6 topology from an initialized Process Group.

        The default Process Group doubles as the EP group, so ``ep_size`` is
        the world size and ``ep_rank`` is the global rank.  Dedicated EP
        subgroups (``ep_size`` smaller than the world size) are introduced in
        the scaling lesson.
        """

        return cls(
            num_experts=num_experts,
            ep_size=context.world_size,
            ep_rank=context.global_rank,
        )


def _require_integer(tensor: torch.Tensor, name: str) -> None:
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use int32 or int64, got {tensor.dtype}")


def compute_send_splits(
    topology: ExpertParallelTopology,
    tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    """Count how many rows this rank sends to each destination rank.

    Args:
        topology: Placement used to group experts by destination rank.
        tokens_per_expert: Assignment counts in global expert order, shape
            ``[num_experts]``, integer dtype, on any device.  These are
            assignment counts (each Token contributes ``top_k`` entries), not
            unique Token counts.

    Returns:
        A contiguous ``int64`` Tensor with shape ``[ep_size]`` on the same
        device.  Entry ``d`` is the number of packed rows that must be sent to
        destination rank ``d``.
    """

    if tokens_per_expert.ndim != 1:
        raise ValueError(
            "tokens_per_expert must have shape [num_experts], got "
            f"{tuple(tokens_per_expert.shape)}"
        )
    if tokens_per_expert.numel() != topology.num_experts:
        raise ValueError(
            f"tokens_per_expert must have {topology.num_experts} entries, got "
            f"{tokens_per_expert.numel()}"
        )
    _require_integer(tokens_per_expert, "tokens_per_expert")

    per_rank = tokens_per_expert.to(torch.int64).reshape(
        topology.ep_size, topology.experts_per_rank
    )
    return per_rank.sum(dim=1).contiguous()


@dataclass(frozen=True)
class ReceiveLayout:
    """Structure of the buffer that this rank receives from All-to-All #1.

    A rank-packed send buffer is grouped by destination rank, and inside one
    destination group the rows keep expert-major order.  After the exchange the
    received buffer is therefore grouped by *source* rank, and inside one
    source group the rows are grouped by the global expert ID of the receiving
    rank.  Because uniform placement gives every rank a contiguous range of
    global expert IDs, one local expert owns exactly ``ep_size`` contiguous
    blocks: one per source rank, in ascending source order.

    Attributes:
        recv_splits: Rows received from each source rank, shape ``[ep_size]``
            ``int64``, on the same device as the exchanged counts.
        recv_offsets: Exclusive prefix sum of ``recv_splits``, shape
            ``[ep_size + 1]``.  Rows from source rank ``s`` occupy
            ``[recv_offsets[s], recv_offsets[s + 1])``.
        local_counts: Rows per (source rank, local expert),
            shape ``[ep_size, num_local_experts]``, ``int64``.
        rows_by_expert: One ``int64`` index Tensor per local expert, in
            local expert order.  The Tensors partition
            ``range(sum(recv_splits))``: together they select every received
            row exactly once, so ``recv_tokens.index_select(0, rows)`` is that
            expert's local batch.
    """

    recv_splits: torch.Tensor
    recv_offsets: torch.Tensor
    local_counts: torch.Tensor
    rows_by_expert: tuple[torch.Tensor, ...]

    @property
    def num_recv_rows(self) -> int:
        """Return how many rows All-to-All #1 delivers to this rank."""

        return int(self.recv_splits.sum().item())


def build_receive_layout(
    topology: ExpertParallelTopology,
    tokens_per_expert_all: torch.Tensor,
    *,
    local_tokens_per_expert: torch.Tensor | None = None,
) -> ReceiveLayout:
    """Derive the received buffer layout from every rank's expert counts.

    Args:
        topology: Placement used to decide which experts belong to this rank.
        tokens_per_expert_all: Counts gathered from all EP ranks, shape
            ``[ep_size, num_experts]``, integer dtype.  Row ``s`` is rank
            ``s``'s ``tokens_per_expert`` in global expert order.
        local_tokens_per_expert: Optional cross-check Tensor with shape
            ``[num_experts]``.  When given it must equal row ``ep_rank`` of
            ``tokens_per_expert_all``; this catches a rank that gathered counts
            from a different routing result than the one it dispatched.

    Returns:
        A :class:`ReceiveLayout` describing the splits and the per-expert row
        index for All-to-All #1.

    Raises:
        ValueError: If shapes or dtypes are wrong, if any count is negative, or
            if the optional cross-check fails.
    """

    if tokens_per_expert_all.ndim != 2:
        raise ValueError(
            "tokens_per_expert_all must have shape [ep_size, num_experts], got "
            f"{tuple(tokens_per_expert_all.shape)}"
        )
    if tokens_per_expert_all.shape != (
        topology.ep_size,
        topology.num_experts,
    ):
        raise ValueError(
            "tokens_per_expert_all must have shape "
            f"[{topology.ep_size}, {topology.num_experts}], got "
            f"{tuple(tokens_per_expert_all.shape)}"
        )
    _require_integer(tokens_per_expert_all, "tokens_per_expert_all")
    if bool((tokens_per_expert_all < 0).any()):
        raise ValueError("tokens_per_expert_all must not contain negative counts")

    device = tokens_per_expert_all.device
    counts = tokens_per_expert_all.to(torch.int64)

    if local_tokens_per_expert is not None:
        local_row = counts[topology.ep_rank]
        expected = local_tokens_per_expert.to(torch.int64).to(device)
        if local_row.shape != expected.shape or not torch.equal(local_row, expected):
            raise ValueError(
                "exchanged counts for this rank do not match the local routing "
                "counts; every rank must gather the counts of the same routing "
                "result it dispatched"
            )

    local_ids = torch.tensor(
        topology.local_expert_ids, dtype=torch.int64, device=device
    )
    local_counts = counts.index_select(1, local_ids).contiguous()
    recv_splits = local_counts.sum(dim=1).contiguous()

    recv_offsets = torch.zeros(
        topology.ep_size + 1, dtype=torch.int64, device=device
    )
    torch.cumsum(recv_splits, dim=0, out=recv_offsets[1:])

    # Inside one source group the block of a local expert starts after all
    # smaller local expert IDs of the same source, because the sender packed
    # rows in ascending global expert ID order.
    rows_by_expert: list[torch.Tensor] = []
    for local_id in range(topology.num_local_experts):
        blocks: list[torch.Tensor] = []
        for source_rank in range(topology.ep_size):
            prefix = int(local_counts[source_rank, :local_id].sum().item())
            start = int(recv_offsets[source_rank].item()) + prefix
            count = int(local_counts[source_rank, local_id].item())
            blocks.append(
                torch.arange(start, start + count, dtype=torch.int64, device=device)
            )
        rows_by_expert.append(
            torch.cat(blocks)
            if blocks
            else torch.empty(0, dtype=torch.int64, device=device)
        )

    return ReceiveLayout(
        recv_splits=recv_splits,
        recv_offsets=recv_offsets,
        local_counts=local_counts,
        rows_by_expert=tuple(rows_by_expert),
    )
