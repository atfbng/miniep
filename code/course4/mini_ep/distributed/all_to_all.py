"""All-to-All wrappers and the expert-count exchange used by EP Dispatch.

Dim 0 of every buffer is the row axis; all remaining dimensions are the
per-row payload.  Split sizes are expressed in **rows** (dim 0), which is what
``torch.distributed.all_to_all_single`` expects for
``input_split_sizes`` / ``output_split_sizes``.

Lesson 4 uses the equal-split wrapper only.  The variable-split wrapper and the
count exchange belong to the lesson-5/6 EP data path.  Asynchronous submission,
CUDA Streams, and communication benchmarks belong to later lessons; every
wrapper here is synchronous and returns no Work handle.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.distributed as dist


_INTEGER_DTYPES = (torch.int32, torch.int64)


def _split_sizes_as_int_list(
    splits: torch.Tensor | Sequence[int],
    name: str,
    world_size: int,
) -> list[int]:
    """Validate one split vector and return it as a Python list of rows."""

    if isinstance(splits, torch.Tensor):
        if splits.dtype not in _INTEGER_DTYPES:
            raise TypeError(
                f"{name} must use int32 or int64, got {splits.dtype}"
            )
        if splits.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D with one entry per rank, got shape "
                f"{tuple(splits.shape)}"
            )
        # Split sizes are host-side metadata for the collective entry point,
        # so the device Tensor is copied to the host here, once per call.
        values = splits.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        values = list(splits)

    if len(values) != world_size:
        raise ValueError(
            f"{name} must have one entry per Process-Group rank "
            f"({world_size}), got {len(values)}"
        )

    resolved: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} entries must be integers, got {value!r}")
        if float(value) != int(value):
            raise TypeError(f"{name} entries must be whole numbers, got {value!r}")
        if int(value) < 0:
            raise ValueError(
                f"{name} entries must be non-negative, got {value!r}"
            )
        resolved.append(int(value))
    return resolved


def _require_send_buffer(send_buffer: torch.Tensor, name: str) -> None:
    if send_buffer.ndim == 0:
        raise ValueError(
            f"{name} must have shape [rows, ...], got a scalar Tensor"
        )
    if not send_buffer.is_contiguous():
        raise ValueError(f"{name} must be contiguous and rank-packed")


def equal_split_all_to_all(
    send_buffer: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Exchange equal-size dim-0 chunks with every Process-Group rank.

    Args:
        send_buffer: Contiguous Tensor with shape ``[rows, ...]``.  Rows are
            rank-packed: chunk ``r`` is sent to destination rank ``r``.
            ``rows`` must be divisible by the Process-Group size.  dtype and
            device are preserved.
        group: Participating Process Group.  ``None`` selects the default
            Process Group.  The group's backend determines the required
            Tensor device; lesson 4 uses NCCL and therefore CUDA Tensors.

    Returns:
        A newly allocated contiguous Tensor with the same shape, dtype, and
        device.  Received chunks are concatenated in source-rank order, so
        chunk ``s`` came from source rank ``s``.

    This function uses the synchronous API form and returns no Work handle.
    CUDA completion still follows the Process Group's stream semantics; the
    lesson example establishes an explicit device synchronization before Host
    reporting.  This wrapper intentionally exposes neither variable splits
    nor async_op.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "a torch.distributed Process Group must be initialized first"
        )
    _require_send_buffer(send_buffer, "send_buffer")

    world_size = dist.get_world_size(group)
    if send_buffer.shape[0] % world_size != 0:
        raise ValueError(
            "send_buffer dim 0 must divide evenly by Process-Group size, got "
            f"{send_buffer.shape[0]} rows and world_size={world_size}"
        )

    recv_buffer = torch.empty_like(send_buffer)
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        group=group,
        async_op=False,
    )
    return recv_buffer


def variable_split_all_to_all(
    send_buffer: torch.Tensor,
    send_splits: torch.Tensor | Sequence[int],
    recv_splits: torch.Tensor | Sequence[int],
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Exchange variable-size dim-0 chunks with every Process-Group rank.

    Args:
        send_buffer: Contiguous Tensor with shape ``[num_send_rows, ...]`` in
            rank-packed order: rows are grouped by destination rank in
            ascending rank order, so destination ``d`` owns rows
            ``[sum(send_splits[:d]), sum(send_splits[:d + 1]))``.  dtype and
            device are preserved.
        send_splits: Row count this rank sends to each destination rank, one
            entry per Process-Group rank.  ``sum(send_splits)`` must equal
            ``num_send_rows``.  This vector must agree with what every peer
            expects in its own ``recv_splits``.
        recv_splits: Row count this rank receives from each source rank, one
            entry per Process-Group rank.  A vector that disagrees with the
            peers' sends makes the collective hang or corrupt data, so it must
            be derived from the same exchanged counts on every rank.
        group: Participating Process Group.  ``None`` selects the default
            Process Group.

    Returns:
        A newly allocated contiguous Tensor with shape
        ``[sum(recv_splits), *send_buffer.shape[1:]]``, the same dtype and
        device as ``send_buffer``, and received chunks concatenated in
        ascending source-rank order.

    The call is synchronous (``async_op=False``) and therefore returns no Work
    handle; CUDA completion still follows the Process Group's stream
    semantics, so a device synchronization is needed only when the Host reads
    the values back.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "a torch.distributed Process Group must be initialized first"
        )
    _require_send_buffer(send_buffer, "send_buffer")

    world_size = dist.get_world_size(group)
    send_rows = _split_sizes_as_int_list(send_splits, "send_splits", world_size)
    recv_rows = _split_sizes_as_int_list(recv_splits, "recv_splits", world_size)
    if sum(send_rows) != send_buffer.shape[0]:
        raise ValueError(
            "send_splits must sum to send_buffer.shape[0], got "
            f"{sum(send_rows)} rows for a buffer with {send_buffer.shape[0]} rows"
        )

    recv_buffer = torch.empty(
        (sum(recv_rows), *send_buffer.shape[1:]),
        dtype=send_buffer.dtype,
        device=send_buffer.device,
    )
    dist.all_to_all_single(
        recv_buffer,
        send_buffer,
        output_split_sizes=recv_rows,
        input_split_sizes=send_rows,
        group=group,
        async_op=False,
    )
    return recv_buffer


def all_gather_expert_counts(
    tokens_per_expert: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Gather every rank's per-expert assignment counts.

    Expert Parallel Dispatch has to exchange counts *before* it exchanges
    Tokens: a sender knows which expert each packed row belongs to, but a
    receiver can only size and slice its receive buffer if it knows how many
    rows each source rank sent to each of its local experts.

    Args:
        tokens_per_expert: This rank's counts in global expert order, shape
            ``[num_experts]``, integer dtype.  The Tensor must already live on
            the device the Process Group uses.
        group: Participating Process Group.  ``None`` selects the default
            Process Group.

    Returns:
        A contiguous ``int64`` Tensor with shape ``[world_size, num_experts]``
        on the input's device.  Row ``s`` is source rank ``s``'s counts.
    """

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "a torch.distributed Process Group must be initialized first"
        )
    if tokens_per_expert.ndim != 1:
        raise ValueError(
            "tokens_per_expert must have shape [num_experts], got "
            f"{tuple(tokens_per_expert.shape)}"
        )
    if tokens_per_expert.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            "tokens_per_expert must use int32 or int64, got "
            f"{tokens_per_expert.dtype}"
        )

    world_size = dist.get_world_size(group)
    counts = tokens_per_expert.detach().to(torch.int64).contiguous()
    # ``all_gather_into_tensor`` fills a flat buffer on every backend, so the
    # rank-by-expert matrix is built by reshaping that buffer afterwards.
    flattened = torch.empty(
        world_size * counts.numel(),
        dtype=torch.int64,
        device=counts.device,
    )
    dist.all_gather_into_tensor(flattened, counts, group=group, async_op=False)
    return flattened.view(world_size, counts.numel())
