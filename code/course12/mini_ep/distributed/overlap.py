"""Chunked Pure EP with asynchronous NCCL and explicit stream dependencies."""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.distributed as dist
from mini_ep.distributed import (
    dispatch_reference,
    pack_by_destination_rank,
    all_gather_expert_counts,
    build_receive_layout,
    combine_reference,
)
from mini_ep.profiler.nvtx import nvtx_range


@dataclass
class EPChunk:
    """One original Token interval; buffers remain owned until final drain.

    dispatch: Expert-major rows [N*K,H], rank_pack: destination-major rows.
    layout: source-major receive [R,H], int64 Expert/source indices.
    send/recv splits are host tuples in rows, one entry per EP rank.
    All payloads preserve the input floating dtype and CUDA device.
    """

    start: int
    end: int
    dispatch: object
    rank_pack: object
    layout: object
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    received: torch.Tensor
    expert_output: torch.Tensor
    returned: torch.Tensor


def prepare_ep_chunks(
    model, tokens: torch.Tensor, chunk_size: int, *, group=None
) -> list[EPChunk]:
    """Route once, then pack chunks; all ranks use the same chunk count/order."""
    if chunk_size <= 0 or tokens.ndim != 2 or not tokens.is_cuda:
        raise ValueError("chunk_size must be positive; tokens must be CUDA [N,H]")
    n = tokens.shape[0]
    sizes = torch.tensor([n, chunk_size], dtype=torch.int64, device=tokens.device)
    gathered = [torch.empty_like(sizes) for _ in range(dist.get_world_size(group))]
    dist.all_gather(gathered, sizes, group=group)
    if any(not torch.equal(sizes, peer) for peer in gathered):
        raise ValueError("all EP ranks must use equal local N and chunk_size")
    routing = model.router(tokens)
    chunks = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        dispatch = dispatch_reference(
            tokens[start:end],
            routing.expert_ids[start:end],
            routing.routing_weights[start:end],
            model.num_experts,
        )
        packed = pack_by_destination_rank(dispatch, model.topology)
        counts = all_gather_expert_counts(dispatch.tokens_per_expert, group=group)
        layout = build_receive_layout(
            model.topology, counts, local_tokens_per_expert=dispatch.tokens_per_expert
        )
        send, recv = tuple(packed.send_splits.tolist()), tuple(
            layout.recv_splits.tolist()
        )
        received = torch.empty(
            (sum(recv), tokens.shape[1]), dtype=tokens.dtype, device=tokens.device
        )
        chunks.append(
            EPChunk(
                start,
                end,
                dispatch,
                packed,
                layout,
                send,
                recv,
                received,
                torch.empty_like(received),
                torch.empty_like(packed.packed_tokens),
            )
        )
    return chunks


def launch_a2a(
    output: torch.Tensor,
    source: torch.Tensor,
    send: tuple[int, ...],
    recv: tuple[int, ...],
    *,
    group=None,
):
    """Enqueue on NCCL; Work.wait() connects completion to the consuming stream."""
    return dist.all_to_all_single(
        output,
        source,
        output_split_sizes=list(recv),
        input_split_sizes=list(send),
        group=group,
        async_op=True,
    )


def run_chunk_pipeline(
    model,
    tokens: torch.Tensor,
    chunks: list[EPChunk],
    *,
    group=None,
    compute_stream=None,
    comm_stream=None,
) -> torch.Tensor:
    """Execute prepared chunks, then join both streams onto the caller stream.

    Each chunk owns distinct buffers; no slot reuse occurs inside this route.
    At entry previous caller work has produced the input and weights. At exit
    caller stream waits for all returns/combinations; Host reads need a boundary
    synchronization. Preparation is measured separately or included by caller.
    """
    caller = torch.cuda.current_stream(tokens.device)
    compute = compute_stream or torch.cuda.Stream(device=tokens.device)
    comm = comm_stream or torch.cuda.Stream(device=tokens.device)
    comm.wait_stream(caller)
    compute.wait_stream(caller)
    result = torch.empty_like(tokens)
    work, returns, ready_events = {}, [], []

    def forward(index: int):
        c = chunks[index]
        with torch.cuda.stream(comm), nvtx_range(f"CHUNK_{index}_COMM"):
            work[index] = launch_a2a(
                c.received,
                c.rank_pack.packed_tokens,
                c.send_splits,
                c.recv_splits,
                group=group,
            )

    if chunks:
        forward(0)
    for index, c in enumerate(chunks):
        with torch.cuda.stream(compute), nvtx_range(f"CHUNK_{index}_COMPUTE"):
            work[index].wait()
            for expert, rows in zip(model.experts, c.layout.local_expert_index):
                c.expert_output.index_copy_(
                    0, rows, expert(c.received.index_select(0, rows))
                )
            ready = torch.cuda.Event()
            ready.record(compute)
            ready_events.append(ready)
        # Queue next receive while current Expert work can still be executing.
        if index + 1 < len(chunks):
            forward(index + 1)
        with torch.cuda.stream(comm), nvtx_range(f"CHUNK_{index}_RETURN"):
            comm.wait_event(ready)
            returned = launch_a2a(
                c.returned, c.expert_output, c.recv_splits, c.send_splits, group=group
            )
            returns.append(returned)
        with torch.cuda.stream(compute), nvtx_range(f"CHUNK_{index}_COMBINE"):
            returned.wait()
            expert_major = torch.empty_like(c.returned)
            expert_major.index_copy_(0, c.rank_pack.row_origin_index, c.returned)
            result[c.start : c.end].copy_(combine_reference(expert_major, c.dispatch))
    caller.wait_stream(comm)
    caller.wait_stream(compute)
    # The allocator sees every side-stream access, including temporary outputs.
    for c in chunks:
        for tensor in (
            c.received,
            c.expert_output,
            c.returned,
            c.rank_pack.packed_tokens,
        ):
            tensor.record_stream(comm)
            tensor.record_stream(compute)
    for c in chunks:
        for tensor in (
            *c.layout.local_expert_index,
            c.rank_pack.row_origin_index,
            c.dispatch.restore_index,
            c.dispatch.packed_routing_weights,
        ):
            tensor.record_stream(compute)
    result.record_stream(compute)
    return result
