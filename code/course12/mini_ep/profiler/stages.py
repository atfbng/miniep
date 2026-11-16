"""Stage ranges and CUDA Event samples for the synchronous EP baseline."""

from __future__ import annotations
from contextlib import contextmanager
from collections.abc import Iterator, Callable
import torch
import torch.distributed as dist
from mini_ep.profiler.nvtx import nvtx_range


class StageRecorder:
    def __init__(self, device: torch.device, timing: bool = False):
        self.device = torch.device(device)
        if timing and self.device.type != "cuda":
            raise ValueError("CUDA Event stage timing needs a CUDA device")
        self.timing = timing
        self.events: dict[str, list] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Label host scope; CUDA events bracket dependent current-stream work."""
        pair = None
        if self.timing:
            with torch.cuda.device(self.device):
                pair = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                pair[0].record()
        with torch.profiler.record_function(name), nvtx_range(name):
            yield
        if pair is not None:
            with torch.cuda.device(self.device):
                pair[1].record()
            self.events.setdefault(name, []).append(pair)

    def samples_ms(self) -> dict[str, list[float]]:
        """Call after the measurement boundary has synchronized this device."""
        return {
            name: [a.elapsed_time(b) for a, b in pairs]
            for name, pairs in self.events.items()
        }


from mini_ep.model.moe import ExpertParallelForwardResult, ExpertParallelMoE
from mini_ep.model.router import RouterOutput
from mini_ep.distributed import (
    DispatchOutput,
    dispatch_reference,
    pack_by_destination_rank,
    all_gather_expert_counts,
    build_receive_layout,
    variable_split_all_to_all,
    combine_reference,
)


def run_ep_stages(
    model: ExpertParallelMoE,
    tokens: torch.Tensor,
    recorder: StageRecorder,
    *,
    group: dist.ProcessGroup | None = None,
    dispatch_impl=dispatch_reference,
    combine_impl=combine_reference,
    expert_compute=None,
) -> ExpertParallelForwardResult:
    if tokens.ndim != 2 or tokens.shape[1] != model.hidden_size:
        raise ValueError(
            f"tokens must have shape [num_local_tokens, hidden_size] with hidden_size={model.hidden_size}, got {tuple(tokens.shape)}"
        )
    if tokens.device != model.router.gate.weight.device:
        raise ValueError(
            f"tokens and expert weights must share a device, got {tokens.device} and {model.router.gate.weight.device}"
        )
    with recorder.stage("ROUTER"):
        routing: RouterOutput = model.router(tokens)
    with recorder.stage("DISPATCH"):
        dispatch: DispatchOutput = dispatch_impl(
            tokens, routing.expert_ids, routing.routing_weights, model.num_experts
        )
        rank_pack = pack_by_destination_rank(dispatch, model.topology)
        tokens_per_expert_all = all_gather_expert_counts(
            routing.tokens_per_expert, group=group
        )
        layout = build_receive_layout(
            model.topology,
            tokens_per_expert_all,
            local_tokens_per_expert=routing.tokens_per_expert,
        )
    send_rows = rank_pack.send_splits.tolist()
    recv_rows = layout.recv_splits.tolist()
    with recorder.stage("A2A_FORWARD"):
        recv_tokens = variable_split_all_to_all(
            rank_pack.packed_tokens, send_rows, recv_rows, group=group
        )
    with recorder.stage("EXPERT"):
        if expert_compute is None:
            expert_output = torch.empty_like(recv_tokens)
            local_expert_rows = []
            for local_expert_id, rows in enumerate(layout.local_expert_index):
                local_tokens = recv_tokens.index_select(0, rows)
                expert_output.index_copy_(
                    0, rows, model.experts[local_expert_id](local_tokens)
                )
                local_expert_rows.append(int(rows.numel()))
        else:
            expert_output = expert_compute(model, recv_tokens, layout)
            if (
                expert_output.shape != recv_tokens.shape
                or expert_output.dtype != recv_tokens.dtype
                or expert_output.device != recv_tokens.device
            ):
                raise ValueError(
                    "Expert compute must preserve receive-buffer shape/dtype/device"
                )
            local_expert_rows = [
                int(rows.numel()) for rows in layout.local_expert_index
            ]
        if sum(local_expert_rows) != recv_tokens.shape[0]:
            raise RuntimeError(
                f"receive layout must cover every received row exactly once, got {sum(local_expert_rows)} rows for a buffer with {recv_tokens.shape[0]} rows"
            )
    with recorder.stage("A2A_RETURN"):
        returned = variable_split_all_to_all(
            expert_output, recv_rows, send_rows, group=group
        )
    with recorder.stage("COMBINE"):
        expert_major_outputs = torch.empty_like(returned)
        expert_major_outputs.index_copy_(0, rank_pack.row_origin_index, returned)
        hidden_states = combine_impl(expert_major_outputs, dispatch)
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
