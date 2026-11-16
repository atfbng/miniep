"""Same-input prepared NCCL execution: ordinary and two-stream chunk pipeline.

Read inputs exported by large_expert.cu. Both routes use FP16 Up/GELU/Down
and FP32 weighted combine. Route/pack/count exchange is outside timing.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import argparse
import json
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


@dataclass
class PreparedChunk:
    """Contiguous CUDA FP16 payloads; splits are token rows in EP-rank order.

    packed: [M*K,H] destination-major; received: [R,H] source-major.
    returned: [M*K,H] expert-major; inverse restores q=token*K+slot.
    ready is recorded after Expert compute and waited by the return launcher.
    local_expert_rows: Q int64 CUDA index arrays into source-major received rows.
    send_expert_rows: [P,Q] destination/local-expert counts; recv_expert_rows:
    [P,Q] source/local-expert counts. Global Expert e maps to rank e//Q.
    Buffers/events are reused only after the previous invocation has drained.
    """
    start: int
    end: int
    packed: torch.Tensor
    received: torch.Tensor
    returned: torch.Tensor
    inverse: torch.Tensor
    send_splits: list[int]
    recv_splits: list[int]
    ready: torch.cuda.Event
    local_expert_rows: tuple[torch.Tensor, ...]
    send_expert_rows: list[int]
    recv_expert_rows: list[int]


def prepare_chunks(x: torch.Tensor, routes: torch.Tensor, size: int, num_experts: int | None = None) -> list[PreparedChunk]:
    """Route IDs [N,K] are fixed; prepare destination packing and NCCL splits."""
    if size < 1:
        raise ValueError(f'chunk size must be positive, got {size}')
    chunks = []
    p, k = dist.get_world_size(), routes.shape[1]
    num_experts = num_experts or p
    if num_experts % p:
        raise ValueError(f'num_experts={num_experts} must be divisible by ep_size={p}')
    q = num_experts // p
    for start in range(0, len(x), size):
        end = min(start + size, len(x))
        ids = routes[start:end].flatten()
        order = torch.argsort(ids, stable=True)
        send = torch.bincount(ids, minlength=num_experts)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send)
        send_experts, recv_experts = send.tolist(), recv.tolist()
        send_splits = send.reshape(p, q).sum(dim=1).tolist()
        recv_splits = recv.reshape(p, q).sum(dim=1).tolist()
        local_rows = [[] for _ in range(q)]
        offset = 0
        for source in range(p):
            for expert in range(q):
                count = recv_experts[source*q + expert]
                local_rows[expert].extend(range(offset, offset + count))
                offset += count
        indices = tuple(torch.tensor(rows, dtype=torch.long, device=x.device) for rows in local_rows)
        packed = x[start:end][order // k].contiguous()
        chunks.append(PreparedChunk(
            start, end, packed,
            torch.empty((sum(recv_splits), x.shape[1]), device=x.device, dtype=x.dtype),
            torch.empty_like(packed), torch.argsort(order), send_splits, recv_splits,
            torch.cuda.Event(), indices, send_experts, recv_experts,
        ))
    return chunks


def expert_mlp(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
               local_expert_rows: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """FP16 inputs [R,H], weights [Q,F,H]/[Q,H,F], output [R,H].

    Rows stay source-major across communication. Each local Expert gathers its
    rows, computes Up/GELU/Down, then scatters to the same receive-row positions.
    """
    if len(up) == 1:
        return F.gelu(x @ up[0].T) @ down[0].T
    output = torch.empty_like(x)
    for expert, rows in enumerate(local_expert_rows):
        tokens = x.index_select(0, rows)
        values = F.gelu(tokens @ up[expert].T) @ down[expert].T
        output.index_copy_(0, rows, values)
    return output


def expert_reference(x: torch.Tensor, up: torch.Tensor, down: torch.Tensor,
                     local_expert_rows: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Independent FP32 GEMMs with explicit FP16 rounding at MLP boundaries."""
    output = torch.empty_like(x)
    for expert, rows in enumerate(local_expert_rows):
        tokens = x.index_select(0, rows).float()
        hidden = (tokens @ up[expert].float().T).half().float()
        hidden = F.gelu(hidden).half().float()
        values = (hidden @ down[expert].float().T).half()
        output.index_copy_(0, rows, values)
    return output


def combine(chunk: PreparedChunk, top_k: int) -> torch.Tensor:
    """Restore original expanded order, then weight/reduce in FP32."""
    return chunk.returned[chunk.inverse].float().reshape(
        chunk.end - chunk.start, top_k, -1
    ).sum(dim=1) / top_k


def run_ordinary(chunk: PreparedChunk, up: torch.Tensor, down: torch.Tensor,
                 top_k: int) -> torch.Tensor:
    """Whole-batch synchronous NCCL -> MLP -> return NCCL -> Combine."""
    dist.all_to_all_single(chunk.received, chunk.packed, chunk.recv_splits, chunk.send_splits)
    computed = expert_mlp(chunk.received, up, down, chunk.local_expert_rows)
    dist.all_to_all_single(chunk.returned, computed, chunk.send_splits, chunk.recv_splits)
    return combine(chunk, top_k)


def run_stream_pipeline(chunks: list[PreparedChunk], up: torch.Tensor,
                        down: torch.Tensor, top_k: int, compute: torch.cuda.Stream,
                        comm: torch.cuda.Stream) -> torch.Tensor:
    """Two-stream pipeline; all ranks issue F0,F1,R0,F2,R1,... in that order.

    NCCL uses its internal stream; Work.wait() establishes dependencies on the
    current consumer stream. Return i can overlap Expert i+1. Combine is drained
    after all Expert work, avoiding a return wait between consecutive GEMMs.
    Each chunk owns separate buffers; no in-flight slot is overwritten.
    """
    caller = torch.cuda.current_stream(up.device)
    compute.wait_stream(caller)
    comm.wait_stream(caller)
    forwards, returns, computed_outputs = {}, [], []
    result = torch.empty((chunks[-1].end, up.shape[-1]), dtype=torch.float32, device=up.device)

    def launch_forward(index: int) -> None:
        c = chunks[index]
        with torch.cuda.stream(comm), torch.cuda.nvtx.range(f'CHUNK_{index}_COMM'):
            forwards[index] = dist.all_to_all_single(
                c.received, c.packed, c.recv_splits, c.send_splits, async_op=True
            )

    launch_forward(0)
    for index, c in enumerate(chunks):
        with torch.cuda.stream(compute), torch.cuda.nvtx.range(f'CHUNK_{index}_COMPUTE'):
            forwards[index].wait()
            computed = expert_mlp(c.received, up, down, c.local_expert_rows)
            computed_outputs.append(computed)  # Own source storage until drain.
            c.ready.record(compute)
        if index + 1 < len(chunks):
            launch_forward(index + 1)
        with torch.cuda.stream(comm), torch.cuda.nvtx.range(f'CHUNK_{index}_RETURN'):
            comm.wait_event(c.ready)
            returns.append(dist.all_to_all_single(
                c.returned, computed, c.send_splits, c.recv_splits, async_op=True
            ))
            computed.record_stream(comm)

    for index, (c, work) in enumerate(zip(chunks, returns)):
        with torch.cuda.stream(compute), torch.cuda.nvtx.range(f'CHUNK_{index}_COMBINE'):
            work.wait()
            result[c.start:c.end].copy_(combine(c, top_k))
    caller.wait_stream(comm)
    caller.wait_stream(compute)
    for c in chunks:
        for tensor in (c.packed, c.received, c.returned):
            tensor.record_stream(comm)
            tensor.record_stream(compute)
        c.inverse.record_stream(compute)
        for rows in c.local_expert_rows:
            rows.record_stream(compute)
    for tensor in computed_outputs:
        tensor.record_stream(compute)
    result.record_stream(compute)
    return result


def measure(function, warmup: int, repeats: int) -> list[float]:
    """CUDA Event timing after caller joins both streams; MAX rank per sample."""
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for iteration in range(warmup + repeats):
        torch.cuda.synchronize()
        dist.barrier()
        begin.record()
        function()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(begin.elapsed_time(end), device='cuda')
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if iteration >= warmup:
            samples.append(elapsed.item())
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--chunk-sizes', type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--profile-chunk', type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1 or any(c < 1 for c in args.chunk_sizes):
        raise ValueError('warmup, repeats and chunk sizes must be positive')
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=120), device_id=torch.device('cuda', local_rank))
    torch.backends.cuda.matmul.allow_tf32 = False
    prefix = f'{args.prefix}-rank{rank}'
    meta = json.loads(Path(prefix + '.json').read_text())
    n, h, f, k, p = (meta[key] for key in ('n', 'h', 'f', 'k', 'p'))
    assert p == dist.get_world_size()
    experts = meta.get('e', p)
    q = experts // p
    assert experts % p == 0
    device = torch.device('cuda', local_rank)

    def load(suffix, dtype, shape):
        return torch.from_numpy(np.fromfile(prefix + suffix + '.bin', dtype=dtype).reshape(shape).copy()).to(device)

    x, up, down = load('-x', np.float16, (n, h)), load('-up', np.float16, (q, f, h)), load('-down', np.float16, (q, h, f))
    routes = load('-routes', np.int32, (n, k)).long()
    whole = prepare_chunks(x, routes, n, experts)[0]
    assert whole.send_splits == meta['send_rows'] and whole.recv_splits == meta['recv_rows']
    if 'send_expert_rows' in meta:
        assert whole.send_expert_rows == meta['send_expert_rows']
        assert whole.recv_expert_rows == meta['recv_expert_rows']
    ordinary = lambda: run_ordinary(whole, up, down, k)
    expected = ordinary()
    reference_computed = expert_reference(whole.received, up, down, whole.local_expert_rows)
    dist.all_to_all_single(whole.returned, reference_computed, whole.send_splits, whole.recv_splits)
    reference = combine(whole, k)
    errors = {'ordinary_reference': (expected - reference).abs().max().item()}
    torch.testing.assert_close(expected, reference, atol=0.003, rtol=0.02)
    for name in ('fused', 'split'):
        got = load('-' + name, np.float32, (n, h))
        torch.testing.assert_close(got, reference, atol=0.003, rtol=0.02)
        errors[name + '_reference'] = (got - reference).abs().max().item()
    compute, comm = torch.cuda.Stream(), torch.cuda.Stream()
    sizes = sorted(set(min(n, size) for size in args.chunk_sizes))
    prepared = {size: prepare_chunks(x, routes, size, experts) for size in sizes}
    functions = {'ordinary': ordinary}
    for size, chunks in prepared.items():
        assert [sum(c.send_splits[j] for c in chunks) for j in range(p)] == whole.send_splits
        assert [sum(c.recv_splits[j] for c in chunks) for j in range(p)] == whole.recv_splits
        function = lambda chunks=chunks: run_stream_pipeline(chunks, up, down, k, compute, comm)
        for _ in range(2):
            output = function()
            assert output.shape == (n, h) and output.dtype == torch.float32 and output.device == device
            torch.testing.assert_close(output, reference, atol=0.003, rtol=0.02)
        errors[f'stream_{size}_reference'] = (output - reference).abs().max().item()
        functions[f'stream_{size}'] = function
    # Change every packed input, verify buffer updates, then restore exact bytes.
    originals = [(c, c.packed.clone()) for chunks in prepared.values() for c in chunks]
    original_whole = whole.packed.clone()
    whole.packed.copy_(original_whole * 0.5)
    for c, original in originals:
        c.packed.copy_(original * 0.5)
    changed_expected = ordinary()
    for name, function in functions.items():
        torch.testing.assert_close(function(), changed_expected, atol=0.003, rtol=0.02)
    torch.cuda.synchronize()
    whole.packed.copy_(original_whole)
    for c, original in originals:
        c.packed.copy_(original)
    torch.cuda.synchronize()
    # Forward then reverse order gives each route two independent measurement blocks.
    samples = {name: [] for name in functions}
    for names in (list(functions), list(reversed(functions))):
        for name in names:
            samples[name].extend(measure(functions[name], args.warmup, args.repeats))
    if args.profile_chunk:
        chosen = min(n, args.profile_chunk)
        if chosen not in prepared:
            raise ValueError(f'profile chunk {chosen} must be among measured chunks {sizes}')
        for _ in range(3):
            torch.cuda.synchronize()
            dist.barrier()
            with torch.cuda.nvtx.range('PIPELINE_PROFILE'):
                functions[f'stream_{chosen}']()
                torch.cuda.synchronize()  # Keep the entire GPU execution in the NVTX scope.
    result = dict(meta, samples_ms=samples, errors=errors, chunk_sizes=sizes,
                  e=experts, experts_per_rank=q, dtype='float16', output_dtype='float32', warmup_per_block=args.warmup,
                  repeats_per_block=args.repeats, changed_input_check='PASS',
                  scope='prepared buffers through FP32 original-order weighted output',
                  config={'tokens': n*p, 'local_tokens': n, 'batch_size': 1,
                          'seq_len': n, 'hidden_size': h, 'expert_hidden_size': f,
                          'num_experts': experts, 'experts_per_rank': q, 'top_k': k, 'ep_size': p,
                          'dtype': 'float16', 'chunk_sizes': sizes},
                  communication_bytes=2*sum(whole.send_splits[j] for j in range(p)
                                            if j != rank)*h*x.element_size())
    Path(prefix + '-stream.json').write_text(json.dumps(result, indent=2))
    print(f'PASS rank={rank} shape=({n},{h},{f}) E={experts} Q={q} chunks={sizes} errors={errors}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    with torch.no_grad():
        main()
