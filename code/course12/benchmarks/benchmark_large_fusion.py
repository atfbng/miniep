"""Compare exported NVSHMEM FP16 MLP results with NCCL + PyTorch.

Run after large_expert.cu with the same prefix and EP size, using torchrun.
Prepared destination-packed tokens and split counts are outside the timer.
The timed interval includes two NCCL A2As, Up/GELU/Down and FP32 combine.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        raise ValueError('warmup and repeats must be positive')
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32 = False
    prefix = f'{args.prefix}-rank{rank}'
    meta = json.loads(Path(prefix + '.json').read_text())
    n, h, f, k, p = (meta[key] for key in ('n', 'h', 'f', 'k', 'p'))
    assert p == dist.get_world_size()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))

    def load(suffix: str, dtype: np.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        array = np.fromfile(prefix + suffix + '.bin', dtype=dtype).reshape(shape)
        return torch.from_numpy(array.copy()).to(device)

    x = load('-x', np.float16, (n, h))
    up = load('-up', np.float16, (f, h))
    down = load('-down', np.float16, (h, f))
    routes = load('-routes', np.int32, (n, k)).long()
    order = torch.argsort(routes.flatten(), stable=True)
    inverse = torch.argsort(order)
    send_splits = torch.bincount(routes.flatten(), minlength=p).tolist()
    recv_splits_tensor = torch.empty(p, device=device, dtype=torch.int64)
    dist.all_to_all_single(recv_splits_tensor, torch.tensor(send_splits, device=device))
    recv_splits = recv_splits_tensor.tolist()
    assert send_splits == meta['send_rows'] and recv_splits == meta['recv_rows']
    # Compact source-major receive layout, preserving each source's packed order.
    packed = x[torch.div(order, k, rounding_mode='floor')].contiguous()
    received = torch.empty((sum(recv_splits), h), dtype=x.dtype, device=device)
    returned = torch.empty_like(packed)

    def combine_returned() -> torch.Tensor:
        # Restore q=t*K+slot before FP32 weighting and Top-K reduction.
        return returned[inverse].float().reshape(n, k, h).sum(dim=1) / k

    def forward() -> torch.Tensor:
        dist.all_to_all_single(received, packed, recv_splits, send_splits)
        computed = F.gelu(received @ up.T) @ down.T
        dist.all_to_all_single(returned, computed.contiguous(), send_splits, recv_splits)
        return combine_returned()

    output = forward()
    # Independent FP32 GEMMs, with explicit FP16 rounding at the same boundaries.
    hidden_reference = (received.float() @ up.float().T).half().float()
    hidden_reference = F.gelu(hidden_reference).half().float()
    computed_reference = (hidden_reference @ down.float().T).half()
    dist.all_to_all_single(returned, computed_reference, send_splits, recv_splits)
    reference = combine_returned()
    errors = {}
    for name in ('fused', 'split'):
        got = load('-' + name, np.float32, (n, h))
        assert got.dtype == torch.float32 and got.device == x.device
        torch.testing.assert_close(got, reference, atol=0.003, rtol=0.02)
        errors[name + '_reference_max_abs'] = (got - reference).abs().max().item()
        torch.testing.assert_close(got, output, atol=0.003, rtol=0.02)
        errors[name + '_nccl_max_abs'] = (got - output).abs().max().item()
    torch.testing.assert_close(output, reference, atol=0.003, rtol=0.02)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for block in range(2):
        for iteration in range(args.warmup + args.repeats):
            torch.cuda.synchronize()
            dist.barrier()
            begin.record()
            output = forward()
            end.record()
            end.synchronize()
            sample = torch.tensor(begin.elapsed_time(end), device=device)
            dist.all_reduce(sample, op=dist.ReduceOp.MAX)
            if iteration >= args.warmup:
                samples.append(sample.item())
    result = dict(meta, nccl_ms=samples, errors=errors, dtype='float16', output_dtype='float32',
                  nccl_warmup_per_block=args.warmup, nccl_repeats_per_block=args.repeats)
    Path(prefix + '-nccl.json').write_text(json.dumps(result, indent=2))
    print(f'PASS rank={rank} shape=({n},{h},{f}) errors={errors}', flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
