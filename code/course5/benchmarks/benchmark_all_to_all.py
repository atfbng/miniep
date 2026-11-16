"""Preallocated payload-only variable-split NCCL All-to-All benchmark."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_ep.distributed import equal_split_all_to_all, nccl_distributed
from mini_ep.profiler.timer import measure_latency


def split_rows(rows: int, size: int, pattern: str) -> list[int]:
    """Split nonnegative rows over positive size, with remainder in rank order."""
    if rows < 0 or size <= 0 or pattern not in ("uniform", "imbalanced"):
        raise ValueError(f"invalid rows={rows}, size={size}, pattern={pattern}")
    if pattern == "uniform" or size == 1:
        base, remainder = divmod(rows, size)
        return [base + (rank < remainder) for rank in range(size)]
    hot = math.ceil(rows * .8)
    rest, remainder = divmod(rows - hot, size - 1)
    return [hot] + [rest + (rank < remainder) for rank in range(size - 1)]


def remote_payload_bytes(matrix: list[list[int]], row_bytes: int) -> int:
    """Count each source-to-destination payload once, excluding self edges."""
    return sum(rows for source, counts in enumerate(matrix)
               for destination, rows in enumerate(counts) if source != destination) * row_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes-bytes", nargs="+", type=int,
                        default=[1024, 10240, 102400, 1048576, 10485760, 104857600])
    parser.add_argument("--patterns", nargs="+", choices=("uniform", "imbalanced"), default=["uniform", "imbalanced"])
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=51)
    args = parser.parse_args()
    if args.hidden_size <= 0 or min(args.sizes_bytes) <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise ValueError("hidden size, message sizes and repeats must be positive; warmup >= 0")
    torch.manual_seed(args.seed)
    with nccl_distributed(timeout_seconds=60) as context, torch.inference_mode():
        rank, size, device = context.global_rank, context.world_size, context.device
        dtype = getattr(torch, args.dtype)
        row_bytes = args.hidden_size * torch.empty((), dtype=dtype).element_size()
        for requested in args.sizes_bytes:
            rows = math.ceil(requested / row_bytes)
            for pattern in args.patterns:
                sends = split_rows(rows, size, pattern)
                count_tensor = torch.tensor(sends, device=device, dtype=torch.int64)
                receives = equal_split_all_to_all(count_tensor, group=context.group).cpu().tolist()
                send = torch.full((rows, args.hidden_size), rank + 1, device=device, dtype=dtype)
                receive = torch.empty((sum(receives), args.hidden_size), device=device, dtype=dtype)

                def operation():
                    dist.all_to_all_single(receive, send, output_split_sizes=receives,
                                           input_split_sizes=sends, group=context.group)
                    return receive

                operation()
                offset = 0
                for source, length in enumerate(receives):
                    torch.testing.assert_close(receive[offset:offset + length],
                                               torch.full((length, args.hidden_size), source + 1, device=device, dtype=dtype))
                    offset += length
                torch.cuda.synchronize(device)
                dist.barrier(group=context.group)  # Measurement boundary, before warmup.
                stats = measure_latency(operation, device=device, warmup=args.warmup, repeats=args.repeats)
                samples = torch.tensor(stats.samples_ms, dtype=torch.float64, device=device)
                dist.all_reduce(samples, op=dist.ReduceOp.MAX, group=context.group)
                slowest = samples.cpu().tolist()
                median = statistics.median(slowest)
                matrix = [sends for _ in range(size)]
                remote = remote_payload_bytes(matrix, row_bytes)
                if rank == 0:
                    print(json.dumps(dict(pattern=pattern, requested_bytes=requested,
                        actual_send_bytes=rows * row_bytes, send_count_matrix=matrix,
                        recv_splits_by_rank=[list(column) for column in zip(*matrix)],
                        recv_bytes_by_rank=[sum(column) * row_bytes for column in zip(*matrix)],
                        hidden_size=args.hidden_size, dtype=args.dtype, ep_size=size,
                        warmup=args.warmup, repeats=args.repeats, seed=args.seed,
                        median_ms=median, mean_ms=statistics.fmean(slowest),
                        min_ms=min(slowest), max_ms=max(slowest), communication_bytes=remote,
                        effective_bandwidth_GB_s=remote / (median * 1e6) if median else 0,
                        device=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                        cuda_version=torch.version.cuda, nccl_version=torch.cuda.nccl.version()), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
