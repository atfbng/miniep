"""比较固定容量 Dispatch 的 PyTorch 与 Triton 实现。

只测“行重排 + 元数据生成”，不包含 NCCL 通信和 Expert 计算。每个配置先用
独立 Reference 检查结果完全一致，再交替测量两条实现，减少顺序带来的漂移。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import statistics
import time
import torch
from mini_ep.distributed.dispatch import pack_by_destination_rank_fixed
from mini_ep.distributed.topology import ExpertParallelTopology
from reference.fixed_dispatch import dispatch_fixed_reference
from mini_ep.kernels.dispatch_triton import pack_by_destination_rank_fixed_triton


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("repeats must be >=2")
    torch.manual_seed(13)
    rows = []
    for n in (1, 4, 8):
        for p in (2, 4):
            topology = ExpertParallelTopology(128, p, 0)
            x = torch.randn(n, 2048, device="cuda", dtype=torch.bfloat16)
            ids = torch.randint(128, (n, 8), device="cuda")
            weights = torch.softmax(torch.randn(n, 8, device="cuda"), -1)
            capacity = n * 8
            functions = dict(
                torch=pack_by_destination_rank_fixed,
                triton=pack_by_destination_rank_fixed_triton,
            )
            expected = dispatch_fixed_reference(x, ids, weights, topology, capacity)
            actual = pack_by_destination_rank_fixed_triton(
                x, ids, weights, topology, capacity
            )
            for field in (
                "packed_tokens",
                "packed_expert_ids",
                "packed_routing_weights",
                "restore_index",
                "valid_rows",
            ):
                torch.testing.assert_close(
                    getattr(actual, field), getattr(expected, field), atol=0, rtol=0
                )
            for fn in functions.values():
                for _ in range(20):
                    fn(x, ids, weights, topology, capacity)
            samples = {name: [] for name in functions}
            for iteration in range(args.repeats):
                order = (
                    list(functions) if iteration % 2 == 0 else list(reversed(functions))
                )
                for name in order:
                    torch.cuda.synchronize()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    begin = time.perf_counter()
                    start.record()
                    functions[name](x, ids, weights, topology, capacity)
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        dict(
                            cuda_ms=start.elapsed_time(end),
                            wall_ms=(time.perf_counter() - begin) * 1000,
                        )
                    )
            rows.append(
                dict(
                    tokens=n,
                    hidden_size=2048,
                    num_experts=128,
                    top_k=8,
                    ep_size=p,
                    dtype="bfloat16",
                    capacity=capacity,
                    samples=samples,
                    medians={
                        name: {
                            metric: statistics.median(s[metric] for s in values)
                            for metric in values[0]
                        }
                        for name, values in samples.items()
                    },
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            dict(
                seed=13,
                warmup=20,
                repeats=args.repeats,
                scope="single GPU fixed dispatch only; alternating order; boundary synchronization",
                device=torch.cuda.get_device_name(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                rows=rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
