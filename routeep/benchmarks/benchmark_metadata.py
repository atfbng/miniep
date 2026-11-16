"""单独测量 Expert 元数据规划，不包含 EP、GEMM 或整模型吞吐。

两条路径使用同一份 ids 和稳定的 argsort 结果，只比较 Host/PyTorch 规划器与
CUDA 设备规划器的耗时。单卡运行示例：
    python3 benchmarks/benchmark_metadata.py --output results/metadata.json

--trace-dir 会在预热后导出 CPU/CUDA trace，便于分析规划开销。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import platform
import statistics
import time
import torch
import triton
from mini_ep.kernels.grouped_gemm.triton_impl import build_device_plan
from reference.grouped_plan import build_grouped_plan_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 1024, 4096])
    parser.add_argument("--experts", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    if min(*args.rows, *args.experts, args.block_m, args.warmup, args.repeats) < 1:
        parser.error("rows, experts, block-m, warmup and repeats must be positive")
    torch.manual_seed(13)
    records = []
    for experts in args.experts:
        for rows in args.rows:
            ids = torch.randint(experts, (rows,), device="cuda", dtype=torch.int64)
            order = torch.argsort(ids, stable=True)

            def host():
                counts = torch.bincount(ids, minlength=experts).tolist()
                return build_grouped_plan_reference(counts, ids.device, args.block_m)

            def device():
                plan = build_device_plan(ids, order, experts, args.block_m)
                return plan.offsets, plan.tile_experts, plan.tile_rows

            expected = host()
            actual = device()
            valid = expected[1].numel()
            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
            for index in (1, 2):
                torch.testing.assert_close(
                    actual[index][:valid], expected[index], atol=0, rtol=0
                )
            assert (actual[1][valid:] == -1).all().item()
            for _ in range(args.warmup):
                host()
                device()
            samples = {"host": [], "triton": []}
            # 交替两条路径的顺序以减少机器漂移；只在测量边界同步。
            for repeat in range(args.repeats):
                names = ("host", "triton") if repeat % 2 == 0 else ("triton", "host")
                for name in names:
                    torch.cuda.synchronize()
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    start = time.perf_counter()
                    result = host() if name == "host" else device()
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        dict(
                            cuda_ms=begin.elapsed_time(end),
                            wall_ms=1000 * (time.perf_counter() - start),
                        )
                    )
                    del result
            records.append(
                dict(
                    received_rows=rows,
                    local_experts=experts,
                    block_m=args.block_m,
                    valid_tiles=valid,
                    allocated_tiles=actual[1].numel(),
                    milliseconds={
                        name: {
                            key: statistics.median(s[key] for s in values)
                            for key in ("cuda_ms", "wall_ms")
                        }
                        for name, values in samples.items()
                    },
                    samples=samples,
                )
            )
            if args.trace_dir and rows == args.rows[-1] and experts == args.experts[-1]:
                args.trace_dir.mkdir(parents=True, exist_ok=True)
                for name, function in (("host", host), ("triton", device)):
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        with torch.profiler.record_function(f"METADATA_{name}"):
                            for _ in range(5):
                                result = function()
                        torch.cuda.synchronize()  # 只在导出 trace 的边界同步
                    prof.export_chrome_trace(str(args.trace_dir / f"{name}.json"))
                    (args.trace_dir / f"{name}-events.json").write_text(
                        json.dumps(
                            [
                                dict(name=event.key, count=event.count)
                                for event in prof.key_averages()
                            ],
                            indent=2,
                        )
                    )
    report = dict(
        workload="Expert offsets/tile metadata only; stable argsort excluded from BOTH paths",
        seed=13,
        warmup=args.warmup,
        repeats=args.repeats,
        statistic="median, single GPU CUDA Event and synchronized wall time; alternating A/B order",
        environment=dict(
            torch=torch.__version__,
            triton=triton.__version__,
            cuda=torch.version.cuda,
            device=torch.cuda.get_device_name(),
            python=platform.python_version(),
        ),
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            [{k: v for k, v in row.items() if k != "samples"} for row in records],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
