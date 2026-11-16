"""NCCL 回归：只要一个 Rank 发现精度或有限性失败，所有 EP Rank 都必须失败。

测试构造相等、1 ULP、NaN、Inf、slot 顺序、shape、dtype 和空张量等用例，
验证分布式验收不会因为某个 Rank 先通过而掩盖另一个 Rank 的失败。
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from validation.exact import exact_tensor_flag, exact_tensor_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    rank, size = dist.get_rank(), dist.get_world_size()
    records = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for case in (
                "equal",
                "ulp",
                "nan",
                "inf",
                "slot_order",
                "shape",
                "dtype",
                "empty",
            ):
                expected = torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=dtype
                )
                actual = expected.clone()
                if case == "empty":
                    actual = actual[:0]
                    expected = expected[:0]
                if rank == size - 1:
                    if case == "ulp":
                        actual[0, 0] = torch.nextafter(
                            actual[0, 0], torch.tensor(2.0, device="cuda", dtype=dtype)
                        )
                    if case == "nan":
                        actual[0, 0] = float("nan")
                    if case == "inf":
                        actual[0, 0] = expected[0, 0] = float("inf")
                    if case == "slot_order":
                        actual = actual.flip(-1)
                    if case == "shape":
                        actual = actual[:1]
                    if case == "dtype":
                        actual = actual.double()
                local = exact_tensor_flag(actual, expected)
                failure = (~local).int().reshape(1)
                dist.all_reduce(failure, op=dist.ReduceOp.MAX)
                failed = bool(failure.item())
                assert failed == (case not in ("equal", "empty")), (rank, case, failed)
                stats = exact_tensor_metrics(actual, expected)
                assert bool(local.item()) == stats["exact_equal"]
                records.append(
                    dict(case=case, dtype=str(dtype), global_failure=failed, **stats)
                )
        root = Path(a.output)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"rank{rank}.json").write_text(
            json.dumps(records, indent=2, allow_nan=False)
        )
        print(
            f"PASS rank={rank} world={size} device_predicates={len(records)}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
