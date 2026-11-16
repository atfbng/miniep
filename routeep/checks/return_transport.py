"""返回精度路径的 torchrun NCCL 回归；所有 Rank 执行完全相同的 case。

用不同 H/F、Top-K、倾斜路由和 return dtype 检查“先加权再舍入”的边界，
并与独立 Reference 对齐。需要 2 个或更多 NCCL Rank。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from datetime import timedelta
import json
import os
import traceback
import torch
import torch.distributed as dist
from mini_ep.model.moe import ep_forward
from reference.moe import ep_forward_reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank, size = dist.get_rank(), dist.get_world_size()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    records = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for h, f in ((32, 48), (128, 192)):
                e = max(8, size)
                torch.manual_seed(13)
                w13 = (torch.randn(e, 2 * f, h, device="cuda") * 0.08).to(dtype)
                w2 = (torch.randn(e, h, f, device="cuda") * 0.08).to(dtype)
                local = (
                    w13.chunk(size)[rank].contiguous(),
                    w2.chunk(size)[rank].contiguous(),
                )
                for k in (1, 2, 8):
                    for skew in (False, True):
                        print(
                            f"CASE rank={rank} dtype={dtype} H={h} K={k} skew={skew}",
                            flush=True,
                        )
                        sizes = [
                            0 if r == 0 and size > 1 else 3 + r for r in range(size)
                        ]
                        n = sizes[rank]
                        torch.manual_seed(130 + rank)
                        x = torch.randn(n, h, device="cuda").to(dtype)
                        ids = (
                            torch.zeros(n, k, device="cuda", dtype=torch.long)
                            if skew
                            else torch.randint(e, (n, k), device="cuda")
                        )
                        weights = torch.softmax(torch.randn(n, k, device="cuda"), -1)
                        base, _ = ep_forward(
                            x,
                            ids,
                            weights,
                            *local,
                            dist.group.WORLD,
                            return_dtype=torch.float32,
                            fuse_return_weight=False,
                        )
                        default, meta = ep_forward(
                            x, ids, weights, *local, dist.group.WORLD
                        )
                        assert meta["return_dtype"] == str(x.dtype)
                        assert (
                            meta["return_send_bytes"]
                            == meta["recv_rows"] * h * x.element_size()
                        )
                        assert meta["preweighted"] and meta["fuse_return_weight"]
                        torch.testing.assert_close(default, base, atol=0, rtol=0)
                        for rd in (torch.bfloat16, torch.float16, torch.float32):
                            ref, _ = ep_forward_reference(
                                x,
                                ids,
                                weights,
                                *local,
                                dist.group.WORLD,
                                return_dtype=rd,
                            )
                            for fixed in (None, max(1, max(sizes) * k)):
                                for backend in ("loop", "grouped"):
                                    for fused in (
                                        (False, True)
                                        if backend == "grouped"
                                        else (False,)
                                    ):
                                        y, meta = ep_forward(
                                            x,
                                            ids,
                                            weights,
                                            *local,
                                            dist.group.WORLD,
                                            return_dtype=rd,
                                            fuse_return_weight=fused,
                                            expert_backend=backend,
                                            fixed_rows_per_rank=fixed,
                                        )
                                        assert (
                                            y.shape == x.shape
                                            and y.dtype == dtype
                                            and y.device == x.device
                                        )
                                        # 独立 GEMM 的结果可能落在传输精度中点的两侧；
                                        # 这个 Kernel-Reference 检查不证明与 DeepEP 完全一致。
                                        tol = (
                                            0.02
                                            if dtype == torch.bfloat16
                                            or rd == torch.bfloat16
                                            else 0.002 if rd == torch.float16 else 1e-5
                                        )
                                        torch.testing.assert_close(
                                            y, ref, atol=tol, rtol=tol
                                        )
                                        if (
                                            dtype == torch.bfloat16
                                            and rd == torch.bfloat16
                                            and backend == "grouped"
                                        ):
                                            assert torch.equal(
                                                y, base
                                            ), f"rank={rank} weighted BF16 changed identical-compute baseline"
                                        records.append(
                                            dict(
                                                dtype=str(dtype),
                                                h=h,
                                                k=k,
                                                skew=skew,
                                                fixed=fixed,
                                                backend=backend,
                                                fused=fused,
                                                exact_baseline=bool(
                                                    torch.equal(y, base)
                                                ),
                                            )
                                            | meta
                                        )
        path = Path(a.output)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"rank{rank}.json").write_text(json.dumps(records, indent=2))
        print(f"PASS rank={rank} ep={size} cases={len(records)}", flush=True)
    except BaseException:
        print(
            f'FAIL rank={rank} rd={locals().get("rd")} '
            f'fixed={locals().get("fixed")} backend={locals().get("backend")} fused={locals().get("fused")}',
            flush=True,
        )
        traceback.print_exc()
        # 让 torchrun 立即终止其它 Rank；失败 Rank 不能在 destroy_process_group
        # 里等待一个已经进入下一次 A2A 的 Rank。
        sys.stderr.flush()
        os._exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
