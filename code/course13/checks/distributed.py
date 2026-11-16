"""Run with torchrun --standalone --nproc-per-node=2; NCCL timeout 120s."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import argparse
from datetime import timedelta
import torch
import torch.distributed as dist
from validation.backends import ep_forward
from reference.moe import moe_reference
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.integration.input_layout import token_range
from mini_ep.integration.input_layout import gather_tp_rows
from reference.router import router_reference


def align_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["reference", "triton"], default="reference"
    )
    parser.add_argument(
        "--expert-backend", choices=["loop", "grouped"], default="grouped"
    )
    parser.add_argument(
        "--fixed-pack-backend", choices=["torch", "triton"], default="torch"
    )
    parser.add_argument("--experts", type=int, default=8)
    args = parser.parse_args()
    if args.experts < 1:
        parser.error("--experts must be positive")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        size = dist.get_world_size()
        device = torch.device("cuda", rank)
        torch.manual_seed(13)
        count = 0
        for dtype in (torch.float32, torch.bfloat16):
            for n in (0, 1, 7, 16):
                for k in (1, 2, 8):
                    if k > args.experts:
                        continue
                    for skew in (False, True):
                        e, h, f = args.experts, 32, 48
                        topology = ExpertParallelTopology(e, size, dist.get_rank())
                        x = torch.randn(n, h, device=device, dtype=dtype)
                        w13 = (torch.randn(e, 2 * f, h, device=device) * 0.1).to(dtype)
                        w2 = (torch.randn(e, h, f, device=device) * 0.1).to(dtype)
                        logits = torch.randn(n, e, device=device)
                        if skew:
                            logits[:, 0] += 20
                        ids, weights = router_reference(logits, k)
                        start, end = topology.token_range(n)
                        offset = topology.ep_rank * topology.num_local_experts
                        y, meta = ep_forward(
                            x[start:end],
                            ids[start:end],
                            weights[start:end],
                            w13[offset : offset + topology.num_local_experts],
                            w2[offset : offset + topology.num_local_experts],
                            dist.group.WORLD,
                            backend=args.backend,
                            expert_backend=args.expert_backend,
                        )
                        actual = gather_tp_rows(y, n, dist.group.WORLD)
                        expected = moe_reference(x, ids, weights, w13, w2, ep_size=size)
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                        )
                        assert meta.get("layout", "variable") == "variable"
                        assert sum(meta["send_splits"]) == (end - start) * k
                        assert len(meta["recv_splits"]) == size
                        if args.backend == "triton":
                            capacity = max(8, align_up(n * k, 8))
                            fixed_y, fixed_meta = ep_forward(
                                x[start:end],
                                ids[start:end],
                                weights[start:end],
                                w13[offset : offset + topology.num_local_experts],
                                w2[offset : offset + topology.num_local_experts],
                                dist.group.WORLD,
                                backend=args.backend,
                                expert_backend=args.expert_backend,
                                fixed_rows_per_rank=capacity,
                                fixed_pack_backend=args.fixed_pack_backend,
                                source_rows=[
                                    token_range(n, r, size)[1]
                                    - token_range(n, r, size)[0]
                                    for r in range(size)
                                ],
                            )
                            fixed_actual = gather_tp_rows(fixed_y, n, dist.group.WORLD)
                            torch.testing.assert_close(
                                fixed_actual,
                                expected,
                                atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                                rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                            )
                            assert fixed_meta["layout"] == "fixed"
                            expected_pack = (
                                "triton"
                                if args.fixed_pack_backend == "triton"
                                and (end - start) * k <= 64
                                else "torch"
                            )
                            assert fixed_meta["fixed_pack_backend"] == expected_pack
                            assert fixed_meta["fixed_rows_per_rank"] == capacity
                            assert fixed_meta["send_rows"] == size * capacity
                            assert fixed_meta["recv_rows"] == size * capacity
                            assert fixed_meta["send_padding_rows"] == (
                                size * capacity - (end - start) * k
                            )
                            assert fixed_meta["received_branches"] is None
                            assert fixed_meta["recv_padding_rows"] is None
                            assert fixed_actual.shape == expected.shape
                            assert fixed_actual.dtype == expected.dtype
                            assert fixed_actual.device == expected.device
                        count += 1
        print(
            f"PASS rank={rank} ep_size={size} backend={args.backend} "
            f"expert_backend={args.expert_backend} experts={args.experts} cases={count}",
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
