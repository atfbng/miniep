"""Four real NCCL ranks: independent DP inputs, TP-local ownership/reassembly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from datetime import timedelta
import torch
import torch.distributed as dist
from mini_ep.integration.input_layout import token_range, gather_tp_rows
from mini_ep.model.moe import ep_forward
from reference.moe import moe_reference
from reference.router import router_reference


def align_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        rank, size = dist.get_rank(), dist.get_world_size()
        if size != 4:
            raise ValueError(f"this test requires four NCCL ranks, got {size}")
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        dp, tp = rank // 2, rank % 2
        cases = 0
        for dtype in (torch.float32, torch.bfloat16):
            for index, lengths in enumerate(
                [(0, 0), (0, 1), (1, 0), (1, 13), (7, 23), (64, 3)]
            ):
                for k in (1, 2, 8):
                    for h, f in ((31, 45), (64, 96)):
                        torch.manual_seed(13 + index)
                        all13 = (torch.randn(8, 2 * f, h, device="cuda") * 0.1).to(
                            dtype
                        )
                        all2 = (torch.randn(8, h, f, device="cuda") * 0.1).to(dtype)
                        torch.manual_seed(100 + dp)
                        n = lengths[dp]
                        x = torch.randn(n, h, device="cuda").to(dtype)
                        logits = torch.randn(n, 8, device="cuda")
                        if index < 3:
                            logits[:, 0] += 20
                        ids, weights = router_reference(logits, k)
                        begin, end = token_range(n, tp, 2)
                        out, meta = ep_forward(
                            x[begin:end],
                            ids[begin:end],
                            weights[begin:end],
                            all13[rank * 2 : rank * 2 + 2],
                            all2[rank * 2 : rank * 2 + 2],
                            dist.group.WORLD,
                        )
                        actual = gather_tp_rows(out, n, groups[dp])
                        expected = moe_reference(
                            x, ids, weights, all13, all2, ep_size=4
                        )
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                        )
                        assert (
                            actual.shape == (n, h)
                            and actual.dtype == dtype
                            and actual.device == x.device
                        )
                        assert sum(meta["send_splits"]) == (end - begin) * k
                        assert len(meta["recv_splits"]) == 4
                        capacity = max(8, align_up(max(lengths) * k, 8))
                        ep_sizes = [
                            token_range(lengths[d], t, 2)[1]
                            - token_range(lengths[d], t, 2)[0]
                            for d in range(2)
                            for t in range(2)
                        ]
                        fixed_out, fixed_meta = ep_forward(
                            x[begin:end],
                            ids[begin:end],
                            weights[begin:end],
                            all13[rank * 2 : rank * 2 + 2],
                            all2[rank * 2 : rank * 2 + 2],
                            dist.group.WORLD,
                            fixed_rows_per_rank=capacity,
                            fixed_pack_backend=os.environ.get(
                                "MINIEP_FIXED_PACK", "torch"
                            ),
                            source_rows=ep_sizes,
                        )
                        fixed_actual = gather_tp_rows(fixed_out, n, groups[dp])
                        torch.testing.assert_close(
                            fixed_actual,
                            expected,
                            atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                            rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                        )
                        assert fixed_meta["layout"] == "fixed"
                        assert fixed_meta["send_rows"] == 4 * capacity
                        assert fixed_meta["recv_rows"] == 4 * capacity
                        assert fixed_meta["recv_padding_rows"] is None
                        if lengths == (0, 1) and rank == 0:
                            assert (
                                sum(meta["recv_splits"]) > 0
                            ), "idle DP must still compute remote tokens"
                        cases += 1
        print(f"PASS TP=2 DP=2 EP=4 rank={rank} cases={cases}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
