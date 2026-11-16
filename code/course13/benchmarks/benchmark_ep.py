"""Two-GPU stage timing of the course EP reference; separate from unit tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from datetime import timedelta
import json
import importlib.metadata
import os
import platform
import statistics
import torch
import torch.distributed as dist
from validation.backends import ep_forward
from reference.moe import moe_reference
from reference.router import router_reference
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.all_to_all import fixed_rows_per_rank
from benchmarks.timing import TimedStages


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--expert-hidden-size", type=int, default=768)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--backend", choices=["reference", "triton"], default="triton")
    p.add_argument("--expert-backend", choices=["loop", "grouped"], default="grouped")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument(
        "--a2a-mode",
        choices=["auto", "on", "off"],
        default="off",
        help="fixed-capacity A2A selection; auto uses it only for short decode",
    )
    p.add_argument(
        "--fixed-pack-backend", choices=["torch", "triton"], default="triton"
    )
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--return-dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto"
    )
    p.add_argument(
        "--fuse-return-weight", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (
        min(
            a.tokens, a.hidden_size, a.expert_hidden_size, a.experts, a.top_k, a.repeats
        )
        < 1
        or a.warmup < 1
    ):
        p.error("sizes/repeats/warmup must be positive")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=120))
    try:
        rank = dist.get_rank()
        size = dist.get_world_size()
        topology = ExpertParallelTopology(a.experts, size, rank)
        fixed_rows = fixed_rows_per_rank(a.a2a_mode, [a.tokens] * size, a.top_k)
        transport = {}
        transport.update(
            return_dtype={
                "auto": None,
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[a.return_dtype]
        )
        if a.backend == "triton":
            transport.update(
                fixed_rows_per_rank=fixed_rows,
                source_rows=[a.tokens] * size,
                fuse_return_weight=a.fuse_return_weight,
            )
            if a.fixed_pack_backend == "triton":
                transport["fixed_pack_backend"] = "triton"
        elif a.a2a_mode != "off":
            raise ValueError("Reference uses variable A2A; select --a2a-mode off")
        torch.manual_seed(13)
        h, f = a.hidden_size, a.expert_hidden_size
        # Identical full weights only for correctness setup; benchmark uses local shard.
        all13 = (torch.randn(a.experts, 2 * f, h, device="cuda") * 0.02).bfloat16()
        all2 = (torch.randn(a.experts, h, f, device="cuda") * 0.02).bfloat16()
        offset = rank * topology.num_local_experts
        w13 = all13[offset : offset + topology.num_local_experts].clone()
        w2 = all2[offset : offset + topology.num_local_experts].clone()
        torch.manual_seed(13 + rank)
        x = torch.randn(a.tokens, h, device="cuda", dtype=torch.bfloat16)
        gate = torch.randn(a.experts, h, device="cuda", dtype=torch.bfloat16) * 0.02
        logits = torch.nn.functional.linear(x, gate)
        ids, weights = router_reference(logits, a.top_k)
        expected = moe_reference(x, ids, weights, all13, all2, ep_size=size)
        actual, _ = ep_forward(
            x,
            ids,
            weights,
            w13,
            w2,
            dist.group.WORLD,
            backend=a.backend,
            expert_backend=a.expert_backend,
            **transport,
        )
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        del all13, all2, expected
        for _ in range(a.warmup):
            ep_forward(
                x,
                ids,
                weights,
                w13,
                w2,
                dist.group.WORLD,
                validate=False,
                backend=a.backend,
                expert_backend=a.expert_backend,
                **transport,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(a.repeats):
            stages = TimedStages()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            begin.record()
            with stages.stage("ROUTER"):
                logits = torch.nn.functional.linear(x, gate)
                ids, weights = router_reference(logits, a.top_k)
            _, metadata = ep_forward(
                x,
                ids,
                weights,
                w13,
                w2,
                dist.group.WORLD,
                stages=stages,
                validate=False,
                backend=a.backend,
                expert_backend=a.expert_backend,
                **transport,
            )
            end.record()
            end.synchronize()  # measurement boundary only
            stats = stages.milliseconds()
            stats["TOTAL"] = begin.elapsed_time(end)
            measured = torch.tensor([stats[k] for k in sorted(stats)], device="cuda")
            dist.all_reduce(measured, op=dist.ReduceOp.MAX)
            samples.append(dict(zip(sorted(stats), measured.tolist())))
        counts = torch.bincount(ids.flatten(), minlength=a.experts)
        dist.all_reduce(counts)
        # Read actual routing counts only after every timed iteration ends.
        counts_cpu = counts.tolist()
        received_rows = [
            sum(
                counts_cpu[
                    r
                    * topology.num_local_experts : (r + 1)
                    * topology.num_local_experts
                ]
            )
            for r in range(size)
        ]
        return_bytes = 4 if a.return_dtype == "fp32" else 2
        weight_bytes = 4
        payload_bytes = [
            (
                (size * fixed_rows * (h * (2 + return_bytes) + 8 + weight_bytes))
                if fixed_rows
                else (
                    a.tokens * a.top_k * (h * 2 + 8 + weight_bytes)
                    + received * h * return_bytes
                )
            )
            for received in received_rows
        ]
        config = dict(
            tokens=a.tokens,
            batch_size=1,
            seq_len=a.tokens,
            hidden_size=h,
            expert_hidden_size=f,
            num_experts=a.experts,
            top_k=a.top_k,
            ep_size=size,
            dtype="bfloat16",
            chunk_size=None,
            backend=a.backend,
            expert_backend=a.expert_backend,
            a2a_mode=a.a2a_mode,
            fixed_pack_backend=a.fixed_pack_backend,
            fixed_rows_per_rank=fixed_rows,
            return_dtype=a.return_dtype,
            fuse_return_weight=a.fuse_return_weight,
        )
        report = dict(
            workload="synthetic seeded EP with distinct source tokens per rank",
            config=config,
            seed=13,
            warmup=a.warmup,
            repeats=a.repeats,
            expert_backend=a.expert_backend,
            statistic="median of per-iteration maximum rank CUDA event time; per-stage maxima need not add to total",
            milliseconds={
                k: statistics.median(s[k] for s in samples) for k in samples[0]
            },
            samples=samples,
            tokens_per_expert=counts_cpu,
            received_valid_rows_by_rank=received_rows,
            received_padding_rows_by_rank=(
                [size * fixed_rows - n for n in received_rows]
                if fixed_rows
                else [0] * size
            ),
            imbalance_ratio=(counts.max() / counts.float().mean()).item(),
            actual_payload_bytes_by_rank=payload_bytes,
            send_padding_rows=(
                (size * fixed_rows - a.tokens * a.top_k) if fixed_rows else 0
            ),
            logical_sent_bytes_per_rank=a.tokens
            * a.top_k
            * (h * (2 + return_bytes) + 8 + weight_bytes),
            communication_bytes_note=(
                "actual_payload_bytes_by_rank counts each rank's forward and return sends, "
                "including padding/self and excluding count exchange. "
                "logical_sent_bytes_per_rank attributes the round trip to source tokens; "
                f"return uses {a.return_dtype}; forward branch weight bytes={weight_bytes}"
            ),
            environment=dict(
                torch=torch.__version__,
                cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(),
                device=torch.cuda.get_device_name(),
                python=platform.python_version(),
                cpu_affinity=(
                    sorted(os.sched_getaffinity(0))
                    if hasattr(os, "sched_getaffinity")
                    else None
                ),
                triton=importlib.metadata.version("triton"),
            ),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        report["tokens_per_second"] = (
            a.tokens * size / (report["milliseconds"]["TOTAL"] / 1000)
        )
        if rank == 0:
            path = Path(a.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps(report["milliseconds"], indent=2))
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
