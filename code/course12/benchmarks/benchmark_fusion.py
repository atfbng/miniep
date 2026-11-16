"""Same-input FP32 prepared execution: sync NCCL, Chunk NCCL, NVSHMEM."""

from __future__ import annotations
import argparse, json, sys, statistics
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_ep.distributed import (
    nccl_distributed,
    variable_split_all_to_all,
    combine_reference,
)
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.distributed.overlap import prepare_ep_chunks, run_chunk_pipeline
from benchmarks.benchmark_ep import measure_forward
from benchmarks.benchmark_kernels import kernel_counts


class FixedRouter:
    def __init__(self, p: int, k: int, skew: bool):
        self.p, self.k, self.skew = p, k, skew

    def __call__(self, tokens):
        original = (torch.round(tokens[:, 0] * 100).to(torch.int64) - 1) // 4
        ids = (
            original[:, None] * self.k
            + torch.arange(self.k, device=tokens.device)[None, :]
        ) % self.p
        if self.skew:
            ids[:, 0] = 0
            if self.k == 2:
                ids[:, 1] = self.p - 1
        return SimpleNamespace(
            expert_ids=ids,
            routing_weights=torch.full(ids.shape, 1 / self.k, device=tokens.device),
        )


class DeviceExpertMLP(torch.nn.Module):
    def __init__(self, pe: int, device):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.arange(1, 17, device=device, dtype=torch.float32).reshape(4, 4)
            * (0.01 * (pe + 1)),
        )

    def forward(self, x):
        return torch.nn.functional.gelu(x @ self.weight.T) @ self.weight.T


def run_sync_prepared(model, tokens, chunks, group):
    c = chunks[0]
    received = variable_split_all_to_all(
        c.rank_pack.packed_tokens, c.send_splits, c.recv_splits, group=group
    )
    computed = model.experts[0](received)
    returned = variable_split_all_to_all(
        computed, c.recv_splits, c.send_splits, group=group
    )
    expert_major = torch.empty_like(returned)
    expert_major.index_copy_(0, c.rank_pack.row_origin_index, returned)
    return combine_reference(expert_major, c.dispatch)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix", required=True)
    p.add_argument("--output-dir", default="results/fusion")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    with nccl_distributed(timeout_seconds=120) as ctx, torch.no_grad():
        cases = json.load(open(a.prefix + f"-rank{ctx.global_rank}.json"))
        report = []
        x = (
            torch.arange(
                ctx.global_rank * 28 + 1,
                ctx.global_rank * 28 + 29,
                device=ctx.device,
                dtype=torch.float32,
            )
            * 0.01
        ).reshape(7, 4)
        compute = torch.cuda.Stream(device=ctx.device)
        comm = torch.cuda.Stream(device=ctx.device)
        for case in cases:
            if case.get("expert") != "mlp":
                raise ValueError("Expected NVSHMEM results with expert=mlp")
            if case.get("input_scale", 1) != 1:
                raise ValueError("Benchmark requires fixed input across all samples")
            model = SimpleNamespace(
                router=FixedRouter(
                    ctx.world_size, case["top_k"], bool(case["scenario"])
                ),
                experts=[DeviceExpertMLP(ctx.global_rank, ctx.device)],
                num_experts=ctx.world_size,
                topology=ExpertParallelTopology(
                    ctx.world_size, ctx.world_size, ctx.global_rank
                ),
            )
            one = prepare_ep_chunks(model, x, 7, group=ctx.group)
            parts = prepare_ep_chunks(model, x, 3, group=ctx.group)
            sync = lambda: run_sync_prepared(model, x, one, ctx.group)
            pipe = lambda: run_chunk_pipeline(
                model,
                x,
                parts,
                group=ctx.group,
                compute_stream=compute,
                comm_stream=comm,
            )
            expected = torch.tensor(
                case["values"], device=ctx.device, dtype=torch.float32
            ).reshape(7, 4)
            for fn in (sync, pipe):
                torch.testing.assert_close(fn(), expected, rtol=2e-5, atol=1e-5)
            result = {key: case[key] for key in ["expert", "top_k", "scenario"]}
            result["global_tokens"] = 7 * ctx.world_size
            result["config"] = {
                "tokens": 7 * ctx.world_size,
                "batch_size": 1,
                "seq_len": 7 * ctx.world_size,
                "hidden_size": 4,
                "expert_hidden_size": 4,
                "num_experts": ctx.world_size,
                "top_k": case["top_k"],
                "ep_size": ctx.world_size,
                "dtype": "float32",
                "chunk_size": {"sync": 7, "pipeline": 3, "nvshmem": 7},
                "warmup": {"sync": 2, "pipeline": 2, "nvshmem": 1},
                "repeats": 5,
            }
            result["environment"] = {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(ctx.device),
            }
            result["scope"] = (
                "prepared buffers through original-order weighted output; excludes routing/packing/metadata and Host result copy; NCCL includes temporary allocation, NVSHMEM reuses workspace"
            )
            result["nvshmem_samples_ms"] = case["prepared_samples_ms"][1:]
            result["kernel_count"] = {"nvshmem": 2}
            for name, fn in [("sync", sync), ("pipeline", pipe)]:
                result[name + "_samples_ms"] = measure_forward(
                    lambda unused, **kw: fn(), x, ctx.group, 2, 5
                )
                result["kernel_count"][name] = kernel_counts(fn, ctx.device)
            result["communication_bytes"] = (
                2
                * sum(
                    one[0].send_splits[d]
                    for d in range(ctx.world_size)
                    if d != ctx.global_rank
                )
                * 4
                * 4
            )
            for backend in ["sync", "pipeline", "nvshmem"]:
                median = statistics.median(result[backend + "_samples_ms"])
                result[backend + "_median_ms"] = median
                result[backend + "_tokens_per_second"] = (
                    result["global_tokens"] * 1000 / median
                )
            result["correctness"] = "PASS"
            report.append(result)
        (out / f"ep{ctx.world_size}-rank{ctx.global_rank}.json").write_text(
            json.dumps(report, indent=2)
        )
        print("FUSION", ctx.world_size, ctx.global_rank, "PASS", flush=True)


if __name__ == "__main__":
    main()
