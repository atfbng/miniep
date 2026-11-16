"""Correctness and matching end-to-end measurements for NCCL Chunk Pipeline."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_ep.distributed import nccl_distributed
from mini_ep.model.moe import distribute_single_gpu_moe
from mini_ep.utils.course_workload import make_workload
from mini_ep.distributed.overlap import prepare_ep_chunks, run_chunk_pipeline
from mini_ep.profiler.nvtx import nvtx_range
from benchmarks.benchmark_ep import measure_forward, max_rank_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--expert-hidden-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--output-dir", default="results/pipeline")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    with nccl_distributed(timeout_seconds=120) as ctx, torch.no_grad():
        cases = []
        for dtype in (torch.float32, torch.bfloat16):
            for k in (1, 2):
                ref, x = make_workload(
                    a.tokens, a.hidden_size, a.expert_hidden_size, 8, k, "hotspot"
                )
                model = distribute_single_gpu_moe(
                    ref, ep_size=ctx.world_size, ep_rank=ctx.global_rank
                ).to(ctx.device, dtype)
                local = x.tensor_split(ctx.world_size)[ctx.global_rank].to(
                    ctx.device, dtype
                )
                compute = torch.cuda.Stream(device=ctx.device)
                comm = torch.cuda.Stream(device=ctx.device)
                chunks = prepare_ep_chunks(model, local, a.chunk_size, group=ctx.group)
                expected = model(local, group=ctx.group).hidden_states
                for repeat in range(3):
                    got = run_chunk_pipeline(
                        model,
                        local,
                        chunks,
                        group=ctx.group,
                        compute_stream=compute,
                        comm_stream=comm,
                    )
                    torch.testing.assert_close(
                        got,
                        expected,
                        rtol=1e-4 if dtype == torch.float32 else 0.02,
                        atol=1e-5 if dtype == torch.float32 else 0.008,
                    )

                def pipeline():
                    prepared = prepare_ep_chunks(
                        model, local, a.chunk_size, group=ctx.group
                    )
                    return run_chunk_pipeline(
                        model,
                        local,
                        prepared,
                        group=ctx.group,
                        compute_stream=compute,
                        comm_stream=comm,
                    )

                timings = {}
                for name, fn in [
                    ("sync", lambda: model(local, group=ctx.group)),
                    ("pipeline", pipeline),
                ]:
                    samples = measure_forward(
                        lambda unused, **kwargs: fn(), local, ctx.group, 2, 5
                    )
                    rank_samples = samples
                    timings[name] = rank_samples
                cases.append(
                    {
                        "dtype": str(dtype),
                        "top_k": k,
                        "correctness": "PASS",
                        "samples_ms": timings,
                        "chunk_size": a.chunk_size,
                        "global_tokens": a.tokens,
                    }
                )
                if a.profile and dtype == torch.float32 and k == 2:
                    for _ in range(3):
                        with nvtx_range("PIPELINE_PROFILE"):
                            pipeline()
                            torch.cuda.synchronize()
        (out / f"rank{ctx.global_rank}.json").write_text(json.dumps(cases, indent=2))
        print("PIPELINE", ctx.world_size, ctx.global_rank, "PASS", flush=True)


if __name__ == "__main__":
    main()
