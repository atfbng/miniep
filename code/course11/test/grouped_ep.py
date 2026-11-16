"""Validate every optimization inside real NCCL EP with the same oracle."""

import json, torch, argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_ep.distributed import nccl_distributed
from mini_ep.model.moe import distribute_single_gpu_moe
from mini_ep.utils.course_workload import make_workload
from mini_ep.profiler.stages import StageRecorder, run_ep_stages
from mini_ep.kernels.dispatch_triton import dispatch_triton
from mini_ep.kernels.combine_triton import combine_triton
from mini_ep.kernels.grouped_gemm.triton_impl import compute_grouped_local_experts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/ep-variants")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    with nccl_distributed(timeout_seconds=120) as ctx, torch.no_grad():
        cases = []
        for dtype in [torch.float32, torch.bfloat16]:
            for k in [1, 2]:
                reference, global_tokens = make_workload(32, 17, 31, 8, k, "hotspot")
                model = distribute_single_gpu_moe(
                    reference, ep_size=ctx.world_size, ep_rank=ctx.global_rank
                ).to(device=ctx.device, dtype=dtype)
                tokens = global_tokens.chunk(ctx.world_size)[ctx.global_rank].to(
                    device=ctx.device, dtype=dtype
                )
                expected = reference.to(device=ctx.device, dtype=dtype)(
                    tokens
                ).hidden_states
                recorder = StageRecorder(ctx.device)
                variants = [
                    ("reference", {}),
                    ("dispatch", {"dispatch_impl": dispatch_triton}),
                    ("combine", {"combine_impl": combine_triton}),
                    ("grouped", {"expert_compute": compute_grouped_local_experts}),
                    (
                        "all",
                        {
                            "dispatch_impl": dispatch_triton,
                            "combine_impl": combine_triton,
                            "expert_compute": compute_grouped_local_experts,
                        },
                    ),
                ]
                for name, kwargs in variants:
                    result = run_ep_stages(
                        model, tokens, recorder, group=ctx.group, **kwargs
                    )
                    rtol, atol = (
                        (1e-4, 1e-5) if dtype == torch.float32 else (2e-2, 8e-3)
                    )
                    torch.testing.assert_close(
                        result.hidden_states, expected, rtol=rtol, atol=atol
                    )
                    assert result.send_splits.sum().item() == tokens.shape[0] * k
                    assert (
                        result.local_expert_rows.sum().item()
                        == result.recv_splits.sum().item()
                    )
                    cases.append(
                        {
                            "dtype": str(dtype),
                            "top_k": k,
                            "variant": name,
                            "status": "PASS",
                        }
                    )
        output = output_dir / f"ep{ctx.world_size}-variants-rank{ctx.global_rank}.json"
        open(output, "w").write(json.dumps(cases, indent=2))
        print(
            "EP",
            ctx.world_size,
            "rank",
            ctx.global_rank,
            "PASS",
            len(cases),
            flush=True,
        )


if __name__ == "__main__":
    main()
