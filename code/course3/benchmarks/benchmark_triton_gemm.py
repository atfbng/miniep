"""比较 PyTorch、固定 Triton、调优 Triton 的单次 GEMM 与 Expert MLP。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from mini_ep.model.expert import expert_mlp_reference, gemm_reference
from mini_ep.profiler import measure_latency


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=128, help="输入行数")
    parser.add_argument("--n", type=int, default=2048, help="输出特征数 / MLP 中间宽度")
    parser.add_argument("--k", type=int, default=1024, help="输入宽度 / GEMM 累加维度")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--backend", choices=("fixed", "tuned", "both"), default="both")
    parser.add_argument("--operation", choices=("gemm", "mlp", "both"), default="both")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path, help="保存配置、数值核对和完整计时样本")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.m < 0 or args.n <= 0 or args.k <= 0:
        parser.error(f"需要 M>=0、N>0、K>0，实际为 {(args.m, args.n, args.k)}")
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("需要 warmup>=0、repeats>0")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("此示例需要一张可见 CUDA GPU")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not 0 <= device.index < torch.cuda.device_count():
        parser.error(f"CUDA 设备编号越界：{device.index}")
    try:
        import triton
        from mini_ep.kernels.gemm_triton import (
            expert_mlp_triton, selected_config, triton_gemm, triton_gemm_tuned,
        )
    except ImportError as error:
        parser.error(f"请安装本课 requirements.txt 中的 Triton：{error}")

    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    # FP32 对照使用 IEEE 精度，FP16 对照关闭低精度部分和累加。
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dtype = getattr(torch, args.dtype)
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        parser.error("所选 GPU 不支持 BF16")
    x = torch.randn((args.m, args.k), dtype=dtype, device=device)
    up = (torch.randn((args.n, args.k), dtype=dtype, device=device) / args.k**0.5).contiguous()
    down = (torch.randn((args.k, args.n), dtype=dtype, device=device) / args.n**0.5).contiguous()
    operations = []
    if args.operation in ("gemm", "both"):
        operations += [("gemm", "torch", lambda: gemm_reference(x, up))]
        if args.backend in ("fixed", "both"):
            operations += [("gemm", "fixed", lambda: triton_gemm(x, up))]
        if args.backend in ("tuned", "both"):
            operations += [("gemm", "tuned", lambda: triton_gemm_tuned(x, up))]
    if args.operation in ("mlp", "both"):
        operations += [("mlp", "torch", lambda: expert_mlp_reference(x, up, down))]
        if args.backend in ("fixed", "both"):
            operations += [("mlp", "fixed", lambda: expert_mlp_triton(x, up, down))]
        if args.backend in ("tuned", "both"):
            operations += [("mlp", "tuned", lambda: expert_mlp_triton(x, up, down, tuned=True))]

    tolerance = {torch.float32: 1e-4, torch.float16: 1e-2, torch.bfloat16: 2e-2}[dtype]
    checks = []
    with torch.inference_mode():
        references = {"gemm": gemm_reference(x, up), "mlp": expert_mlp_reference(x, up, down)}
        # 首次调用包含 JIT 与 AutoTune，全部完成后才开始任何正式计时。
        for kind, backend, operation in operations:
            actual = operation()
            expected = references[kind]
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            assert actual.shape == expected.shape and actual.dtype == dtype
            assert actual.device == device and actual.is_contiguous()
            checks.append({"operation": kind, "backend": backend, "passed": True,
                           "shape": list(actual.shape), "max_abs_error":
                           float((actual - expected).abs().max()) if actual.numel() else 0.0})
        torch.cuda.synchronize(device)
        print(f"PASS: {len(checks)} 项数值与输出属性对照；X={tuple(x.shape)}，W={tuple(up.shape)}")
        configs = {"up": selected_config(x, up)}
        if args.operation in ("mlp", "both"):
            activated = torch.empty((args.m, args.n), dtype=dtype, device=device)
            configs["down"] = selected_config(activated, down)
        print("选定配置：" + json.dumps(configs, ensure_ascii=False))
        records = []
        for kind, backend, operation in operations:
            stats = measure_latency(operation, device=device, warmup=args.warmup, repeats=args.repeats)
            flops = 2 * args.m * args.n * args.k * (2 if kind == "mlp" else 1)
            record = {"operation": kind, "backend": backend, "median_us": stats.median_ms * 1000,
                      "mean_us": stats.mean_ms * 1000, "samples_ms": list(stats.samples_ms),
                      "flops": flops, "tflops": flops / (stats.median_ms * 1e9) if stats.median_ms else 0.0}
            records.append(record)
            print(f"{kind:4s} {backend:5s}: {record['median_us']:9.3f} μs  {record['tflops']:8.3f} TFLOP/s")
    report = {"config": {"tokens": args.m, "batch_size": None, "seq_len": None,
              "hidden_size": args.k, "expert_hidden_size": args.n, "num_experts": 1,
              "top_k": 1, "ep_size": 1, "chunk_size": None, "dtype": args.dtype,
              "device": str(device), "seed": args.seed, "warmup": args.warmup,
              "repeats": args.repeats, "operation": args.operation, "backend": args.backend},
              "environment": {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "triton": triton.__version__, "cuda": torch.version.cuda, "python": platform.python_version()},
              "timing": "单 GPU CUDA Event 完整调用；取中位数，包含输出分配与流内提交间隙；JIT/调优在计时前完成",
              "selected_configs": configs, "tolerance": {"rtol": tolerance, "atol": tolerance},
              "checks": checks, "results": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(f"结果保存到 {args.output}")


if __name__ == "__main__":
    main()
