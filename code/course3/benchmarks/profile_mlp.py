"""先调优、验证和预热，再用 Proton 采集 Expert MLP 的三个阶段。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from mini_ep.model.expert import expert_mlp_reference, gemm_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--backend", choices=("fixed", "tuned", "both"), default="tuned")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--output", type=Path, default=Path("lesson03_mlp_warm"))
    args = parser.parse_args()
    if min(args.m, args.n, args.k, args.warmup, args.repeats) <= 0:
        parser.error("M、N、K、warmup、repeats 必须大于 0")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("Proton MLP 分析需要 CUDA GPU")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.index >= torch.cuda.device_count():
        parser.error(f"device={device}，仅有 {torch.cuda.device_count()} 张可见 GPU")
    import triton
    import triton.profiler as proton
    from mini_ep.kernels.gemm_triton import selected_config, triton_gemm, triton_gemm_tuned

    torch.cuda.set_device(device)
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        parser.error("所选 GPU 不支持 BF16")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dtype = getattr(torch, args.dtype)
    # 连续行主序 X[M,K]、up[N,K]、down[K,N]；输出保持输入行序。
    x = torch.randn((args.m, args.k), device=device, dtype=dtype)
    up = (torch.randn((args.n, args.k), device=device, dtype=dtype) / args.k**0.5).contiguous()
    down = (torch.randn((args.k, args.n), device=device, dtype=dtype) / args.n**0.5).contiguous()
    variants = [("TORCH_MLP", gemm_reference)]
    if args.backend in ("fixed", "both"):
        variants.append(("TRITON_FIXED_MLP", triton_gemm))
    if args.backend in ("tuned", "both"):
        variants.append(("TRITON_TUNED_MLP", triton_gemm_tuned))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tolerance = {torch.float16: 1e-2, torch.bfloat16: 2e-2, torch.float32: 1e-4}[dtype]
    checks = []
    with torch.inference_mode():
        expected = expert_mlp_reference(x, up, down)
        for label, project in variants:
            # 首次上下投影完成各自的 JIT / AutoTune，采集尚未启动。
            actual = project(F.gelu(project(x, up)), down)
            torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
            assert actual.shape == x.shape and actual.dtype == dtype
            assert actual.device == device and actual.is_contiguous()
            checks.append({"backend": label, "passed": True,
                           "max_abs_error": float((actual - expected).abs().max())})
            for _ in range(args.warmup):
                project(F.gelu(project(x, up)), down)
        # 等全部预热 GPU 工作完成，随后启动采集。
        torch.cuda.synchronize(device)
        print("PASS: 调优、数值检查与预热完成；开始 Proton 采集", flush=True)
        # 首次启用 CUPTI 后，让各 Kernel 的采集路径也先执行一次。
        # 初始化记录单独保存，正式报告仍只有 repeats 次稳态调用。
        setup = proton.start(str(args.output) + "_setup", hook="triton")
        try:
            with proton.scope("PROFILER_SETUP"):
                for _ in range(10):
                    for _, project in variants:
                        project(F.gelu(project(x, up)), down)
            torch.cuda.synchronize(device)
            proton.deactivate(setup)
        except BaseException:
            proton.finalize(setup)
            raise
        session = proton.start(str(args.output), hook="triton")
        try:
            # 每轮交替顺序，各路径使用同一份输入、权重和 PyTorch GELU。
            flops_key = "flops32" if dtype == torch.float32 else "flops16"
            flops = 2 * args.m * args.n * args.k
            size = x.element_size()
            projection_bytes = size * (args.m * args.k + args.n * args.k + args.m * args.n)
            for repeat in range(args.repeats):
                ordered = variants if repeat % 2 == 0 else variants[::-1]
                for label, project in ordered:
                    with proton.scope(label):
                        with proton.scope("UP", {flops_key: flops, "bytes": projection_bytes}), proton.scope(f"CALL_{repeat:04d}"):
                            hidden = project(x, up)
                        with proton.scope("GELU", {"bytes": 2 * size * args.m * args.n}), proton.scope(f"CALL_{repeat:04d}"):
                            activated = F.gelu(hidden)
                        with proton.scope("DOWN", {flops_key: flops, "bytes": projection_bytes}), proton.scope(f"CALL_{repeat:04d}"):
                            result = project(activated, down)
            torch.cuda.synchronize(device)
        finally:
            proton.finalize(session)
            proton.finalize(setup)
    config = vars(args).copy()
    config["output"] = str(args.output)
    metadata = {"config": config, "environment": {
        "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda, "python": platform.python_version()},
        "selected_configs": {"up": selected_config(x, up), "down": selected_config(activated, down)},
        "checks": checks, "tensor_layout": "X[M,K], up[N,K], down[K,N]；连续行主序",
        "profiler_setup": "采集路径预热 10 次，记录另存 *_setup.hatchet；不计入正式报告",
        "num_experts": 1, "top_k": 1, "ep_size": 1,
        "timing": "预热后采集；各阶段 GPU Kernel 时间累加 / repeats；不含调优与预热；bytes 为张量大小估算"}
    args.output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"报告：{args.output}.hatchet", flush=True)


if __name__ == "__main__":
    main()
