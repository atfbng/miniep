#!/usr/bin/env python3
"""Benchmark equivalent torch.matmul and bias-free torch.nn.Linear GEMMs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import sys
from pathlib import Path

import torch
from torch import nn

# Keep the benchmark runnable before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_ep.profiler import LatencyStats, measure_latency  # noqa: E402


SUPPORTED_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
CORRECTNESS_TOLERANCES = {
    torch.float32: (1.0e-5, 1.0e-5),
    torch.float16: (1.0e-3, 1.0e-3),
    torch.bfloat16: (1.0e-2, 1.0e-2),
}
SUPPORTED_OPERATIONS = ("matmul", "linear")


@dataclass(frozen=True)
class GemmShape:
    """Logical GEMM dimensions for ``[M, K] @ [K, N] -> [M, N]``."""

    m: int
    n: int
    k: int

    def __post_init__(self) -> None:
        for name, value in (("M", self.m), ("N", self.n), ("K", self.k)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

    @property
    def output_shape(self) -> tuple[int, int]:
        """Return the logical output shape ``[M, N]``."""

        return (self.m, self.n)

    @property
    def flops(self) -> int:
        """Return the conventional GEMM count ``2 * M * N * K``."""

        return 2 * self.m * self.n * self.k


@dataclass(frozen=True)
class GemmBenchmarkResult:
    """One operation's measured latency and derived GEMM throughput."""

    operation: str
    shape: GemmShape
    dtype: torch.dtype
    device: torch.device
    latency: LatencyStats

    @property
    def median_tflops(self) -> float:
        """Compute achieved TFLOPS from the median latency.

        This is a workload throughput metric, not GPU utilization and not a
        percentage of the device's advertised peak throughput.
        """

        if self.latency.median_ms == 0.0:
            return math.inf
        return self.shape.flops / (self.latency.median_ms * 1.0e9)


def matmul_gemm(tokens: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``tokens @ weight.T``.

    ``tokens`` must be floating point ``[M, K]`` and contiguous. ``weight``
    must be floating point ``[N, K]`` on the same device and with the same
    dtype. The returned Tensor has shape ``[M, N]``.
    """

    _validate_operands(tokens, weight)
    return torch.matmul(tokens, weight.transpose(0, 1))


def create_linear(weight: torch.Tensor) -> nn.Linear:
    """Create a bias-free ``nn.Linear`` that owns a copy of ``weight [N, K]``."""

    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K], got {tuple(weight.shape)}")
    if not weight.is_floating_point():
        raise TypeError("weight must use a floating-point dtype")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")

    linear = nn.Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        device=weight.device,
        dtype=weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(weight)
    linear.requires_grad_(False)
    return linear


def _validate_operands(tokens: torch.Tensor, weight: torch.Tensor) -> None:
    if tokens.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            "tokens and weight must have shapes [M, K] and [N, K], got "
            f"{tuple(tokens.shape)} and {tuple(weight.shape)}"
        )
    if tokens.shape[1] != weight.shape[1]:
        raise ValueError(
            "tokens and weight must share K, got "
            f"{tokens.shape[1]} and {weight.shape[1]}"
        )
    if not tokens.is_floating_point() or not weight.is_floating_point():
        raise TypeError("tokens and weight must use floating-point dtypes")
    if tokens.dtype != weight.dtype:
        raise TypeError(
            f"tokens and weight must share dtype, got {tokens.dtype} and {weight.dtype}"
        )
    if tokens.device != weight.device:
        raise ValueError(
            "tokens and weight must be on the same device, got "
            f"{tokens.device} and {weight.device}"
        )
    if not tokens.is_contiguous() or not weight.is_contiguous():
        raise ValueError("tokens and weight must be contiguous")


def create_operands(
    shape: GemmShape,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create reproducible contiguous operands ``[M, K]`` and ``[N, K]``."""

    if dtype not in SUPPORTED_DTYPES.values():
        raise TypeError(
            "dtype must be float32, float16, or bfloat16, "
            f"got {dtype}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")

    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(seed)
    tokens = torch.randn(
        shape.m,
        shape.k,
        dtype=dtype,
        device=resolved_device,
        generator=generator,
    )
    weight = torch.randn(
        shape.n,
        shape.k,
        dtype=dtype,
        device=resolved_device,
        generator=generator,
    )
    return tokens.contiguous(), weight.contiguous()


def run_gemm_benchmark(
    operation: str,
    tokens: torch.Tensor,
    weight: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> GemmBenchmarkResult:
    """Measure one GEMM implementation using fixed input and weight Tensors."""

    _validate_operands(tokens, weight)
    if operation not in SUPPORTED_OPERATIONS:
        raise ValueError(
            f"operation must be one of {SUPPORTED_OPERATIONS}, got {operation!r}"
        )

    shape = GemmShape(
        m=tokens.shape[0],
        n=weight.shape[0],
        k=tokens.shape[1],
    )
    if operation == "matmul":
        # Shape/layout validation and View creation belong to setup. The timed
        # callable contains only the public Matmul operation under study.
        transposed_weight = weight.transpose(0, 1)
        measured_operation = lambda: torch.matmul(tokens, transposed_weight)
    else:
        # Constructing the Module and copying its weight are setup costs, not
        # steady-state GEMM work, so both happen before the timer starts.
        linear = create_linear(weight)
        measured_operation = lambda: linear(tokens)
    latency = measure_latency(
        measured_operation,
        device=tokens.device,
        warmup=warmup,
        repeats=repeats,
    )
    return GemmBenchmarkResult(
        operation=operation,
        shape=shape,
        dtype=tokens.dtype,
        device=tokens.device,
        latency=latency,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024, help="output row count")
    parser.add_argument("--n", type=int, default=2048, help="output column count")
    parser.add_argument("--k", type=int, default=1024, help="reduction dimension")
    parser.add_argument(
        "--operation",
        choices=("matmul", "linear", "both"),
        default="both",
    )
    parser.add_argument(
        "--dtype",
        choices=tuple(SUPPORTED_DTYPES),
        default=None,
        help="default: float16 on CUDA, float32 on CPU",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="auto uses one CUDA GPU when available, otherwise CPU",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    """Resolve the CLI device selection without silently ignoring CUDA requests."""

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    shape = GemmShape(args.m, args.n, args.k)
    if args.warmup < 0:
        raise ValueError(f"--warmup must be non-negative, got {args.warmup}")
    if args.repeats <= 0:
        raise ValueError(f"--repeats must be positive, got {args.repeats}")
    if args.seed < 0:
        raise ValueError(f"--seed must be non-negative, got {args.seed}")

    device = resolve_device(args.device)
    dtype_name = args.dtype or ("float16" if device.type == "cuda" else "float32")
    dtype = SUPPORTED_DTYPES[dtype_name]
    tokens, weight = create_operands(
        shape,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    operation_names = (
        SUPPORTED_OPERATIONS
        if args.operation == "both"
        else (args.operation,)
    )
    results = [
        run_gemm_benchmark(
            operation,
            tokens,
            weight,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for operation in operation_names
    ]

    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU"
    )
    print("Mini-EP lesson 3 GEMM benchmark")
    print(
        f"device={device} ({device_name}), dtype={dtype}, "
        f"shape=[{shape.m},{shape.k}] @ [{shape.k},{shape.n}], "
        f"warmup={args.warmup}, repeats={args.repeats}, seed={args.seed}"
    )
    print(
        f"torch={torch.__version__}, "
        f"cuda_runtime={torch.version.cuda}, "
        f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
    )
    print("\noperation  median_ms  mean_ms  min_ms  max_ms  achieved_TFLOPS")
    for result in results:
        stats = result.latency
        print(
            f"{result.operation:<9}  {stats.median_ms:9.4f}  "
            f"{stats.mean_ms:7.4f}  {stats.min_ms:6.4f}  {stats.max_ms:6.4f}  "
            f"{result.median_tflops:15.4f}"
        )

    if len(results) == 2:
        with torch.inference_mode():
            matmul_output = matmul_gemm(tokens, weight)
            linear_output = create_linear(weight)(tokens)
        max_difference = float(
            (matmul_output.float() - linear_output.float()).abs().max().item()
        )
        rtol, atol = CORRECTNESS_TOLERANCES[dtype]
        torch.testing.assert_close(
            matmul_output,
            linear_output,
            rtol=rtol,
            atol=atol,
        )
        print(
            "\nCorrectness check: PASS, "
            f"output_shape={tuple(matmul_output.shape)}, "
            f"max_abs_difference={max_difference:.6g}, "
            f"rtol={rtol:g}, atol={atol:g}"
        )

    print(
        "\nNote: achieved TFLOPS is derived from 2*M*N*K and median latency; "
        "it is not a GPU-utilization percentage."
    )


if __name__ == "__main__":
    main()
