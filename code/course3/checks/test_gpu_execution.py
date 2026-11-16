"""Correctness tests for lesson 3's GEMM and timing experiment."""

from __future__ import annotations

import unittest

import torch

from benchmarks.benchmark_gemm import (
    GemmBenchmarkResult,
    GemmShape,
    create_linear,
    create_operands,
    matmul_gemm,
    run_gemm_benchmark,
)
from mini_ep.profiler import LatencyStats, measure_latency


class TestLatencyStats(unittest.TestCase):
    def test_summary_statistics_use_milliseconds(self) -> None:
        stats = LatencyStats((1.0, 4.0, 2.0, 3.0))

        self.assertEqual(stats.repeats, 4)
        self.assertEqual(stats.median_ms, 2.5)
        self.assertEqual(stats.mean_ms, 2.5)
        self.assertEqual(stats.min_ms, 1.0)
        self.assertEqual(stats.max_ms, 4.0)

    def test_rejects_empty_or_negative_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            LatencyStats(())
        with self.assertRaisesRegex(ValueError, "non-negative"):
            LatencyStats((0.1, -0.1))
        with self.assertRaisesRegex(ValueError, "finite"):
            LatencyStats((float("nan"),))


class TestGemmReference(unittest.TestCase):
    def test_matmul_and_linear_are_equivalent(self) -> None:
        shape = GemmShape(m=3, n=5, k=4)
        tokens, weight = create_operands(
            shape,
            dtype=torch.float32,
            device="cpu",
            seed=7,
        )

        matmul_output = matmul_gemm(tokens, weight)
        linear = create_linear(weight)
        linear_output = linear(tokens)

        self.assertEqual(matmul_output.shape, (3, 5))
        self.assertEqual(matmul_output.dtype, torch.float32)
        self.assertEqual(matmul_output.device.type, "cpu")
        self.assertIsNone(linear.bias)
        self.assertFalse(linear.weight.requires_grad)
        torch.testing.assert_close(linear.weight, weight)
        torch.testing.assert_close(matmul_output, linear_output)

    def test_shape_and_flop_count(self) -> None:
        shape = GemmShape(m=2, n=3, k=4)

        self.assertEqual(shape.output_shape, (2, 3))
        self.assertEqual(shape.flops, 48)

        result = GemmBenchmarkResult(
            operation="matmul",
            shape=shape,
            dtype=torch.float32,
            device=torch.device("cpu"),
            latency=LatencyStats((2.0,)),
        )
        self.assertAlmostEqual(result.median_tflops, 48 / (2.0e9))

    def test_seed_reproduces_contiguous_operands(self) -> None:
        shape = GemmShape(m=4, n=3, k=2)
        first = create_operands(
            shape, dtype=torch.float32, device="cpu", seed=11
        )
        second = create_operands(
            shape, dtype=torch.float32, device="cpu", seed=11
        )

        for first_tensor, second_tensor in zip(first, second, strict=True):
            self.assertTrue(first_tensor.is_contiguous())
            torch.testing.assert_close(first_tensor, second_tensor)

    def test_rejects_invalid_shapes_and_layouts(self) -> None:
        for dimensions in ((0, 2, 3), (2, -1, 3), (2, 3, False)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    GemmShape(*dimensions)

        with self.assertRaisesRegex(ValueError, "share K"):
            matmul_gemm(torch.randn(2, 3), torch.randn(4, 2))
        with self.assertRaisesRegex(TypeError, "share dtype"):
            matmul_gemm(torch.randn(2, 3), torch.randn(4, 3).half())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            matmul_gemm(torch.randn(3, 4).transpose(0, 1), torch.randn(2, 3))


class TestMeasurement(unittest.TestCase):
    def test_cpu_timer_honors_warmup_and_repeat_counts(self) -> None:
        calls = 0
        value = torch.ones(128)

        def operation() -> torch.Tensor:
            nonlocal calls
            calls += 1
            return value + 1.0

        stats = measure_latency(operation, device="cpu", warmup=2, repeats=3)

        self.assertEqual(calls, 5)
        self.assertEqual(stats.repeats, 3)
        self.assertTrue(all(sample >= 0.0 for sample in stats.samples_ms))

    def test_cpu_gemm_benchmark_preserves_metadata(self) -> None:
        shape = GemmShape(m=16, n=12, k=8)
        tokens, weight = create_operands(
            shape, dtype=torch.float32, device="cpu", seed=13
        )

        result = run_gemm_benchmark(
            "linear", tokens, weight, warmup=1, repeats=2
        )

        self.assertEqual(result.operation, "linear")
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, torch.device("cpu"))
        self.assertEqual(result.latency.repeats, 2)
        self.assertGreaterEqual(result.median_tflops, 0.0)

    def test_rejects_invalid_measurement_configuration(self) -> None:
        operation = lambda: torch.ones(1)
        for warmup, repeats in ((-1, 1), (0, 0), (True, 1), (0, False)):
            with self.subTest(warmup=warmup, repeats=repeats):
                with self.assertRaisesRegex(ValueError, "warmup|repeats"):
                    measure_latency(
                        operation,
                        device="cpu",
                        warmup=warmup,
                        repeats=repeats,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_event_timer_and_fp16_gemm(self) -> None:
        shape = GemmShape(m=64, n=48, k=32)
        tokens, weight = create_operands(
            shape, dtype=torch.float16, device="cuda", seed=17
        )

        result = run_gemm_benchmark(
            "matmul", tokens, weight, warmup=2, repeats=3
        )
        expected = create_linear(weight)(tokens)
        actual = matmul_gemm(tokens, weight)

        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.dtype, torch.float16)
        self.assertEqual(result.latency.repeats, 3)
        self.assertTrue(all(sample >= 0.0 for sample in result.latency.samples_ms))
        self.assertEqual(actual.shape, (64, 48))
        self.assertEqual(actual.device.type, "cuda")
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA GPUs required")
    def test_cuda_timer_selects_the_operand_device(self) -> None:
        original_device = torch.cuda.current_device()
        shape = GemmShape(m=16, n=12, k=8)
        tokens, weight = create_operands(
            shape, dtype=torch.float32, device="cuda:1", seed=19
        )

        result = run_gemm_benchmark(
            "matmul", tokens, weight, warmup=1, repeats=2
        )

        self.assertEqual(result.device, torch.device("cuda:1"))
        self.assertEqual(torch.cuda.current_device(), original_device)


if __name__ == "__main__":
    unittest.main()
