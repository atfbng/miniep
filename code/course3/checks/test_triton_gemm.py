"""Triton 投影的边界、数值与布局；CPU Reference 可独立验证。"""

from __future__ import annotations

import importlib.util
import unittest

import torch
import torch.nn.functional as F

from mini_ep.model.expert import expert_mlp_reference, gemm_reference


class TestExpertReference(unittest.TestCase):
    def test_projection_preserves_rows_and_weight_orientation(self) -> None:
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        weight = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        actual = gemm_reference(x, weight)
        torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 7.0]]))
        self.assertEqual(actual.shape, (2, 3))
        self.assertEqual(actual.dtype, x.dtype)
        self.assertEqual(actual.device, x.device)
        self.assertTrue(actual.is_contiguous())

    def test_mlp_and_empty_expert(self) -> None:
        torch.manual_seed(31)
        up, down = torch.randn(5, 3), torch.randn(3, 5)
        for m in (0, 1, 7):
            with self.subTest(m=m):
                x = torch.randn(m, 3)
                actual = expert_mlp_reference(x, up, down)
                torch.testing.assert_close(actual, F.gelu(x @ up.T) @ down.T)
                self.assertEqual(actual.shape, (m, 3))
                self.assertTrue(actual.is_contiguous())

    def test_invalid_shapes_dtype_and_layout(self) -> None:
        x, up = torch.ones(2, 3), torch.ones(5, 3)
        with self.assertRaisesRegex(ValueError, "输入宽度"):
            gemm_reference(x, torch.ones(5, 4))
        with self.assertRaisesRegex(ValueError, "X\\[M,K\\]"):
            gemm_reference(x[0], up)
        with self.assertRaisesRegex(TypeError, "dtype"):
            gemm_reference(x, up.to(torch.float16))
        with self.assertRaisesRegex(ValueError, "连续"):
            gemm_reference(torch.ones(2, 6)[:, ::2], up)
        with self.assertRaisesRegex(ValueError, "下投影权重"):
            expert_mlp_reference(x, up, torch.ones(4, 5))
        with self.assertRaisesRegex(TypeError, "dtype"):
            expert_mlp_reference(x, up, torch.ones(3, 5, dtype=torch.float16))


@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("triton"),
                     "需要 CUDA GPU 与 Triton")
class TestTritonProjection(unittest.TestCase):
    def setUp(self) -> None:
        self.old_tf32 = torch.backends.cuda.matmul.allow_tf32
        self.old_reduction = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.manual_seed(31)

    def tearDown(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = self.old_tf32
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self.old_reduction

    def assert_output(self, actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device, expected.device)
        self.assertTrue(actual.is_contiguous())

    def test_fixed_masks_dtypes_and_mlp(self) -> None:
        from mini_ep.kernels.gemm_triton import expert_mlp_triton, triton_gemm

        for dtype, tolerance in [(torch.float32, 1e-4), (torch.float16, 1e-2),
                                 (torch.bfloat16, 2e-2)]:
            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                continue
            for m, h, f in [(0, 7, 5), (1, 7, 5), (17, 65, 97), (32, 64, 64)]:
                with self.subTest(dtype=dtype, shape=(m, h, f)), torch.inference_mode():
                    x = torch.randn(m, h, device="cuda", dtype=dtype)
                    up = torch.randn(f, h, device="cuda", dtype=dtype) / h**0.5
                    down = torch.randn(h, f, device="cuda", dtype=dtype) / f**0.5
                    self.assert_output(triton_gemm(x, up), gemm_reference(x, up), tolerance)
                    self.assert_output(expert_mlp_triton(x, up, down),
                                       expert_mlp_reference(x, up, down), tolerance)

    def test_tuned_non_aligned_mlp_and_cache(self) -> None:
        from mini_ep.kernels.gemm_triton import (
            expert_mlp_triton, selected_config, triton_gemm_tuned,
        )

        with torch.inference_mode():
            x = torch.randn(17, 65, device="cuda", dtype=torch.float16)
            up = torch.randn(97, 65, device="cuda", dtype=torch.float16) / 65**0.5
            down = torch.randn(65, 97, device="cuda", dtype=torch.float16) / 97**0.5
            expected = gemm_reference(x, up)
            for _ in range(2):
                self.assert_output(triton_gemm_tuned(x, up), expected, 1e-2)
            self.assertIsNotNone(selected_config(x, up))
            for _ in range(2):
                self.assert_output(expert_mlp_triton(x, up, down, tuned=True),
                                   expert_mlp_reference(x, up, down), 1e-2)
            activated = torch.empty(17, 97, device="cuda", dtype=torch.float16)
            self.assertIsNotNone(selected_config(activated, down))
            empty = x[:0]
            self.assert_output(expert_mlp_triton(empty, up, down, tuned=True),
                               expert_mlp_reference(empty, up, down), 1e-2)

    def test_cuda_requirement_and_input_validation(self) -> None:
        from mini_ep.kernels.gemm_triton import triton_gemm

        with self.assertRaisesRegex(ValueError, "CUDA"):
            triton_gemm(torch.ones(2, 3), torch.ones(5, 3))
        x = torch.ones(2, 3, device="cuda")
        with self.assertRaisesRegex(ValueError, "device"):
            triton_gemm(x, torch.ones(5, 3))
        with self.assertRaisesRegex(ValueError, "连续"):
            triton_gemm(x, torch.ones(3, 5, device="cuda").T)


if __name__ == "__main__":
    unittest.main()
