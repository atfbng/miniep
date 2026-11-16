"""新增下投影的边界、缓存与完整 MLP 对照。"""
import importlib.util
import unittest
import torch
from mini_ep.model.expert import expert_mlp_reference, gemm_reference

@unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec('triton'),
                     '需要 CUDA GPU 与 Triton')
class TestDownProjection(unittest.TestCase):
    def test_hint_config_masks_and_candidates(self):
        from mini_ep.kernels.down_projection import DOWN_CANDIDATES
        from mini_ep.kernels.gemm_triton import CANDIDATES, gemm_kernel
        from triton.runtime.errors import OutOfResources
        self.assertEqual(len(DOWN_CANDIDATES), len(set(DOWN_CANDIDATES)))
        self.assertTrue(set(CANDIDATES).issubset(DOWN_CANDIDATES))
        self.assertIn((128, 128, 64, 4, 4), DOWN_CANDIDATES)
        torch.manual_seed(31)
        # 非整齐行数与 K，检查建议配置的行、列、K 三种 mask。
        x = torch.randn(17, 97, device='cuda', dtype=torch.float16)
        weight = torch.randn(65, 97, device='cuda', dtype=torch.float16)
        output = torch.empty(17, 65, device='cuda', dtype=torch.float16)
        try:
            gemm_kernel[(1, 1)](x, weight, output, 17, 65, 97,
                BM=128, BN=128, BK=64, num_warps=4, num_stages=4)
        except OutOfResources as error:
            self.skipTest(f'建议配置超出当前 GPU 资源：{error}')
        torch.testing.assert_close(output, x @ weight.T, rtol=1e-2, atol=1e-2)
        self.assertEqual(output.shape, (17, 65))
        self.assertEqual(output.dtype, x.dtype)
        self.assertEqual(output.device, x.device)
        self.assertTrue(output.is_contiguous())

    def test_values_layout_cache_and_mlp(self):
        from mini_ep.kernels.down_projection import (
            down_projection_tuned, down_selected_config, expert_mlp_down_tuned)
        flags = (torch.backends.cuda.matmul.allow_tf32,
                 torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        try:
            torch.manual_seed(31)
            for dtype, tol in ((torch.float32, 1e-4), (torch.float16, 1e-2),
                               (torch.bfloat16, 2e-2)):
                if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                    continue
                for m in (0, 1, 17):
                    with self.subTest(dtype=dtype, m=m), torch.inference_mode():
                        x = torch.randn(m, 65, device='cuda', dtype=dtype)
                        up = (torch.randn(97, 65, device='cuda', dtype=dtype)/65**.5).contiguous()
                        down = (torch.randn(65, 97, device='cuda', dtype=dtype)/97**.5).contiguous()
                        hidden = torch.randn(m, 97, device='cuda', dtype=dtype)
                        for _ in range(2):
                            pairs = [(down_projection_tuned(hidden, down), gemm_reference(hidden, down)),
                                     (expert_mlp_down_tuned(x, up, down), expert_mlp_reference(x, up, down))]
                            for actual, expected in pairs:
                                torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
                                self.assertEqual(actual.shape, expected.shape)
                                self.assertEqual(actual.dtype, dtype)
                                self.assertEqual(actual.device, x.device)
                                self.assertTrue(actual.is_contiguous())
                        if m:
                            self.assertIsNotNone(down_selected_config(hidden, down))
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                down_projection_tuned(torch.ones(1, 3), torch.ones(2, 3))
            with self.assertRaisesRegex(ValueError, '连续'):
                down_projection_tuned(torch.ones(1, 3, device='cuda'), torch.ones(3, 2, device='cuda').T)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = flags

if __name__ == '__main__':
    unittest.main()
