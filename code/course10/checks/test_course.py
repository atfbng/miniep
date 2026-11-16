"""从既有测试中保留与本课主题相关的用例。"""
import unittest
import torch
from mini_ep.distributed.dispatch import dispatch_reference, combine_reference
from mini_ep.kernels.combine_triton import combine_triton
from mini_ep.kernels.histogram_triton import triton

@unittest.skipUnless(torch.cuda.is_available() and triton is not None, 'CUDA and Triton required')
class TestAdvancedCUDAKernels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_tf32

    def test_combine_numerics_tail_and_empty(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for (t, h, k) in [(0, 33, 2), (1, 7, 1), (3, 5, 2), (129, 127, 2)]:
                with self.subTest(dtype=dtype, t=t, h=h, k=k):
                    x = torch.randn(t, h, device='cuda', dtype=dtype)
                    ids = torch.stack([(torch.arange(t, device='cuda') + s) % 4 for s in range(k)], 1)
                    weights = torch.rand(t, k, device='cuda')
                    weights /= weights.sum(1, keepdim=True)
                    dispatch = dispatch_reference(x, ids, weights, 4)
                    y = torch.randn(t * k, h * 2, device='cuda', dtype=dtype)[:, ::2]
                    torch.testing.assert_close(combine_triton(y, dispatch), combine_reference(y, dispatch), rtol=0.0001, atol=1e-05)

if __name__ == "__main__":
    unittest.main()
