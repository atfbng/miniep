"""从既有测试中保留与本课主题相关的用例。"""
import unittest
import torch
from mini_ep.distributed.dispatch import dispatch_reference
from mini_ep.kernels.dispatch_triton import dispatch_triton
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

    def test_dispatch_metadata_all_layouts(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for (t, h, e, k) in [(0, 17, 4, 2), (1, 33, 4, 1), (3, 5, 4, 2), (129, 127, 8, 2), (257, 65, 3, 2)]:
                with self.subTest(dtype=dtype, t=t, h=h, e=e, k=k):
                    x = torch.randn(t, h * 2, device='cuda', dtype=dtype)[:, ::2]
                    ids = torch.stack([(torch.arange(t, device='cuda') + s) % min(2, e) for s in range(k)], 1).to(torch.int64 if t == 257 else torch.int32)
                    weights = torch.rand(t, k, device='cuda')
                    weights /= weights.sum(1, keepdim=True)
                    reference = dispatch_reference(x, ids, weights, e)
                    actual = dispatch_triton(x, ids, weights, e)
                    for field in vars(reference):
                        (a, b) = (getattr(actual, field), getattr(reference, field))
                        if isinstance(a, torch.Tensor):
                            torch.testing.assert_close(a, b, rtol=0, atol=0)
                        else:
                            self.assertEqual(a, b)
                    self.assertTrue(actual.packed_tokens.is_contiguous())
                    records = torch.arange(t * k, device='cuda', dtype=torch.int64)
                    torch.testing.assert_close(actual.restore_index.argsort().index_select(0, actual.restore_index), records)

if __name__ == "__main__":
    unittest.main()
