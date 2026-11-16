"""从既有测试中保留与本课主题相关的用例。"""
import unittest
import torch
from mini_ep.utils.course_workload import make_workload, load_metrics
from mini_ep.kernels.histogram_triton import triton
from mini_ep.kernels.grouped_gemm.reference import create_grouped_plan, expert_loop, expert_mlp_loop, pack_local_experts
from mini_ep.kernels.grouped_gemm.triton_impl import grouped_gemm, grouped_expert_mlp

class TestWorkloadAndGroupedReference(unittest.TestCase):

    def test_tiles_empty_experts_and_reference(self):
        for counts in [(2, 0, 1, 3), (0, 0, 0), (1, 17, 0, 33)]:
            plan = create_grouped_plan(counts, 'cpu')
            self.assertEqual(plan.offsets[-1].item(), sum(counts))
            x = torch.arange(sum(counts) * 3, dtype=torch.float32).reshape(-1, 3)
            w = torch.stack([torch.full((2, 3), float(e + 1)) for e in range(len(counts))])
            y = expert_loop(x, w, plan)
            start = 0
            for (e, m) in enumerate(counts):
                torch.testing.assert_close(y[start:start + m], x[start:start + m].sum(1, keepdim=True).expand(m, 2) * (e + 1))
                start += m
            self.assertEqual(y.shape, (sum(counts), 2))
            self.assertEqual(y.dtype, x.dtype)
            coverage = [0] * sum(counts)
            for (e, first) in zip(plan.tile_experts.tolist(), plan.tile_rows.tolist()):
                for row in range(first, min(first + 16, counts[e])):
                    coverage[plan.offsets[e].item() + row] += 1
            self.assertEqual(coverage, [1] * sum(counts))

    def test_receive_order_roundtrip(self):
        x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        indices = [torch.tensor([0, 3]), torch.tensor([], dtype=torch.int64), torch.tensor([1, 2, 4, 5])]
        (packed, origin, counts) = pack_local_experts(x, indices)
        returned = torch.empty_like(x)
        returned.index_copy_(0, origin, packed)
        torch.testing.assert_close(returned, x)
        self.assertEqual(counts, (2, 0, 4))

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            make_workload(10, 4, 8, 8, 2, 'uniform')
        with self.assertRaises(ValueError):
            create_grouped_plan([1, -1], 'cpu')
        with self.assertRaises(ValueError):
            load_metrics(torch.ones(8, dtype=torch.int64), 3)

@unittest.skipUnless(torch.cuda.is_available() and triton is not None, 'CUDA and Triton required')
class TestAdvancedCUDAKernels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.previous_tf32

    def test_grouped_gemm_and_mlp(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            for (counts, h, f) in [((2, 0, 1, 3), 13, 17), ((0, 0, 0), 32, 64), ((1, 17, 0, 33), 33, 65)]:
                with self.subTest(dtype=dtype, counts=counts, h=h, f=f):
                    plan = create_grouped_plan(counts, 'cuda')
                    x = torch.randn(sum(counts), h, device='cuda', dtype=dtype)
                    up = torch.randn(len(counts), f, h, device='cuda', dtype=dtype) * 0.1
                    down = torch.randn(len(counts), h, f, device='cuda', dtype=dtype) * 0.1
                    (rtol, atol) = (0.0001, 1e-05) if dtype == torch.float32 else (0.01, 0.002) if dtype == torch.float16 else (0.02, 0.008)
                    torch.testing.assert_close(grouped_gemm(x, up, plan), expert_loop(x, up, plan), rtol=rtol, atol=atol)
                    torch.testing.assert_close(grouped_expert_mlp(x, up, down, plan), expert_mlp_loop(x, up, down, plan), rtol=rtol, atol=atol)

if __name__ == "__main__":
    unittest.main()
