"""从既有测试中保留与本课主题相关的用例。"""
import unittest
import torch
from mini_ep.utils.course_workload import make_workload, load_metrics

class TestWorkloadAndGroupedReference(unittest.TestCase):

    def test_fixed_global_routing_and_load(self):
        for k in [1, 2]:
            (model, x) = make_workload(32, 16, 32, 8, k, 'uniform')
            routing = model.router(x)
            self.assertEqual(routing.tokens_per_expert.tolist(), [4 * k] * 8)
            for p in [1, 2, 4]:
                parts = [model.router(part).tokens_per_expert for part in x.chunk(p)]
                torch.testing.assert_close(torch.stack(parts).sum(0), routing.tokens_per_expert)
                self.assertEqual(load_metrics(torch.stack(parts), p)['rank_imbalance'], 1)
        (model, x) = make_workload(32, 16, 32, 8, 2, 'hotspot')
        counts = model.router(x).tokens_per_expert
        self.assertEqual(counts[0].item(), 32)
        self.assertEqual(load_metrics(counts, 4)['rows_per_rank'], [37, 10, 9, 8])

if __name__ == "__main__":
    unittest.main()
