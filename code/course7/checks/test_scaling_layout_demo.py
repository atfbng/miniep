"""Check that conceptual tables are derived from the fixed Router."""
import unittest
from test.demo_scaling_layout import layout_data


class TestScalingLayoutDemo(unittest.TestCase):
    def test_strong_scaling_placement_routes_and_bytes(self):
        for ep, remote in [(1,0),(2,32),(4,48)]:
            plan=layout_data(32,8,2,ep)
            self.assertEqual(plan['tokens_per_expert'],[8]*8)
            self.assertEqual(plan['recv_rows'],[64//ep]*ep)
            self.assertEqual(plan['remote_rows'],remote)
            self.assertEqual(plan['expert_ids'][10],[4,5])
            self.assertEqual(sum(map(sum,plan['send_matrix'])),64)

    def test_weak_scaling_expert_batches(self):
        for ep in [1,2,4]:
            plan=layout_data(32*ep,8,2,ep)
            self.assertEqual(plan['tokens_per_expert'],[8*ep]*8)
            self.assertEqual(plan['recv_rows'],[64]*ep)

    def test_top1_counts(self):
        for ep in [1,2,4]:
            plan=layout_data(32,8,1,ep)
            self.assertEqual(plan['tokens_per_expert'],[4]*8)
            self.assertEqual(sum(map(sum,plan['send_matrix'])),32)
