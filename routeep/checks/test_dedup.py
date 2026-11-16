"""CPU 去重布局与精度检查，不依赖传输后端或可选 GPU 模块。

验证内容：
    - 唯一向量打包与 rank 展开后的记录完全一致；
    - lookup/restore_index 能把分支和 Token 映射回原位置；
    - owner 归约保持“分支先舍入、slot 顺序 FP32 累加、owner 再舍入”的规则；
    - 空输入、缺失 owner 和抵消反例。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from mini_ep.distributed.dedup import (
    pack_token_owners, expand_owner_branches, reduce_owner_branches, combine_owner_partials,
)
from mini_ep.distributed.dispatch import pack_by_destination_rank
from mini_ep.distributed.topology import ExpertParallelTopology
from reference.dedup import pack_token_owners_reference
from reference.dispatch import combine_reference


class DedupTests(unittest.TestCase):
    def test_user_example_and_logical_payload(self):
        x = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1, 2, 3, 128, 129, 130, 131]])
        weights = torch.full((1, 8), 1/8)
        topology = ExpertParallelTopology(256, 2, 0)
        packed = pack_token_owners(x, ids, weights, topology)
        self.assertEqual(packed.send_splits.tolist(), [1, 1])
        self.assertEqual(packed.packed_tokens.tolist(), [[1., 2.], [1., 2.]])
        self.assertEqual(packed.packed_expert_ids[1].tolist(), [-1]*4 + [128, 129, 130, 131])
        self.assertEqual(packed.restore_index.tolist(), [[0], [1]])
        self.assertEqual(pack_by_destination_rank(x, ids, weights, topology).send_splits.tolist(), [4, 4])
        large = torch.zeros(1, 2048, dtype=torch.bfloat16)
        unique = pack_token_owners(large, ids, weights, topology)
        expanded = pack_by_destination_rank(large, ids, weights, topology)
        vector_bytes = unique.packed_tokens.shape[1] * unique.packed_tokens.element_size()
        self.assertEqual(2 * int(unique.send_splits[1]) * vector_bytes, 8192)
        self.assertEqual(2 * int(expanded.send_splits[1]) * vector_bytes, 32768)

    def test_independent_pack_and_branch_expansion(self):
        torch.manual_seed(926)
        for dtype in (torch.float32, torch.bfloat16):
            for p in (1, 2, 4):
                topo = ExpertParallelTopology(8, p, 0)
                for n in (0, 1, 9):
                    for k in (1, 2, 8):
                        for skew in (False, True):
                            x = torch.randn(n, 14).to(dtype)[:, ::2]
                            ids = torch.randint(8, (n, 2*k))[:, ::2]
                            if skew: ids.zero_()
                            weights = torch.rand(n, 2*k)[:, ::2]
                            actual = pack_token_owners(x, ids, weights, topo)
                            expected = pack_token_owners_reference(x, ids, weights, topo)
                            for key in actual.__dataclass_fields__:
                                a, b = getattr(actual, key), getattr(expected, key)
                                torch.testing.assert_close(a, b, rtol=0, atol=0)
                                self.assertEqual((a.dtype, a.device), (b.dtype, b.device))
                                self.assertTrue(a.is_contiguous())
                            branch = pack_by_destination_rank(x, ids, weights, topo)
                            expanded, eids, eweights, lookup = expand_owner_branches(
                                actual.packed_tokens, actual.packed_expert_ids, actual.packed_routing_weights)
                            for a, b in ((expanded, branch.packed_tokens), (eids, branch.packed_expert_ids), (eweights, branch.packed_routing_weights)):
                                torch.testing.assert_close(a, b, atol=0, rtol=0)
                            self.assertEqual(int((lookup >= 0).sum()), n*k)

    def test_rounding_and_order_match_original_combine(self):
        torch.manual_seed(27)
        for dtype in (torch.float32, torch.bfloat16):
            for rd in (torch.float32, torch.bfloat16, torch.float16):
                for p in (1, 2, 4):
                    n, k, h = 9, 8, 7
                    ids = torch.randint(8, (n, k));weights = torch.rand(n, k)
                    topo = ExpertParallelTopology(8, p, 0)
                    packed = pack_token_owners(torch.zeros(n,h,dtype=dtype), ids, weights, topo)
                    _, _, _, lookup = expand_owner_branches(packed.packed_tokens, packed.packed_expert_ids, packed.packed_routing_weights)
                    original = pack_by_destination_rank(torch.zeros(n,h,dtype=dtype), ids, weights, topo)
                    # 合成输出已包含路由权重；归约不能再次加权，也不能改变分支/owner 舍入边界。
                    outputs = (torch.randn(n*k,h)*300).to(rd)
                    actual = combine_owner_partials(reduce_owner_branches(outputs, lookup, model_dtype=dtype), packed.restore_index)
                    expected = combine_reference(outputs, original.restore_index, ids, topo, model_dtype=dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        trap = torch.tensor([[256.], [1.], [-256.], [1.]], dtype=torch.bfloat16)
        owner = reduce_owner_branches(trap, torch.tensor([[0,1,2,3]]), model_dtype=torch.bfloat16)
        self.assertEqual(owner.item(), 2.)
        self.assertEqual(reduce_owner_branches(torch.tensor([[1.],[2.],[3.],[4.]]), torch.tensor([[0,1,2,3]]), model_dtype=torch.bfloat16).item(), 10.)

    def test_missing_owners_and_empty(self):
        for n in (0, 3):
            actual = combine_owner_partials(torch.empty(0, 2, dtype=torch.bfloat16), torch.full((4,n), -1))
            self.assertEqual(actual.shape, (n, 2))
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(actual.count_nonzero().item(), 0)
        for k in (1,8):
            reduced = reduce_owner_branches(torch.empty(0,2), torch.empty(0,k,dtype=torch.long), model_dtype=torch.float32)
            self.assertEqual(reduced.shape, (0,2))


if __name__ == '__main__':
    unittest.main()


class FixedOwnerPackTests(unittest.TestCase):
    def test_fixed_pack_matches_variable_layout(self):
        from mini_ep.distributed.dedup import pack_token_owners_fixed
        torch.manual_seed(1027)
        for p in (1, 2, 4):
            topo = ExpertParallelTopology(8, p, 0)
            for n in (0, 1, 9):
                for k in (1, 2, 8):
                    for skew in (False, True):
                        for extra in (0, 3):
                            x = torch.randn(n, 14, dtype=torch.bfloat16)[:, ::2]
                            ids = torch.randint(8, (n, 2 * k))[:, ::2]
                            if skew: ids.zero_()
                            w = torch.rand(n, 2 * k)[:, ::2]
                            c = max(n, 1) + extra
                            var = pack_token_owners(x, ids, w, topo)
                            fix = pack_token_owners_fixed(x, ids, w, topo, c)
                            self.assertEqual(fix.packed_tokens.shape, (p * c, 7))
                            self.assertEqual(fix.send_splits.tolist(), var.send_splits.tolist())
                            # 每个目标段的前 count 行与变长布局逐元素相同；之后是 padding。
                            start = 0
                            for r, count in enumerate(var.send_splits.tolist()):
                                seg = slice(r * c, r * c + count)
                                for a, b in ((fix.packed_tokens, var.packed_tokens),
                                             (fix.packed_expert_ids, var.packed_expert_ids),
                                             (fix.packed_routing_weights, var.packed_routing_weights)):
                                    torch.testing.assert_close(a[seg], b[start:start + count], atol=0, rtol=0)
                                pad = slice(r * c + count, (r + 1) * c)
                                self.assertTrue((fix.packed_expert_ids[pad] == -1).all())
                                self.assertTrue((fix.packed_routing_weights[pad] == 0).all())
                                # restore 指向同一行内容：变长行号 start+i <-> 固定行号 r*C+i。
                                v = var.restore_index[r]
                                expected = torch.where(v >= 0, v - start + r * c, -1)
                                torch.testing.assert_close(fix.restore_index[r], expected, atol=0, rtol=0)
                                start += count
                            self.assertEqual(fix.packed_expert_ids.dtype, torch.int64)
                            self.assertEqual(fix.packed_routing_weights.dtype, torch.float32)

    def test_fixed_capacity_too_small_fails(self):
        from mini_ep.distributed.dedup import pack_token_owners_fixed
        with self.assertRaisesRegex(ValueError, "capacity"):
            pack_token_owners_fixed(torch.zeros(3, 2), torch.zeros(3, 1, dtype=torch.long),
                                    torch.ones(3, 1), ExpertParallelTopology(2, 2, 0), 2)
