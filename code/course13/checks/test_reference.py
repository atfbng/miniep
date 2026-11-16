"""CPU semantic tests; CUDA/NCCL coverage lives in distributed.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch
import torch.nn.functional as F
from mini_ep.distributed.topology import ExpertParallelTopology
from reference.dispatch import dispatch_reference, combine_reference
from mini_ep.distributed.dispatch import pack_by_destination_rank_fixed
from reference.moe import moe_reference
from reference.router import router_reference, routing_matches_reference


class ReferenceTests(unittest.TestCase):
    def test_grouped_plan_reference(self):
        from reference.grouped_plan import build_grouped_plan_reference

        plan = build_grouped_plan_reference([0, 1, 64, 65, 0], torch.device("cpu"), 64)
        expected = ([0, 0, 1, 65, 130, 130], [1, 2, 3, 3], [0, 0, 0, 64])
        for actual, values in zip(plan, expected, strict=True):
            torch.testing.assert_close(actual, torch.tensor(values), atol=0, rtol=0)
            self.assertEqual(actual.dtype, torch.int64)
            self.assertEqual(actual.device.type, "cpu")
            self.assertTrue(actual.is_contiguous())
        empty = build_grouped_plan_reference([0, 0], torch.device("cpu"), 64)
        self.assertEqual(empty[0].tolist(), [0, 0, 0])
        self.assertEqual(empty[1].numel(), 0)
        self.assertEqual(empty[2].numel(), 0)
        padded = build_grouped_plan_reference(
            [2, 0, 1, 3], torch.device("cpu"), 16, padding_rows=3
        )
        self.assertEqual(padded[0].tolist(), [3, 5, 5, 6, 9])
        self.assertEqual(padded[1].tolist(), [0, 2, 3])
        with self.assertRaises(ValueError):
            build_grouped_plan_reference([1], torch.device("cpu"), 16, padding_rows=-1)

    def test_values_layout_and_empty(self):
        torch.manual_seed(13)
        for n, h, f, e, k in [
            (0, 4, 7, 8, 8),
            (1, 4, 7, 8, 1),
            (7, 5, 9, 8, 2),
            (9, 8, 4, 8, 8),
        ]:
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.randn(n, h).to(dtype)
                w13 = (torch.randn(e, 2 * f, h) * 0.2).to(dtype)
                w2 = (torch.randn(e, h, f) * 0.2).to(dtype)
                ids, weights = router_reference(torch.randn(n, e), k)
                actual = moe_reference(x, ids, weights, w13, w2)
                expected = torch.zeros(n, h, dtype=torch.float32)
                for t in range(n):
                    for slot in range(k):
                        expert = int(ids[t, slot])
                        # Independent token/slot loop, intentionally no dispatch helper.
                        z = F.linear(x[t], w13[expert])
                        activation = F.silu(z[:f]) * z[f:]
                        value = F.linear(activation.float(), w2[expert].float())
                        expected[t] += (value * weights[t, slot]).to(dtype).float()
                torch.testing.assert_close(
                    actual, expected.to(dtype), atol=1e-5, rtol=1e-4
                )
                self.assertEqual(actual.shape, (n, h))
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, x.device)

    def test_inverse_and_weight_once(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ids = torch.tensor([[3, 0], [3, 1], [0, 1]])
        weights = torch.tensor([[0.2, 0.8], [0.3, 0.7], [0.4, 0.6]])
        packed = dispatch_reference(x, ids, weights, ExpertParallelTopology(4, 2, 0))
        torch.testing.assert_close(
            packed.pack_index[packed.restore_index], torch.arange(6)
        )
        self.assertEqual(packed.send_splits.tolist(), [4, 2])
        actual = combine_reference(
            packed.packed_tokens * packed.packed_routing_weights[:, None],
            packed.restore_index,
            ids,
            ExpertParallelTopology(4, 2, 0),
            model_dtype=x.dtype,
        )
        torch.testing.assert_close(actual, x)
        self.assertTrue(packed.packed_tokens.is_contiguous())

    def test_variable_pack_records_and_strides(self):
        from mini_ep.distributed.dispatch import pack_by_destination_rank

        torch.manual_seed(7)
        for n in (0, 1, 17):
            tokens = torch.randn(n, 14)[:, ::2]
            ids = torch.randint(8, (n, 4), dtype=torch.int32)[:, ::2]
            weights = torch.rand(n, 4)[:, ::2]
            topology = ExpertParallelTopology(8, 4, 0)
            actual = pack_by_destination_rank(tokens, ids, weights, topology)
            expected = dispatch_reference(tokens, ids, weights, topology)
            for field in (
                "packed_tokens",
                "packed_expert_ids",
                "packed_routing_weights",
                "pack_index",
                "restore_index",
                "send_splits",
            ):
                x, y = getattr(actual, field), getattr(expected, field)
                self.assertTrue(torch.equal(x, y))
                self.assertEqual(x.dtype, y.dtype)
                self.assertEqual(x.device, tokens.device)
                self.assertTrue(x.is_contiguous())
            self.assertTrue(
                torch.equal(
                    actual.packed_routing_weights[actual.restore_index],
                    weights.reshape(-1),
                )
            )
        with self.assertRaises(TypeError):
            topology.destination_rank(torch.tensor([1.5]))
        with self.assertRaises(TypeError):
            topology.local_expert_id(torch.tensor([1.5]))

    def test_fixed_rank_packing(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        ids = torch.tensor([[3, 0], [3, 1], [0, 1]])
        topology = ExpertParallelTopology(4, 2, 0)
        weights = torch.full(ids.shape, 0.5)
        packed = pack_by_destination_rank_fixed(
            x, ids, weights, topology, rows_per_rank=6
        )
        from reference.fixed_dispatch import dispatch_fixed_reference

        reference = dispatch_fixed_reference(x, ids, weights, topology, rows_per_rank=6)
        for field in (
            "packed_tokens",
            "packed_expert_ids",
            "packed_routing_weights",
            "restore_index",
            "valid_rows",
        ):
            torch.testing.assert_close(
                getattr(packed, field), getattr(reference, field), atol=0, rtol=0
            )
        self.assertEqual(packed.packed_tokens.shape, (12, 2))
        self.assertEqual(packed.packed_expert_ids.shape, (12,))
        self.assertEqual(packed.restore_index.tolist(), [6, 0, 7, 1, 2, 3])
        self.assertEqual(packed.valid_rows.tolist(), [4, 2])
        torch.testing.assert_close(
            packed.packed_tokens.index_select(0, packed.restore_index),
            x[torch.tensor([0, 0, 1, 1, 2, 2])],
        )
        # Rows after each destination's valid prefix are zero/-1 padding.
        self.assertTrue((packed.packed_expert_ids[8:] == -1).all())
        self.assertTrue((packed.packed_tokens[8:] == 0).all())
        self.assertEqual(packed.packed_tokens.dtype, x.dtype)
        self.assertEqual(packed.packed_expert_ids.dtype, torch.int64)
        with self.assertRaises(ValueError):
            pack_by_destination_rank_fixed(x, ids, weights, topology, rows_per_rank=0)

    def test_fixed_padding_local_id(self):
        ids = torch.tensor([-1, 0, 3, 4, 7])
        actual = ExpertParallelTopology(8, 2, 0).local_expert_id(ids, padding=True)
        torch.testing.assert_close(
            actual, torch.tensor([-1, 0, 3, 0, 3]), atol=0, rtol=0
        )
        single = ExpertParallelTopology(4, 4, 0).local_expert_id(
            torch.tensor([-1, 0, 1, 2, 3]), padding=True
        )
        torch.testing.assert_close(
            single, torch.tensor([-1, 0, 0, 0, 0]), atol=0, rtol=0
        )

    def test_fixed_capacity_policy(self):
        from mini_ep.distributed.all_to_all import fixed_rows_per_rank

        self.assertEqual(fixed_rows_per_rank("auto", [0, 8], 8), 64)
        self.assertIsNone(fixed_rows_per_rank("auto", [0, 9], 8))
        self.assertIsNone(fixed_rows_per_rank("on", [0, 0], 8))
        self.assertIsNone(fixed_rows_per_rank("off", [4, 4], 8))
        for sizes in ([], [-1, 4]):
            with self.assertRaises(ValueError):
                fixed_rows_per_rank("auto", sizes, 8)
        with self.assertRaises(ValueError):
            fixed_rows_per_rank("on", [8193], 1)
        # Capacity must be safe even when every branch goes to one destination.
        with self.assertRaises(ValueError):
            pack_by_destination_rank_fixed(
                torch.ones(3, 2),
                torch.zeros(3, 2, dtype=torch.int64),
                torch.ones(3, 2),
                ExpertParallelTopology(4, 2, 0),
                rows_per_rank=5,
            )

    def test_silu_rounding_before_multiply(self):
        from reference.expert import swiglu_reference

        x = torch.ones(1, 1, dtype=torch.bfloat16)
        w13 = torch.tensor([[0.5], [1.5]], dtype=torch.bfloat16)
        w2 = torch.ones(1, 1, dtype=torch.bfloat16)
        actual = swiglu_reference(x, w13, w2)
        expected = (
            F.silu(torch.tensor(0.5, dtype=torch.bfloat16))
            * torch.tensor(1.5, dtype=torch.bfloat16)
        ).float()
        self.assertEqual(actual.item(), expected.item())
        fused = (F.silu(torch.tensor(0.5)) * 1.5).bfloat16().float()
        self.assertNotEqual(expected.item(), fused.item())

    def test_ordered_topk_midpoint(self):
        # Captured cancellation pattern: a tiny slot changes a BF16 tie when
        # torch.sum reassociates the strided K dimension. Native uses slot order.
        values = [
            -0.0380859375,
            -0.06787109375,
            9.255018085241318e-09,
            -0.037353515625,
            -0.0185546875,
            -0.027099609375,
            0.0,
            0.0,
        ]
        returned = torch.tensor(values).reshape(8, 1)
        result = combine_reference(
            returned,
            torch.arange(8),
            torch.arange(8).reshape(1, 8),
            ExpertParallelTopology(8, 1, 0),
            model_dtype=torch.bfloat16,
        )
        self.assertEqual(result.item(), -0.189453125)

    def test_deepep_owner_rounding(self):
        # Each owner's BF16 store precedes the FP32 rank-order merge.
        values = torch.tensor([[1.0], [1 / 256], [-1.0]], dtype=torch.bfloat16)
        ids = torch.tensor([[0, 1, 2]])
        result = combine_reference(
            values,
            torch.arange(3),
            ids,
            ExpertParallelTopology(4, 2, 0),
            model_dtype=torch.bfloat16,
        )
        flat = combine_reference(
            values,
            torch.arange(3),
            ids,
            ExpertParallelTopology(4, 1, 0),
            model_dtype=torch.bfloat16,
        )
        self.assertEqual(result.item(), 0.0)
        self.assertEqual(flat.item(), 1 / 256)

    def test_fp32_owner_merge_order(self):
        values = torch.tensor([[2.0**25], [1.0], [-(2.0**25)], [1.0]])
        result = combine_reference(
            values,
            torch.arange(4),
            torch.arange(4).reshape(1, 4),
            ExpertParallelTopology(4, 4, 0),
            model_dtype=torch.float32,
        )
        self.assertEqual(result.item(), 1.0)

    def test_tied_router(self):
        logits = torch.zeros(2, 8)
        self.assertTrue(
            routing_matches_reference(
                logits, torch.tensor([[0, 1], [6, 7]]), torch.full((2, 2), 0.5)
            )
        )
        self.assertFalse(
            routing_matches_reference(
                logits, torch.tensor([[0, 0], [6, 7]]), torch.full((2, 2), 0.5)
            )
        )

    def test_validation_and_router(self):
        with self.assertRaises(ValueError):
            ExpertParallelTopology(7, 2, 0)
        with self.assertRaises(ValueError):
            ExpertParallelTopology(8, 2, 2)
        with self.assertRaises(ValueError):
            router_reference(torch.zeros(3, 4), 5)
        ids, w = router_reference(torch.tensor([[1.0, 4.0, 2.0, 3.0]]), 2)
        self.assertEqual(ids.tolist(), [[1, 3]])
        torch.testing.assert_close(w.sum(-1), torch.ones(1))
        self.assertEqual(w.dtype, torch.float32)
        # Single-token decode: rank 0 owns zero rows, rank 1 owns one.
        self.assertEqual(ExpertParallelTopology(128, 2, 0).token_range(1), (0, 0))
        self.assertEqual(ExpertParallelTopology(128, 2, 1).token_range(1), (0, 1))


if __name__ == "__main__":
    unittest.main()
