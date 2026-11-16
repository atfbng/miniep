"""CPU regression: rounding before/after routing is observable."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from reference.dispatch import combine_reference
from mini_ep.distributed.topology import ExpertParallelTopology


class ReturnTransportTests(unittest.TestCase):
    def test_weight_before_bf16_return_preserves_branch_rounding(self):
        torch.manual_seed(13)
        for n, h, k in ((1, 7, 1), (9, 31, 2), (5, 64, 8), (0, 7, 2)):
            ids = torch.randint(8, (n, k))
            weights = torch.softmax(torch.randn(n, k), -1)
            expert = torch.randn(n * k, h)
            restore = torch.randperm(n * k)
            packed_weights = torch.zeros(n * k)
            packed_weights.index_copy_(0, restore, weights.flatten())
            weighted = (expert * packed_weights[:, None]).bfloat16()
            actual = combine_reference(
                weighted,
                restore,
                ids,
                ExpertParallelTopology(8, 4, 0),
                model_dtype=torch.bfloat16,
            )
            # Independent original-record loop observes weighting exactly once.
            restored = (
                expert.index_select(0, restore) * weights.reshape(-1, 1)
            ).bfloat16()
            expected = combine_reference(
                restored,
                torch.arange(n * k),
                ids,
                ExpertParallelTopology(8, 4, 0),
                model_dtype=torch.bfloat16,
            )
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.dtype, torch.bfloat16)
            self.assertEqual(actual.shape, (n, h))

    def test_reference_default_follows_activation_and_sends_preweighted_values(self):
        from reference import moe

        # A one-rank identity transport isolates the dtype/rounding contract;
        # this CPU test makes no NCCL or GPU execution claim.
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(13)
            x = torch.randn(3, 7).to(dtype)
            w13 = torch.randn(2, 10, 7).to(dtype) * 0.1
            w2 = torch.randn(2, 7, 5).to(dtype) * 0.1
            ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
            weights = torch.tensor([[0.7, 0.3]]).expand(3, 2).contiguous()
            sent = []

            def identity(data, *args):
                sent.append(data.clone())
                return data.clone()

            with patch.object(moe.dist, "get_world_size", return_value=1), patch.object(
                moe.dist, "get_rank", return_value=0
            ), patch.object(
                moe, "exchange_splits", return_value=([6], [6])
            ), patch.object(
                moe, "variable_split_all_to_all", side_effect=identity
            ):
                actual, _ = moe.ep_forward_reference(x, ids, weights, w13, w2, object())
                self.assertEqual(sent[-1].dtype, dtype)
                self.assertEqual(
                    sent[-1].numel() * sent[-1].element_size(), 6 * 7 * x.element_size()
                )
                # Independently compute FP32 down values and weight before casting.
                expected = moe.moe_reference(x, ids, weights, w13, w2)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_diagnostic_transport_does_not_enable_fp16_model(self):
        from reference import moe
        from mini_ep.distributed.topology import ExpertParallelTopology

        x = torch.ones(1, 2, dtype=torch.float16)
        ids = torch.zeros(1, 1, dtype=torch.long)
        weights = torch.ones(1, 1)
        args = (
            x,
            ids,
            weights,
            torch.ones(1, 4, 2, dtype=x.dtype),
            torch.ones(1, 2, 2, dtype=x.dtype),
        )
        with self.assertRaises(TypeError):
            moe.validate_inputs(*args, ExpertParallelTopology(1, 1, 0))
        for dtype in (None, torch.float32, torch.float16, torch.bfloat16):
            with patch.object(moe.dist, "get_world_size", return_value=1), patch.object(
                moe.dist, "get_rank", return_value=0
            ):
                with self.assertRaisesRegex(TypeError, "FP32/BF16"):
                    moe.ep_forward_reference(
                        *args, object(), validate=False, return_dtype=dtype
                    )

    def test_direct_cast_changes_values_and_fp16_can_overflow(self):
        # Keep this witness explicit: the optimization must not become a plain cast.
        expert = torch.tensor([[1.003]])
        weights = torch.tensor([[0.7]])
        correctly_weighted = (expert * weights).bfloat16()
        prematurely_rounded = (expert.bfloat16().float() * weights).bfloat16()
        self.assertFalse(torch.equal(correctly_weighted, prematurely_rounded))
        self.assertTrue(torch.isinf(torch.tensor([70000.0]).half()).item())


if __name__ == "__main__":
    unittest.main()
