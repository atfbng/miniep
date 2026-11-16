"""Correctness tests for lesson 2's packed Dispatch and Combine Reference."""

from __future__ import annotations

import unittest

import torch

from mini_ep.distributed import combine_reference, dispatch_reference
from mini_ep.model import TopKRouter


class TestDispatchReference(unittest.TestCase):
    def setUp(self) -> None:
        self.tokens = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        self.expert_ids = torch.tensor(
            [[2, 0], [1, 2], [0, 1]], dtype=torch.int64
        )
        self.routing_weights = torch.tensor(
            [[0.25, 0.75], [0.60, 0.40], [0.20, 0.80]]
        )

    def test_stable_expert_major_layout_and_metadata(self) -> None:
        result = dispatch_reference(
            self.tokens,
            self.expert_ids,
            self.routing_weights,
            num_experts=4,
        )

        expected_packed_tokens = self.tokens.index_select(
            0, torch.tensor([0, 2, 1, 2, 0, 1])
        )
        torch.testing.assert_close(result.packed_tokens, expected_packed_tokens)
        torch.testing.assert_close(
            result.packed_expert_ids, torch.tensor([0, 0, 1, 1, 2, 2])
        )
        torch.testing.assert_close(
            result.packed_token_indices, torch.tensor([0, 2, 1, 2, 0, 1])
        )
        torch.testing.assert_close(
            result.packed_route_slots, torch.tensor([1, 0, 0, 1, 0, 1])
        )
        torch.testing.assert_close(
            result.packed_routing_weights,
            torch.tensor([0.75, 0.20, 0.60, 0.80, 0.25, 0.40]),
        )
        torch.testing.assert_close(
            result.tokens_per_expert, torch.tensor([2, 2, 2, 0])
        )
        torch.testing.assert_close(
            result.expert_offsets, torch.tensor([0, 2, 4, 6, 6])
        )
        torch.testing.assert_close(
            result.restore_index, torch.tensor([4, 0, 2, 5, 1, 3])
        )
        self.assertEqual(result.num_tokens, 3)
        self.assertEqual(result.top_k, 2)
        self.assertEqual(result.num_assignments, 6)
        self.assertTrue(result.packed_tokens.is_contiguous())
        self.assertEqual(result.packed_tokens.dtype, self.tokens.dtype)
        self.assertEqual(result.packed_tokens.device, self.tokens.device)

    def test_restore_index_inverts_the_pack_permutation(self) -> None:
        result = dispatch_reference(
            self.tokens,
            self.expert_ids,
            self.routing_weights,
            num_experts=4,
        )

        token_major = result.packed_tokens.index_select(0, result.restore_index)
        expected = self.tokens.repeat_interleave(2, dim=0)
        torch.testing.assert_close(token_major, expected)

        restored_expert_ids = result.packed_expert_ids.index_select(
            0, result.restore_index
        )
        torch.testing.assert_close(restored_expert_ids, self.expert_ids.reshape(-1))

    def test_accepts_noncontiguous_tokens_and_returns_contiguous_pack(self) -> None:
        storage = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        tokens = storage[:, ::2]
        self.assertFalse(tokens.is_contiguous())

        result = dispatch_reference(
            tokens,
            self.expert_ids,
            self.routing_weights,
            num_experts=4,
        )

        self.assertEqual(result.packed_tokens.shape, (6, 4))
        self.assertTrue(result.packed_tokens.is_contiguous())

    def test_top1_highly_skewed_routing_and_empty_experts(self) -> None:
        tokens = torch.randn(5, 3)
        expert_ids = torch.full((5, 1), 2, dtype=torch.int64)
        routing_weights = torch.ones(5, 1)

        result = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=4
        )

        torch.testing.assert_close(
            result.tokens_per_expert, torch.tensor([0, 0, 5, 0])
        )
        torch.testing.assert_close(
            result.expert_offsets, torch.tensor([0, 0, 0, 5, 5])
        )
        torch.testing.assert_close(result.packed_tokens, tokens)

    def test_zero_tokens_preserve_shapes_and_offsets(self) -> None:
        tokens = torch.empty(0, 7)
        expert_ids = torch.empty(0, 2, dtype=torch.int64)
        routing_weights = torch.empty(0, 2)

        result = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=3
        )
        combined = combine_reference(result.packed_tokens, result)

        self.assertEqual(result.packed_tokens.shape, (0, 7))
        self.assertEqual(result.restore_index.shape, (0,))
        torch.testing.assert_close(
            result.expert_offsets, torch.tensor([0, 0, 0, 0])
        )
        self.assertEqual(combined.shape, (0, 7))

    def test_rejects_invalid_routing_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            dispatch_reference(
                self.tokens,
                self.expert_ids,
                self.routing_weights[:, :1],
                num_experts=4,
            )
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            dispatch_reference(
                self.tokens,
                self.expert_ids.float(),
                self.routing_weights,
                num_experts=4,
            )
        with self.assertRaisesRegex(TypeError, "tokens must use"):
            dispatch_reference(
                self.tokens.to(torch.int64),
                self.expert_ids,
                self.routing_weights,
                num_experts=4,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 2\)"):
            dispatch_reference(
                self.tokens,
                self.expert_ids,
                self.routing_weights,
                num_experts=2,
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed num_experts"):
            dispatch_reference(
                self.tokens,
                torch.zeros(3, 3, dtype=torch.int64),
                torch.full((3, 3), 1.0 / 3.0),
                num_experts=2,
            )


class TestCombineReference(unittest.TestCase):
    def test_identity_experts_restore_original_tokens(self) -> None:
        tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expert_ids = torch.tensor([[0, 2], [1, 0]])
        routing_weights = torch.tensor([[0.30, 0.70], [0.40, 0.60]])
        dispatch = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=3
        )

        combined = combine_reference(dispatch.packed_tokens, dispatch)

        torch.testing.assert_close(combined, tokens)
        self.assertEqual(combined.shape, tokens.shape)
        self.assertEqual(combined.dtype, tokens.dtype)
        self.assertEqual(combined.device, tokens.device)

    def test_weighted_top2_combine_matches_manual_result(self) -> None:
        tokens = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        expert_ids = torch.tensor([[2, 0], [1, 2], [0, 1]])
        routing_weights = torch.tensor(
            [[0.25, 0.75], [0.60, 0.40], [0.20, 0.80]]
        )
        dispatch = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=4
        )
        expert_scale = (dispatch.packed_expert_ids + 1).to(tokens.dtype)
        packed_outputs = dispatch.packed_tokens * expert_scale.unsqueeze(-1)

        combined = combine_reference(packed_outputs, dispatch)

        expected = torch.stack(
            (tokens[0] * 1.50, tokens[1] * 2.40, tokens[2] * 1.80)
        )
        torch.testing.assert_close(combined, expected)

    def test_supports_strided_expert_outputs(self) -> None:
        tokens = torch.randn(4, 3)
        expert_ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0]])
        weights = torch.full((4, 2), 0.5)
        dispatch = dispatch_reference(tokens, expert_ids, weights, num_experts=2)
        storage = torch.empty(8, 6)
        storage[:, ::2] = dispatch.packed_tokens
        packed_outputs = storage[:, ::2]
        self.assertFalse(packed_outputs.is_contiguous())

        combined = combine_reference(packed_outputs, dispatch)

        torch.testing.assert_close(combined, tokens)

    def test_backward_reaches_tokens_and_routing_weights(self) -> None:
        tokens = torch.randn(4, 3, requires_grad=True)
        expert_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 0]])
        raw_weights = torch.rand(4, 2, requires_grad=True)
        routing_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
        dispatch = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=3
        )
        scales = (dispatch.packed_expert_ids + 1).to(tokens.dtype).unsqueeze(-1)

        combined = combine_reference(dispatch.packed_tokens * scales, dispatch)
        combined.square().mean().backward()

        self.assertIsNotNone(tokens.grad)
        self.assertIsNotNone(raw_weights.grad)
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        self.assertTrue(bool(torch.isfinite(raw_weights.grad).all()))

    def test_rejects_output_with_wrong_layout_shape(self) -> None:
        tokens = torch.randn(2, 3)
        expert_ids = torch.tensor([[0], [1]])
        weights = torch.ones(2, 1)
        dispatch = dispatch_reference(tokens, expert_ids, weights, num_experts=2)

        with self.assertRaisesRegex(ValueError, "match packed_tokens shape"):
            combine_reference(torch.randn(3, 3), dispatch)


class TestRouterDispatchIntegration(unittest.TestCase):
    def test_lesson_configuration_has_64_top2_assignments(self) -> None:
        torch.manual_seed(23)
        router = TopKRouter(hidden_size=8, num_experts=4, top_k=2)
        tokens = torch.randn(32, 8)

        routing = router(tokens)
        dispatch = dispatch_reference(
            tokens,
            routing.expert_ids,
            routing.routing_weights,
            num_experts=4,
        )
        combined = combine_reference(dispatch.packed_tokens, dispatch)

        self.assertEqual(dispatch.packed_tokens.shape, (64, 8))
        self.assertEqual(int(dispatch.tokens_per_expert.sum()), 64)
        self.assertEqual(int(dispatch.expert_offsets[-1]), 64)
        torch.testing.assert_close(combined, tokens)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_fp16_preserves_layout_dtype_and_device(self) -> None:
        torch.manual_seed(29)
        router = TopKRouter(hidden_size=8, num_experts=4, top_k=2).cuda().half()
        tokens = torch.randn(7, 8, device="cuda", dtype=torch.float16)

        routing = router(tokens)
        dispatch = dispatch_reference(
            tokens,
            routing.expert_ids,
            routing.routing_weights,
            num_experts=4,
        )
        combined = combine_reference(dispatch.packed_tokens, dispatch)

        self.assertEqual(dispatch.packed_tokens.device.type, "cuda")
        self.assertEqual(dispatch.packed_tokens.dtype, torch.float16)
        self.assertEqual(dispatch.packed_routing_weights.dtype, torch.float32)
        self.assertEqual(dispatch.expert_offsets.device.type, "cuda")
        self.assertEqual(combined.dtype, torch.float16)
        torch.testing.assert_close(combined, tokens, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
