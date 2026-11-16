"""Correctness tests for lesson 1's pure PyTorch MoE."""

from __future__ import annotations

import unittest

import torch

from mini_ep.model import ExpertMLP, SingleGPUMoE, TopKRouter
from mini_ep.model.moe import combine_expert_outputs, dispatch_tokens


class TestExpertMLP(unittest.TestCase):
    def test_preserves_token_layout_dtype_and_device(self) -> None:
        expert = ExpertMLP(hidden_size=8, expert_hidden_size=16)
        tokens = torch.randn(5, 8)

        output = expert(tokens)

        self.assertEqual(output.shape, tokens.shape)
        self.assertEqual(output.dtype, tokens.dtype)
        self.assertEqual(output.device, tokens.device)


class TestTopKRouter(unittest.TestCase):
    def test_top1_metadata(self) -> None:
        router = TopKRouter(hidden_size=8, num_experts=4, top_k=1)
        result = router(torch.randn(7, 8))

        self.assertEqual(result.logits.shape, (7, 4))
        self.assertEqual(result.expert_ids.shape, (7, 1))
        self.assertEqual(result.routing_weights.shape, (7, 1))
        self.assertEqual(result.tokens_per_expert.shape, (4,))
        self.assertEqual(int(result.tokens_per_expert.sum()), 7)
        torch.testing.assert_close(
            result.routing_weights.sum(dim=-1), torch.ones(7)
        )
        self.assertGreaterEqual(int(result.expert_ids.min()), 0)
        self.assertLess(int(result.expert_ids.max()), 4)

    def test_top2_weights_are_renormalized(self) -> None:
        router = TopKRouter(hidden_size=8, num_experts=4, top_k=2)
        result = router(torch.randn(7, 8))

        self.assertEqual(result.expert_ids.shape, (7, 2))
        self.assertEqual(int(result.tokens_per_expert.sum()), 14)
        torch.testing.assert_close(
            result.routing_weights.sum(dim=-1), torch.ones(7)
        )
        self.assertTrue(
            bool(torch.all(result.expert_ids[:, 0] != result.expert_ids[:, 1]))
        )


class TestDispatchAndCombine(unittest.TestCase):
    def test_weighted_top2_combine_restores_original_order(self) -> None:
        tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expert_ids = torch.tensor([[0, 1], [1, 0]])
        routing_weights = torch.tensor([[0.25, 0.75], [0.60, 0.40]])
        assignments = dispatch_tokens(
            tokens, expert_ids, routing_weights, num_experts=3
        )

        # E0 is identity, E1 doubles its input, and E2 receives no tokens.
        expert_outputs = [
            assignment.tokens * (assignment.expert_id + 1)
            for assignment in assignments
        ]
        combined = combine_expert_outputs(tokens, assignments, expert_outputs)

        expected = torch.stack((tokens[0] * 1.75, tokens[1] * 1.60))
        torch.testing.assert_close(combined, expected)
        self.assertEqual(assignments[2].tokens.shape, (0, 2))


class TestSingleGPUMoE(unittest.TestCase):
    def test_forward_matches_explicit_token_by_token_reference(self) -> None:
        torch.manual_seed(11)
        model = SingleGPUMoE(
            hidden_size=6,
            expert_hidden_size=12,
            num_experts=4,
            top_k=2,
        )
        tokens = torch.randn(5, 6)

        result = model(tokens)
        expected = torch.zeros_like(tokens)
        for token_id in range(tokens.shape[0]):
            for slot in range(2):
                expert_id = int(result.expert_ids[token_id, slot])
                weight = result.routing_weights[token_id, slot]
                expert_output = model.experts[expert_id](
                    tokens[token_id : token_id + 1]
                )[0]
                expected[token_id] += weight * expert_output

        torch.testing.assert_close(result.hidden_states, expected)
        self.assertEqual(result.hidden_states.shape, tokens.shape)
        self.assertEqual(int(result.tokens_per_expert.sum()), 10)

    def test_backward_reaches_input_router_and_selected_experts(self) -> None:
        torch.manual_seed(13)
        model = SingleGPUMoE(
            hidden_size=6,
            expert_hidden_size=12,
            num_experts=4,
            top_k=2,
        )
        tokens = torch.randn(8, 6, requires_grad=True)

        result = model(tokens)
        result.hidden_states.square().mean().backward()

        self.assertIsNotNone(tokens.grad)
        self.assertTrue(bool(torch.isfinite(tokens.grad).all()))
        self.assertIsNotNone(model.router.gate.weight.grad)
        self.assertTrue(bool(torch.isfinite(model.router.gate.weight.grad).all()))
        selected_experts = set(result.expert_ids.reshape(-1).tolist())
        for expert_id in selected_experts:
            self.assertIsNotNone(model.experts[expert_id].up_proj.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_device_is_preserved(self) -> None:
        model = SingleGPUMoE(
            hidden_size=8,
            expert_hidden_size=16,
            num_experts=4,
            top_k=2,
        ).cuda().half()
        tokens = torch.randn(6, 8, device="cuda", dtype=torch.float16)

        result = model(tokens)

        self.assertEqual(result.hidden_states.device.type, "cuda")
        self.assertEqual(result.expert_ids.device.type, "cuda")
        self.assertEqual(result.routing_weights.device.type, "cuda")
        self.assertEqual(result.hidden_states.dtype, torch.float16)
        self.assertEqual(result.routing_weights.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
