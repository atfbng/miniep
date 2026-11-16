"""小型 Triton Kernel 必须与独立 PyTorch Reference 一致。

覆盖固定容量 Dispatch、Grouped GEMM 计划、SwiGLU 和 Combine 的空专家、
tile 边界、非连续输入和 padding 行。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from reference.expert import swiglu_reference


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton")
class KernelTests(unittest.TestCase):
    def test_fixed_dispatch_triton(self):
        from mini_ep.kernels.dispatch_triton import (
            pack_by_destination_rank_fixed_triton,
        )
        from mini_ep.distributed.dispatch import pack_by_destination_rank_fixed
        from reference.fixed_dispatch import dispatch_fixed_reference
        from mini_ep.distributed.topology import ExpertParallelTopology

        torch.manual_seed(13)
        for p in (1, 2, 4):
            topology = ExpertParallelTopology(128, p, 0)
            for n, k in ((0, 8), (1, 1), (1, 2), (4, 8), (8, 8)):
                for dtype in (torch.float32, torch.bfloat16, torch.float16):
                    x = torch.randn(n, 62, device="cuda", dtype=dtype)[:, ::2]
                    for id_dtype in (torch.int32, torch.int64):
                        for skew in (False, True):
                            ids = torch.randint(
                                128, (n, k * 2), device="cuda", dtype=id_dtype
                            )[:, ::2]
                            if skew:
                                ids.fill_(127)
                            c = max(8, n * k)
                            weights = torch.softmax(
                                torch.randn(n, k, device="cuda"), -1
                            )
                            expected = dispatch_fixed_reference(
                                x, ids, weights, topology, c
                            )
                            fallback = pack_by_destination_rank_fixed(
                                x, ids, weights, topology, c
                            )
                            actual = pack_by_destination_rank_fixed_triton(
                                x, ids, weights, topology, c
                            )
                            for field in (
                                "packed_tokens",
                                "packed_expert_ids",
                                "packed_routing_weights",
                                "restore_index",
                                "valid_rows",
                            ):
                                result, ref = getattr(actual, field), getattr(
                                    expected, field
                                )
                                torch.testing.assert_close(result, ref, atol=0, rtol=0)
                                torch.testing.assert_close(
                                    getattr(fallback, field), ref, atol=0, rtol=0
                                )
                                self.assertEqual(result.dtype, ref.dtype)
                                self.assertEqual(result.device, ref.device)
                                self.assertTrue(result.is_contiguous())

    def test_padding_activation_skips_nan_rows(self):
        from mini_ep.kernels.swiglu_triton import swiglu_activation_triton

        torch.manual_seed(13)
        for dtype in (torch.float32, torch.bfloat16):
            for first in (0, 1, 8, 9):
                gate_up = torch.randn(9, 62, device="cuda", dtype=dtype)
                gate_up[:first] = float("nan")
                offsets = torch.tensor([first, 9], device="cuda", dtype=torch.int64)
                expected = swiglu_activation_triton(gate_up)
                expected[:first] = 0
                actual = swiglu_activation_triton(gate_up, valid_start=offsets)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                self.assertTrue(torch.isfinite(actual).all())
                self.assertEqual(actual.dtype, dtype)
                self.assertEqual(actual.device, gate_up.device)

    def test_grouped_plan_padding_prefix(self):
        from mini_ep.distributed.topology import ExpertParallelTopology
        from mini_ep.kernels.grouped_gemm.triton_impl import build_device_plan
        from reference.grouped_plan import build_grouped_plan_reference

        ids = torch.tensor([3, -1, 0, 3, -1, 2, 0, -1, 3], device="cuda")
        origin = torch.argsort(ids, stable=True)
        plan = build_device_plan(ids, origin, 4, 16)
        offsets, experts, rows = plan.offsets, plan.tile_experts, plan.tile_rows
        expected = build_grouped_plan_reference(
            [2, 0, 1, 3], ids.device, 16, padding_rows=3
        )
        torch.testing.assert_close(offsets, expected[0], atol=0, rtol=0)
        torch.testing.assert_close(experts[:3], expected[1], atol=0, rtol=0)
        torch.testing.assert_close(rows[:3], expected[2], atol=0, rtol=0)
        torch.testing.assert_close(
            origin,
            torch.tensor([1, 4, 7, 2, 6, 5, 0, 3, 8], device="cuda"),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            offsets, torch.tensor([3, 5, 5, 6, 9], device="cuda"), atol=0, rtol=0
        )
        torch.testing.assert_close(
            experts, torch.tensor([0, 2, 3, -1], device="cuda"), atol=0, rtol=0
        )
        torch.testing.assert_close(
            rows[:3], torch.zeros(3, dtype=torch.int64, device="cuda"), atol=0, rtol=0
        )
        single_ids = ExpertParallelTopology(1, 1, 0).local_expert_id(
            torch.tensor([-1, 0, -1], device="cuda"),
            padding=True,
        )
        torch.testing.assert_close(
            single_ids, torch.tensor([-1, 0, -1], device="cuda"), atol=0, rtol=0
        )
        single_order = torch.argsort(single_ids, stable=True)
        single_offsets = build_device_plan(single_ids, single_order, 1, 16).offsets
        torch.testing.assert_close(
            single_offsets, torch.tensor([2, 3], device="cuda"), atol=0, rtol=0
        )

    def test_grouped_device_plan(self):
        from mini_ep.kernels.grouped_gemm.triton_impl import build_device_plan
        from reference.grouped_plan import build_grouped_plan_reference

        torch.manual_seed(13)
        # 非 2 的幂 L、tile 边界、单个非空 Expert 和大量空 Expert，
        # 同时覆盖下界二分和 padding tile。
        for experts in (1, 3, 32, 64):
            for rows in (1, 16, 63, 64, 65, 129, 4097):
                for skew in (False, True):
                    for dtype in (torch.int32, torch.int64):
                        ids = torch.randint(
                            experts, (rows * 2,), device="cuda", dtype=dtype
                        )[::2]
                        if skew:
                            ids.fill_(experts - 1)
                        order = torch.argsort(ids, stable=True)
                        counts = torch.bincount(ids.long(), minlength=experts).tolist()
                        for block_m in (16, 64, 128):
                            actual = build_device_plan(ids, order, experts, block_m)
                            expected = build_grouped_plan_reference(
                                counts, ids.device, block_m
                            )
                            offsets, tile_experts, tile_rows = (
                                actual.offsets,
                                actual.tile_experts,
                                actual.tile_rows,
                            )
                            valid = expected[1].numel()
                            torch.testing.assert_close(
                                offsets, expected[0], atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                tile_experts[:valid], expected[1], atol=0, rtol=0
                            )
                            torch.testing.assert_close(
                                tile_rows[:valid], expected[2], atol=0, rtol=0
                            )
                            self.assertTrue((tile_experts[valid:] == -1).all().item())
                            for tensor in (offsets, tile_experts, tile_rows):
                                self.assertEqual(tensor.dtype, torch.int64)
                                self.assertEqual(tensor.device, ids.device)
                                self.assertTrue(tensor.is_contiguous())

    def test_ordered_combine(self):
        from mini_ep.kernels.combine_triton import combine_triton
        from reference.dispatch import combine_reference
        from mini_ep.distributed.topology import ExpertParallelTopology

        torch.manual_seed(13)
        for n in (0, 1, 17):
            h, k = 31, 8
            returned = torch.randn(n * k, h, device="cuda")
            restore = torch.randperm(n * k, device="cuda")
            ids = torch.arange(k, device="cuda").expand(n, k).contiguous()
            weights = torch.softmax(torch.randn(n, k, device="cuda"), -1)
            topology = ExpertParallelTopology(8, 2, 0)
            for dtype in (torch.float32, torch.bfloat16):
                expected = combine_reference(
                    returned, restore, ids, topology, model_dtype=dtype
                )
                actual = combine_triton(
                    returned, restore, ids, topology, model_dtype=dtype
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_swiglu(self):
        from mini_ep.kernels.swiglu_triton import swiglu_triton

        torch.manual_seed(13)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for dtype in (torch.float32, torch.bfloat16):
            for m, h, f in [(0, 32, 48), (1, 31, 45), (17, 64, 96), (7, 2048, 768)]:
                x = torch.randn(m, h, device="cuda").to(dtype)
                w13 = (torch.randn(2 * f, h, device="cuda") * 0.02).to(dtype)
                w2 = (torch.randn(h, f, device="cuda") * 0.02).to(dtype)
                actual = swiglu_triton(x, w13, w2)
                expected = swiglu_reference(x, w13, w2)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                    rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                )
                self.assertEqual(actual.shape, (m, h))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device, x.device)

    def test_compute_grouped_local_experts(self):
        from mini_ep.kernels.grouped_gemm.triton_impl import (
            compute_grouped_local_experts,
        )
        from mini_ep.model.expert import compute_local_experts_loop

        torch.manual_seed(13)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        cases = [
            (0, 32, 48, 4, []),
            (1, 31, 45, 4, [0]),
            (17, 64, 96, 8, [0, 0, 0, 2, 2, 4, 7]),
            (33, 64, 96, 8, [1, 1, 1, 1, 3, 3, 3, 6, 6, 7]),
            (129, 31, 45, 3, [2]),
            (257, 64, 96, 8, [0, 0, 0, 2, 2, 4, 7]),
            (7, 2048, 768, 4, [0, 2, 3]),
        ]
        for dtype in (torch.float32, torch.bfloat16):
            for m, h, f, experts, pattern in cases:
                x = torch.randn(m, h, device="cuda", dtype=dtype)
                w13 = (torch.randn(experts, 2 * f, h, device="cuda") * 0.02).to(dtype)
                w2 = (torch.randn(experts, h, f, device="cuda") * 0.02).to(dtype)
                ids = torch.tensor(
                    [pattern[i % len(pattern)] for i in range(m)],
                    device="cuda",
                    dtype=torch.int64,
                )
                expected = compute_local_experts_loop(x, ids, w13, w2)
                actual = compute_grouped_local_experts(x, ids, w13, w2)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                    rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                )
                self.assertEqual(actual.shape, (m, h))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device, x.device)

    def test_grouped_swiglu_padding(self):
        from mini_ep.kernels.grouped_gemm.triton_impl import (
            compute_grouped_local_experts,
        )
        from mini_ep.model.expert import compute_local_experts_loop

        torch.manual_seed(13)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for dtype in (torch.float32, torch.bfloat16):
            for m, h, f, experts in [
                (9, 32, 48, 4),
                (17, 64, 96, 8),
                (129, 31, 45, 3),
            ]:
                x = torch.randn(m, h, device="cuda", dtype=dtype)
                w13 = (torch.randn(experts, 2 * f, h, device="cuda") * 0.02).to(dtype)
                w2 = (torch.randn(experts, h, f, device="cuda") * 0.02).to(dtype)
                ids = torch.randint(experts, (m,), device="cuda", dtype=torch.int64)
                # Padding 可能交错出现，不一定位于接收顺序的前缀。
                ids[::4] = -1
                expected = compute_local_experts_loop(x, ids, w13, w2)
                actual = compute_grouped_local_experts(x, ids, w13, w2, padding=True)
                torch.testing.assert_close(
                    actual,
                    expected,
                    atol=0.02 if dtype == torch.bfloat16 else 1e-5,
                    rtol=0.02 if dtype == torch.bfloat16 else 1e-4,
                )
                self.assertTrue((actual[ids < 0] == 0).all())
                self.assertEqual(actual.shape, (m, h))
                self.assertEqual(actual.dtype, torch.float32)
                self.assertEqual(actual.device, x.device)
        # 全 padding 输入不能进入任何 Expert tile。
        x = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
        w13 = torch.randn(4, 96, 32, device="cuda", dtype=torch.bfloat16)
        w2 = torch.randn(4, 32, 48, device="cuda", dtype=torch.bfloat16)
        ids = torch.full((5,), -1, device="cuda", dtype=torch.int64)
        actual = compute_grouped_local_experts(x, ids, w13, w2, padding=True)
        self.assertTrue((actual == 0).all())


if __name__ == "__main__":
    unittest.main()
