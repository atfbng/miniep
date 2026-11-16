"""Grouped GEMM 新 tile 策略与融合 gate_up+SwiGLU 必须与原 128x128x64 两步实现逐位一致。

原实现（已通过 DeepEP / 原生后端整模型精确对齐）：BM=128、BN=128、BK=64、8 warps，
gate_up 写 [R,2F] 模型 dtype，再由 swiglu_activation_triton 激活，down 投影加权。
这里直接用该配置启动同一个 grouped_gemm_kernel 作为基准，覆盖三档行数区间、
Qwen3（F=768）与 Qwen3.5（F=512）形状、padding 前缀、空专家和单专家热点。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch


def _legacy_mlp(x, w13, w2, flat_ids, origin, row_weights, output_dtype):
    """原固定 128x128x64/8 warps 配置的两步实现，返回 Expert-major [R,H]。"""
    import triton
    from mini_ep.kernels.grouped_gemm.triton_impl import build_device_plan, grouped_gemm_kernel
    from mini_ep.kernels.swiglu_triton import swiglu_activation_triton
    plan = build_device_plan(flat_ids, origin, w13.shape[0], 128)

    def gemm(a, w, out_dtype, rw):
        n = w.shape[1]
        y = torch.empty((a.shape[0], n), device=a.device, dtype=out_dtype)
        grouped_gemm_kernel[(plan.tile_experts.numel(), triton.cdiv(n, 128))](
            a, w, y, plan.offsets, plan.tile_experts, plan.tile_rows,
            rw if rw is not None else a, rw is not None, a.shape[1], n, 128, 128, 64,
            num_warps=8, num_stages=3, enable_fp_fusion=rw is None)
        return y

    act = swiglu_activation_triton(gemm(x, w13, x.dtype, None), valid_start=plan.offsets)
    return gemm(act, w2, output_dtype, row_weights), int(plan.offsets[0])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton")
class GroupedTileTests(unittest.TestCase):
    def test_new_tiles_and_fused_swiglu_match_legacy(self):
        from mini_ep.kernels.grouped_gemm.triton_impl import (
            _tile_config, build_device_plan, grouped_expert_mlp)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        gen = torch.Generator(device="cuda").manual_seed(2027)
        for h, f, local in ((2048, 768, 32), (2048, 512, 64)):
            w13 = (torch.randn(local, 2 * f, h, generator=gen, device="cuda") * .02).bfloat16()
            w2 = (torch.randn(local, h, f, generator=gen, device="cuda") * .02).bfloat16()
            # 行数覆盖三档 tile 区间边界两侧。
            for rows in (1, 7, 64, 2048, 2049, 9000, 16384, 16385, 40000):
                for case in ("uniform", "hot", "padding"):
                    ids = torch.randint(local, (rows,), generator=gen, device="cuda")
                    if case == "hot":
                        ids[: rows - rows // 8] = local - 1
                    if case == "padding":
                        ids[torch.rand(rows, generator=gen, device="cuda") < .4] = -1
                    origin = torch.argsort(ids, stable=True)
                    x = torch.randn(rows, h, generator=gen, device="cuda").bfloat16().index_select(0, origin)
                    rw = torch.rand(rows, generator=gen, device="cuda")
                    expected, start = _legacy_mlp(x, w13, w2, ids, origin, rw, torch.bfloat16)
                    plan = build_device_plan(ids, origin, local, _tile_config(torch.bfloat16, rows)[0])
                    for fuse in (True, False):
                        actual = grouped_expert_mlp(x, w13, w2, plan, row_weights=rw,
                                                    output_dtype=torch.bfloat16, padding=True,
                                                    fuse_swiglu=fuse)
                        msg = f"h={h} f={f} rows={rows} {case} fuse={fuse}"
                        self.assertEqual((actual.shape, actual.dtype), (expected.shape, expected.dtype), msg)
                        self.assertTrue(torch.equal(actual[start:], expected[start:]), msg)

    def test_peer_style_bm128_plan_keeps_legacy_kernel(self):
        """Peer/TMA 路径用固定 BM=128 计划调用 grouped_gemm，必须仍可运行且与原配置逐位相同。"""
        import triton
        from mini_ep.kernels.grouped_gemm.triton_impl import (
            build_device_plan, grouped_gemm, grouped_gemm_kernel)
        gen = torch.Generator(device="cuda").manual_seed(3)
        local, h, f = 8, 2048, 512
        w2 = (torch.randn(local, h, f, generator=gen, device="cuda") * .02).bfloat16()
        for rows in (128, 1024, 4096):
            ids = torch.randint(local, (rows,), generator=gen, device="cuda")
            origin = torch.argsort(ids, stable=True)
            plan = build_device_plan(ids, origin, local, 128)
            x = torch.randn(rows, f, generator=gen, device="cuda").bfloat16()
            rw = torch.rand(rows, generator=gen, device="cuda")
            actual = grouped_gemm(x, w2, plan, output_dtype=torch.bfloat16, row_weights=rw)
            expected = torch.empty_like(actual)
            grouped_gemm_kernel[(plan.tile_experts.numel(), triton.cdiv(h, 128))](
                x, w2, expected, plan.offsets, plan.tile_experts, plan.tile_rows, rw, True,
                f, h, 128, 128, 64, num_warps=8, num_stages=3, enable_fp_fusion=False)
            self.assertTrue(torch.equal(actual, expected), f"rows={rows}")

    def test_fp32_fused_matches_unfused(self):
        from mini_ep.kernels.grouped_gemm.triton_impl import (
            _tile_config, build_device_plan, grouped_expert_mlp)
        gen = torch.Generator(device="cuda").manual_seed(7)
        local, h, f = 4, 96, 80
        w13 = torch.randn(local, 2 * f, h, generator=gen, device="cuda") * .05
        w2 = torch.randn(local, h, f, generator=gen, device="cuda") * .05
        for rows in (1, 33, 300):
            ids = torch.randint(-1, local, (rows,), generator=gen, device="cuda")
            origin = torch.argsort(ids, stable=True)
            x = torch.randn(rows, h, generator=gen, device="cuda")
            plan = build_device_plan(ids, origin, local, _tile_config(torch.float32, rows)[0])
            a = grouped_expert_mlp(x, w13, w2, plan, padding=True, fuse_swiglu=True)
            b = grouped_expert_mlp(x, w13, w2, plan, padding=True, fuse_swiglu=False)
            start = int(plan.offsets[0])
            self.assertTrue(torch.equal(a[start:], b[start:]), f"rows={rows}")


if __name__ == "__main__":
    unittest.main()
