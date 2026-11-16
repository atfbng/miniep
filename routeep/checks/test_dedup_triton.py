"""融合的去重 EXPERT/COMBINE Triton 路径必须与分步 PyTorch 实现逐元素一致。

分步实现：expand_owner_branches -> compute_grouped_local_experts ->
reduce_owner_branches，以及 combine_owner_partials。覆盖 FP32/BF16、
Top-1/2/8、空输入、全部 slot 落在同一专家、某些专家为空和 return dtype。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for Triton")
class DedupTritonTests(unittest.TestCase):
    def _received(self, u, k, experts, ep, rank, h, dtype, case, gen):
        local = experts // ep
        logits = torch.randn(u, experts, generator=gen, device="cuda")
        logits[:, rank * local] += 30  # 每个收到的唯一行至少一个 slot 属于本卡
        if case == "hot":
            logits[:, rank * local:(rank + 1) * local] += 30
        ids = torch.topk(logits, k, -1).indices
        if case == "single_expert":
            ids = torch.full((u, k), -1, device="cuda", dtype=torch.long)
            ids[:, 0] = rank * local
        own = ids // local == rank
        ids = torch.where(own, ids, -1).long().contiguous()
        weights = torch.where(own, torch.rand(u, k, generator=gen, device="cuda"), 0.0).contiguous()
        x = torch.randn(u, h, generator=gen, device="cuda").to(dtype)
        return x, ids, weights

    def test_owner_partials_match_unfused(self):
        from mini_ep.distributed.dedup import expand_owner_branches, reduce_owner_branches
        from mini_ep.model.expert import compute_grouped_local_experts
        from mini_ep.kernels.dedup_triton import compute_owner_partials_triton
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        gen = torch.Generator(device="cuda").manual_seed(31)
        experts, ep, rank = 32, 4, 1
        local = experts // ep
        for dtype in (torch.float32, torch.bfloat16):
            for h, f in ((64, 96), (2048, 768)):
                w13 = (torch.randn(local, 2 * f, h, generator=gen, device="cuda") * .05).to(dtype)
                w2 = (torch.randn(local, h, f, generator=gen, device="cuda") * .05).to(dtype)
                for k in (1, 2, 8):
                    for u in (0, 1, 7, 300):
                        for case in ("uniform", "hot", "single_expert"):
                            for rd in (dtype, torch.float32, torch.float16):
                                x, ids, w = self._received(u, k, experts, ep, rank, h, dtype, case, gen)
                                fused = compute_owner_partials_triton(x, ids, w, w13, w2, return_dtype=rd)
                                if u:
                                    e, bid, bw, lookup = expand_owner_branches(x, ids, w)
                                    branches = compute_grouped_local_experts(
                                        e, bid % local, w13, w2, branch_weights=bw, output_dtype=rd)
                                    expected = reduce_owner_branches(branches, lookup, model_dtype=dtype)
                                else:
                                    expected = x.new_empty((0, h))
                                msg = f"{dtype} h={h} k={k} u={u} {case} rd={rd}"
                                self.assertEqual((fused.shape, fused.dtype, fused.device),
                                                 (expected.shape, expected.dtype, expected.device), msg)
                                torch.testing.assert_close(fused, expected, atol=0, rtol=0, msg=msg)

    def test_combine_matches_reference(self):
        from mini_ep.distributed.dedup import combine_owner_partials
        from mini_ep.kernels.dedup_triton import combine_owner_partials_triton
        gen = torch.Generator(device="cuda").manual_seed(7)
        for dtype in (torch.float32, torch.bfloat16):
            for p in (1, 2, 4):
                for n in (0, 1, 33):
                    present = torch.rand(p, n, generator=gen, device="cuda") < .6
                    rows = int(present.sum())
                    restore = torch.full((p, n), -1, dtype=torch.long, device="cuda")
                    restore[present] = torch.randperm(rows, device="cuda")
                    returned = (torch.randn(rows, 130, generator=gen, device="cuda") * 100).to(dtype)
                    a = combine_owner_partials_triton(returned, restore)
                    b = combine_owner_partials(returned, restore)
                    self.assertEqual((a.shape, a.dtype), (b.shape, b.dtype))
                    torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_fixed_owner_pack_matches_reference(self):
        from mini_ep.distributed.dedup import pack_token_owners_fixed
        from mini_ep.distributed.topology import ExpertParallelTopology
        from mini_ep.kernels.dedup_triton import pack_token_owners_fixed_triton, split_owner_rows
        gen = torch.Generator(device="cuda").manual_seed(11)
        for dtype in (torch.float32, torch.bfloat16):
            for p in (1, 2, 4):
                topo = ExpertParallelTopology(16, p, 0)
                for n in (0, 1, 5, 256):
                    for k in (1, 2, 8):
                        for skew in (False, True):
                            for h in (64, 2048):
                                x = torch.randn(n, h, generator=gen, device="cuda").to(dtype)
                                ids = torch.randint(16, (n, k), generator=gen, device="cuda", dtype=torch.int32)
                                if skew:
                                    ids.fill_(15)
                                w = torch.rand(n, k, generator=gen, device="cuda")
                                c = max(n, 1) + 2
                                ref = pack_token_owners_fixed(x, ids, w, topo, c)
                                buf, restore, counts = pack_token_owners_fixed_triton(x, ids, w, topo.num_local_experts, p, c)
                                tx, tid, tw = split_owner_rows(buf, h, k, dtype)
                                msg = f"{dtype} p={p} n={n} k={k} skew={skew} h={h}"
                                torch.testing.assert_close(restore, ref.restore_index, atol=0, rtol=0, msg=msg)
                                self.assertEqual(counts.tolist(), ref.send_splits.tolist(), msg)
                                torch.testing.assert_close(tid.long(), ref.packed_expert_ids, atol=0, rtol=0, msg=msg)
                                torch.testing.assert_close(tw, ref.packed_routing_weights, atol=0, rtol=0, msg=msg)
                                valid = ref.packed_expert_ids.ge(0).any(1)  # padding 行激活未初始化
                                torch.testing.assert_close(tx[valid], ref.packed_tokens[valid], atol=0, rtol=0, msg=msg)


if __name__ == "__main__":
    unittest.main()
