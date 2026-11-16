"""以独立 CPU 算术核对 GPU Peer 地址计划，覆盖稀疏 Expert 和无效尾部 tile。"""
import unittest

import torch
import triton
from mini_ep.kernels.peer_triton import _layout, _plan


@unittest.skipUnless(torch.cuda.is_available(), 'requires CUDA')
class PeerPlanTests(unittest.TestCase):
    def test_addresses_and_invalid_tiles(self):
        for p, e in ((1, 8), (2, 8), (4, 16), (4, 128), (4, 256)):
            l = e // p
            for empty in (True, False):
                counts_cpu = torch.zeros(p, e, dtype=torch.int64)
                if not empty:
                    # 专家数多于行数，用于检查二分查找的哨兵处理；
                    # 大部分已分配的 tile 槽位都必须是 -1。
                    for source in range(p):
                        counts_cpu[source, source % e] = source + 1
                        counts_cpu[source, -1] = 2
                counts = counts_cpu.cuda()
                capacity = max(128, triton.cdiv(int(counts_cpu.sum()), 128) * 128)
                tile_capacity = triton.cdiv(capacity, 128) + l - 1
                for rank in range(p):
                    starts = torch.empty(e, device='cuda', dtype=torch.int64)
                    targets, prefix, tile_starts = [torch.empty_like(starts) for _ in range(3)]
                    offsets = torch.empty(l + 1, device='cuda', dtype=torch.int64)
                    experts = torch.empty(tile_capacity, device='cuda', dtype=torch.int64)
                    rows, expected = torch.empty_like(experts), torch.empty_like(experts)
                    _layout[(1,)](counts, starts, targets, prefix, tile_starts, offsets,
                                  e, p, l, rank, 128, e, triton.next_power_of_2(p))
                    _plan[(1,)](offsets, experts, rows, expected, l, 128, tile_capacity,
                                triton.next_power_of_2(l), triton.next_power_of_2(tile_capacity), 8)
                    total = counts_cpu.sum(0).tolist()
                    offsets_ref = [0]
                    for n in total[rank*l:(rank+1)*l]:
                        offsets_ref.append(offsets_ref[-1] + n)
                    self.assertEqual(offsets.cpu().tolist(), offsets_ref)
                    send = [sum(counts_cpu[rank, :q]).item() if q else 0 for q in range(e)]
                    self.assertEqual(starts.cpu().tolist(), send)
                    for q in range(e):
                        base = sum(total[q//l*l:q])
                        prior = int(counts_cpu[:rank, q].sum())
                        self.assertEqual(targets[q].item(), base + prior)
                        self.assertEqual(prefix[q].item(), prior)
                        self.assertEqual(tile_starts[q].item(), sum(triton.cdiv(n,128) for n in total[q//l*l:q]))
                    expert_ref, row_ref, expected_ref = [], [], []
                    for q in range(l):
                        n = total[rank*l+q]
                        for r in range(0,n,128):
                            expert_ref.append(q); row_ref.append(r)
                            expected_ref.append(min(128,n-r)*8)
                    active = len(expert_ref)
                    self.assertEqual(experts.cpu().tolist(), expert_ref + [-1]*(tile_capacity-active))
                    self.assertEqual(rows[:active].cpu().tolist(), row_ref)
                    self.assertEqual(expected[:active].cpu().tolist(), expected_ref)


if __name__ == '__main__':
    unittest.main()
