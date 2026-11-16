"""单卡拆解 EXPERT 阶段（展开、Grouped GEMM、SwiGLU、归约）的 Kernel 耗时。

启动方式（1 卡）：
    python3 benchmarks/expert_breakdown.py --rows 16 7456 29836

用合成的去重接收缓冲（模拟 EP=4 时本卡收到的 [U,K] ID/权重）调用与
deduplicated_moe.ep_forward 相同的 EXPERT 子步骤，输出 torch.profiler
CUDA 时间表和 CPU 墙钟。只用于定位热点，不替代端到端 Benchmark。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import time

import torch
from torch.profiler import profile, ProfilerActivity

from mini_ep.distributed.dedup import expand_owner_branches, reduce_owner_branches
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.model.expert import compute_grouped_local_experts


def synthetic_received(u, k, experts, ep, rank, h, device):
    """构造本卡收到的 U 个唯一行：每行至少一个 slot 属于本卡。"""
    g = torch.Generator(device=device).manual_seed(5)
    logits = torch.randn(u, experts, generator=g, device=device)
    local = experts // ep
    # 保证每行至少选中一个本卡专家，模拟“只有属于本卡的行才会被发来”。
    logits[:, rank * local] += 20
    w, ids = torch.topk(torch.softmax(logits, -1), k, -1)
    own = ids // local == rank
    ids = torch.where(own, ids, -1).long()
    w = torch.where(own, w, 0.0).float()
    x = torch.randn(u, h, generator=g, device=device).bfloat16()
    return x, ids.contiguous(), w.contiguous()


def expert_stage(x, ids, w, w13, w2, topo):
    expanded, bids, bw, lookup = expand_owner_branches(x, ids, w)
    local_ids = topo.local_expert_id(bids)
    out = compute_grouped_local_experts(expanded, local_ids, w13, w2,
                                        branch_weights=bw, output_dtype=x.dtype)
    return reduce_owner_branches(out, lookup, model_dtype=x.dtype)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[4, 117, 7456, 29836])
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--expert-hidden-size", type=int, default=768)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--ep", type=int, default=4)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--top", type=int, default=14)
    a = p.parse_args()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    local = a.experts // a.ep
    topo = ExpertParallelTopology(a.experts, a.ep, 0)
    w13 = (torch.randn(local, 2 * a.expert_hidden_size, a.hidden_size, device=device) * .02).bfloat16()
    w2 = (torch.randn(local, a.hidden_size, a.expert_hidden_size, device=device) * .02).bfloat16()
    for u in a.rows:
        x, ids, w = synthetic_received(u, a.top_k, a.experts, a.ep, 0, a.hidden_size, device)
        for _ in range(5):
            expert_stage(x, ids, w, w13, w2, topo)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            expert_stage(x, ids, w, w13, w2, topo)
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / 20 * 1e3
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            for _ in range(10):
                expert_stage(x, ids, w, w13, w2, topo)
            torch.cuda.synchronize()
        print(f"\n=== U={u} branches={int((ids >= 0).sum())} wall={wall:.3f} ms/iter")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=a.top,
                                        max_name_column_width=60))


if __name__ == "__main__":
    main()
