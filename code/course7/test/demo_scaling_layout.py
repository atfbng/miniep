"""Print the lesson's placement, routing, payload and metric examples on CPU."""
from __future__ import annotations
import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_ep.distributed.topology import ExpertParallelTopology
from mini_ep.utils.course_workload import make_workload


def layout_data(tokens: int, experts: int, top_k: int, ep: int, seed: int = 0) -> dict:
    """Derive source-to-target row counts from the actual CPU Router.

    Inputs are global Token/Expert counts. C is int64 [P,E] (source, Expert),
    S is int64 [P,P] (source, destination); their entries count expanded rows.
    This prints layout arithmetic; vector communication is run by test/demo.py.
    """
    if tokens <= 0 or ep <= 0 or tokens % ep or experts % ep:
        raise ValueError("Token and Expert counts must divide evenly by EP size")
    reference, x = make_workload(tokens, max(8, experts), 16, experts, top_k, "uniform", seed)
    with torch.no_grad():
        routing = reference.router(x)
        counts = torch.stack([reference.router(part).tokens_per_expert for part in x.chunk(ep)])
    send_matrix = counts.reshape(ep, ep, experts // ep).sum(-1)
    return {"tokens": tokens, "experts": experts, "top_k": top_k, "ep": ep, "local_tokens": tokens // ep, "expert_ids": routing.expert_ids.tolist(), "tokens_per_expert": counts.sum(0).tolist(), "send_matrix": send_matrix.tolist(), "recv_rows": send_matrix.sum(0).tolist(), "remote_rows": int((send_matrix.sum()-send_matrix.diag().sum()).item())}


def print_table(headers, rows):
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(map(str, row)) + " |")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", choices=["all", "placement", "routes", "strong", "weak", "communication", "timing"], default="all")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ep-sizes", type=int, nargs="+", default=[1,2,4])
    parser.add_argument("--tokens-per-rank", type=int, default=32)
    parser.add_argument("--token-id", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--element-bytes", type=int, default=4)
    parser.add_argument("--single-ms", type=float, default=4)
    parser.add_argument("--multi-ms", type=float, default=2.5)
    parser.add_argument("--compare-ep", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.token_id < args.tokens:
        raise ValueError("token-id must lie within the global input")
    if min(args.hidden_size, args.element_bytes, args.tokens_per_rank, args.compare_ep, args.single_ms, args.multi_ms) <= 0:
        raise ValueError("sizes and example latencies must be positive")
    plans = [layout_data(args.tokens, args.num_experts, args.top_k, ep, args.seed) for ep in args.ep_sizes]
    if args.show in ("all", "placement", "strong"):
        print("固定全组输入：每个 Expert 的行数保持不变")
        rows = []
        for plan in plans:
            ep = plan["ep"]
            for rank in range(ep):
                topology = ExpertParallelTopology(args.num_experts, ep, rank)
                start = rank * plan["local_tokens"]
                rows.append([ep, rank, f"T{start}～T{start+plan['local_tokens']-1}", "/".join(f"E{i}" for i in topology.local_expert_ids), plan["local_tokens"] * args.top_k, plan["recv_rows"][rank], str([plan["tokens_per_expert"][i] for i in topology.local_expert_ids])])
        print_table(["EP", "Rank", "原始 Token", "本地 Expert", "发送行数", "接收行数", "各本地 Expert 行数"], rows)
    if args.show in ("all", "routes"):
        rows=[]
        for plan in plans:
            source=args.token_id // plan["local_tokens"]
            chosen=plan["expert_ids"][args.token_id]
            targets=[expert // (args.num_experts // plan["ep"]) for expert in chosen]
            rows.append([plan["ep"],f"Rank {source}",args.token_id % plan["local_tokens"],str(chosen),str(targets)])
        print_table(["EP",f"T{args.token_id} 来源", "本地行号", "选择的 Expert", "Expert 所在 Rank"],rows)
    if args.show in ("all", "strong"):
        speedup=args.single_ms/args.multi_ms
        print("指标代入示例（使用给定延迟）：")
        print_table(["加速比", "并行效率", "全组吞吐 Token/s"],[[f"{speedup:.2f}",f"{100*speedup/args.compare_ep:.1f}%",f"{1000*args.tokens/args.multi_ms:.0f}"]])
    if args.show in ("all", "weak"):
        rows=[]
        for ep in args.ep_sizes:
            total=args.tokens_per_rank*ep
            plan=layout_data(total,args.num_experts,args.top_k,ep,args.seed)
            counts=plan["tokens_per_expert"]
            rows.append([ep,args.tokens_per_rank,total,args.num_experts//ep,counts[0] if len(set(counts))==1 else counts])
        print_table(["EP", "每卡 Token N", "全组 Token T", "每卡 Expert 数", "每 Expert 行数"],rows)
    if args.show in ("all", "communication"):
        total=args.tokens*args.top_k
        rows=[[p['ep'],total-p['remote_rows'],p['remote_rows'],2*p['remote_rows']*args.hidden_size*args.element_bytes] for p in plans]
        print_table(["EP", "本地行数", "远端行数", "远端往返 bytes"],rows)
    if args.show in ("all", "timing"):
        samples=[[4,1],[1,4]]
        maxima=[max(v) for v in zip(*samples)]
        print_table(["Rank 0 样本 ms", "Rank 1 样本 ms", "逐轮最大值 ms", "中位数 ms"],[[samples[0],samples[1],maxima,statistics.median(maxima)]])


if __name__ == "__main__":
    main()
