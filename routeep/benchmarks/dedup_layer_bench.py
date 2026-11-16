"""按层测量 NCCL 去重 EP 前向（不含 vLLM），用于优化前后对比。

启动方式（4 卡 H100，EP=4）：
    torchrun --standalone --nproc-per-node=4 benchmarks/dedup_layer_bench.py \
        --output /results/dedup-layer.json

每次重复连续执行 --layers 个 MoE 层，模拟模型中层与层背靠背下发；
Host 同步点会让 CPU 下发间隙暴露在 GPU 时间线上，因此整段耗时比单层
孤立计时更接近真实推理。报告值：每次重复取最慢 Rank 的平均每层耗时，
再取重复间中位数。路由为合成数据（均匀随机 Top-K 或热点），不是真实模型路由。
分阶段计时（--stages）单独运行，包含依赖和下发间隙，不能直接相加。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from datetime import timedelta
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

from mini_ep.model.deduplicated_moe import ep_forward
from mini_ep.profiler.stages import Stages


def make_routing(n: int, experts: int, top_k: int, skew: float, generator: torch.Generator, device):
    """生成 [N,K] 全局专家编号（无重复）和归一化 FP32 权重。

    skew>0 时给前 1/8 的专家加 logit 偏置，制造热点 owner。
    """
    logits = torch.randn(n, experts, generator=generator, device=device)
    if skew:
        logits[:, : max(experts // 8, 1)] += skew
    weights, ids = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return ids.to(torch.int32).contiguous(), weights.float().contiguous()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 32, 256, 2048, 8192])
    p.add_argument("--hidden-size", type=int, default=2048)
    p.add_argument("--expert-hidden-size", type=int, default=768)
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--skew", type=float, default=0.0)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--expert-backend", choices=["grouped", "grouped_unfused"], default="grouped")
    p.add_argument("--fixed", action="store_true", help="固定容量去重 A2A，容量=组内最大 Token 数")
    p.add_argument("--transport", choices=["nccl", "peer"], default="nccl",
                   help="peer 使用 Peer/TMA 去重路径（需 SM90+ 与 CUDA P2P）")
    p.add_argument("--stages", action="store_true", help="另跑一轮 CUDA Event 分阶段计时")
    p.add_argument("--label", default="current")
    p.add_argument("--output", required=True)
    a = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    if a.experts % world:
        raise ValueError(f"experts={a.experts} must divide by EP size={world}")
    local = a.experts // world
    h, f = a.hidden_size, a.expert_hidden_size
    torch.manual_seed(1234 + rank)
    w13 = (torch.randn(local, 2 * f, h, device=device) * 0.02).bfloat16().contiguous()
    w2 = (torch.randn(local, h, f, device=device) * 0.02).bfloat16().contiguous()

    if a.transport == "peer":
        # 延迟导入：Peer 路径依赖 symmetric memory，只在显式选择时加载。
        from mini_ep.distributed.peer_dedup import deduplicated_peer_forward as peer_forward
    results = []
    for n in a.tokens:
        gen = torch.Generator(device=device).manual_seed(77 + rank * 1000 + n)
        x = (torch.randn(n, h, generator=gen, device=device)).bfloat16()
        ids, weights = make_routing(n, a.experts, a.top_k, a.skew, gen, device)

        def run_layers(stages=None):
            y = x
            for _ in range(a.layers):
                # 每层都使用同一组路由，输出只用于保持数据依赖。
                if a.transport == "peer":
                    y, meta = peer_forward(x, ids.long(), weights, w13, w2, group,
                                           source_rows=[n] * world, stages=stages, validate=False)
                    continue
                y, meta = ep_forward(x, ids, weights, w13, w2, group,
                                     stages=stages, validate=False,
                                     expert_backend=a.expert_backend,
                                     fixed_capacity=n if a.fixed else None)
            return y, meta

        for _ in range(a.warmup):
            run_layers()
        torch.cuda.synchronize()
        samples, host_samples = [], []
        for _ in range(a.repeats):
            dist.barrier(device_ids=[local_rank])
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            begin.record()
            _, meta = run_layers()
            end.record()
            end.synchronize()
            host_samples.append((time.perf_counter() - t0) * 1e3 / a.layers)
            samples.append(begin.elapsed_time(end) / a.layers)
        local_t = torch.tensor(samples, device=device)
        dist.all_reduce(local_t, op=dist.ReduceOp.MAX)
        per_layer = local_t.tolist()
        row = dict(
            tokens_per_rank=n, ms_per_layer_median=statistics.median(per_layer),
            ms_per_layer_min=min(per_layer), ms_per_layer_max=max(per_layer),
            host_ms_per_layer_rank0=statistics.median(host_samples),
            send_rows=meta.get("send_rows"), recv_rows=meta.get("recv_rows"),
            received_branches=meta.get("received_branches"),
        )
        if a.stages:
            stage_rows = {}
            for _ in range(a.repeats):
                dist.barrier(device_ids=[local_rank])
                st = Stages(timings=True)
                run_layers(st)
                for name, values in st.milliseconds().items():
                    stage_rows.setdefault(name, []).append(sum(values) / len(values))
            stage_max = {}
            for name in sorted(stage_rows):
                t = torch.tensor(stage_rows[name], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                stage_max[name] = round(statistics.median(t.tolist()), 4)
            row["stages_ms"] = stage_max
        results.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)

    if rank == 0:
        out = Path(a.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(
            label=a.label, expert_backend=a.expert_backend, fixed=a.fixed, transport=a.transport, ep_size=world, hidden_size=h, expert_hidden_size=f,
            num_experts=a.experts, top_k=a.top_k, skew=a.skew, dtype="bfloat16",
            layers_per_repeat=a.layers, warmup=a.warmup, repeats=a.repeats,
            statistic="per-repeat max-rank mean ms/layer, median over repeats",
            routing="synthetic softmax top-k", device=torch.cuda.get_device_name(),
            torch=torch.__version__, nccl=".".join(map(str, torch.cuda.nccl.version())),
            results=results,
        ), indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
