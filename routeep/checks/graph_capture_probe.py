"""可行性检查：NCCL 去重固定容量 EP 前向能否被 CUDA Graph capture，回放是否与 eager 逐位一致。

启动方式（4 卡）：
    torchrun --standalone --nproc-per-node=4 checks/graph_capture_probe.py

做法：每个 Rank 用相同形状、不同内容的输入，先 eager 跑一次得到参考输出；
再在 torch.cuda.graph 下 capture 同一个前向（输入/输出为静态缓冲），
往静态输入里拷入新数据后回放，与同一份新数据的 eager 结果逐位比较。
不涉及 vLLM，只回答“c10d all_to_all_single + 固定容量路径能否 capture”。
"""
import os
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist

from mini_ep.model.deduplicated_moe import ep_forward


def main():
    rank_local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank_local)
    device = torch.device("cuda", rank_local)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", timeout=timedelta(seconds=120), device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    h, f, e, k = 2048, 768, 128, 8
    local = e // world
    torch.manual_seed(7 + rank)
    w13 = (torch.randn(local, 2 * f, h, device=device) * .02).bfloat16()
    w2 = (torch.randn(local, h, f, device=device) * .02).bfloat16()
    ok = True
    for n in (1, 4, 32):
        def make(seed):
            g = torch.Generator(device=device).manual_seed(seed * 100 + rank)
            x = torch.randn(n, h, generator=g, device=device).bfloat16()
            ids = torch.topk(torch.randn(n, e, generator=g, device=device), k, -1).indices.int()
            w = torch.softmax(torch.randn(n, k, generator=g, device=device), -1)
            return x, ids.contiguous(), w.contiguous()

        run = lambda x, ids, w: ep_forward(x, ids, w, w13, w2, group, validate=False,
                                           fixed_capacity=n)[0]
        x0, i0, w0 = make(1)
        sx, si, sw = x0.clone(), i0.clone(), w0.clone()
        # 预热：触发 Triton JIT 和 NCCL 通信器初始化，必须在 capture 之前完成。
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                run(sx, si, sw)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        status = "captured"
        try:
            with torch.cuda.graph(graph):
                out = run(sx, si, sw)
        except Exception as exc:  # 记录 capture 失败原因，不中断其他规模
            status = f"capture failed: {type(exc).__name__}: {str(exc)[:160]}"
            ok = False
        if status == "captured":
            x1, i1, w1 = make(2)
            expected = run(x1, i1, w1)
            sx.copy_(x1); si.copy_(i1); sw.copy_(w1)
            graph.replay()
            torch.cuda.synchronize()
            equal = torch.equal(out, expected)
            ok &= equal
            status = f"replay {'bit-exact' if equal else 'MISMATCH max_abs=%.3g' % (out.float() - expected.float()).abs().max().item()}"
        print(f"rank{rank} N={n}: {status}", flush=True)
        dist.barrier()
    print(f"rank{rank} {'PASS' if ok else 'FAIL'}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
