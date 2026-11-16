"""真实 NCCL 与逐分支 Peer 的回归检查：用 torchrun 启动1/2/4张H100。"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from mini_ep.distributed.peer import PeerSwiGLU, peer_forward, clear_peer_workspaces
from mini_ep.model.moe import ep_forward
from reference.moe import moe_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl', timeout=timedelta(seconds=120))
    rank, p = dist.get_rank(), dist.get_world_size()
    device = torch.device('cuda', torch.cuda.current_device())
    torch.manual_seed(2512)
    records = []
    for h, f, e in [(128, 128, 8), (256, 256, 16), (128, 128, 256)]:
        torch.manual_seed(2512 + h)
        full_up = (torch.randn(e, 2*f, h, device=device) / h**0.5).bfloat16()
        full_down = (torch.randn(e, h, f, device=device) / f**0.5).bfloat16()
        l = e // p
        up = full_up[rank*l:(rank+1)*l].contiguous()
        down = full_down[rank*l:(rank+1)*l].contiguous()
        for k in (1, 2, 8):
            workspace = PeerSwiGLU(dist.group.WORLD, h, l, k, [137]*p, device)
            for case, sizes in [('uneven', [17+r for r in range(p)]),
                                ('skew', [137]*p), ('empty_source', [0]+[3]*(p-1)),
                                ('all_empty', [0]*p), ('single', [1]*p)]:
                n = sizes[rank]
                for iteration in range(3):
                    torch.manual_seed(600 + rank * 13 + iteration)
                    x = torch.randn(n, h, device=device).bfloat16()
                    ids = (torch.arange(n*k, device=device).reshape(n,k) + rank + iteration) % e
                    if case in ('skew', 'single'):
                        ids = torch.arange(k, device=device).expand(n,k).contiguous()
                    weights = torch.rand(n,k, device=device)
                    weights = (weights / weights.sum(1, keepdim=True)).contiguous()
                    y, metadata = workspace.forward(x, ids, weights, up, down)
                    # 复用 workspace 前先 clone 输出；后续轮次会覆盖内部缓冲。
                    y = y.clone()
                    original, _ = ep_forward(x, ids, weights, up, down, dist.group.WORLD)
                    reference = moe_reference(x, ids, weights, full_up, full_down, ep_size=p)
                    assert y.shape == (n,h) and y.dtype == x.dtype and y.device == x.device
                    assert metadata['branches'] == n*k and metadata['epoch'] > 0
                    local_counts = workspace.counts.sum(0)[rank*l:(rank+1)*l]
                    expected_offsets = torch.cat((local_counts.new_zeros(1), local_counts.cumsum(0)))
                    torch.testing.assert_close(workspace.offsets, expected_offsets, atol=0, rtol=0)
                    assert ((workspace.experts >= -1) & (workspace.experts < l)).all().item()
                    torch.testing.assert_close(y, original, atol=0.02, rtol=0.03)
                    torch.testing.assert_close(y, reference, atol=0.02, rtol=0.03)
                    error = (y.float()-reference.float()).abs().max().item() if n else 0.0
                    records.append(dict(h=h,f=f,e=e,k=k,case=case,iteration=iteration,
                                        tokens=n,max_abs_error=error,
                                        exact_original=torch.equal(y,original)))
                    print(f'rank={rank} H={h} E={e} K={k} {case} iter={iteration} PASS error={error}', flush=True)
            torch.cuda.synchronize()
            del workspace
    # 模拟框架的动态尺寸：长 prefill 不能让后续 decode 仍启动 prefill 大小的
    # grid；回到某个容量桶时，继续复用该桶自己的 epoch。
    bucket_capacities = []
    for n in (137, 1, 17, 1, 0):
        x = torch.randn(n, h, device=device).bfloat16()
        ids = torch.arange(n*8, device=device).reshape(n,8) % e
        weights = torch.full((n,8), 1/8, device=device)
        y, meta = peer_forward(x, ids, weights, up, down, dist.group.WORLD, source_rows=[n]*p)
        baseline, _ = ep_forward(x, ids, weights, up, down, dist.group.WORLD)
        torch.testing.assert_close(y, baseline, atol=0.02, rtol=0.03)
        bucket_capacities.append(meta['input_capacity'])
        records.append(dict(case='capacity_bucket',tokens=n,capacity=meta['input_capacity'],epoch=meta['epoch']))
    assert bucket_capacities[0] > bucket_capacities[1]
    assert bucket_capacities[1] == bucket_capacities[3] == bucket_capacities[4]
    clear_peer_workspaces()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    (root/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,ep_size=p,records=records),indent=2))
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
