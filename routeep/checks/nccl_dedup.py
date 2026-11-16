"""真实 NCCL/CUDA 去重正确性检查；--cpu 时改用 Gloo，只验证通信语义。

启动方式（GPU）：
    torchrun --standalone --nproc-per-node=2 checks/nccl_dedup.py --output ...

覆盖非均匀路由、空来源、全空输入、单 Token、热点 owner、Top-K=1/2/8，
并用相同 Grouped GEMM 对照要求逐元素完全一致。--peer-smoke 额外比较
NCCL 去重与 Peer 去重路径。
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from mini_ep.model.deduplicated_moe import ep_forward
from reference.moe import ep_forward_reference


def exercise(group, device, records, *, quick=False):
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    cpu = device.type == 'cpu'
    if not cpu:
        from mini_ep.model.moe import ep_forward as original
    for dtype in (torch.float32, torch.bfloat16):
        for h, f in ((7,9),) if cpu else ((32,48),(128,128)):
            e=8;local=e//size
            torch.manual_seed(904)
            w13=(torch.randn(e,2*f,h,device=device)*.08).to(dtype)
            w2=(torch.randn(e,h,f,device=device)*.08).to(dtype)
            up=w13[rank*local:(rank+1)*local].contiguous()
            down=w2[rank*local:(rank+1)*local].contiguous()
            for k in (1,2,8):
                cases=['uneven','empty_source','all_empty','single','hot_owner'] if not quick else ['empty_source']
                for case in cases:
                    n=rank+3
                    if case=='empty_source' and rank==0:n=0
                    if case=='all_empty':n=0
                    if case=='single':n=1
                    torch.manual_seed(213+rank)
                    x=torch.randn(n,h,device=device).to(dtype)
                    ids=(torch.arange(n*k,device=device).reshape(n,k)+rank)%e
                    if case=='hot_owner':ids=(torch.arange(k,device=device)%local).expand(n,k).contiguous()
                    weights=torch.softmax(torch.randn(n,k,device=device),-1)
                    for rd in (None, torch.float32, torch.float16):
                        y,meta=ep_forward(x,ids,weights,up,down,group,
                            expert_backend='torch' if cpu else 'grouped',return_dtype=rd)
                        ref,_=ep_forward_reference(x,ids,weights,up,down,group,return_dtype=rd)
                        tol=.02 if dtype==torch.bfloat16 else .002 if rd==torch.float16 else 1e-5
                        torch.testing.assert_close(y,ref,atol=tol,rtol=tol)
                        exact=None
                        # 相同 Grouped GEMM 对照要求逐元素相等；独立 PyTorch 使用上面的容差。
                        if not cpu:
                            baseline,_=original(x,ids,weights,up,down,group,return_dtype=rd)
                            torch.testing.assert_close(y,baseline,atol=0,rtol=0)
                            exact=True
                        assert y.shape==x.shape and y.dtype==dtype and y.device==device
                        expected=[sum(bool((ids[t]//local==r).any()) for t in range(n)) for r in range(size)]
                        assert meta['send_splits']==expected
                        assert meta['transport']=='nccl' and meta['deduplicated']
                        assert meta['return_buffer_dtype']==str(dtype)
                        assert meta['return_send_bytes']==meta['recv_rows']*h*x.element_size()
                        assert meta['send_rows']<=ids.numel()
                        assert meta['branches']==n*k
                        if not cpu:
                            # 固定容量去重：容量取组内最大来源 Token 数（含 0），必须与变长路径逐元素一致。
                            sizes=torch.tensor([n],device=device);dist.all_reduce(sizes,op=dist.ReduceOp.MAX,group=group)
                            cap=max(int(sizes.item()),1)
                            yf,fm=ep_forward(x,ids,weights,up,down,group,return_dtype=rd,fixed_capacity=cap)
                            torch.testing.assert_close(yf,y,atol=0,rtol=0)
                            assert yf.shape==y.shape and yf.dtype==y.dtype and fm['fixed_rows_per_rank']==cap
                            assert fm['layout']=='unique-token-owner-fixed' and fm['send_rows']==size*cap
                        records.append(dict(ep_size=size,ep_rank=rank,global_rank=dist.get_rank(),
                            case=case,h=h,k=k,dtype=str(dtype),return_dtype=str(rd),
                            exact_original=exact,max_abs_error=(y.float()-ref.float()).abs().max().item() if n else 0.,
                            send_rows=meta['send_rows'],branches=n*k))


def peer_smoke(device, records):
    from mini_ep.distributed.peer_dedup import deduplicated_peer_forward,clear_deduplicated_peer_workspaces
    rank,size=dist.get_rank(),dist.get_world_size();h=f=128;e=8;k=8;l=e//size
    torch.manual_seed(6)
    up=(torch.randn(e,2*f,h,device=device)*.05).bfloat16()
    down=(torch.randn(e,h,f,device=device)*.05).bfloat16()
    local=(up[rank*l:(rank+1)*l].contiguous(),down[rank*l:(rank+1)*l].contiguous())
    for case,sizes in [('uneven',[r+3 for r in range(size)]),('empty_source',[0]+[1]*(size-1)),('all_empty',[0]*size)]:
        n=sizes[rank];x=torch.randn(n,h,device=device).bfloat16()
        ids=(torch.arange(n*k,device=device).reshape(n,k)+rank)%e
        weights=torch.full((n,k),1/k,device=device)
        nccl,meta=ep_forward(x,ids,weights,*local,dist.group.WORLD)
        peer,pm=deduplicated_peer_forward(x,ids,weights,*local,dist.group.WORLD,source_rows=sizes)
        torch.testing.assert_close(nccl,peer,atol=0,rtol=0)
        assert pm['transport']=='peer' and pm['deduplicated']
        records.append(dict(case='peer_'+case,ep_size=size,exact_peer=True))
    clear_deduplicated_peer_workspaces()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu',action='store_true')
    p.add_argument('--peer-smoke',action='store_true');p.add_argument('--output',required=True);a=p.parse_args()
    device=torch.device('cpu')
    if not a.cpu:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',torch.cuda.current_device())
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    dist.init_process_group('gloo' if a.cpu else 'nccl',timeout=timedelta(seconds=180))
    records=[]
    try:
        exercise(dist.group.WORLD,device,records)
        if dist.get_world_size()==4:
            # 交错 global rank 建 EP 子组，验证布局/通信没有把 EP rank 当作 CUDA 序号。
            groups=[dist.new_group(ranks=ranks) for ranks in ([0,2],[1,3])]
            group=groups[dist.get_rank()%2];exercise(group,device,records,quick=True)
        if a.peer_smoke:
            if a.cpu:raise ValueError('peer smoke requires CUDA')
            peer_smoke(device,records)
        if not a.cpu:torch.cuda.synchronize()
        output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
        (output/f'rank{dist.get_rank()}.json').write_text(json.dumps(dict(passed=True,
            backend=dist.get_backend(),records=records),indent=2))
        print(f'PASS rank={dist.get_rank()} backend={dist.get_backend()} cases={len(records)}',flush=True)
    except BaseException:
        print(f'FAIL rank={dist.get_rank()}',flush=True);traceback.print_exc();sys.stderr.flush();os._exit(1)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
