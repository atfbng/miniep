"""Peer 去重与原分支实现/独立 Reference 的定向精度检查，包含舍入反例。"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import triton
from mini_ep.distributed.peer import PeerSwiGLU
from mini_ep.distributed.peer_dedup import DeduplicatedPeerSwiGLU
from mini_ep.kernels.peer_dedup_triton import push_owner_partials_kernel, combine_owner_partials_kernel
from mini_ep.kernels.peer_triton import _signal
from reference.moe import moe_reference


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    a=parser.parse_args();torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    rank,p=dist.get_rank(),dist.get_world_size();device=torch.device('cuda',torch.cuda.current_device())
    records=[]
    for h,f,e,k in ((128,128,8,1),(128,128,8,2),(128,128,8,8),
                    (256,256,256,8),(2048,512,256,8)):
        torch.manual_seed(2600+h+e)
        all_up=(torch.randn(e,2*f,h,device=device)/h**.5).bfloat16()
        all_down=(torch.randn(e,h,f,device=device)/f**.5).bfloat16()
        l=e//p;up=all_up[rank*l:(rank+1)*l].contiguous();down=all_down[rank*l:(rank+1)*l].contiguous()
        current=PeerSwiGLU(dist.group.WORLD,h,l,k,[137]*p,device)
        optimized=DeduplicatedPeerSwiGLU(dist.group.WORLD,h,l,k,[137]*p,device)
        for name,sizes in (('owner_duplicates',[1]+[0]*(p-1)),('uneven',[17+r for r in range(p)]),
                           ('skew',[137]*p),('all_empty',[0]*p)):
            n=sizes[rank]
            for iteration in range(2):
                torch.manual_seed(700+rank*31+iteration)
                x=torch.randn(n,h,device=device).bfloat16()
                ids=(torch.arange(n*k,device=device).reshape(n,k)+rank+iteration)%e
                if name=='owner_duplicates' and e==256:
                    ids=torch.tensor([[0,1,2,3,l,l+1,l+2,l+3]],device=device,dtype=torch.int64).expand(n,k).contiguous()
                if name=='skew':ids=torch.arange(k,device=device).expand(n,k).contiguous()
                weights=torch.rand(n,k,device=device);weights=(weights/weights.sum(1,keepdim=True)).contiguous()
                expected,_=current.forward(x,ids,weights,up,down)
                actual,meta=optimized.forward(x,ids,weights,up,down)
                reference=moe_reference(x,ids,weights,all_up,all_down,ep_size=p)
                assert actual.shape==expected.shape==(n,h) and actual.dtype==x.dtype and actual.device==device
                # 使用同一套 Up/Down Kernel，并保持完全相同的分支/owner 舍入。
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                torch.testing.assert_close(actual,reference,atol=.02,rtol=.03)
                owners=ids//l
                remote_branches=int((owners!=rank).sum())
                remote_pairs=sum(int(((owners==owner).any(1)).sum()) for owner in range(p) if owner!=rank)
                if name=='owner_duplicates' and e==256 and rank==0:
                    assert remote_branches==4 and remote_pairs==1
                records.append(dict(h=h,f=f,e=e,k=k,case=name,iteration=iteration,tokens=n,
                    exact_peer=True,max_abs_reference=float((actual.float()-reference.float()).abs().max()) if n else 0,
                    remote_expert_branches=remote_branches,remote_token_owner_pairs=remote_pairs,
                    large_vector_roundtrip_bytes_before=remote_branches*h*2*2,
                    large_vector_roundtrip_bytes_after=remote_pairs*h*2*2))
                print(f'rank={rank} H={h} E={e} K={k} {name} iter={iteration} PASS',flush=True)
        torch.cuda.synchronize();dist.barrier();del current,optimized
    # 直接回归 owner 尾声：BF16 递增求和会把 [256,1,-256,1] 算成错误值；
    # 按 slot 顺序的 FP32 求和必须得到 2。
    h,e,k=128,256,8;l=e//p
    workspace=DeduplicatedPeerSwiGLU(dist.group.WORLD,h,l,k,[1]*p,device)
    workspace.row_lookup.fill_(-1)
    if rank==0:workspace.row_lookup[0,:4]=torch.arange(4,device=device)
    if rank==1:workspace.row_lookup[0,4:]=torch.arange(4,device=device)
    ids=torch.tensor([[0,1,2,3,l,l+1,l+2,l+3]],device=device)
    for epoch,(values,wanted) in enumerate([([1,2,3,4],10),([256,1,-256,1],2)],1):
        outputs=torch.zeros(4,h,dtype=torch.bfloat16,device=device)
        if rank==1:outputs[:,0]=torch.tensor(values,dtype=torch.bfloat16,device=device)
        torch.cuda.synchronize();dist.barrier()
        push_owner_partials_kernel[(triton.cdiv(p,16),triton.cdiv(h,256))](
            outputs,workspace.row_lookup,workspace.dedup_ptrs[2],h,k,1,p,rank,256,enable_fp_fusion=False)
        _signal[(1,)](workspace.ptrs[5],epoch,p,rank,True)
        if rank==0:
            result=torch.empty(1,h,dtype=torch.bfloat16,device=device)
            combine_owner_partials_kernel[(triton.cdiv(h,256),)](
                workspace.owner_returned,ids,result,1,h,k,1,p,l,256,enable_fp_fusion=False)
            expected=torch.zeros_like(result);expected[0,0]=wanted
            torch.testing.assert_close(result,expected,atol=0,rtol=0)
        torch.cuda.synchronize();dist.barrier()
        records.append(dict(case='owner_rounding',values=values,expected=wanted,passed=True))
    del workspace
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    (root/f'rank{rank}.json').write_text(json.dumps(dict(passed=True,ep_size=p,records=records),indent=2))
    dist.destroy_process_group()

if __name__=='__main__':main()
