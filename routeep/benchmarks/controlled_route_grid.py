"""受控路由去重实验，不是 DeepEP 算子基准。

两条 Peer 路径使用相同 BF16 权重、输入和路由，逐项检查输出完全一致；
小输入另与独立 PyTorch 对齐。CUDA Event 取最慢 rank，采样交替 AB/BA。
字节数仅为逻辑大向量载荷，不是物理链路计数或实测带宽。"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import triton
from mini_ep.distributed.peer import PeerSwiGLU
from mini_ep.distributed.peer_dedup import DeduplicatedPeerSwiGLU
from reference.moe import moe_reference

MODELS={'qwen3':(2048,768,128),'qwen35':(2048,512,256)}
PATTERNS=('random','spread','clustered','hot_owner','empty_source')


def routes(n,k,e,p,rank,pattern,device):
    l=e//p
    token=torch.arange(n,device=device)[:,None];slot=torch.arange(k,device=device)[None,:]
    if pattern=='spread':return (((token+slot+rank)%p)*l+(token*k+slot//p)%l).long().contiguous()
    if pattern=='clustered':return (((token+rank+1)%p)*l+(token*k+slot)%l).long().contiguous()
    if pattern=='hot_owner':return slot.expand(n,k).long().contiguous()
    return torch.rand(n,e,device=device).topk(k,dim=1).indices.contiguous()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--tokens',type=int,nargs='+',default=[1,8,128,1024,4096])
    p.add_argument('--pairs',type=int,default=10);p.add_argument('--warmup',type=int,default=5)
    a=p.parse_args();torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    rank,size=dist.get_rank(),dist.get_world_size();device=torch.device('cuda',torch.cuda.current_device())
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);records=[]
    environment=dict(torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                     nccl=torch.cuda.nccl.version(),gpu=torch.cuda.get_device_name(),ep_size=size,
                     command=sys.argv,scope='original tokens/routes to BF16 output; dynamic metadata included',
                     pairing='alternating AB/BA per sample; maximum EP-rank CUDA Event duration',
                     source_sha256={str(q.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(q.read_bytes()).hexdigest()
                        for q in (Path(__file__).resolve().parents[1]/'mini_ep').rglob('*.py')})
    (root/f'environment-rank{rank}.json').write_text(json.dumps(environment,indent=2))
    for model,(h,f,e) in MODELS.items():
        torch.manual_seed(2601)
        full_up=(torch.randn(e,2*f,h,device=device)/h**.5).bfloat16()
        full_down=(torch.randn(e,h,f,device=device)/f**.5).bfloat16()
        l=e//size;up=full_up[rank*l:(rank+1)*l].contiguous();down=full_down[rank*l:(rank+1)*l].contiguous()
        for n in a.tokens:
            for k in (1,2,8):
                base=PeerSwiGLU(dist.group.WORLD,h,l,k,[n]*size,device)
                dedup=DeduplicatedPeerSwiGLU(dist.group.WORLD,h,l,k,[n]*size,device)
                clock=torch.ones(2048,2048,dtype=torch.bfloat16,device=device);until=time.perf_counter()+1
                while time.perf_counter()<until:
                    _=clock@clock;torch.cuda.synchronize()
                del clock
                for seed in (117,293):
                    for pattern in PATTERNS:
                        local_n=0 if pattern=='empty_source' and rank==size-1 else n
                        torch.manual_seed(seed+rank*31)
                        x=torch.randn(local_n,h,device=device).bfloat16()
                        ids=routes(local_n,k,e,size,rank,pattern,device)
                        weights=torch.rand(local_n,k,device=device);weights=(weights/weights.sum(1,keepdim=True)).contiguous()
                        def run(label):
                            return (base if label=='peer' else dedup).forward(x,ids,weights,up,down,validate=False)[0]
                        old=run('peer');actual=run('dedup')
                        torch.testing.assert_close(actual,old,atol=0,rtol=0)
                        assert actual.shape==(local_n,h) and actual.dtype==x.dtype and actual.device==device
                        reference_error=None
                        if n<=8:
                            reference=moe_reference(x,ids,weights,full_up,full_down,ep_size=size)
                            torch.testing.assert_close(actual,reference,atol=.02,rtol=.03)
                            reference_error=float((actual.float()-reference.float()).abs().max()) if local_n else 0
                        for label in ('peer','dedup'):
                            for _ in range(a.warmup):run(label)
                        torch.cuda.synchronize();samples={'peer':[],'dedup':[]}
                        for pair in range(a.pairs):
                            for label in (('peer','dedup') if pair%2==0 else ('dedup','peer')):
                                dist.barrier();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                                start.record();output=run(label);end.record();end.synchronize()
                                value=torch.tensor(start.elapsed_time(end),device=device,dtype=torch.float64)
                                dist.all_reduce(value,op=dist.ReduceOp.MAX);samples[label].append(value.item())
                        owners=ids//l
                        branches=int((owners!=rank).sum())
                        pairs=sum(int((owners==q).any(1).sum()) for q in range(size) if q!=rank)
                        totals=torch.tensor([branches,pairs],device=device,dtype=torch.int64)
                        dist.all_reduce(totals);global_branches,global_pairs=totals.cpu().tolist()
                        histogram=torch.bincount(ids.flatten(),minlength=e);dist.all_reduce(histogram)
                        histogram=histogram.cpu().tolist();mean=sum(histogram)/e
                        median={label:statistics.median(values) for label,values in samples.items()}
                        records.append(dict(model=model,tokens=n,local_tokens=local_n,batch_size=1,seq_len=n,
                            hidden_size=h,expert_hidden_size=f,num_experts=e,top_k=k,ep_size=size,
                            dtype='bfloat16',chunk_size=None,seed=seed,pattern=pattern,
                            exact_peer=True,max_abs_reference=reference_error,tokens_per_expert=histogram,
                            imbalance_ratio=max(histogram)/mean if mean else 0,
                            global_remote_branches=global_branches,global_remote_token_owner_pairs=global_pairs,
                            logical_roundtrip_bytes_before=global_branches*h*4,
                            logical_roundtrip_bytes_after=global_pairs*h*4,
                            samples_max_rank_ms=samples,median_max_rank_ms=median,
                            latency_reduction=1-median['dedup']/median['peer']))
                        if len(records)%10==0:
                            (root/f'rank{rank}.json').write_text(json.dumps(dict(completed=False,records=records),indent=2))
                        if rank==0:print(model,n,k,seed,pattern,median,flush=True)
                torch.cuda.synchronize();dist.barrier();del base,dedup
    (root/f'rank{rank}.json').write_text(json.dumps(dict(completed=True,records=records),indent=2))
    dist.destroy_process_group()

if __name__=='__main__':main()
