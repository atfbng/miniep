"""动态 MoE 层的 Peer/原分支 NCCL 对照，包含分发和元数据开销。

相同输入、路由、权重下先检查精度，再预热并用 CUDA Event 测量；
分布式延迟关注最慢 rank，不把 JIT 或初始化时间计入稳态。"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
import triton
from mini_ep.distributed.peer import PeerSwiGLU
from mini_ep.model.moe import ep_forward
from mini_ep.profiler.stages import Stages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens',type=int,nargs='+',default=[1,8,128,1024])
    p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--intermediate',type=int,default=512)
    p.add_argument('--experts',type=int,default=256)
    p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--repeats',type=int,default=30)
    p.add_argument('--output',required=True)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    if a.warmup<1 or a.repeats<3 or min(a.tokens)<0:
        p.error('require warmup>=1 repeats>=3 tokens>=0')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    rank,size=dist.get_rank(),dist.get_world_size()
    device=torch.device('cuda',torch.cuda.current_device())
    if a.experts%size:
        p.error('experts must divide EP size')
    l=a.experts//size
    torch.manual_seed(25+rank)
    w13=(torch.randn(l,2*a.intermediate,a.hidden,device=device)/a.hidden**0.5).bfloat16()
    w2=(torch.randn(l,a.hidden,a.intermediate,device=device)/a.intermediate**0.5).bfloat16()
    root=Path(a.output); root.mkdir(parents=True,exist_ok=True)
    results=[]
    for n in a.tokens:
        for routing in ('uniform','skew'):
            torch.manual_seed(100+rank)
            x=torch.randn(n,a.hidden,device=device).bfloat16()
            if routing=='uniform':
                ids=torch.randint(a.experts,(n,a.top_k),device=device)
            else:
                ids=torch.arange(a.top_k,device=device).expand(n,a.top_k).contiguous()
            weights=torch.rand(n,a.top_k,device=device)
            weights=(weights/weights.sum(1,keepdim=True)).contiguous()
            peer=PeerSwiGLU(dist.group.WORLD,a.hidden,l,a.top_k,[n]*size,device)
            def run(label,stages=None):
                if label=='peer':
                    return peer.forward(x,ids,weights,w13,w2,validate=False,stages=stages)[0]
                c=((n*a.top_k+7)//8)*8 if 0<n<=8 else None
                return ep_forward(x,ids,weights,w13,w2,dist.group.WORLD,validate=False,
                                  fixed_rows_per_rank=c,stages=stages)[0]
            expected=run('nccl')
            actual=run('peer')
            torch.testing.assert_close(actual,expected,atol=0.02,rtol=0.03)
            error=(actual.float()-expected.float()).abs().max().item() if n else 0
            samples={'nccl':[],'peer':[]}
            for order in [('nccl','peer'),('peer','nccl')]:
                for label in order:
                    for _ in range(a.warmup):run(label)
                    torch.cuda.synchronize()
                    for _ in range(a.repeats):
                        dist.barrier()
                        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                        start.record(); y=run(label);end.record();end.synchronize()
                        value=torch.tensor(start.elapsed_time(end),device=device,dtype=torch.float64)
                        dist.all_reduce(value,op=dist.ReduceOp.MAX)
                        samples[label].append(value.item())
            med={k:statistics.median(v) for k,v in samples.items()}
            stages_result={}
            for label in ('nccl','peer'):
                stages=Stages(timings=True)
                run(label,stages)
                torch.cuda.synchronize()
                stages_result[label]=stages.milliseconds()
            record=dict(tokens=n,batch_size=1,seq_len=n,hidden_size=a.hidden,
                expert_hidden_size=a.intermediate,num_experts=a.experts,top_k=a.top_k,
                ep_size=size,dtype='torch.bfloat16',chunk_size=None,routing=routing,
                max_abs_error=error,exact=torch.equal(actual,expected),samples_max_rank_ms=samples,
                median_max_rank_ms=med,latency_reduction=1-med['peer']/med['nccl'],
                stages_local_ms=stages_result,
                tokens_per_expert=torch.bincount(ids.flatten(),minlength=a.experts).cpu().tolist())
            results.append(record)
            (root/f'rank{rank}.json').write_text(json.dumps(dict(
                torch=torch.__version__,triton=triton.__version__,cuda=torch.version.cuda,
                nccl=torch.cuda.nccl.version(),device=torch.cuda.get_device_name(),seed=25,
                args=vars(a),statistics='median of max EP rank per sample, forward/reverse order',
                source_sha256={str(q.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(q.read_bytes()).hexdigest()
                  for q in (Path(__file__).resolve().parents[1]/'mini_ep').rglob('*.py')},
                results=results),indent=2))
            print(f'rank={rank} N={n} {routing} {med}',flush=True)
            if a.profile:
                torch.cuda.profiler.start()
                for label in ('nccl','peer'):
                    with torch.cuda.nvtx.range('DYNAMIC_'+label.upper()):
                        for _ in range(5):run(label)
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            del peer
    dist.destroy_process_group()

if __name__=='__main__':main()
