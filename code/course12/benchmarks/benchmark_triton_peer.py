"""Same-input H100 comparison of synchronous NCCL and cooperative Triton EP."""
from __future__ import annotations
import argparse,json,os,sys,statistics
from datetime import timedelta
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from benchmarks.benchmark_stream_fusion import prepare_chunks,expert_mlp,expert_reference,measure,run_ordinary
from mini_ep.kernels.comm_compute.triton_peer import TritonPeerMLP,TritonPeerSeparated


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--prefix')
    a.add_argument('--output',required=True)
    a.add_argument('--tokens','--n',dest='n',type=int,default=17);a.add_argument('--hidden-size','--h',dest='h',type=int,default=128);a.add_argument('--expert-hidden-size','--f',dest='f',type=int,default=256)
    a.add_argument('--experts',type=int,default=8);a.add_argument('--top-k',type=int,default=2)
    a.add_argument('--skew',action='store_true');a.add_argument('--weighted',action='store_true')
    a.add_argument('--bm',type=int,default=128);a.add_argument('--bn',type=int,default=256);a.add_argument('--bk',type=int,default=64)
    a.add_argument('--warps',type=int,default=8);a.add_argument('--stages',type=int,default=3);a.add_argument('--grid',type=int);a.add_argument('--comm',type=int,default=16);a.add_argument('--copy-n',type=int,default=256)
    a.add_argument('--profile',action='store_true');a.add_argument('--warmup',type=int,default=5);a.add_argument('--repeats',type=int,default=10)
    args=a.parse_args()
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=120),device_id=device)
    p=dist.get_world_size();torch.manual_seed(20261007+rank)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.prefix:
        prefix=f'{args.prefix}-rank{rank}';meta=json.loads(Path(prefix+'.json').read_text())
        n,h,f,k,e=(meta[z] for z in ('n','h','f','k','e'))
        def load(suffix,dtype,shape):return torch.from_numpy(np.fromfile(prefix+suffix+'.bin',dtype=dtype).reshape(shape).copy()).to(device)
        x=load('-x',np.float16,(n,h));up=load('-up',np.float16,(e//p,f,h));down=load('-down',np.float16,(e//p,h,f))
        routes=load('-routes',np.int32,(n,k)).long()
    else:
        n,h,f,k,e=args.n,args.h,args.f,args.top_k,args.experts
        if e%p or k not in (1,2) or k>e:raise ValueError('invalid expert placement or Top-K')
        x=torch.empty((n,h),device=device,dtype=torch.float16).uniform_(-1,1)
        up=torch.empty((e//p,f,h),device=device,dtype=torch.float16).uniform_(-h**-.5,h**-.5)
        down=torch.empty((e//p,h,f),device=device,dtype=torch.float16).uniform_(-f**-.5,f**-.5)
        routes=((torch.arange(n*k,device=device).reshape(n,k)+rank)%e).long()
        if args.skew:routes=torch.arange(k,device=device).expand(n,k).contiguous()
    weights=torch.full((n,k),1/k,device=device,dtype=torch.float32)
    if args.weighted:
        weights=torch.rand((n,k),device=device);weights/=weights.sum(1,keepdim=True)
    whole=prepare_chunks(x,routes,n,e)[0]
    def combine():
        return (whole.returned[whole.inverse].float().reshape(n,k,h)*weights[:,:,None]).sum(1)
    def ordinary():
        dist.all_to_all_single(whole.received,whole.packed,whole.recv_splits,whole.send_splits)
        y=expert_mlp(whole.received,up,down,whole.local_expert_rows)
        dist.all_to_all_single(whole.returned,y,whole.send_splits,whole.recv_splits)
        return combine()
    if not args.weighted:
        ordinary=lambda:run_ordinary(whole,up,down,k)
    expected=ordinary()
    y=expert_reference(whole.received,up,down,whole.local_expert_rows)
    dist.all_to_all_single(whole.returned,y,whole.send_splits,whole.recv_splits)
    reference=combine()
    torch.testing.assert_close(expected,reference,atol=.003,rtol=.02)
    fused=TritonPeerMLP(x,routes,weights,up,down,bm=args.bm,bn=args.bn,bk=args.bk,warps=args.warps,stages=args.stages,grid=args.grid,comm=args.comm,copy_n=args.copy_n)
    print(f'prepared rank={rank} M={fused.m} rows={fused.rows.tolist()}',flush=True)
    for _ in range(3):
        got=fused.run();torch.cuda.synchronize()
        assert got.shape==(n,h) and got.dtype==torch.float32 and got.device==device
        torch.testing.assert_close(got,reference,atol=.003,rtol=.02)
    error=(got-reference).abs().max().item()
    # Refresh the payload and repeat without host barriers between fused calls.
    fused.update_input(x*.5)
    whole.packed.mul_(.5);changed=ordinary()
    queued=[fused.run().clone() for _ in range(6)]
    torch.cuda.synchronize()
    for output in queued:torch.testing.assert_close(output,changed,atol=.003,rtol=.02)
    fused.update_input(x);whole.packed.copy_(x[torch.argsort(routes.flatten(),stable=True)//k])
    torch.cuda.synchronize();dist.barrier()
    separated=TritonPeerSeparated(x,routes,weights,up,down,bm=args.bm,bn=args.bn,bk=args.bk,warps=args.warps,stages=args.stages,copy_n=args.copy_n)
    sep_output=separated.run()
    assert sep_output.shape==(n,h) and sep_output.dtype==torch.float32 and sep_output.device==device
    torch.testing.assert_close(sep_output,reference,atol=.003,rtol=.02)
    errors={'ordinary_reference':(expected-reference).abs().max().item(),
            'triton_fused_reference':error,'triton_separated_reference':(sep_output-reference).abs().max().item()}
    queued=[separated.run().clone() for _ in range(6)]
    torch.cuda.synchronize()
    for output in queued:torch.testing.assert_close(output,reference,atol=.003,rtol=.02)
    separated.update_input(x*.25);whole.packed.mul_(.25)
    changed_sep=ordinary()
    queued=[separated.run().clone() for _ in range(6)]
    torch.cuda.synchronize()
    for output in queued:torch.testing.assert_close(output,changed_sep,atol=.003,rtol=.02)
    separated.update_input(x);whole.packed.copy_(x[torch.argsort(routes.flatten(),stable=True)//k])
    torch.cuda.synchronize();dist.barrier()
    functions={'ordinary':ordinary,'triton_fused':fused.run,'triton_separated':separated.run,'gemm_only':separated.compute,'pull_only':separated.pull,'push_only':separated.push,'input_push_only':lambda:separated.send_input(signal=False)}
    samples={name:[] for name in functions}
    for names in (list(functions),list(reversed(functions))):
        for name in names:
            samples[name].extend(measure(functions[name],args.warmup,args.repeats))
    if args.profile:
        for name in ('ordinary','triton_separated','triton_fused'):
            for _ in range(3):
                torch.cuda.synchronize();dist.barrier()
                with torch.cuda.nvtx.range(name.upper()+'_PROFILE'):
                    functions[name]()
                    torch.cuda.synchronize()
    fused.run(profile=True);torch.cuda.synchronize();dist.barrier()
    times=fused.times.cpu().numpy()
    phase_ns={"start":int(times[:,0].min()),"comm_end":int(times[:fused.comm,1].max()),"up_end":int(times[:,1].max()),"down_start":int(times[:,2].min()),"down_end":int(times[:,3].max()),"last_barrier":int(times[:,4].max())}
    result={'rank':rank,'ep_size':p,'n':n,'h':h,'f':f,'e':e,'k':k,'dtype':'float16','output_dtype':'float32',
            'bm':args.bm,'bn':args.bn,'bk':args.bk,'warps':args.warps,'stages':args.stages,'grid':fused.grid,'comm':fused.comm,'copy_n':fused.copy_n,
            'compile':{'registers':fused.last_kernel.n_regs,'spills':fused.last_kernel.n_spills,'shared':fused.last_kernel.metadata.shared,'wgmma_count':fused.last_kernel.asm['ptx'].count('wgmma.mma_async'),'mma_sync_count':fused.last_kernel.asm['ptx'].count('mma.sync')},
            'samples_ms':samples,'median_ms':{k:statistics.median(v) for k,v in samples.items()},'max_abs_error':error,'errors':errors,'phase_ns':phase_ns,
            'warmup_per_block':args.warmup,'repeats_per_block':args.repeats,'statistics':'median of per-iteration maximum-rank CUDA Event milliseconds; forward/reverse blocks',
            'scope':'prepared input through FP32 original-token-order weighted output; routing/packing/count exchange excluded; workspace reused',
            'communication_bytes':2*sum(whole.send_splits[j] for j in range(p) if j!=rank)*h*x.element_size(),
            'software':{'torch':torch.__version__,'cuda':torch.version.cuda,'triton':__import__('triton').__version__},
            'device':torch.cuda.get_device_name(device),'device_uuid':str(torch.cuda.get_device_properties(device).uuid),
            'skew':args.skew,'weighted':args.weighted,'changed_input_check':'PASS','repeat_check':'PASS'}
    out=Path(args.output+f'-rank{rank}.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
    print(f'PASS rank={rank} {result["median_ms"]} error={error}',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':
    with torch.no_grad():main()
