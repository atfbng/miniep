"""五个独立引擎配对块，AB/BA 交替比较 Peer 去重与原生 DeepEP。

请求矩阵在运行前固定；每块 seed/语料、GPU、配置与字面 Token IDs 一致。
保留生成差异并排除对应性能结论，后端回退或配置不一致直接判定比较无效。"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.extended_workloads import experiment_cases

SEEDS=(13,37,123,321,777)


def devices(environment):
    return [{k.strip():v.strip() for k,v in row.items() if k.strip() in ('index','uuid','name','driver_version')}
            for row in csv.DictReader(io.StringIO(environment['gpu_before']))]


def validate_pair(native,optimized,world):
    """核对后端、环境及字面输入；排除 TP 副本重复计数，逐项保留生成差异。"""
    errors=[];differences=[];positions=0
    for rank in range(world):
        n,o=Path(native)/f'rank{rank}',Path(optimized)/f'rank{rank}'
        x,y=[json.loads((p/'environment.json').read_text()) for p in (n,o)]
        for key in ('versions','options','checkpoint_digest','model_parameters','cuda_version','nccl_version','warmup','repeats','clock_warmup_seconds','cpu_quota','source_sha256'):
            if x[key]!=y[key]:errors.append(dict(rank=rank,error=f'{key} differs'))
        if devices(x)!=devices(y):errors.append(dict(rank=rank,error='GPU identities differ'))
        if json.loads((n/'manifest.json').read_text())!=json.loads((o/'manifest.json').read_text()):
            errors.append(dict(rank=rank,error='literal request IDs differ'))
        proof=json.loads((n/'backend.json').read_text());other=json.loads((o/'backend.json').read_text())
        if proof.get('manager')!='DeepEPHTAll2AllManager' or proof.get('runner')!='MoERunner':
            errors.append(dict(rank=rank,error='native fallback',proof=proof))
        transfer=json.loads((o/'transport.json').read_text())
        meta=transfer['metadata']
        valid_peer=meta.get('transport') in ('peer', 'peer_dedup')
        valid_dedup=meta.get('deduplicated', meta.get('transport')=='peer_dedup')
        if other.get('runner')!='MiniEPRunner' or not valid_peer or not valid_dedup:
            errors.append(dict(rank=rank,error='wrong optimized implementation'))
        assert proof['startup_wrapper_removed'] and other['startup_wrapper_removed'] and transfer['startup_wrapper_removed']
        a,b=[json.loads((p/'samples.json').read_text()) for p in (n,o)]
        if len(a)!=len(b):errors.append(dict(rank=rank,error='sample count differs'));continue
        for aa,bb in zip(a,b,strict=True):
            if (aa['case'],aa['repetition'])!=(bb['case'],bb['repetition']):
                errors.append(dict(rank=rank,error='sample ordering differs'));continue
            # TP 副本含相同输出，只按每 DP 的一个 TP rank 计生成位置，避免重复统计。
            if rank%(world//2)==0:
                positions+=sum(map(len,aa['token_ids']))
                if aa['token_ids']!=bb['token_ids']:
                    mismatches=sum(sum(v!=w for v,w in zip(vv,ww,strict=True)) for vv,ww in zip(aa['token_ids'],bb['token_ids'],strict=True))
                    differences.append(dict(rank=rank,case=aa['case'],repetition=aa['repetition'],mismatched_positions=mismatches))
    return dict(valid_comparison=not errors,errors=errors,token_differences=differences,
                all_generated_tokens_equal=not differences,unique_dp_generated_positions=positions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',required=True);p.add_argument('--manifest-dir',required=True)
    p.add_argument('--output',required=True);p.add_argument('--tp-size',type=int,choices=[1,2],required=True)
    p.add_argument('--blocks',type=int,default=5);p.add_argument('--warmup',type=int,default=2)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--cases',nargs='+')
    a=p.parse_args()
    if not 1<=a.blocks<=len(SEEDS) or a.warmup<1 or a.repeats<3:p.error('require 1..5 blocks, warmup>=1, repeats>=3')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    cases=experiment_cases()
    if a.cases:
        cases=[q for q in cases if q['name'] in a.cases]
        if set(a.cases)!={q['name'] for q in cases}:p.error('unknown workload')
    world=a.tp_size*2;kv=12288 if a.tp_size==1 else 8192
    journal=dict(status='running',models=a.models,tp_size=a.tp_size,ep_size=world,blocks=a.blocks,
                 warmup=a.warmup,repeats=a.repeats,cases=cases,jobs=[],pairs=[],started=time.time())
    def save():
        temp=root/'suite.tmp';temp.write_text(json.dumps(journal,indent=2));temp.replace(root/'suite.json')
    save()
    for model in a.models:
        model_name=Path(model).name
        for block,seed in enumerate(SEEDS[:a.blocks]):
            spec=root/f'{model_name}-block{block}-workloads.json'
            spec.write_text(json.dumps(dict(seed=seed,corpus=block%3,cases=cases),indent=2))
            # 第二个模型也使用相反顺序，抵消固定先后顺序带来的漂移。
            reverse=(block+a.models.index(model))%2
            order=('peer_dedup','native') if reverse else ('native','peer_dedup')
            pair={}
            for label in order:
                name=f'{model_name}-ep{world}-block{block}-{label}';dest=root/name
                command=['torchrun','--standalone',f'--nproc-per-node={world}','--monitor_interval=1',
                    str(Path(__file__).with_name('deepep_latency_run.py')),'--model',model,
                    '--checkpoint-manifest',str(Path(a.manifest_dir)/(model_name+'-checkpoint.json')),
                    '--backend','native' if label=='native' else 'miniep','--transport','nccl' if label=='native' else 'peer_dedup',
                    '--label',label,'--output',str(dest),'--tp-size',str(a.tp_size),
                    '--workload-spec',str(spec),'--max-model-len','4608','--max-num-batched-tokens','32768',
                    '--max-num-seqs','32','--kv-cache-mib',str(kv),'--gpu-memory-utilization','0.85' if world==2 else '0.65',
                    '--warmup',str(a.warmup),'--repeats',str(a.repeats),'--clock-warmup-seconds','3',
                    '--seed',str(seed),'--safetensors-load-strategy','lazy']
                job=dict(name=name,path=str(dest),model=model,block=block,seed=seed,label=label,command=command,started=time.time())
                journal['jobs'].append(job);save();print('START',name,flush=True)
                with (root/(name+'.log')).open('w') as log:
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    job['pid']=proc.pid;save()
                    try:status=proc.wait(timeout=2700)
                    except BaseException:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        status=124
                job.update(exit_code=status,finished=time.time());save()
                if status or any(not(dest/f'rank{r}/completed.json').exists() for r in range(world)):
                    journal.update(status='failed',failed_job=name);save();return 1
                pair[label]=str(dest);print('DONE',name,flush=True)
            proof=validate_pair(pair['native'],pair['peer_dedup'],world)
            (root/f'{model_name}-block{block}-pair.json').write_text(json.dumps(proof,indent=2))
            journal['pairs'].append(dict(model=model,block=block,paths=pair,**proof));save()
            if not proof['valid_comparison']:
                journal.update(status='invalid_comparison');save();return 1
    journal.update(status='completed',finished=time.time());save();return 0

if __name__=='__main__':sys.exit(main())
