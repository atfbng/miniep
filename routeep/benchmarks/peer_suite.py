"""新建引擎比较 NCCL去重、Peer/TMA去重及原生DeepEP HT。

先运行正确性检查。每个任务独立启动进程组和引擎，保留 GPU/软件/源码/
checkpoint 身份与字面输入；正常计时不保留精度探针或 profiling 钩子。"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',required=True)
    parser.add_argument('--manifest-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--trials',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--job-timeout',type=int,default=1800)
    a=parser.parse_args()
    if a.trials<3 or a.warmup<1 or a.repeats<3:
        parser.error('require trials>=3 warmup>=1 repeats>=3')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for model in a.models:
        for trial in range(a.trials):
            labels=['nccl','peer','deepep']
            labels=labels[trial%3:]+labels[:trial%3]
            for label in labels:
                name=f'{Path(model).name}-trial{trial}-{label}'
                path=root/name
                command=['torchrun','--standalone','--nproc-per-node=4','--monitor_interval=1',
                  str(Path(__file__).with_name('deepep_latency_run.py')),
                  '--model',model,'--checkpoint-manifest',str(Path(a.manifest_dir)/(Path(model).name+'-checkpoint.json')),
                  '--backend','native' if label=='deepep' else 'miniep',
                  '--transport','peer' if label=='peer' else 'nccl',
                  '--label',label,'--output',str(path),'--warmup',str(a.warmup),
                  '--repeats',str(a.repeats),'--clock-warmup-seconds','3',
                  '--safetensors-load-strategy','lazy','--kv-cache-mib','4096']
                record=dict(model=model,trial=trial,label=label,name=name,path=str(path),command=command,started=time.time())
                jobs.append(record)
                (root/'suite.json').write_text(json.dumps(dict(status='running',jobs=jobs),indent=2))
                print('START',name,flush=True)
                with (root/(name+'.log')).open('w') as log:
                    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:status=proc.wait(timeout=a.job_timeout)
                    except BaseException:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        status=124
                record.update(exit_code=status,finished=time.time())
                if status or any(not(path/f'rank{r}/completed.json').exists() for r in range(4)):
                    (root/'suite.json').write_text(json.dumps(dict(status='failed',jobs=jobs),indent=2))
                    print('FAILED',name,flush=True);return 1
                print('DONE',name,flush=True)
    summary=summarize(jobs)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    (root/'suite.json').write_text(json.dumps(dict(status='completed',jobs=jobs),indent=2))
    return 0


def summarize(jobs):
    grouped=defaultdict(list);tokens={};manifests={};differences=[];proofs=[]
    for job in jobs:
        path=Path(job['path']);model=Path(job['model']).name
        for r in range(4):
            d=path/f'rank{r}'
            env=json.loads((d/'environment.json').read_text())
            proof=json.loads((d/'backend.json').read_text())
            proofs.append(dict(job=job['name'],rank=r,proof=proof))
            if job['label']=='deepep':
                assert proof['manager']=='DeepEPHTAll2AllManager' and proof['runner']=='MoERunner'
            else:
                transport=json.loads((d/'transport.json').read_text())
                assert transport['metadata'].get('transport','nccl')==job['label']
                assert transport['metadata']['deduplicated']
            assert env['options']['tensor_parallel_size']==2 and env['options']['data_parallel_size']==2
            signature=(model,r)
            manifest=json.loads((d/'manifest.json').read_text())
            if signature in manifests:assert manifests[signature]==manifest
            manifests[signature]=manifest
            samples=json.loads((d/'samples.json').read_text())
            for row in samples:
                key=(model,r,row['case'],row['repetition'])
                value=row['token_ids']
                if key in tokens and tokens[key][1]!=value:
                    differences.append(dict(model=model,rank=r,case=row['case'],trial=job['trial'],
                        left=tokens[key][0],right=job['label'],
                        mismatched_positions=sum(sum(x!=y for x,y in zip(xx,yy)) for xx,yy in zip(tokens[key][1],value))))
                tokens[key]=(job['label'],value)
                if r==0:grouped[(model,row['case'],job['label'])].append(row)
    cases={}
    for (model,case,label),rows in grouped.items():
        cases.setdefault(model,{}).setdefault(case,{})[label]=dict(
            samples=len(rows),e2e_ms=statistics.median(q['max_rank_ms'][0] for q in rows),
            prefill_ms=statistics.median(q['max_rank_ms'][2] for q in rows),
            decode_step_ms=statistics.median(q['max_rank_ms'][4] for q in rows),
            raw_max_rank_ms=[q['max_rank_ms'] for q in rows])
    for model,rows in cases.items():
        for case,values in rows.items():
            for metric in ('e2e_ms','prefill_ms','decode_step_ms'):
                peer=values['peer'][metric]
                for label in ('nccl','deepep'):
                    baseline=values[label][metric]
                    values.setdefault('peer_latency_reduction',{}).setdefault(label,{})[metric]=1-peer/baseline if baseline else None
    return dict(cases=cases,backend_proofs=proofs,free_generation_token_differences=differences,
                all_free_generation_tokens_equal=not differences,
                statistics='median of 9 samples (3 fresh engines x 3 repeats), each sample maximum rank',
                scope='full offline model generation including dynamic routing, packing and metadata')

if __name__=='__main__':sys.exit(main())
