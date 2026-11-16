"""历史去重补测工具：复用已完成的冻结对照，不自动重跑原生/NCCL。

每模型三个新引擎，配置与字面输入对齐 peer_suite；不追加微基准或 Timeline。
历史对照对应其源码快照，不能当作当前 NCCL 去重版本的测量。"""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',required=True);p.add_argument('--manifest-dir',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    baseline=Path(a.baseline).resolve();jobs=[]
    for model in a.models:
        for trial in range(3):
            name=f'{Path(model).name}-trial{trial}-peer_dedup';dest=root/name
            command=['torchrun','--standalone','--nproc-per-node=4','--monitor_interval=1',
                str(Path(__file__).with_name('deepep_latency_run.py')),
                '--model',model,'--checkpoint-manifest',str(Path(a.manifest_dir)/(Path(model).name+'-checkpoint.json')),
                '--backend','miniep','--transport','peer_dedup','--label','peer_dedup',
                '--output',str(dest),'--warmup','2','--repeats','3','--clock-warmup-seconds','3',
                '--safetensors-load-strategy','lazy','--kv-cache-mib','4096']
            job=dict(name=name,model=model,trial=trial,path=str(dest),command=command,started=time.time())
            jobs.append(job);save(root,jobs,'running');print('START',name,flush=True)
            with (root/(name+'.log')).open('w') as log:
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:status=proc.wait(timeout=1800)
                except BaseException:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    status=124
            job.update(exit_code=status,finished=time.time())
            if status:
                save(root,jobs,'failed');return 1
            proof=check_job(dest,baseline,Path(model).name,trial)
            (dest/'acceptance.json').write_text(json.dumps(proof,indent=2))
            job['passed']=proof['passed'];save(root,jobs,'running')
            if not proof['passed']:save(root,jobs,'failed');return 1
            print('DONE',name,flush=True)
    rows={}
    for model in a.models:
        label=Path(model).name;cases={}
        for trial in range(3):
            data=json.loads((root/f'{label}-trial{trial}-peer_dedup/rank0/samples.json').read_text())
            for r in data:cases.setdefault(r['case'],[]).append(r)
        rows[label]={}
        for case,data in cases.items():
            rows[label][case]=dict(samples=len(data),
                e2e_ms=statistics.median(r['max_rank_ms'][0] for r in data),
                prefill_ms=statistics.median(r['max_rank_ms'][2] for r in data),
                decode_step_ms=statistics.median(r['max_rank_ms'][4] for r in data),
                raw_max_rank_ms=[r['max_rank_ms'] for r in data])
    (root/'summary.json').write_text(json.dumps(dict(cases=rows,all_tokens_equal=True,
        scope='supplemental runs after frozen Peer/NCCL/DeepEP baseline suite; not interleaved',
        statistics='median of max-rank durations, 3 fresh engines x 3 repeats'),indent=2))
    save(root,jobs,'completed');return 0


def save(root,jobs,status):
    (root/'suite.json').write_text(json.dumps(dict(status=status,jobs=jobs),indent=2))


def check_job(root,baseline,model,trial):
    errors=[];positions=0
    for rank in range(4):
        d=root/f'rank{rank}';b=baseline/f'{model}-trial{trial}-peer/rank{rank}'
        try:
            completed=json.loads((d/'completed.json').read_text())
            assert completed['samples']==18
            env=json.loads((d/'environment.json').read_text());old=json.loads((b/'environment.json').read_text())
            for key in ('versions','options','checkpoint_digest','model_parameters','cuda_version','nccl_version','warmup','repeats','cpu_quota'):
                if env[key]!=old[key]:errors.append(dict(rank=rank,error=f'{key} differs'))
            if json.loads((d/'manifest.json').read_text())!=json.loads((b/'manifest.json').read_text()):
                errors.append(dict(rank=rank,error='literal inputs differ'))
            backend=json.loads((d/'backend.json').read_text());transport=json.loads((d/'transport.json').read_text())
            assert backend['runner']=='MiniEPRunner' and backend['ep_size']==4
            assert 'deduplicated' in backend['transport'] and backend['startup_wrapper_removed']
            assert transport['metadata']['transport'] in ('peer', 'peer_dedup') and transport['startup_wrapper_removed']
            assert transport['metadata'].get('deduplicated', transport['metadata']['transport'] == 'peer_dedup')
            data=json.loads((d/'samples.json').read_text());previous=json.loads((b/'samples.json').read_text())
            for x,y in zip(data,previous,strict=True):
                assert (x['case'],x['repetition'])==(y['case'],y['repetition'])
                if x['token_ids']!=y['token_ids']:errors.append(dict(rank=rank,case=x['case'],error='generated tokens differ'))
                if rank%2==0:positions+=sum(map(len,x['token_ids']))
        except (AssertionError,FileNotFoundError) as e:
            errors.append(dict(rank=rank,error=repr(e)))
    return dict(passed=not errors,errors=errors,unique_dp_generated_positions=positions)

if __name__=='__main__':sys.exit(main())
