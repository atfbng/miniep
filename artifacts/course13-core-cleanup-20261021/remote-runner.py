from pathlib import Path
import subprocess,json,os,time,sys
base=Path(__file__).resolve().parent
state=dict(status='running',jobs=[])
def save():
 (base/'progress.json').write_text(json.dumps(state,indent=2))
def job(name,source,argv,timeout=7200):
 record=dict(name=name,command=argv,cwd=str(base/source),started=time.time())
 state['jobs'].append(record); save()
 with (base/(name+'.log')).open('w') as log:
  p=subprocess.run(argv,cwd=base/source,env={**os.environ,'PYTHONPATH':str(base/source),'MINIEP_STAGE_TIMING':'0'},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
 record.update(exit_code=p.returncode,ended=time.time());save()
 if p.returncode: raise RuntimeError(name+' failed')
try:
 job('unit','after',['python3','-m','unittest','discover','-s','checks','-p','test_*.py'])
 job('tp-dp','after',['torchrun','--standalone','--nproc-per-node=4','checks/tp_dp.py'])
 job('return-transport','after',['torchrun','--standalone','--nproc-per-node=4','checks/return_transport.py'])
 job('exact-corpus','after',['python3','test/deepep_alignment.py','--model','/model','--output',str(base/'accuracy'),'--checkpoint-manifest','/experiment/default-exact-v3/deep4-full/checkpoint.json','--jobs','candidate-rotated-13-r1','--reuse-completed','/experiment/default-exact-v3/deep4-full','--audit-native-reuse'])
 for trial,order in enumerate([['before','after'],['after','before']]):
  for source in order:
   name=f'bench-{trial}-{source}'
   job(name,source,['torchrun','--standalone','--nproc-per-node=4','benchmarks/deepep_latency_run.py','--model','/model','--output',str(base/name),'--checkpoint-manifest','/experiment/default-exact-v3/deep4-full/checkpoint.json','--backend','miniep','--label',source,'--warmup','3','--repeats','5','--cases','decode_b1','decode_b32','prefill_b4_long'])
 state['status']='passed';save()
except Exception as e:
 state.update(status='failed',error=repr(e));save();raise
