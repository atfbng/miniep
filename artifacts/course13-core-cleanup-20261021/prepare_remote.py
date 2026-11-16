from pathlib import Path
import tarfile,io,base64,json
root=Path('code/course13'); art=Path('artifacts/course13-core-cleanup-20261021')
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as a:
 for p in sorted(root.rglob('*.py')):
  if not any(x.startswith('draft') or x in ('history','__pycache__') for x in p.relative_to(root).parts):
   a.add(p,arcname=p.relative_to(root))
(art/'after-source.tgz').write_bytes(buf.getvalue())
payload=base64.b64encode(buf.getvalue()).decode()
before=base64.b64encode((art/'before.tgz').read_bytes()).decode()
script='''from pathlib import Path
import base64,io,tarfile,shutil,subprocess,json,os
base=Path('/experiment/core-cleanup-20261021')
base.mkdir(exist_ok=False)
for label,payload in PAYLOADS:
 target=base/label
 shutil.copytree('/experiment/deepep-delivery-v7',target,ignore=shutil.ignore_patterns('__pycache__'))
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload)),mode='r:gz') as a:
  a.extractall(target,filter='data')
(base/'run.py').write_text(RUNNER)
log=(base/'controller.log').open('w')
p=subprocess.Popen(['python3',str(base/'run.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','NCCL_NVLS_ENABLE':'0'})
print(json.dumps(dict(pid=p.pid,root=str(base))))
'''
runner='''from pathlib import Path
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
'''
# Before gets exactly the pre-cleanup core; other tooling is current in both.
script=script.replace('PAYLOADS',repr([('before',payload),('after',payload)]))
script=script.replace('(base/\'run.py\').write_text(RUNNER)',"with tarfile.open(fileobj=io.BytesIO(base64.b64decode("+repr(before)+")),mode='r:gz') as a:\n a.extractall(base/'before',filter='data')\n(base/'run.py').write_text("+repr(runner)+")")
(art/'remote-upload.py').write_text(script)
(art/'remote-runner.py').write_text(runner)
