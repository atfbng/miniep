"""Check four-rank timing proofs, fairness, tokens, and paired cleanup timings."""
from pathlib import Path
import csv,io,json,sys,hashlib
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code/course13'))
from benchmarks.deepep_report import assess_job
from benchmarks.deepep_metrics import distribution
ART=Path(__file__).resolve().parent
raw=ART/'remote-raw'
load=lambda p:json.loads(p.read_text())
state=load(raw/'progress.json')
assert state['status']=='passed'
errors=[];jobs=[];groups=defaultdict(list)
old_sha=load(ART/'before-sha256.json')
new_sha={str(p.relative_to(ROOT/'code/course13')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'code/course13/mini_ep').rglob('*.py')}
for trial in range(2):
 pair={}
 for label in ('before','after'):
  folder=raw/f'bench-{trial}-{label}'
  proof=assess_job(folder)
  if not proof['passed']:errors.append(dict(job=folder.name,errors=proof['errors']))
  jobs.append(dict(name=folder.name,**proof))
  pair[label]=folder
  for sample in load(folder/'rank0/samples.json'):
   groups[(label,sample['case'])].append(sample)
 for rank in range(4):
  left=pair['before']/f'rank{rank}';right=pair['after']/f'rank{rank}'
  x,y=load(left/'environment.json'),load(right/'environment.json')
  for env,wanted in ((x,old_sha),(y,new_sha)):
   actual={k:v for k,v in env['source_sha256'].items() if k.startswith('mini_ep/')}
   if actual!=wanted:errors.append(f'core source snapshot rank{rank} trial{trial}')
  if load(left/'manifest.json')!=load(right/'manifest.json'):errors.append(f'input rank{rank} trial{trial}')
  for key in ('options','versions','checkpoint_digest','model_parameters','cpu_quota','cuda_version','nccl_version','profile','warmup','repeats'):
   if x.get(key)!=y.get(key):errors.append(f'{key} rank{rank} trial{trial}')
  def devices(env):
   rows=csv.DictReader(io.StringIO(env['gpu_before']))
   return sorted((r[next(k for k in r if k.strip()=='uuid')].strip(),r[next(k for k in r if k.strip()=='driver_version')].strip()) for r in rows)
  if devices(x)!=devices(y):errors.append(f'devices rank{rank} trial{trial}')
  def tokens(p):return [(s['case'],s['repetition'],s['token_ids']) for s in load(p/'samples.json')]
  if tokens(left)!=tokens(right):errors.append(f'tokens rank{rank} trial{trial}')
  ta,tb=load(left/'transport.json'),load(right/'transport.json')
  if ta['collectives']!=tb['collectives']:errors.append(f'startup collective trace rank{rank} trial{trial}')
  keys=(set(ta['metadata'])|set(tb['metadata']))-{'router_logits_reused','fixed_pack_backend_requested'}
  if any(ta['metadata'].get(k)!=tb['metadata'].get(k) for k in keys):errors.append(f'startup layout/transport metadata rank{rank} trial{trial}')
rows={}
for name in sorted({key[1] for key in groups}):
 a,b=groups['before',name],groups['after',name]
 metrics={}
 for key,index in [('e2e_ms',0),('prefill_ms',2),('decode_step_ms',4)]:
  if not a[0]['max_rank_ms'][index]:
   assert all(s['max_rank_ms'][index]==0 for s in a+b)
   metrics[key]=None;continue
  before=distribution([s['max_rank_ms'][index] for s in a]);after=distribution([s['max_rank_ms'][index] for s in b])
  metrics[key]=dict(before=before,after=after,median_change_percent=(after['median']/before['median']-1)*100 if before['median'] else None,p10_p90_overlap=max(before['p10'],after['p10'])<=min(before['p90'],after['p90']))
 rows[name]=metrics
accuracy_full=load(raw/'accuracy/candidate-rotated-13-r1/acceptance.json')
assert accuracy_full['passed'],accuracy_full
for rank in range(4):
 env=load(raw/f'accuracy/candidate-rotated-13-r1/rank{rank}/environment.json')
 actual={k:v for k,v in env['spec']['source_sha256'].items() if k.startswith('mini_ep/')}
 if actual!=new_sha:errors.append(f'precision core source snapshot rank{rank}')
accuracy={k:v for k,v in accuracy_full.items() if k!='ranks'}
accuracy['ranks']={k:{n:v for n,v in q.items() if n!='model_calls'} for k,q in accuracy_full['ranks'].items()}
report=dict(passed=not errors,errors=errors,timing_jobs=jobs,cases=rows,accuracy=accuracy,statistic='2 independent engines/version; AB then BA; 3 warmups plus 5 repeats/case/engine, 10 samples/version/case; maximum rank per repetition; median/p10/p90. Profiling/shadow disabled.',scope='Incremental dead-code cleanup; not a fresh nine-configuration accuracy matrix or a new native DeepEP performance sweep.')
(ART/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=report['passed'],errors=errors,cases=rows),indent=2))
assert report['passed']
