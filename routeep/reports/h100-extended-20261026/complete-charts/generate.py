"""逐场景绝对延迟图：保留所有已测配置，包含不确定和回退点。"""
from pathlib import Path
import csv,json,gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE=Path(__file__).resolve().parent.parent
OUT=Path(__file__).resolve().parent;OUT.mkdir(parents=True,exist_ok=True)
summary=json.loads((BASE/'summary.json').read_text());rows=summary['rows'];raw=json.loads(gzip.decompress((BASE/'raw-samples.json.gz').read_bytes()))
assert summary['completed'] and len(rows)==108
models=['Qwen3.5-35B-A3B','Qwen3-30B-A3B']
colors={'native':'#6366f1','peer_dedup':'#0d9488'}
plt.rcParams.update({'font.size':12})
manifest=[];coverage=set();csv_rows=[]

def shape_label(r):
 w=r['workload'];s='mixed 64..4096' if w.get('lengths') else str(w['seq_len'])
 return f"B{w['batch_size']}  S{s}  O{w['steps']}"
def ordering(r):
 w=r['workload'];return (bool(w.get('lengths')),w.get('seq_len') or 99999,w['batch_size'])

def extended(model,ep,phase):
 metric='prefill_batch_ms' if phase=='prefill' else 'decode_step_ms'
 selected=sorted([r for r in rows if r['model']==model and r['ep_size']==ep and r['metric']==metric],key=ordering)
 assert len(selected)==(19 if phase=='prefill' else 8)
 n=len(selected);height=2.8+.55*n
 fig,axes=plt.subplots(1,2,figsize=(12.0,height),sharey=True,gridspec_kw={'width_ratios':[2.5,1.2]})
 ax,rel=axes;y=np.arange(n)
 for offset,(label,title) in enumerate([('native','native DeepEP HT'),('peer_dedup','dedup + Peer/TMA')]):
  values=[];lower=[];upper=[]
  for r in selected:
   data=[q['max_rank_ms'][2 if phase=='prefill' else 4] for q in raw if q['model']==model and q['ep_size']==ep and q['case']==r['case'] and q['label']==label]
   assert len(data)==15
   m=float(np.median(data));values.append(m);lower.append(m-float(np.quantile(data,.25)));upper.append(float(np.quantile(data,.75))-m)
  bars=ax.barh(y+(offset-.5)*.30,values,.28,xerr=[lower,upper],capsize=2,color=colors[label],label=title,error_kw={'elinewidth':.8})
  ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=4,fontsize=11.5,color=colors[label])
 ax.set_yticks(y,[shape_label(r) for r in selected]);ax.invert_yaxis();ax.set_xlim(0,max(r['native']['median_ms'] for r in selected)*1.23)
 ax.set_xlabel('prefill batch latency (ms)' if phase=='prefill' else 'decode-step latency (ms)')
 ax.set_title('Absolute latency: median of 15 measurements',fontsize=12)
 ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 rel.axvspan(-2,2,color='#e2e8f0',alpha=.65);rel.axvline(0,color='#334155',lw=.9)
 for i,r in enumerate(selected):
  value=100*r['latency_reduction'];lo=100*r['reduction_ci95_low'];hi=100*r['reduction_ci95_high']
  cls=r['classification'];color='#0d9488' if cls=='stable_gain' else '#dc2626' if cls=='stable_regression' else '#64748b'
  marker='o' if cls=='stable_gain' else 'X' if cls=='stable_regression' else 's'
  rel.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt=marker,color=color,capsize=3,markersize=6)
  csv_rows.append(dict(model=model,ep_size=ep,case=r['case'],phase=phase,native_ms=r['native']['median_ms'],dedup_ms=r['peer_dedup']['median_ms'],paired_reduction=r['latency_reduction'],ci95_low=r['reduction_ci95_low'],ci95_high=r['reduction_ci95_high'],classification=cls))
  coverage.add((model,ep,r['case']))
 rel.set_xlabel('paired latency reduction (%)\npositive = dedup faster');rel.set_title('95% paired-block interval',fontsize=12)
 rel.grid(axis='x',alpha=.16);rel.spines[['top','right','left']].set_visible(False);rel.tick_params(axis='y',left=False,labelleft=False)
 handles,labels=ax.get_legend_handles_labels()
 handles += [Line2D([0],[0],marker='o',ls='',color='#0d9488',label='stable gain'),Line2D([0],[0],marker='s',ls='',color='#64748b',label='not beyond criterion')]
 labels += ['stable gain','not beyond criterion']
 fig.legend(handles,labels,loc='upper center',ncol=4,frameon=False,fontsize=11)
 fig.suptitle(f'{model} · EP{ep} / TP{ep//2} / DP2 · {phase.upper()} · ALL {n} SCENARIOS',y=.91 if phase=='decode' else .95,fontsize=14)
 fig.subplots_adjust(left=.22,right=.98,top=.78 if phase=='decode' else .87,bottom=.20 if phase=='decode' else .12,wspace=.16)
 fig.text(.5,.025,'Absolute bars: median and 25%–75% range (not CI). Relative points: 5 fresh-engine paired blocks, 95% block bootstrap.\nGray squares include uncertain / small / slower cases. Stable gain requires all 5 blocks agree and the entire interval > +2%.',ha='center',fontsize=9.5)
 tag=('qwen35' if '3.5' in model else 'qwen3')+f'-ep{ep}-{phase}-all'
 for ext in ('png','svg'):fig.savefig(OUT/f'{tag}.{ext}',dpi=190)
 plt.close(fig)
 manifest.append(dict(filename=f'{tag}.png',dataset='extended',model=model,ep_size=ep,phase=phase,scenarios=len(selected),cases=[r['case'] for r in selected],classifications={k:sum(r['classification']==k for r in selected) for k in ('stable_gain','not_beyond_margin','stable_regression','accuracy_mismatch')}))

for m in models:
 for ep in (2,4):
  for phase in ('prefill','decode'):extended(m,ep,phase)
assert coverage=={(r['model'],r['ep_size'],r['case']) for r in rows}
# The older pilot is retained separately: it used different engine settings and
# 128 output tokens for decode, and its dedup runs were staged, not paired.
old=json.loads((BASE.parent/'h100-peer-dedup-20261026/combined-summary.json').read_text())['cases']
pilot_coverage=set()
for model in models:
 for phase in ('prefill','decode'):
  cases=[c for c in old[model] if c.startswith(phase)]
  index=2 if phase=='prefill' else 4;metric='prefill_ms' if phase=='prefill' else 'decode_step_ms'
  fig,ax=plt.subplots(figsize=(12,5.6));y=np.arange(len(cases))
  for off,(key,color,title) in enumerate([('nccl','#94a3b8','original NCCL RouteEP'),('peer','#f59e0b','Peer / TMA'),('peer_dedup','#0d9488','dedup + Peer / TMA'),('deepep','#6366f1','native DeepEP HT')]):
   values=[old[model][c][key][metric] for c in cases]
   data=[[v[index] for v in old[model][c][key]['raw_max_rank_ms']] for c in cases]
   assert all(len(q)==9 for q in data)
   low=[v-np.quantile(a,.25) for v,a in zip(values,data)];high=[np.quantile(a,.75)-v for v,a in zip(values,data)]
   bars=ax.barh(y+(off-1.5)*.18,values,.17,xerr=[low,high],capsize=2,color=color,label=title)
   ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3,fontsize=10)
  ax.set_yticks(y,[{'decode_b1':'B1 S128 O128','decode_b4':'B4 S128 O128','decode_b32':'B32 S128 O128','prefill_b1_long':'B1 S2048 O1','prefill_b4_short':'B4 S128 O1','prefill_b4_long':'B4 S2048 O1'}[c] for c in cases]);ax.invert_yaxis();ax.set_xlim(0,max(old[model][c][k][metric] for c in cases for k in ('nccl','peer','peer_dedup','deepep'))*1.18)
  ax.set_xlabel('prefill batch latency (ms)' if phase=='prefill' else 'decode-step latency (ms)');ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
  fig.suptitle(f'OLDER EXPLORATORY PROTOCOL · {model} · {phase.upper()} · EP4',fontsize=13,y=.98)
  handles,labels=ax.get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=10)
  fig.subplots_adjust(left=.2,right=.98,top=.72,bottom=.18)
  fig.text(.5,.03,'9 samples per path; error bars are interquartile range. KV4 GiB, max input budget 8192, decode outputs 128.\nDedup was measured later, not interleaved. Do not pool with the extended study (KV8/12 GiB, decode outputs 64).',ha='center',fontsize=9)
  tag=('qwen35' if '3.5' in model else 'qwen3')+f'-pilot-{phase}-all'
  for ext in ('png','svg'):fig.savefig(OUT/f'{tag}.{ext}',dpi=190)
  plt.close(fig)
  pilot_coverage.update((model,c) for c in cases)
  manifest.append(dict(filename=f'{tag}.png',dataset='pilot',model=model,ep_size=4,phase=phase,scenarios=len(cases),cases=cases))
assert len(pilot_coverage)==12
with (OUT/'extended-chart-data.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
proof=dict(passed=True,extended_scenarios=len(coverage),pilot_scenarios=len(pilot_coverage),new_charts=len(manifest),extended_charts=sum(q['dataset']=='extended' for q in manifest),pilot_charts=sum(q['dataset']=='pilot' for q in manifest),omitted_scenarios=[],manifest=manifest)
(OUT/'coverage-proof.json').write_text(json.dumps(proof,indent=2))
print({k:v for k,v in proof.items() if k!='manifest'})
