"""完整折线图：仅连接同模型、EP、输入长度和协议下实际测得的 batch 点。"""
from pathlib import Path
import json, csv, gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
OUT=Path(__file__).resolve().parent
BASE=Path(__file__).resolve().parent.parent
rows=json.loads((BASE/'summary.json').read_text())['rows']
raw=json.loads(gzip.decompress((BASE/'raw-samples.json.gz').read_bytes()))
models=['Qwen3.5-35B-A3B','Qwen3-30B-A3B'];manifest=[];covered=set();pilotcovered=set()
plt.rcParams.update({'font.size':10,'axes.titlesize':11})
colors={'native':'#6366f1','peer_dedup':'#0d9488'}
handles=[Line2D([],[],color=colors['native'],marker='o',label='native DeepEP HT'),Line2D([],[],color=colors['peer_dedup'],marker='o',label='dedup + Peer/TMA'),Line2D([],[],color='#0d9488',marker='o',ls='',label='stable gain'),Line2D([],[],color='#64748b',marker='s',ls='',label='not beyond criterion')]
for model in models:
 for ep in (2,4):
  for phase in ('prefill','decode'):
   selected=[r for r in rows if r['model']==model and r['ep_size']==ep and r['metric']==('prefill_batch_ms' if phase=='prefill' else 'decode_step_ms')]
   groups=sorted({r['workload']['seq_len'] for r in selected if r['workload']['seq_len'] is not None})+[None]
   nrows=4 if phase=='prefill' else 2
   fig,axes=plt.subplots(nrows,3,figsize=(14,13 if phase=='prefill' else 7.5),squeeze=False)
   for g,s in enumerate(groups):
    group=sorted([r for r in selected if r['workload']['seq_len']==s],key=lambda r:r['workload']['batch_size'])
    col=g%3;row=(g//3)*2;ax=axes[row,col];rel=axes[row+1,col]
    batches=[r['workload']['batch_size'] for r in group];x=np.arange(len(group));title=f'S = {s}' if s else 'Mixed lengths (64 .. 4096)'
    for label in ('native','peer_dedup'):
     data=[[q['max_rank_ms'][2 if phase=='prefill' else 4] for q in raw if q['model']==model and q['ep_size']==ep and q['case']==r['case'] and q['label']==label] for r in group]
     assert all(len(d)==15 for d in data)
     y=np.array([np.median(d) for d in data]);lo=np.array([np.quantile(d,.25) for d in data]);hi=np.array([np.quantile(d,.75) for d in data])
     ax.errorbar(x,y,yerr=[y-lo,hi-y],color=colors[label],marker='o',lw=1.8,capsize=3)
     for i,v in enumerate(y):ax.annotate(f'{v:.2f}',(x[i],v),xytext=(0,9 if label=='native' else -14),textcoords='offset points',ha='center',color=colors[label],fontsize=8)
    ax.set_title(title);ax.set_ylabel('Prefill batch (ms)' if phase=='prefill' else 'Decode step (ms)');ax.set_ylim(0,max(r[k]['median_ms'] for r in group for k in ('native','peer_dedup'))*1.23)
    v=np.array([r['latency_reduction']*100 for r in group]);lo=np.array([r['reduction_ci95_low']*100 for r in group]);hi=np.array([r['reduction_ci95_high']*100 for r in group])
    rel.axhspan(-2,2,color='#e2e8f0',alpha=.65);rel.axhline(0,color='#334155',lw=.7);rel.plot(x,v,color='#475569',lw=1.3)
    for i,r in enumerate(group):
     c='#0d9488' if r['classification']=='stable_gain' else '#dc2626' if r['classification']=='stable_regression' else '#64748b';m='o' if r['classification']=='stable_gain' else 's'
     rel.errorbar(x[i],v[i],yerr=[[v[i]-lo[i]],[hi[i]-v[i]]],fmt=m,color=c,capsize=4)
     rel.annotate(f'{v[i]:+.1f}%',(x[i],hi[i]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
     covered.add((model,ep,r['case']))
    rel.set_ylabel('Paired reduction (%)');rel.margins(y=.3)
    for a in (ax,rel):
     a.set_xticks(x,[str(b) for b in batches]);a.set_xlabel('Batch size per DP replica');a.set_xlim(-.35,max(len(group)-1,.0)+.35);a.grid(alpha=.18);a.spines[['top','right']].set_visible(False)
   fig.suptitle(f'{model} | EP{ep} / TP{ep//2} / DP2 | {phase.upper()} | ALL {len(selected)} scenarios',y=.985,fontsize=15)
   fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.957),ncol=4,frameon=False,fontsize=10)
   fig.subplots_adjust(top=.89 if phase=='decode' else .92,bottom=.17 if phase=='decode' else .11,hspace=.62,wspace=.30,left=.07,right=.98)
   fig.text(.5,.02,'Top of each pair: median of 15 max-rank measurements; whiskers = IQR. Bottom: paired reduction and 95% block-bootstrap CI.\nPositive = faster. Gray band = +/-2%. All points retained, including slower / uncertain cases. Lines connect measured B only at fixed S; no interpolation.\nB is categorical with actual values labeled. Mixed lengths are isolated. Panel y scales differ; absolute-latency axes start at zero.',ha='center',fontsize=9)
   tag=('qwen35' if '3.5' in model else 'qwen3')+f'-ep{ep}-{phase}-lines'
   for ext in ('png','svg'):fig.savefig(OUT/f'{tag}.{ext}',dpi=180)
   plt.close(fig);manifest.append(dict(filename=tag+'.png',dataset='extended',model=model,ep_size=ep,phase=phase,scenarios=len(selected),cases=[r['case'] for r in selected]))
old=json.loads((BASE.parent/'h100-peer-dedup-20261026/combined-summary.json').read_text())['cases']
paths=[('nccl','#94a3b8','original NCCL RouteEP'),('peer','#f59e0b','Peer / TMA'),('peer_dedup','#0d9488','dedup + Peer/TMA'),('deepep','#6366f1','native DeepEP HT')]
for model in models:
 for phase in ('prefill','decode'):
  groups=[(['prefill_b4_short'],[4],'S128'),(['prefill_b1_long','prefill_b4_long'],[1,4],'S2048')] if phase=='prefill' else [(['decode_b1','decode_b4','decode_b32'],[1,4,32],'S128')]
  fig,axes=plt.subplots(1,len(groups),figsize=(12,5.6),squeeze=False)
  for ax,(cases,bs,title) in zip(axes[0],groups):
   x=np.arange(len(cases));metric='prefill_ms' if phase=='prefill' else 'decode_step_ms';idx=2 if phase=='prefill' else 4
   for key,color,label in paths:
    y=np.array([old[model][c][key][metric] for c in cases]);data=[[v[idx] for v in old[model][c][key]['raw_max_rank_ms']] for c in cases]
    assert all(len(d)==9 for d in data)
    lo=np.array([np.quantile(d,.25) for d in data]);hi=np.array([np.quantile(d,.75) for d in data]);ax.errorbar(x,y,yerr=[y-lo,hi-y],color=color,marker='o',lw=1.8,capsize=3,label=label)
   ax.set_title(title);ax.set_xticks(x,[str(b) for b in bs]);ax.set_xlim(-.3,len(cases)-.7);ax.set_ylim(bottom=0);ax.margins(y=.12);ax.set_xlabel('Batch size per DP replica');ax.set_ylabel('Prefill batch (ms)' if phase=='prefill' else 'Decode step (ms)');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
   pilotcovered.update((model,c) for c in cases)
  fig.suptitle(f'OLDER EXPLORATORY | {model} | EP4 | {phase.upper()} | ALL 3 scenarios',fontsize=13,y=.98)
  fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=4,frameon=False,fontsize=9)
  fig.subplots_adjust(top=.77,bottom=.23,wspace=.28)
  fig.text(.5,.04,'Median of 9 measurements; whiskers = IQR (not confidence interval). O=1 prefill; O=128 decode.\nKV4 GiB / input budget 8192. Dedup measured later, not interleaved. Do not pool with the formal extended study.\nLines connect measured B only at fixed S; categorical spacing. Singleton scenarios remain as points.',ha='center',fontsize=9)
  tag=('qwen35' if '3.5' in model else 'qwen3')+f'-pilot-{phase}-lines'
  for ext in ('png','svg'):fig.savefig(OUT/f'{tag}.{ext}',dpi=180)
  plt.close(fig);manifest.append(dict(filename=tag+'.png',dataset='pilot',model=model,ep_size=4,phase=phase,scenarios=3))
assert covered=={(r['model'],r['ep_size'],r['case']) for r in rows}
assert len(pilotcovered)==12
proof=dict(passed=True,extended_scenarios=len(covered),pilot_scenarios=len(pilotcovered),charts=len(manifest),omitted=[],manifest=manifest)
(OUT/'coverage-proof.json').write_text(json.dumps(proof,indent=2));print({k:v for k,v in proof.items() if k!='manifest'})
