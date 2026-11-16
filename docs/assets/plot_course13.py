"""Recreate README figures from the reported JSON; matplotlib is optional tooling."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

p=argparse.ArgumentParser()
p.add_argument('--summary',type=Path,required=True)
p.add_argument('--stages',type=Path)
p.add_argument('--optimized-key')
a=p.parse_args()
root=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors=['#8094a7','#147d92','#e6a23c']
fig,ax=plt.subplots(figsize=(12,5.3))
ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
ax.text(.1,4.65,'Mini-EP: follow one token round trip',fontsize=16,weight='bold')
ax.text(.2,4.1,'SOURCE RANK',color='#526477',fontsize=10,weight='bold')
ax.text(6.2,4.1,'EXPERT OWNER RANK',color='#526477',fontsize=10,weight='bold')
labels=['01  Router / Top-K','02  Rank packing','03  Forward A2A','04  Expert packing',
        '05  Grouped SwiGLU','06  Restore receive order','07  Return A2A','08  Combine']
positions=[(.2,3.0),(3.2,3.0),(6.2,3.0),(9.2,3.0),(9.2,1.65),(6.2,1.65),(3.2,1.65),(.2,1.65)]
for label,(x,y) in zip(labels,positions):
 ax.add_patch(FancyBboxPatch((x,y),2.55,.75,boxstyle='round,pad=0.04',facecolor='#e9f3f5',edgecolor='#147d92'))
 number,text=label.split('  ')
 ax.text(x+1.275,y+.375,number+'\n'+text,ha='center',va='center',fontsize=10)
for i in range(7):
 x,y=positions[i];xx,yy=positions[i+1]
 if y==yy:
  start=(x+2.6,y+.375) if xx>x else (x-.05,y+.375)
  end=(xx-.05,yy+.375) if xx>x else (xx+2.6,yy+.375)
 else:start=(x+1.275,y-.05);end=(xx+1.275,yy+.8)
 ax.annotate('',end,start,arrowprops={'arrowstyle':'->','color':'#526477'})
ax.text(.2,.95,'Lessons 2 / 9: stable records     •     Lesson 10: mapped loads / combine     •     Lesson 11: tile tables',fontsize=10)
ax.text(.2,.4,'vLLM boundary: TP/DP layout, owner reduction, Attention, KV cache and scheduling',fontsize=10)
fig.tight_layout()
for ext in ('svg','png'):fig.savefig(root/f'course13-dataflow.{ext}',dpi=180)
plt.close(fig)
data=json.loads(a.summary.read_text())
selected_key=a.optimized_key or data.get('selected_variant','d')
models=data.get('model',{})
for dp,rows in models.items():
 if not all(k in rows for k in ('a',selected_key,'n')):continue
 fig,axes=plt.subplots(1,3,figsize=(11,4.4))
 for ax,case,title in zip(axes,['prefill','decode','mixed'],['512 → 1','128 → 32','512 → 16']):
  values=[rows[k][case]['median'] for k in ('a',selected_key,'n')]
  selected=[rows[k][case] for k in ('a',selected_key,'n')]
  errors=[[v-r['minimum'] for v,r in zip(values,selected)], [r['maximum']-v for v,r in zip(values,selected)]]
  bars=ax.bar(['Baseline','Candidate' if data.get('release_status')=='pending_final_acceptance' else 'Optimized','Native'],values,color=colors,width=.65,yerr=errors,capsize=3)
  for i,r in enumerate(selected):
   run_values=r['run_medians']
   ax.scatter([i+(j-(len(run_values)-1)/2)*.10 for j in range(len(run_values))],run_values,s=17,color='#263746',zorder=4)
  ax.bar_label(bars,fmt='%.4f',padding=4,fontsize=10,bbox={'facecolor':'white','edgecolor':'none','pad':1})
  ax.set(title=title,ylabel='Batch completion time (s)',ylim=(0,max(r['maximum'] for r in selected)*1.22))
 fig.suptitle(f"Qwen3-30B-A3B BF16 • TP=2 / DP={int(dp[2:])} / EP={2*int(dp[2:])}\nMaximum-rank batch time • bars: pooled median • whiskers: range • dots: run medians\n{data.get('experiment_date','2026-10-19')} • two requests per DP",fontsize=11)
 fig.tight_layout()
 for ext in ('svg','png'):fig.savefig(root/f'course13-latency-{dp}.{ext}',dpi=180)
 plt.close(fig)
if a.stages:
 stages=json.loads(a.stages.read_text())
 fig,axes=plt.subplots(1,2,figsize=(12,4.5))
 for ax,case in zip(axes,('prefill','decode')):
  names=[n for n in ('ROUTER','DISPATCH','A2A_FORWARD','EXPERT','A2A_RETURN','COMBINE','EP_REDUCE') if n in stages[case]]
  x=list(range(len(names)))
  for j,key in enumerate(('base','optimized')):
   ax.bar([i+(j-.5)*.36 for i in x],[stages[case][n][key] for n in names],width=.36,color=colors[j],label=stages.get('_labels',{}).get(key,key))
  ax.set_xticks(x,names,rotation=35,ha='right',fontsize=8)
  ax.set(title=case,ylabel=stages.get('_metric_label','Median stage CUDA Event time (ms)'))
 fig.suptitle(stages.get('_title','Full-model MoE stages • EP=4 • independent stage timing'),fontsize=12)
 handles,legend_labels=axes[0].get_legend_handles_labels()
 fig.legend(handles,legend_labels,loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False)
 fig.tight_layout(rect=(0,0,1,.86))
 for ext in ('svg','png'):fig.savefig(root/f'course13-stages.{ext}',dpi=180)
