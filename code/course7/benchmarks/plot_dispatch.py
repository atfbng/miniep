"""Plot paired Dispatch attribution and the CPU-metadata control."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "figure.facecolor": "#F8FAFC", "axes.facecolor": "white", "text.color": "#0F172A", "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#CBD5E1", "svg.fonttype": "none"})
    output.mkdir(parents=True, exist_ok=True)
    cases = ["small", "large"]
    keys = ["EXPERT_PACK", "RANK_PACK", "COUNT_ALLGATHER", "RECEIVE_LAYOUT", "BOUNDARY"]
    labels = ["Expert pack", "Rank pack", "Count AllGather", "Receive layout", "Scope boundaries"]
    colors = ["#2563EB", "#0D9488", "#F59E0B", "#7C3AED", "#CBD5E1"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    fig.subplots_adjust(left=.08, right=.98, top=.70, bottom=.29, wspace=.25)
    fig.text(.08, .94, "Inside Dispatch: what takes the most time?", weight="bold", size=22)
    fig.text(.08, .865, "H100  |  FP16  |  8 Experts  |  Top-2  |  matched critical-Rank samples", color="#64748B", size=11)
    for ax, case in zip(axes, cases):
        rows = sorted([r for r in data["rows"] if r["case"] == case], key=lambda r: r["ep_size"])
        left = [0.] * len(rows)
        for key, label, color in zip(keys, labels, colors):
            values = [r["boundary_mean_ms"] if key == "BOUNDARY" else r["parts_mean_ms"][key] for r in rows]
            ax.barh(range(len(rows)), values, left=left, color=color, height=.57, label=label)
            for i, (v, start) in enumerate(zip(values, left)):
                if v >= .18:
                    ax.text(start + v / 2, i, f"{v:.2f}", ha="center", va="center", color="white" if key != "BOUNDARY" else "#0F172A", weight="bold", fontsize=9)
            left = [a+b for a,b in zip(left,values)]
        for i,row in enumerate(rows):
            ax.text(left[i]+.025,i,f"{row['parent_mean_ms']:.2f} ms",va="center",fontsize=10,weight="bold")
        ax.set_xlim(0,max(left)*1.22)
        ax.set_yticks(range(len(rows)),[f"EP={r['ep_size']}" for r in rows]);ax.invert_yaxis()
        ax.set_xlabel("Dispatch mean CUDA interval (ms)")
        ax.set_title("Small matrices" if case=="small" else "Larger matrices",loc="left",weight="bold",pad=18)
        ax.set_axisbelow(True);ax.grid(axis="x",color="#E2E8F0")
    handles, names=axes[0].get_legend_handles_labels();fig.legend(handles,names,loc="lower center",bbox_to_anchor=(.53,.075),ncol=5,frameon=False,fontsize=10)
    fig.text(.08,.018,"Each repetition selects one slowest Dispatch Rank; all child intervals come from that same Rank. Labels are means, not independent maxima.",size=9,color="#64748B")
    for ext in ("png","svg"):fig.savefig(output/f"lesson07_dispatch_parts.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(12,5.0));fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.73,wspace=.26)
    fig.text(.075,.94,"A control experiment: move small counts together",size=21,weight="bold")
    fig.text(.075,.865,"Original and CPU-metadata paths: same payloads, splits, indices and numerical outputs",size=11,color="#64748B")
    for ax,case in zip(axes,cases):
        rows=sorted([r for r in data["rows"] if r["case"]==case],key=lambda r:r["ep_size"])
        for offset,field,label,color in [(-.18,"control_median_ms","Original GPU-metadata path","#7C3AED"),(.18,"cpu_metadata_median_ms","CPU-metadata control","#0D9488")]:
            values=[r[field] for r in rows];pos=[i+offset for i in range(len(rows))]
            ax.bar(pos,values,width=.32,color=color,label=label)
            for x,v in zip(pos,values):ax.annotate(f"{v:.3f}",(x,v),xytext=(0,7),textcoords="offset points",ha="center",fontsize=10,weight="bold")
        ax.set_xticks(range(len(rows)),[f"EP={r['ep_size']}" for r in rows]);ax.set_ylim(0,max(r['control_median_ms'] for r in rows)*1.30)
        ax.set_ylabel("Dispatch latency (ms)");ax.set_title("Small matrices" if case=="small" else "Larger matrices",loc="left",weight="bold",pad=18)
        ax.set_axisbelow(True);ax.grid(axis="y",color="#E2E8F0");ax.legend(frameon=False,fontsize=8,loc="upper right")
    fig.text(.075,.055,"Controls use identical stage-level instrumentation; values are trial medians of repetition-wise max-Rank medians.",size=9,color="#64748B")
    for ext in ("png","svg"):fig.savefig(output/f"lesson07_dispatch_control.{ext}",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=Path("results/scaling/dispatch_summary.json"))
    parser.add_argument("--output-dir",type=Path,default=Path("results/scaling/figures"))
    args=parser.parse_args();plot(json.loads(args.input.read_text()),args.output_dir)


if __name__=="__main__":main()
