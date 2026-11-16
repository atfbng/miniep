"""Render latency, speedup, and stage charts from the saved scaling summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

STAGES = ["ROUTER", "DISPATCH", "A2A_FORWARD", "EXPERT", "A2A_RETURN", "COMBINE"]
STAGE_LABELS = ["Router", "Dispatch", "Forward A2A", "Expert MLP", "Return A2A", "Combine"]
COLORS = ["#2563EB", "#0D9488", "#F59E0B"]


def plot(summary: dict, output_dir: Path) -> list[Path]:
    """Create exportable PNG/SVG figures; read only measured summary values.

    Latency whiskers span independent trial medians. Stage cells are the
    median of trial-wise max-rank stage medians, not additive portions of a
    total. Figures use English labels so no external font download is needed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#CBD5E1", "axes.labelcolor": "#334155", "text.color": "#0F172A", "xtick.color": "#475569", "ytick.color": "#475569", "figure.facecolor": "#F8FAFC", "axes.facecolor": "#FFFFFF", "svg.fonttype": "none"})
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = sorted({r["case"] for r in summary["rows"]}, key=lambda c: (c != "small", c))
    if len(cases) != 2:
        raise ValueError("these comparison figures need the small and large cases")
    grouped = {case: sorted([r for r in summary["rows"] if r["case"] == case], key=lambda r: r["ep_size"]) for case in cases}
    paths = []
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.1), gridspec_kw={"width_ratios": [1, 1, 1.12]})
    fig.subplots_adjust(left=.055, right=.975, bottom=.21, top=.72, wspace=.35)
    fig.text(.055, .93, "More GPUs, the same global MoE workload", size=22, weight="bold")
    fig.text(.055, .86, "H100 SXM 80 GB  |  8 Experts  |  Top-2  |  FP16  |  EP = 1 / 2 / 4", size=11, color="#64748B")
    for ax, case in zip(axes[:2], cases):
        rows = grouped[case]
        values = [r["latency_ms"] for r in rows]
        low = [r["latency_ms"] - r["trial_min_ms"] for r in rows]
        high = [r["trial_max_ms"] - r["latency_ms"] for r in rows]
        positions = list(range(len(rows)))
        ax.bar(positions, values, width=.56, color=COLORS[:len(rows)], zorder=3)
        ax.errorbar(positions, values, yerr=[low, high], fmt="none", ecolor="#0F172A", capsize=4, zorder=4)
        ideal = [values[0] / r["ep_size"] for r in rows]
        ax.plot(positions, ideal, "--", color="#94A3B8", marker="o", markersize=4, label="Ideal 1/P latency", zorder=2)
        for pos, row in zip(positions, rows):
            ax.annotate(f"{row['latency_ms']:.3f}", (pos, row["trial_max_ms"]), xytext=(0, 9), textcoords="offset points", ha="center", weight="bold", size=10)
        conf = rows[0]["config"]
        title = "Small matrices" if case == "small" else "Larger matrices"
        ax.set_title(f"{title}\nT={conf['tokens']:,}, H={conf['hidden_size']}, F={conf['expert_hidden_size']}", loc="left", weight="bold", pad=18)
        ax.set_xticks(positions, [f"EP={r['ep_size']}" for r in rows])
        ax.set_ylabel("Layer latency (ms)")
        ax.set_ylim(0, max(r["trial_max_ms"] for r in rows) * 1.30)
        ax.grid(axis="y", color="#E2E8F0", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax = axes[2]
    ax.plot([1, 2, 4], [1, 2, 4], "--", color="#94A3B8", label="Linear scaling")
    for case, color, label in zip(cases, ["#2563EB", "#0D9488"], ["Small matrices", "Larger matrices"]):
        rows = grouped[case]
        ax.plot([r["ep_size"] for r in rows], [r["speedup"] for r in rows], marker="o", markersize=7, linewidth=2.5, color=color, label=label)
        for row in rows[1:]:
            ax.annotate(f"{row['speedup']:.2f}x", (row["ep_size"], row["speedup"]), xytext=(7, 8 if case == "large" else -16), textcoords="offset points", color=color, weight="bold")
    ax.set_title("Scaling benefit\nSpeedup = EP=1 latency / EP=P latency", loc="left", weight="bold", pad=18)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("EP size")
    ax.set_ylabel("Speedup vs. EP=1")
    ax.set_ylim(0, max(4.25, max(r["speedup"] for r in summary["rows"]) * 1.15))
    ax.grid(color="#E2E8F0")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    trials = summary["rows"][0]["trials"]
    fig.text(.055, .065, f"Bars: median of {trials} independent trials. Whiskers: min-max trial medians. Each trial: repetition-wise max Rank, then median.", color="#64748B", size=9)
    for ext in ("png", "svg"):
        target = output_dir / f"lesson07_scaling.{ext}"
        fig.savefig(target, dpi=220, bbox_inches="tight")
        paths.append(target)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8))
    fig.subplots_adjust(left=.10, right=.88, bottom=.14, top=.78, hspace=.72)
    fig.text(.10, .94, "Where does each stage spend time?", size=22, weight="bold")
    fig.text(.10, .88, "Per-stage max-Rank medians (ms)  |  Measured separately from whole-layer latency", size=11, color="#64748B")
    all_values = [r["stages_ms"][s] for r in summary["rows"] for s in STAGES]
    positive = [v for v in all_values if v > 0]
    norm = LogNorm(vmin=min(positive), vmax=max(positive))
    for ax, case in zip(axes, cases):
        rows = grouped[case]
        values = [[r["stages_ms"][s] for s in STAGES] for r in rows]
        image = ax.imshow(values, cmap="Blues", norm=norm, aspect="auto")
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white" if norm(max(value, norm.vmin)) > .58 else "#0F172A", size=11, weight="bold")
        ax.set_xticks(range(len(STAGES)), STAGE_LABELS)
        ax.set_yticks(range(len(rows)), [f"EP={r['ep_size']}" for r in rows])
        ax.set_title("Small matrices" if case == "small" else "Larger matrices", loc="left", weight="bold", pad=12)
        ax.tick_params(length=0, pad=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes([.91, .19, .015, .54])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label("Stage latency (ms), log color scale", labelpad=12)
    fig.text(.10, .055, "Different stages may be slowest on different Ranks; these cells should not be added to obtain layer latency.", color="#64748B", size=9)
    for ext in ("png", "svg"):
        target = output_dir / f"lesson07_stages.{ext}"
        fig.savefig(target, dpi=220, bbox_inches="tight")
        paths.append(target)
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/scaling/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/scaling/figures"))
    args = parser.parse_args()
    for path in plot(json.loads(args.input.read_text()), args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
