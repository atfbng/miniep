"""绘制 H100 优化实验报告的图（reports/h100-opt-20261028）。

输入（均为报告目录 data/ 下的 JSON）：
    three-way-<label>.json：{model: {case: {deepep|nccl|peer: [样本...]}}}，
        每个样本是一次测量中 4 个 Rank 里最慢的端到端/prefill/decode 时间；
    layer-*.json：benchmarks/dedup_layer_bench.py 的输出。

输出：
    relative-speed-<label>.png/svg：两条 RouteEP 路径相对 DeepEP HT 的速度（DeepEP=100%）；
    layer-latency.png/svg：层级 microbenchmark，优化前后的 ms/层。

启动方式：
    python3 benchmarks/plot_opt_report.py --report reports/h100-opt-20261028

颜色按“实现”固定分配，不随排名或数量变化：DeepEP HT=slot1 蓝，
RouteEP NCCL=slot2 橙，RouteEP Peer/TMA=slot3 青。3 色已用 dataviz 校验器检查
（CVD ΔE 9.2，正常视觉 ΔE 27.6；青色对浅底对比度 2.74 < 3:1，由完整数据表 table-<label>.md
兜底；图上只标注偏离 DeepEP 超过 5 个百分点的柱）。
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import use_chinese_font

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
COLORS = {"deepep": "#2a78d6", "nccl": "#eb6834", "peer": "#1baf7a"}
NAMES = {"deepep": "DeepEP HT", "nccl": "RouteEP NCCL 去重", "peer": "RouteEP Peer/TMA 去重"}
CASES = ["decode_b1", "decode_b4", "decode_b32", "prefill_b1_long", "prefill_b4_short", "prefill_b4_long"]
MODELS = ["Qwen3-30B-A3B", "Qwen3.5-35B-A3B"]
#: 直接数值标签只给偏离 DeepEP 超过该百分点的柱（选择性标注），其余数值在表格里。
LABEL_THRESHOLD = 5


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def median_e2e(rows):
    return statistics.median(q["e2e_ms"] for q in rows)


def relative_speed(path: Path):
    """返回 {model: {case: {label: DeepEP 中位 ÷ 该路径中位}}} 和样本数。"""
    data = json.loads(path.read_text())
    out, counts = {}, set()
    for model in MODELS:
        for case in CASES:
            groups = data[model][case]
            base = median_e2e(groups["deepep"])
            for label in ("nccl", "peer"):
                out.setdefault(model, {}).setdefault(case, {})[label] = base / median_e2e(groups[label])
                counts.add(len(groups[label]))
            counts.add(len(groups["deepep"]))
    return out, sorted(counts)


def plot_relative(path: Path, out_stem: Path, title: str, ymax: float = 150):
    rel, counts = relative_speed(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    width = 0.36
    for ax, model in zip(axes, MODELS):
        _style(ax)
        xs = range(len(CASES))
        firsts = []
        for k, label in enumerate(("nccl", "peer")):
            values = [rel[model][c][label] * 100 for c in CASES]
            pos = [x + (k - 0.5) * (width + 0.02) for x in xs]
            bars = ax.bar(pos, values, width, color=COLORS[label], label=NAMES[label],
                          edgecolor=SURFACE, linewidth=1)
            firsts.append((bars[0], values[0], label))
            for bar, v in zip(bars, values):
                # 只标注与 DeepEP 相差超过 5 个百分点的柱；全部数值见同目录的数据表。
                if abs(v - 100) <= LABEL_THRESHOLD:
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
        # 第一组柱上方直接标出系列名（文字色），两行错开避免重叠；其余组左右顺序相同。
        base = max(max(v, 100) + (7 if abs(v - 100) > LABEL_THRESHOLD else 2) for _, v, _ in firsts)
        for row, (bar, _, label) in enumerate(firsts):
            ax.text(bar.get_x() + bar.get_width() / 2, base + row * 6, "NCCL" if label == "nccl" else "Peer",
                    ha="center", va="bottom", fontsize=8, color=TEXT, fontweight="bold")
        # DeepEP HT 是基准：画成 100% 参考线，而不是一根恒等于 100 的柱。
        ax.axhline(100, color=COLORS["deepep"], linewidth=2, label=f"{NAMES['deepep']}（=100%）")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(CASES, rotation=25, ha="right")
        ax.set_title(model, color=TEXT, fontsize=11)
        # 所有版本共用同一 y 上限，便于并排比较；留出数值标签和两行系列名的空间。
        ax.set_ylim(0, ymax)
    axes[0].set_ylabel("相对 DeepEP HT 的速度（%，越高越快）", color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=TEXT, fontsize=12, y=1.07)
    fig.text(0.5, -0.04, f"EP4（TP2/DP2），4×H100；相对速度 = DeepEP 端到端中位延迟 ÷ 该路径中位延迟；"
             f"每格样本数 {counts}，每个样本取 4 个 Rank 中最慢者；仅标注偏离 >{LABEL_THRESHOLD} 个百分点的柱，完整数值见数据表",
             ha="center", color=TEXT_SECONDARY, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def write_table(path: Path, out: Path):
    """输出 Markdown 表：每个模型 × 场景的三条路径中位端到端延迟和相对速度。"""
    data = json.loads(path.read_text())
    lines = []
    for model in MODELS:
        lines += [f"**{model}**", "",
                  "| 场景 | DeepEP HT (ms) | NCCL 去重 (ms) | Peer/TMA (ms) | NCCL 相对 | Peer 相对 | 样本 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for case in CASES:
            g = data[model][case]
            m = {k: median_e2e(g[k]) for k in ("deepep", "nccl", "peer")}
            fmt = (lambda v: f"{v:.1f}") if m["deepep"] < 1000 else (lambda v: f"{v:.0f}")
            lines.append(f"| `{case}` | {fmt(m['deepep'])} | {fmt(m['nccl'])} | {fmt(m['peer'])} | "
                         f"{m['deepep'] / m['nccl']:.1%} | {m['deepep'] / m['peer']:.1%} | {len(g['peer'])} |")
        lines.append("")
    out.write_text("\n".join(lines))


def layer_series(data_dir: Path, prefix: str, variant: str):
    """同一配置多轮的 {tokens: 平均 ms/层}。"""
    runs = sorted(data_dir.glob(f"layer-{prefix}-r*{variant}.json"))
    if not runs:
        raise FileNotFoundError(f"no layer runs for {prefix} {variant}")
    by_n = {}
    for run in runs:
        for q in json.loads(run.read_text())["results"]:
            by_n.setdefault(q["tokens_per_rank"], []).append(q["ms_per_layer_median"])
    return {n: statistics.mean(v) for n, v in sorted(by_n.items())}


def plot_layers(data_dir: Path, out_stem: Path):
    panels = [
        ("NCCL 变长路径：第 1 项前后", [("ep2-grouped_unfused", "", "优化前（未融合）"), ("ep2-grouped", "", "第 1 项后（融合）")]),
        ("NCCL 固定容量路径：第 6 项前后", [("l6-commit", "-fixed", "9f9ff6f"), ("l6-new", "-fixed", "+第 6 项")]),
        ("Peer/TMA 路径：移植前后", [("p9-commit", "", "9f9ff6f"), ("p9-new", "", "+Peer 移植")]),
    ]
    # 每个面板只比较“之前/之后”两条线：之前 slot5 品红（虚线），之后 slot7 紫（实线）。
    # 不复用 DeepEP/NCCL/Peer 的颜色；该对颜色已用校验器检查（浅色 CVD ΔE 19.9）。
    styles = [dict(color="#e87ba4", linestyle="--", marker="o"), dict(color="#6250d6", linestyle="-", marker="o")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor=SURFACE)
    for ax, (title, series) in zip(axes, panels):
        _style(ax)
        ax.grid(axis="both", color=GRID, linewidth=0.8)
        for (prefix, variant, name), style in zip(series, styles):
            values = layer_series(data_dir, prefix, variant)
            xs, ys = list(values), list(values.values())
            ax.plot(xs, ys, label=name, linewidth=2, markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, **style)
            # 只在最大 Token 数处直接标注数值，避免每个点都写数字。
            ax.annotate(f"{ys[-1]:.2f} {'后' if style is styles[1] else '前'}", (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(6, 0), va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.set_xscale("log", base=2)
        ax.set_title(title, color=TEXT, fontsize=10)
        ax.set_xlabel("每 Rank Token 数（对数）", color=TEXT_SECONDARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("ms / 层（越低越好）", color=TEXT_SECONDARY)
    fig.text(0.5, -0.06, "EP2（2×H100），合成均匀路由，H=2048、F=768、E=128、K=8，BF16；"
             "每轮连续 8 层取最慢 Rank 平均，20 次重复取中位，再对 2 轮取平均",
             ha="center", color=TEXT_SECONDARY, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(out_stem.with_suffix("." + ext), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    use_chinese_font()
    report = Path(a.report)
    data = report / "data"
    titles = {"9f9ff6f": "基线 9f9ff6f（Peer 未移植、无第 6 项）"}
    paths = sorted(data.glob("three-way-*.json"))
    top = max(v * 100 for path in paths for m in relative_speed(path)[0].values()
              for c in m.values() for v in c.values())
    ymax = max(130, (int(top) // 10 + 3) * 10)
    for path in paths:
        label = path.stem.removeprefix("three-way-")
        plot_relative(path, report / f"relative-speed-{label}",
                      titles.get(label, f"{label}（第 6 项 + Peer 移植）"), ymax)
        write_table(path, report / f"table-{label}.md")
    plot_layers(data, report / "layer-latency")


if __name__ == "__main__":
    main()
