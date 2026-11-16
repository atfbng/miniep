"""折线图：横轴为输入规模，纵轴为耗时，DeepEP 与 RouteEP 对比；prefill、decode eager、decode graph 分开。

输入：一个或多个汇总 JSON（由 benchmarks/deepep_latency_run.py 各 rank 的 samples.json 汇总而来，见报告 data/），格式
    {model: {case: {label: [{trial, rep, value}, ...]}}}
    value：decode 场景为稳态每步 ms（所有请求出首 Token 后的区间 ÷ (步数-1)），prefill 场景为 prefill 阶段 ms；
    每个样本已取 4 个 Rank 中最慢者。多个文件按顺序合并，后者覆盖同名 (model, case, label)。
输出：<out>/<model>-<panel>.png 与 table-scaling.md。

启动方式：
    python3 benchmarks/plot_scaling_lines.py --inputs data/full-suite-ep4.json data/llcap-ep4.json --out <dir>

对比组（部署时各阶段实际使用的执行方式）：
    prefill        ：DeepEP HT / RouteEP Peer / RouteEP NCCL，均 eager（prefill 部署不开 graph）；
    decode（graph）：DeepEP LL + Graph / RouteEP NCCL + Graph（RouteEP 的 Peer 路径还不能 capture）；
    decode（eager）：DeepEP HT / RouteEP Peer / RouteEP NCCL，作为 graph 收益的参照。
图表约定：颜色按实现固定（DeepEP HT 蓝、RouteEP NCCL 橙、RouteEP Peer 青、DeepEP LL 紫），
    浅色色阶已用 dataviz 校验器检查；系列名直接标在线尾，标记形状不同，细须为 9 个样本的 min–max。
"""
import argparse
import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import use_chinese_font

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {
    "ht": ("DeepEP HT", "#2a78d6", "o"),
    "peer": ("RouteEP Peer", "#1baf7a", "s"),
    "nccl": ("RouteEP NCCL", "#eb6834", "^"),
    "llg64": ("DeepEP LL + Graph", "#6250d6", "D"),
    "ncclg": ("RouteEP NCCL + Graph", "#eb6834", "^"),
}
PANELS = [
    ("prefill", "Prefill 阶段耗时（eager）", ["ht", "peer", "nccl"], "prefill"),
    ("decode-graph", "Decode 每步耗时（开 CUDA Graph）", ["llg64", "ncclg"], "decode"),
    ("decode-eager", "Decode 每步耗时（eager）", ["ht", "peer", "nccl"], "decode"),
]
# 固定六场景到“每个 DP 副本的输入规模”；扫描场景从名字解析（prefill_b4_s1024 → 4×1024，decode_b8_s128 → batch 8）。
FIXED = {
    "prefill_b4_short": ("prefill", 4 * 128, "4×128"),
    "prefill_b1_long": ("prefill", 1 * 2048, "1×2048"),
    "prefill_b4_long": ("prefill", 4 * 2048, "4×2048"),
    "decode_b1": ("decode", 1, "1"),
    "decode_b4": ("decode", 4, "4"),
    "decode_b32": ("decode", 32, "32"),
}


def case_axis(case: str):
    if case in FIXED:
        return FIXED[case]
    m = re.fullmatch(r"prefill_b(\d+)_s(\d+)", case)
    if m:
        b, s = map(int, m.groups())
        return "prefill", b * s, f"{b}×{s}"
    m = re.fullmatch(r"decode_b(\d+)_s\d+", case)
    if m:
        b = int(m[1])
        return "decode", b, str(b)
    return None


def load(paths):
    merged = {}
    for p in paths:
        for model, cases in json.loads(Path(p).read_text()).items():
            for case, labels in cases.items():
                for label, rows in labels.items():
                    merged.setdefault(model, {}).setdefault(case, {})[label] = rows
    return merged


def points(cases: dict, phase: str, label: str):
    out = []
    for case, labels in cases.items():
        axis = case_axis(case)
        if not axis or axis[0] != phase or label not in labels:
            continue
        v = [r["value"] for r in labels[label]]
        out.append((axis[1], axis[2], statistics.median(v), min(v), max(v), len(v)))
    return sorted(out)


def plot(model: str, cases: dict, panel, out: Path) -> list[str]:
    key, title, labels, phase = panel
    series = {l: points(cases, phase, l) for l in labels}
    series = {l: p for l, p in series.items() if p}
    if not series:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ticks = {}
    ends = []
    for label, pts in series.items():
        name, color, marker = SERIES[label]
        xs = [p[0] for p in pts]
        med = [p[2] for p in pts]
        dashed = label in ("llg64",)
        ax.plot(xs, med, color=color, marker=marker, markersize=5, linewidth=2,
                linestyle=(0, (4, 2)) if dashed else "-", label=name)
        ax.errorbar(xs, med, yerr=[[m - p[3] for m, p in zip(med, pts)], [p[4] - m for m, p in zip(med, pts)]],
                    fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.8)
        ticks.update({p[0]: p[1] for p in pts})
        ends.append([xs[-1], med[-1], name, color])
    # 线尾直接标系列名；纵向过近时依次错开，避免重叠。
    ends.sort(key=lambda e: e[1])
    ylo, yhi = ax.get_ylim()
    gap = (yhi - ylo) * 0.08
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < gap:
            ends[i][1] = ends[i - 1][1] + gap
    for x, y, name, color in ends:
        ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9, color=TEXT)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(ticks))
    ax.set_xticklabels([ticks[x] for x in sorted(ticks)], fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.set_xlim(min(ticks) / 1.3, max(ticks) * 2.6)
    ax.set_xlabel("每个 DP 副本的输入：请求数×长度" if phase == "prefill" else "每个 DP 副本的 decode batch（输入 128）",
                  color=TEXT2, fontsize=9)
    ax.set_ylabel("ms（越低越好）", color=TEXT2, fontsize=9)
    ax.set_title(f"{model} · {title}", color=TEXT, fontsize=11, loc="left")
    n = max(p[5] for pts in series.values() for p in pts)
    fig.text(0.01, -0.02, f"EP4（TP2/DP2），4×H100，各组同机交替运行；点 = {n} 个样本中位数，细须 = min–max，"
             "每个样本取 4 个 Rank 中最慢者", color=TEXT2, fontsize=7.5)
    fig.tight_layout()
    path = out / f"{model}-{key}.png"
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    rows = [f"**{model} · {title}（ms，中位数 [min, max]）**", "",
            "| 输入规模 | " + " | ".join(SERIES[l][0] for l in series) + " |",
            "|---|" + "---:|" * len(series)]
    for x in sorted(ticks):
        cells = []
        for l, pts in series.items():
            hit = [p for p in pts if p[0] == x]
            cells.append(f"{hit[0][2]:.1f} [{hit[0][3]:.1f}, {hit[0][4]:.1f}]" if hit else "—")
        rows.append(f"| {ticks[x]} | " + " | ".join(cells) + " |")
    return rows + [""]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    use_chinese_font()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load(a.inputs)
    table = []
    for model in sorted(data):
        for panel in PANELS:
            table += plot(model, data[model], panel, out)
    (out / "table-scaling.md").write_text("\n".join(table))
    print("\n".join(sorted(str(x.name) for x in out.glob("*.png"))))


if __name__ == "__main__":
    main()
