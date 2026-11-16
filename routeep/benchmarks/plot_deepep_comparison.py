"""绘制 RouteEP 相对原生 DeepEP HT 的端到端速度百分比。

同一 workload 下：相对速度 = DeepEP HT 中位延迟 / RouteEP 中位延迟 * 100。
每个中位数先取每次采样的最慢 Rank。源 JSON 保留完整统计，图中只展示汇总值。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.plot_style import use_chinese_font


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("decode_b1", "解码 · 批次 1"),
    ("decode_b4", "解码 · 批次 4"),
    ("decode_b32", "解码 · 批次 32"),
    ("prefill_b1_long", "预填充 · 批次 1 / 长输入"),
    ("prefill_b4_short", "预填充 · 批次 4 / 短输入"),
    ("prefill_b4_long", "预填充 · 批次 4 / 长输入"),
)
POSITIONS = (6.0, 4.9, 3.8, 2.3, 1.2, 0.1)
COLORS = {"decode": "#177EAA", "prefill": "#8054B8"}


def read_cases(path: Path) -> dict:
    report = json.loads(path.read_text())
    if report.get("passed") is not True or report.get("errors"):
        raise ValueError(f"acceptance report did not pass: {path}")
    cases = report["cases"]
    if set(cases) != {name for name, _ in CASES}:
        raise ValueError(f"unexpected workload cases in {path}")
    for name, _ in CASES:
        row = cases[name]
        native = row["native"]["e2e_ms"]["median"]
        miniep = row["miniep"]["e2e_ms"]["median"]
        ratio = row["miniep_over_deepep"]["e2e_latency"]
        if not all(math.isfinite(x) and x > 0 for x in (native, miniep, ratio)):
            raise ValueError(f"invalid latency in {path}: {name}")
        if not math.isclose(ratio, miniep / native, rel_tol=1e-8):
            raise ValueError(f"ratio does not match measured latencies: {name}")
    return cases


def format_ms(value: float) -> str:
    return f"{value:,.0f}" if value >= 1_000 else f"{value:,.1f}"


def draw_panel(ax, title: str, cases: dict) -> None:
    ax.set_facecolor("white")
    ax.axhspan(-0.43, 2.82, color="#FAF8FD", zorder=0)
    ax.axvline(100, color="#708090", linewidth=1.7, zorder=2)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.56, 6.58)
    ax.set_yticks(POSITIONS, [label for _, label in CASES])
    ax.tick_params(axis="y", length=0, pad=12, labelsize=10, colors="#253547")
    ax.set_xticks((0, 25, 50, 75, 100),
                  ("0%", "25%", "50%", "75%", "100%"))
    ax.tick_params(axis="x", length=0, pad=10, labelsize=9, colors="#637385")
    ax.grid(axis="x", color="#DFE6ED", linewidth=0.9, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", color="#182736", fontsize=15,
                 fontweight="bold", pad=18)

    for (name, _), y in zip(CASES, POSITIONS):
        row = cases[name]
        native = row["native"]["e2e_ms"]["median"]
        miniep = row["miniep"]["e2e_ms"]["median"]
        relative_speed = native / miniep * 100
        color = COLORS[name.split("_", 1)[0]]
        ax.barh(y, 100, height=0.26, color="#E5EBF0",
                edgecolor="none", zorder=3)
        ax.barh(y, relative_speed, height=0.26, color=color,
                edgecolor="none", zorder=4)
        ax.scatter((relative_speed,), (y,), color=color, s=44, zorder=5)
        ax.text(relative_speed + 1.8, y + 0.015, f"{relative_speed:.1f}%", va="center",
                color="#182736", fontsize=10, fontweight="bold")
        ax.text(2, y - 0.31,
                f"DeepEP HT {format_ms(native)} ms   ·   RouteEP {format_ms(miniep)} ms",
                va="center", color="#697A89", fontsize=8.4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3", type=Path,
                        default=PROJECT_ROOT / "reports/qwen3-performance.json")
    parser.add_argument("--qwen35", type=Path,
                        default=PROJECT_ROOT / "reports/qwen35-performance.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "assets/h100-deepep-comparison")
    args = parser.parse_args()

    data = (read_cases(args.qwen3), read_cases(args.qwen35))
    use_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 8.2), facecolor="#F3F6FA")
    fig.subplots_adjust(left=0.16, right=0.975, top=0.75, bottom=0.15,
                        wspace=0.30)
    fig.text(0.055, 0.955, "H100 性能对比", fontsize=11,
             color="#177EAA", weight="bold")
    fig.text(0.055, 0.904, "RouteEP 相对 DeepEP HT 的端到端速度", fontsize=26,
             color="#172635", weight="bold")
    fig.text(0.055, 0.855,
             "4 × NVIDIA H100 80 GB · BF16 · TP=2 / DP=2 / EP=4",
             fontsize=11, color="#607181")
    fig.text(0.055, 0.808, "■  解码", fontsize=10.5, color=COLORS["decode"])
    fig.text(0.155, 0.808, "■  预填充", fontsize=10.5,
             color=COLORS["prefill"])
    fig.text(0.27, 0.808, "DeepEP HT = 100% · 越高越好",
             fontsize=10.5, color="#607181")

    draw_panel(axes[0], "Qwen3-30B-A3B  ·  48 层 MoE", data[0])
    draw_panel(axes[1], "Qwen3.5-35B-A3B  ·  40 层 MoE", data[1])
    fig.text(0.055, 0.073,
             "相对速度 = DeepEP HT 延迟中位数 ÷ RouteEP 延迟中位数 × 100%。"
             "每场景 15 次采样，单次取四个 Rank 中的最大值。",
             fontsize=9, color="#607181")
    fig.text(0.055, 0.045,
             "数据源：reports/qwen3-performance.json 与 reports/qwen35-performance.json",
             fontsize=8.5, color="#8996A3")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = args.output.with_suffix(".svg")
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(args.output.with_suffix(".png"), dpi=180,
                facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
