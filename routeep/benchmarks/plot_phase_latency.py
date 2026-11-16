"""从 H100 报告中绘制 Decode 每步延迟和 Prefill 批次延迟。"""

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
MODELS = (
    ("Qwen3-30B-A3B", "qwen3-performance.json"),
    ("Qwen3.5-35B-A3B", "qwen35-performance.json"),
)
PHASES = (
    (
        "decode",
        "解码每步延迟",
        "decode_step_ms",
        (("decode_b1", "批次 1"), ("decode_b4", "批次 4"),
         ("decode_b32", "批次 32")),
        "#177EAA",
    ),
    (
        "prefill",
        "预填充批次延迟",
        "prefill_batch_ms",
        (("prefill_b1_long", "批次 1 · 长输入"),
         ("prefill_b4_short", "批次 4 · 短输入"),
         ("prefill_b4_long", "批次 4 · 长输入")),
        "#8054B8",
    ),
)


def load_reports(root: Path) -> list[tuple[str, dict]]:
    reports = []
    for title, filename in MODELS:
        path = root / filename
        report = json.loads(path.read_text())
        if report.get("passed") is not True or report.get("errors"):
            raise ValueError(f"benchmark report did not pass: {path}")
        reports.append((title, report["cases"]))
    return reports


def median(cases: dict, case: str, backend: str, metric: str) -> float:
    value = cases[case][backend][metric]["median"]
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {metric} for {case}/{backend}: {value}")
    return value


def render_phase(output: Path, title: str, metric: str,
                 workloads: tuple, miniep_color: str,
                 reports: list[tuple[str, dict]]) -> None:
    all_values = [
        median(cases, case, backend, metric)
        for _, cases in reports
        for case, _ in workloads
        for backend in ("native", "miniep")
    ]
    axis_max = max(all_values) * 1.29
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.4), facecolor="#F3F6FA")
    fig.subplots_adjust(left=0.15, right=0.975, top=0.66, bottom=0.19,
                        wspace=0.29)
    fig.text(0.055, 0.947, "H100 性能对比", fontsize=11,
             color=miniep_color, weight="bold")
    fig.text(0.055, 0.881, title, fontsize=26, color="#172635",
             weight="bold")
    fig.text(0.055, 0.819,
             "4 × NVIDIA H100 80 GB · BF16 · TP=2 / DP=2 / EP=4 · 越低越好",
             fontsize=11, color="#607181")
    fig.text(0.055, 0.757, "■  DeepEP HT", fontsize=10.5,
             color="#8393A3")
    fig.text(0.174, 0.757, "■  RouteEP", fontsize=10.5,
             color=miniep_color)

    for ax, (model, cases) in zip(axes, reports):
        ax.set_facecolor("white")
        ax.set_xlim(0, axis_max)
        ax.set_ylim(-0.5, 3.35)
        ax.set_yticks((2.5, 1.35, 0.2), [label for _, label in workloads])
        ax.tick_params(axis="y", length=0, pad=12, labelsize=10,
                       colors="#253547")
        ax.tick_params(axis="x", length=0, pad=10, labelsize=9,
                       colors="#637385")
        ax.grid(axis="x", color="#DFE6ED", linewidth=0.9, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(model, loc="left", color="#182736", fontsize=15,
                     fontweight="bold", pad=17)
        ax.set_xlabel("毫秒", fontsize=9, color="#637385",
                      labelpad=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for (case, _), y in zip(workloads, (2.5, 1.35, 0.2)):
            native = median(cases, case, "native", metric)
            miniep = median(cases, case, "miniep", metric)
            for value, offset, color in (
                (native, 0.17, "#91A1B0"),
                (miniep, -0.17, miniep_color),
            ):
                ax.barh(y + offset, value, height=0.26, color=color,
                        edgecolor="none", zorder=3)
                ax.text(value + axis_max * 0.012, y + offset,
                        f"{value:,.1f} ms", va="center", fontsize=9.5,
                        color="#223345", fontweight="bold")

    fig.text(0.055, 0.077,
             "每场景 15 次采样，单次取四个 Rank 中的最大值；图中为中位数。",
             fontsize=9, color="#607181")
    fig.text(0.055, 0.046,
             "数据源：reports/qwen3-performance.json 与 reports/qwen35-performance.json",
             fontsize=8.5, color="#8996A3")
    output.parent.mkdir(parents=True, exist_ok=True)
    svg_path = output.with_suffix(".svg")
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(output.with_suffix(".png"), dpi=180,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path,
                        default=PROJECT_ROOT / "reports")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "assets")
    args = parser.parse_args()
    reports = load_reports(args.reports)
    use_chinese_font()
    for stem, title, metric, workloads, color in PHASES:
        render_phase(args.output / f"h100-{stem}-latency", title, metric,
                     workloads, color, reports)


if __name__ == "__main__":
    main()
