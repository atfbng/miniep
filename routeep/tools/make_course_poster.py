"""生成 RouteEP 课程海报，同时输出 PNG 和 SVG。

所有速度数字都从实测 performance JSON 读取；报告未通过时拒绝出图。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.plot_style import use_chinese_font


ROOT = Path(__file__).resolve().parents[1]
INK = "#102633"
PAPER = "#F5F7F5"
NAVY = "#0B2231"
CYAN = "#78D9D4"
SOFT = "#B8CBCF"


def measured_speed(model: str, phase: str) -> float:
    report = json.loads((ROOT / "reports" / f"{model}-performance.json").read_text())
    if report.get("passed") is not True:
        raise ValueError(f"performance report did not pass: {model}")
    values = [
        100 * row["native"]["e2e_ms"]["median"]
        / row["miniep"]["e2e_ms"]["median"]
        for name, row in report["cases"].items()
        if name.startswith(phase + "_")
    ]
    if len(values) != 3:
        raise ValueError(f"expected three {phase} workloads for {model}")
    return statistics.median(values)


def draw(output: Path) -> None:
    use_chinese_font()
    fig = plt.figure(figsize=(9, 13.5), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PAPER)

    ax.text(0.07, 0.958, "RouteEP", fontsize=27, fontweight="bold",
            color=INK, va="center", transform=ax.transAxes)
    ax.text(0.93, 0.958, "EXPERT PARALLEL  /  COURSE",
            fontsize=9.7, color="#58707A", ha="right", va="center",
            transform=ax.transAxes)
    ax.plot((0.07, 0.93), (0.927, 0.927), color="#B4C4C7",
            linewidth=1.0, transform=ax.transAxes)

    ax.add_patch(FancyBboxPatch(
        (0.055, 0.335), 0.89, 0.56,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=NAVY, edgecolor=NAVY, transform=ax.transAxes,
    ))
    ax.text(0.10, 0.844, "VLLM  ×  EXPERT PARALLEL",
            fontsize=10.8, color=CYAN, fontweight="bold",
            va="center", transform=ax.transAxes)
    ax.text(0.10, 0.785, "大模型 MoE", fontsize=45,
            color="#FFFFFF", fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(0.10, 0.724, "专家并行实战", fontsize=47,
            color="#FFFFFF", fontweight="bold", va="top",
            transform=ax.transAxes)
    ax.text(0.10, 0.650, "接入 vLLM 的 MoE 计算与通信流程",
            fontsize=15.7, color="#C9DADD", va="center",
            transform=ax.transAxes)

    ax.plot((0.10, 0.90), (0.607, 0.607), color="#42616E",
            linewidth=0.8, transform=ax.transAxes)
    ax.add_patch(FancyBboxPatch(
        (0.10, 0.424), 0.80, 0.158,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor="#163647", edgecolor="#537783", linewidth=1.0,
        transform=ax.transAxes,
    ))
    ax.text(0.13, 0.559, "vLLM 推理链路", fontsize=11.2,
            color=CYAN, va="center",
            transform=ax.transAxes)
    for x, width, edge, name, detail, large in (
        (0.13, 0.20, "#7E9AA7", "调度 / Attention", "请求与 KV Cache", 10.8),
        (0.365, 0.30, CYAN, "RouteEP", "MoE 计算与通信", 16.0),
        (0.69, 0.18, "#7E9AA7", "模型输出", "继续推理", 11.6),
    ):
        ax.add_patch(FancyBboxPatch(
            (x, 0.473), width, 0.060,
            boxstyle="round,pad=0,rounding_size=0.007",
            facecolor="#122F40", edgecolor=edge, linewidth=1.25,
            transform=ax.transAxes,
        ))
        ax.text(x + 0.018, 0.509, name, fontsize=large,
                color="#FFFFFF", fontweight="bold", va="center",
                transform=ax.transAxes)
        ax.text(x + 0.018, 0.486, detail, fontsize=9.0,
                color=SOFT, va="center", transform=ax.transAxes)
    for x0, x1 in ((0.33, 0.365), (0.665, 0.69)):
        ax.annotate("", xy=(x1, 0.503), xytext=(x0, 0.503),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", color=CYAN,
                                    linewidth=1.5, mutation_scale=10))
    ax.text(0.13, 0.447, "性能对照：DeepEP HT · DeepSeek 高性能 MoE 通信库",
            fontsize=10.1, color="#D6E8E9", va="center",
            transform=ax.transAxes)
    ax.plot((0.10, 0.90), (0.380, 0.380), color="#42616E",
            linewidth=0.8, transform=ax.transAxes)
    ax.text(0.10, 0.357, "同模型同请求  /  BF16  /  TP=2 · DP=2 · EP=4  /  4×H100",
            fontsize=10.4, color=CYAN, va="center", transform=ax.transAxes)

    ax.text(0.07, 0.297, "Qwen3.5 · H100 实测性能", fontsize=16.5, color=INK,
            fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(0.93, 0.297, "DeepEP HT = 100%", fontsize=10.2,
            color="#627884", ha="right", va="center",
            transform=ax.transAxes)
    rows = (
        ("解码", measured_speed("qwen35", "decode"), "#168F8D"),
        ("预填充", measured_speed("qwen35", "prefill"), "#496E82"),
    )
    for index, (label, value, color) in enumerate(rows):
        y = 0.249 - index * 0.051
        ax.text(0.07, y, label, fontsize=13.0, color=INK,
                va="center", transform=ax.transAxes)
        ax.add_patch(FancyBboxPatch(
            (0.37, y - 0.0052), 0.45, 0.0104,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor="#D9E3E3", edgecolor="none", transform=ax.transAxes,
        ))
        ax.add_patch(FancyBboxPatch(
            (0.37, y - 0.0052), 0.45 * value / 100, 0.0104,
            boxstyle="round,pad=0,rounding_size=0.003",
            facecolor=color, edgecolor="none", transform=ax.transAxes,
        ))
        ax.text(0.93, y, f"{value:.1f}%", fontsize=11.8,
                color=INK, fontweight="bold", ha="right", va="center",
                transform=ax.transAxes)
    ax.text(0.07, 0.141, "相对 DeepEP HT 的端到端速度；每组为三个场景的中位数",
            fontsize=9.2, color="#627884", va="center",
            transform=ax.transAxes)

    ax.plot((0.07, 0.93), (0.116, 0.116), color="#B4C4C7",
            linewidth=1.0, transform=ax.transAxes)
    ax.text(0.07, 0.083, "vLLM 接入  ·  NCCL All-to-All  ·  Triton / Grouped GEMM",
            fontsize=11.2, color=INK, va="center",
            transform=ax.transAxes)
    ax.text(0.93, 0.055, "ROUTEEP  /  GPU SYSTEMS",
            fontsize=9.2, color="#82949A", ha="right", va="center",
            transform=ax.transAxes)

    output.parent.mkdir(parents=True, exist_ok=True)
    svg = output.with_suffix(".svg")
    fig.savefig(svg, facecolor=fig.get_facecolor())
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(output.with_suffix(".png"), dpi=180,
                facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "assets/routeep-course-poster")
    args = parser.parse_args()
    draw(args.output)


if __name__ == "__main__":
    main()
