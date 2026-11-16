"""按阶段对比三条路径的延迟：prefill 阶段耗时与 decode 每步耗时（PNG/SVG + 交互 HTML + 数据表）。

输入：报告目录 data/three-way-<label>.json（benchmarks/peer_suite.py 的样本，见 plot_opt_report.py）。
    prefill 场景取 prefill_ms（首次调度到最后一个首 Token），decode 场景取 decode_step_ms
    （首个首 Token 到最后一个 Token 的区间除以步数）。每个样本是 4 个 Rank 中最慢者。
输出：phase-latency-<label>.png/.svg/.html 与 table-phase-<label>.md。

启动方式：
    python3 benchmarks/plot_phase_report.py --report reports/h100-opt-20261028 --label 11876eb

图表约定：
    - 2×2 小图：行 = 模型，列 = 阶段。两个阶段量级不同，各自一条 y 轴（从 0 开始），不做双轴。
    - 每个场景三根柱：DeepEP HT、RouteEP NCCL、RouteEP Peer；柱高为中位数，细须为 min–max。
    - 颜色按实现固定（slot1 蓝 / slot2 橙 / slot3 青），浅/深两套色阶已用 dataviz 校验器检查；
      浅色下青色对比度 2.74 < 3:1，由 Peer 柱的直接数值标签和数据表兜底。
    - 只给 Peer 柱标数值（本报告关注的实现），其余数值在悬停提示和数据表里。
"""
import argparse
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import use_chinese_font

MODELS = ["Qwen3-30B-A3B", "Qwen3.5-35B-A3B"]
PHASES = [
    ("prefill", "Prefill 阶段耗时", "prefill_ms", ["prefill_b1_long", "prefill_b4_short", "prefill_b4_long"]),
    ("decode", "Decode 每步耗时", "decode_step_ms", ["decode_b1", "decode_b4", "decode_b32"]),
]
SERIES = ["deepep", "nccl", "peer"]
NAMES = {"deepep": "DeepEP HT", "nccl": "RouteEP NCCL", "peer": "RouteEP Peer"}
LIGHT = {"deepep": "#2a78d6", "nccl": "#eb6834", "peer": "#1baf7a"}
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def summarize(path: Path) -> dict:
    """{model: {phase: {case: {series: {median,min,max,n}}}}}"""
    raw = json.loads(path.read_text())
    out = {}
    for model in MODELS:
        for phase, _, key, cases in PHASES:
            for case in cases:
                for s in SERIES:
                    v = [q[key] for q in raw[model][case][s]]
                    out.setdefault(model, {}).setdefault(phase, {}).setdefault(case, {})[s] = dict(
                        median=statistics.median(v), min=min(v), max=max(v), n=len(v))
    return out


def plot_png(data: dict, stem: Path, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), facecolor=SURFACE)
    width = 0.26
    for r, model in enumerate(MODELS):
        for c, (phase, ptitle, _, cases) in enumerate(PHASES):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=TEXT2, labelsize=9)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            top = max(d["max"] for cs in data[model][phase].values() for d in cs.values())
            for k, s in enumerate(SERIES):
                xs = [i + (k - 1) * (width + 0.02) for i in range(len(cases))]
                med = [data[model][phase][cs][s]["median"] for cs in cases]
                lo = [m - data[model][phase][cs][s]["min"] for m, cs in zip(med, cases)]
                hi = [data[model][phase][cs][s]["max"] - m for m, cs in zip(med, cases)]
                ax.bar(xs, med, width, color=LIGHT[s], label=NAMES[s], edgecolor=SURFACE, linewidth=1)
                ax.errorbar(xs, med, yerr=[lo, hi], fmt="none", ecolor=TEXT2, elinewidth=1, capsize=2)
                # 系列名直接标在第一组柱下方（文字色），其余组左右顺序相同。
                ax.text(xs[0], top * 0.03, {"deepep": "HT", "nccl": "NCCL", "peer": "Peer"}[s],
                        ha="center", va="bottom", fontsize=7, color=TEXT, rotation=90)
                if s == "peer":
                    for x, m, h in zip(xs, med, hi):
                        ax.text(x, m + h + top * 0.02, f"{m:.1f}", ha="center", va="bottom",
                                fontsize=8, color=TEXT2)
            ax.set_xticks(range(len(cases)))
            ax.set_xticklabels(cases, fontsize=9)
            ax.set_ylim(0, top * 1.15)
            ax.set_title(f"{model} · {ptitle}", color=TEXT, fontsize=10)
            if c == 0:
                ax.set_ylabel("ms（越低越好）", color=TEXT2)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=TEXT, fontsize=12, y=1.04)
    fig.text(0.5, -0.02, "EP4（TP2/DP2），4×H100，eager；柱 = 9 个样本中位数，细须 = min–max；每个样本取 4 个 Rank 中最慢者；"
             "只标注 Peer 数值，完整数值见数据表", ha="center", color=TEXT2, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def write_table(data: dict, out: Path):
    lines = []
    for model in MODELS:
        for phase, ptitle, _, cases in PHASES:
            lines += [f"**{model} · {ptitle}（ms，中位数 [min, max]，n=9）**", "",
                      "| 场景 | DeepEP HT | RouteEP NCCL | RouteEP Peer | Peer 相对 HT |", "|---|---:|---:|---:|---:|"]
            for cs in cases:
                d = data[model][phase][cs]
                cell = lambda s: f"{d[s]['median']:.1f} [{d[s]['min']:.1f}, {d[s]['max']:.1f}]"
                lines.append(f"| `{cs}` | {cell('deepep')} | {cell('nccl')} | {cell('peer')} | "
                             f"{d['peer']['median'] / d['deepep']['median'] - 1:+.1%} |")
            lines.append("")
    out.write_text("\n".join(lines))


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root { color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f3f2ef; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#7a7974; --grid:#e4e3df; --axis:#c9c8c3;
  --deepep:#2a78d6; --nccl:#eb6834; --peer:#1baf7a; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#242423; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#94938b;
  --grid:#33332f; --axis:#4a4945; --deepep:#3987e5; --nccl:#d95926; --peer:#199e70; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface-1:#1a1a19; --surface-2:#242423; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --text-muted:#94938b; --grid:#33332f; --axis:#4a4945; --deepep:#3987e5; --nccl:#d95926; --peer:#199e70; }
body { margin:0; }
.viz-root { background:var(--surface-1); color:var(--text-primary); font:14px/1.5 "Noto Sans SC",system-ui,sans-serif; padding:24px; }
h1 { font-size:18px; margin:0 0 4px; font-weight:600; } .sub { color:var(--text-secondary); margin:0 0 12px; font-size:13px; }
.legend { display:flex; gap:20px; flex-wrap:wrap; margin:0 0 12px; font-size:13px; color:var(--text-secondary); }
.legend span { display:inline-flex; align-items:center; gap:6px; } .legend svg { width:12px; height:12px; flex:none; }
.controls { margin:0 0 12px; } .controls button { font:inherit; font-size:13px; color:var(--text-primary); background:var(--surface-2);
  border:1px solid var(--grid); border-radius:6px; padding:4px 10px; cursor:pointer; }
.panels { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:24px; }
.panel h2 { font-size:14px; font-weight:600; margin:0 0 4px; }
.panel svg { width:100%; height:auto; display:block; overflow:visible; }
.grid-line { stroke:var(--grid); stroke-width:1; } .axis-line { stroke:var(--axis); stroke-width:1; }
.tick { fill:var(--text-muted); font-size:11px; } .value { fill:var(--text-secondary); font-size:11px; }
.series-name { fill:var(--text-primary); font-size:10px; font-weight:600; }
.whisker { stroke:var(--text-secondary); stroke-width:1; }
.bar { outline:none; } .bar:hover .mark, .bar:focus .mark { opacity:.8; } .bar:focus .hit { stroke:var(--text-primary); stroke-width:1.5; }
.texture { display:none; pointer-events:none; } .viz-root[data-texture="on"] .texture { display:inline; }
.hatch line { stroke:var(--text-primary); stroke-opacity:.45; }
@media print { .texture { display:inline; } .controls, .tooltip { display:none !important; } }
@media (forced-colors: active) { .texture { display:inline; } .hatch line { stroke:CanvasText; } .mark { fill:Canvas; stroke:CanvasText; } }
.tooltip { position:fixed; pointer-events:none; background:var(--surface-2); color:var(--text-primary); border:1px solid var(--grid);
  border-radius:6px; padding:8px 10px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.15); display:none; z-index:10; min-width:220px; }
.tooltip .v { font-size:15px; font-weight:600; } .tooltip .row { display:flex; align-items:center; gap:6px; color:var(--text-secondary); }
.tooltip .row b { color:var(--text-primary); font-weight:600; margin-left:auto; } .tooltip .k { width:12px; height:2px; display:inline-block; }
table { border-collapse:collapse; margin-top:8px; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { padding:4px 10px; border-bottom:1px solid var(--grid); text-align:right; } th:first-child, td:first-child { text-align:left; }
th { color:var(--text-secondary); font-weight:600; } .note { color:var(--text-muted); font-size:12px; margin-top:16px; }
details { margin-top:20px; } summary { cursor:pointer; color:var(--text-secondary); }
</style></head>
<body><div class="viz-root">
<h1>__TITLE__</h1>
<p class="sub">EP4（TP2/DP2），4×H100 80GB，三条路径都是 eager。Prefill 场景为 prefill 阶段耗时，Decode 场景为每步耗时；越低越好。柱 = 9 个样本中位数，细须 = min–max。只标出 Peer 的数值，悬停或 Tab 聚焦可看每根柱，完整数值见下方数据表。</p>
<div class="legend" aria-hidden="true">
 <span><svg viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="var(--deepep)"/></svg>DeepEP HT</span>
 <span><svg viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="var(--nccl)"/><rect width="12" height="12" rx="2" fill="url(#h45)" class="texture"/></svg>RouteEP NCCL</span>
 <span><svg viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="var(--peer)"/><rect width="12" height="12" rx="2" fill="url(#h135)" class="texture"/></svg>RouteEP Peer</span>
</div>
<div class="controls"><button type="button" id="tex" aria-pressed="false">纹理：关</button></div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>
 <pattern id="h45" class="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke-width="2"/></pattern>
 <pattern id="h135" class="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(135)"><line x1="0" y1="0" x2="0" y2="6" stroke-width="2"/></pattern>
</defs></svg>
<div class="panels" id="panels"></div>
<details open><summary>数据表</summary><div id="tables"></div></details>
<p class="note">样本来源：benchmarks/peer_suite.py，三条路径在同一台空闲的 4×H100 上交替运行（每模型 3 个 trial × 3 次测量，trial 间轮换顺序）；三条路径的自由生成 Token 完全一致。DeepEP 为 HT 后端，DP&gt;1 时 vLLM 不允许它开 CUDA Graph。</p>
<div class="tooltip" id="tip" role="status" aria-live="polite"></div>
</div>
<script>
const DATA = __DATA__, PHASES = __PHASES__, SERIES = ["deepep","nccl","peer"];
const NAMES = {deepep:"DeepEP HT", nccl:"RouteEP NCCL", peer:"RouteEP Peer"};
const SHORT = {deepep:"HT", nccl:"NCCL", peer:"Peer"}, HATCH = {nccl:"h45", peer:"h135"};
const NS = "http://www.w3.org/2000/svg", W = 520, H = 280, M = {t:20, r:8, b:36, l:44};
const tip = document.getElementById("tip");
function el(t, a, p) { const n = document.createElementNS(NS, t); for (const [k,v] of Object.entries(a||{})) n.setAttribute(k, v); if (p) p.appendChild(n); return n; }
function nice(v) { for (const s of [10,20,25,50,100]) { const step = s * Math.pow(10, Math.floor(Math.log10(v)) - 1); if (v / step <= 6) return step; } return v / 5; }
function show(e, model, phase, cs, s) {
  const d = DATA[model][phase.key][cs]; tip.replaceChildren();
  const h = document.createElement("div"); h.style.color = "var(--text-secondary)"; h.textContent = model + " · " + cs; tip.append(h);
  const v = document.createElement("div"); v.className = "v"; v.textContent = d[s].median.toFixed(1) + " ms  " + NAMES[s]; tip.append(v);
  for (const k of SERIES) { const r = document.createElement("div"); r.className = "row";
    const key = document.createElement("i"); key.className = "k"; key.style.background = "var(--" + k + ")";
    const n = document.createElement("span"); n.textContent = NAMES[k];
    const b = document.createElement("b"); b.textContent = d[k].median.toFixed(1) + " ms [" + d[k].min.toFixed(1) + ", " + d[k].max.toFixed(1) + "]";
    r.append(key, n, b); tip.append(r); }
  const f = document.createElement("div"); f.style.color = "var(--text-muted)"; f.textContent = "中位数 [min, max]，n=" + d[s].n + "；" + phase.title; tip.append(f);
  tip.style.display = "block";
  let x, y; if (e.type === "focus") { const b = e.target.getBoundingClientRect(); x = b.right + 8; y = b.top; } else { x = e.clientX + 14; y = e.clientY + 14; }
  tip.style.left = Math.min(x, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = Math.min(y, innerHeight - tip.offsetHeight - 8) + "px";
}
function hide() { tip.style.display = "none"; }
function panel(model, phase) {
  const wrap = document.createElement("div"); wrap.className = "panel";
  const h = document.createElement("h2"); h.textContent = model + " · " + phase.title; wrap.append(h);
  const svg = el("svg", {viewBox:`0 0 ${W} ${H}`, role:"img", "aria-label": model + " " + phase.title + " 三条路径对比"}, wrap);
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const top = Math.max(...phase.cases.flatMap(c => SERIES.map(s => DATA[model][phase.key][c][s].max)));
  const step = nice(top * 1.15), ymax = Math.ceil(top * 1.15 / step) * step, Y = v => M.t + ih - v / ymax * ih;
  for (let v = 0; v <= ymax + 1e-9; v += step) {
    el("line", {x1:M.l, x2:W-M.r, y1:Y(v), y2:Y(v), class: v ? "grid-line" : "axis-line"}, svg);
    el("text", {x:M.l-6, y:Y(v)+4, "text-anchor":"end", class:"tick"}, svg).textContent = +v.toFixed(1);
  }
  const band = iw / phase.cases.length, bw = Math.min(30, band * 0.24), gap = 2;
  phase.cases.forEach((cs, i) => {
    const cx = M.l + band * (i + .5);
    SERIES.forEach((s, k) => {
      const d = DATA[model][phase.key][cs][s], x = cx + (k - 1) * (bw + gap) - bw / 2, t = Y(d.median), r = 4;
      const g = el("g", {class:"bar", tabindex:0, role:"graphics-symbol", "aria-label": `${cs} ${NAMES[s]} ${d.median.toFixed(1)} ms`}, svg);
      el("rect", {class:"hit", x:x-gap, y:M.t, width:bw+gap*2, height:ih, fill:"transparent", rx:3}, g);
      const shape = `M${x},${Y(0)} V${t+r} Q${x},${t} ${x+r},${t} H${x+bw-r} Q${x+bw},${t} ${x+bw},${t+r} V${Y(0)} Z`;
      el("path", {class:"mark", fill:`var(--${s})`, d:shape}, g);
      if (HATCH[s]) el("path", {class:"texture", fill:`url(#${HATCH[s]})`, d:shape}, g);
      const mx = x + bw / 2;
      el("line", {class:"whisker", x1:mx, x2:mx, y1:Y(d.min), y2:Y(d.max)}, g);
      for (const v of [d.min, d.max]) el("line", {class:"whisker", x1:mx-3, x2:mx+3, y1:Y(v), y2:Y(v)}, g);
      if (s === "peer") el("text", {x:mx, y:Y(d.max)-4, "text-anchor":"middle", class:"value"}, g).textContent = d.median.toFixed(1);
      if (i === 0) el("text", {x:mx, y:Y(0)-6, "text-anchor":"start", class:"series-name",
        transform:`rotate(-90 ${mx+3} ${Y(0)-6})`}, g).textContent = SHORT[s];
      g.addEventListener("pointermove", e => show(e, model, phase, cs, s)); g.addEventListener("pointerleave", hide);
      g.addEventListener("focus", e => show(e, model, phase, cs, s)); g.addEventListener("blur", hide);
    });
    el("text", {x:cx, y:H-M.b+16, "text-anchor":"middle", class:"tick"}, svg).textContent = cs;
  });
  el("text", {x:4, y:M.t-6, class:"tick"}, svg).textContent = "ms";
  return wrap;
}
function table(model, phase) {
  const w = document.createElement("div"), h = document.createElement("h2"); h.style.fontSize = "13px";
  h.textContent = model + " · " + phase.title + "（ms，中位数 [min, max]）"; w.append(h);
  const t = document.createElement("table"), hr = t.createTHead().insertRow();
  for (const s of ["场景", "DeepEP HT", "RouteEP NCCL", "RouteEP Peer", "Peer 相对 HT"]) { const th = document.createElement("th"); th.textContent = s; hr.append(th); }
  const b = t.createTBody();
  for (const cs of phase.cases) { const d = DATA[model][phase.key][cs], r = b.insertRow();
    for (const v of [cs, ...SERIES.map(s => `${d[s].median.toFixed(1)} [${d[s].min.toFixed(1)}, ${d[s].max.toFixed(1)}]`),
        ((d.peer.median / d.deepep.median - 1) * 100).toFixed(1) + "%"]) r.insertCell().textContent = v; }
  w.append(t); return w;
}
const panels = document.getElementById("panels"), tables = document.getElementById("tables");
for (const m of Object.keys(DATA)) for (const p of PHASES) { panels.append(panel(m, p)); tables.append(table(m, p)); }
const root = document.querySelector(".viz-root"), tb = document.getElementById("tex");
tb.addEventListener("click", () => { const on = tb.getAttribute("aria-pressed") !== "true"; tb.setAttribute("aria-pressed", String(on));
  tb.textContent = on ? "纹理：开" : "纹理：关"; root.dataset.texture = on ? "on" : "off"; });
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    p.add_argument("--label", default="11876eb")
    a = p.parse_args()
    use_chinese_font()
    report = Path(a.report)
    data = summarize(report / "data" / f"three-way-{a.label}.json")
    title = f"按阶段对比：DeepEP HT / RouteEP NCCL / RouteEP Peer（{a.label}，EP4）"
    plot_png(data, report / f"phase-latency-{a.label}", title)
    write_table(data, report / f"table-phase-{a.label}.md")
    phases = [dict(key=k, title=t, cases=c) for k, t, _, c in PHASES]
    (report / f"phase-latency-{a.label}.html").write_text(
        HTML.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__PHASES__", json.dumps(phases, ensure_ascii=False)), encoding="utf-8")


if __name__ == "__main__":
    main()
