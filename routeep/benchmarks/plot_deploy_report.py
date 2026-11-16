"""部署基线（实验 C）：decode 每步耗时，RouteEP Peer eager vs DeepEP LL eager vs DeepEP LL graph。

输入：报告目录 data/deploy-baseline-expC-ep4.json，格式
    {model: {case: {peer|lle|llg: [{trial, rep, steady, old}, ...]}}}
    steady = 所有请求都出首 Token 之后的区间 ÷ (步数-1)，每次重复取 4 个 Rank 的最大值。
    LL 两组开了 512 预算的 chunked prefill，decode_b32 的旧口径（从最早首 Token 起算）
    会把后续请求的 prefill 计入 decode；steady 口径排除了这部分，三组一致可比。
输出：deploy-baseline-ep4.png/.svg/.html 与 table-deploy-baseline-ep4.md。

启动方式：
    python3 benchmarks/plot_deploy_report.py --report reports/h100-opt-20261028

图表约定：
    - 两张小图（两个模型），同一阶段同一单位，共用 y 轴范围；柱 = 9 个样本中位数，细须 = min–max。
    - 颜色按实现固定：Peer 用 slot3 青（与报告其它图一致），LL eager 用 slot4 黄，LL graph 用 slot7 紫。
      浅/深两套色阶已用 dataviz 校验器检查（浅色 CVD ΔE 9.1、深色 8.4；浅色下青/黄对比度 < 3:1，
      由直接数值标签与数据表兜底）。
    - 只给 Peer 和 LL graph 标数值（问题的两端），LL eager 的数值在悬停提示和数据表里。
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
CASES = ["decode_b1", "decode_b4", "decode_b32"]
SERIES = ["peer", "lle", "llg"]
NAMES = {"peer": "RouteEP Peer + eager", "lle": "DeepEP LL + eager", "llg": "DeepEP LL + CUDA Graph"}
SHORT = {"peer": "Peer", "lle": "LL", "llg": "LL+G"}
LIGHT = {"peer": "#1baf7a", "lle": "#eda100", "llg": "#6250d6"}
LABELLED = {"peer", "llg"}
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def summarize(path: Path) -> dict:
    raw = json.loads(path.read_text())
    out = {}
    for m in MODELS:
        for c in CASES:
            for s in SERIES:
                v = [q["steady"] for q in raw[m][c][s]]
                out.setdefault(m, {}).setdefault(c, {})[s] = dict(
                    median=statistics.median(v), min=min(v), max=max(v), n=len(v))
    return out


def plot_png(data: dict, stem: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, facecolor=SURFACE)
    top = max(d["max"] for m in data.values() for cs in m.values() for d in cs.values())
    width = 0.26
    for ax, model in zip(axes, MODELS):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT2, labelsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for k, s in enumerate(SERIES):
            xs = [i + (k - 1) * (width + 0.02) for i in range(len(CASES))]
            med = [data[model][c][s]["median"] for c in CASES]
            lo = [m - data[model][c][s]["min"] for m, c in zip(med, CASES)]
            hi = [data[model][c][s]["max"] - m for m, c in zip(med, CASES)]
            ax.bar(xs, med, width, color=LIGHT[s], label=NAMES[s], edgecolor=SURFACE, linewidth=1)
            ax.errorbar(xs, med, yerr=[lo, hi], fmt="none", ecolor=TEXT2, elinewidth=1, capsize=2)
            ax.text(xs[0], top * 0.03, SHORT[s], ha="center", va="bottom", fontsize=7,
                    color=TEXT, rotation=90)
            if s in LABELLED:
                for x, m, h in zip(xs, med, hi):
                    ax.text(x, m + h + top * 0.02, f"{m:.1f}", ha="center", va="bottom",
                            fontsize=8, color=TEXT2)
        ax.set_xticks(range(len(CASES)))
        ax.set_xticklabels(CASES, fontsize=9)
        ax.set_ylim(0, top * 1.18)
        ax.set_title(model, color=TEXT, fontsize=10)
    axes[0].set_ylabel("decode 每步耗时 ms（越低越好）", color=TEXT2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, color=TEXT, fontsize=12, y=1.08)
    fig.text(0.5, -0.04, "EP4（TP2/DP2），gpu3 物理 GPU4–7，三组同机交替；柱 = 9 个样本中位数，细须 = min–max；"
             "口径：所有请求出首 Token 后的稳态区间 ÷ 步数，每次取 4 个 Rank 中最慢者", ha="center",
             color=TEXT2, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(stem.with_suffix("." + ext), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def write_table(data: dict, out: Path):
    lines = []
    for m in MODELS:
        lines += [f"**{m}（decode 每步 ms，中位数 [min, max]，n=9）**", "",
                  "| 场景 | RouteEP Peer + eager | DeepEP LL + eager | DeepEP LL + Graph | Peer ÷ LL+Graph |",
                  "|---|---:|---:|---:|---:|"]
        for c in CASES:
            d = data[m][c]
            cell = lambda s: f"{d[s]['median']:.1f} [{d[s]['min']:.1f}, {d[s]['max']:.1f}]"
            lines.append(f"| `{c}` | {cell('peer')} | {cell('lle')} | {cell('llg')} | "
                         f"{d['peer']['median'] / d['llg']['median']:.2f}× |")
        lines.append("")
    out.write_text("\n".join(lines))


HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root { color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f3f2ef; --text-primary:#0b0b0b;
  --text-secondary:#52514e; --text-muted:#7a7974; --grid:#e4e3df; --axis:#c9c8c3;
  --peer:#1baf7a; --lle:#eda100; --llg:#6250d6; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) .viz-root { color-scheme: dark;
  --surface-1:#1a1a19; --surface-2:#242423; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#94938b;
  --grid:#33332f; --axis:#4a4945; --peer:#199e70; --lle:#c98500; --llg:#9085e9; } }
:root[data-theme="dark"] .viz-root { color-scheme: dark; --surface-1:#1a1a19; --surface-2:#242423; --text-primary:#f0efec;
  --text-secondary:#c3c2b7; --text-muted:#94938b; --grid:#33332f; --axis:#4a4945; --peer:#199e70; --lle:#c98500; --llg:#9085e9; }
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
  border-radius:6px; padding:8px 10px; font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.15); display:none; z-index:10; min-width:240px; }
.tooltip .v { font-size:15px; font-weight:600; } .tooltip .row { display:flex; align-items:center; gap:6px; color:var(--text-secondary); }
.tooltip .row b { color:var(--text-primary); font-weight:600; margin-left:auto; } .tooltip .k { width:12px; height:2px; display:inline-block; }
table { border-collapse:collapse; margin-top:8px; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { padding:4px 10px; border-bottom:1px solid var(--grid); text-align:right; } th:first-child, td:first-child { text-align:left; }
th { color:var(--text-secondary); font-weight:600; } .note { color:var(--text-muted); font-size:12px; margin-top:16px; }
details { margin-top:20px; } summary { cursor:pointer; color:var(--text-secondary); }
</style></head>
<body><div class="viz-root">
<h1>__TITLE__</h1>
<p class="sub">EP4（TP2/DP2），gpu3 物理 GPU4–7，三组在同一组卡上交替运行。纵轴为 decode 每步耗时，越低越好；柱 = 9 个样本中位数，细须 = min–max。只标出 Peer 和 LL+Graph 的数值，悬停或 Tab 聚焦可看每根柱，完整数值见下方数据表。</p>
<div class="legend" aria-hidden="true">
 <span><svg viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="var(--peer)"/></svg>RouteEP Peer + eager</span>
 <span><svg viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="var(--lle)"/><rect width="12" height="12" rx="2" fill="url(#h45)" class="texture"/></svg>DeepEP LL + eager</span>
 <span><svg viewBox="0 0 12 12"><rect width="12" height="12" rx="2" fill="var(--llg)"/><rect width="12" height="12" rx="2" fill="url(#h135)" class="texture"/></svg>DeepEP LL + CUDA Graph</span>
</div>
<div class="controls"><button type="button" id="tex" aria-pressed="false">纹理：关</button></div>
<svg width="0" height="0" style="position:absolute" aria-hidden="true" focusable="false"><defs>
 <pattern id="h45" class="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke-width="2"/></pattern>
 <pattern id="h135" class="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(135)"><line x1="0" y1="0" x2="0" y2="6" stroke-width="2"/></pattern>
</defs></svg>
<div class="panels" id="panels"></div>
<details open><summary>数据表</summary><div id="tables"></div></details>
<p class="note">口径：所有请求都出首 Token 之后的稳态区间 ÷ (步数−1)，每次重复取 4 个 Rank 中最慢者。LL 两组用 512 Token 预算 + chunked prefill（与 decode 部署一致），Peer 用默认调度；稳态口径排除了 chunked prefill 对 decode_b32 的影响。三组 Peer 均实测走 Peer/TMA 去重，LL 两组 48/40 层均为 DeepEPLLAll2AllManager，Graph 组为 FULL_DECODE_ONLY。</p>
<div class="tooltip" id="tip" role="status" aria-live="polite"></div>
</div>
<script>
const DATA = __DATA__, CASES = __CASES__, SERIES = ["peer","lle","llg"], LABELLED = new Set(["peer","llg"]);
const NAMES = {peer:"RouteEP Peer + eager", lle:"DeepEP LL + eager", llg:"DeepEP LL + CUDA Graph"};
const SHORT = {peer:"Peer", lle:"LL", llg:"LL+G"}, HATCH = {lle:"h45", llg:"h135"};
const NS = "http://www.w3.org/2000/svg", W = 520, H = 280, M = {t:20, r:8, b:36, l:44};
const TOP = Math.max(...Object.values(DATA).flatMap(m => Object.values(m).flatMap(c => SERIES.map(s => c[s].max))));
const tip = document.getElementById("tip");
function el(t, a, p) { const n = document.createElementNS(NS, t); for (const [k,v] of Object.entries(a||{})) n.setAttribute(k, v); if (p) p.appendChild(n); return n; }
function show(e, model, cs, s) {
  const d = DATA[model][cs]; tip.replaceChildren();
  const h = document.createElement("div"); h.style.color = "var(--text-secondary)"; h.textContent = model + " · " + cs; tip.append(h);
  const v = document.createElement("div"); v.className = "v"; v.textContent = d[s].median.toFixed(1) + " ms  " + NAMES[s]; tip.append(v);
  for (const k of SERIES) { const r = document.createElement("div"); r.className = "row";
    const key = document.createElement("i"); key.className = "k"; key.style.background = "var(--" + k + ")";
    const n = document.createElement("span"); n.textContent = NAMES[k];
    const b = document.createElement("b"); b.textContent = d[k].median.toFixed(1) + " ms [" + d[k].min.toFixed(1) + ", " + d[k].max.toFixed(1) + "]";
    r.append(key, n, b); tip.append(r); }
  const f = document.createElement("div"); f.style.color = "var(--text-muted)";
  f.textContent = "Peer ÷ LL+Graph = " + (d.peer.median / d.llg.median).toFixed(2) + "×，n=" + d[s].n; tip.append(f);
  tip.style.display = "block";
  let x, y; if (e.type === "focus") { const b = e.target.getBoundingClientRect(); x = b.right + 8; y = b.top; } else { x = e.clientX + 14; y = e.clientY + 14; }
  tip.style.left = Math.min(x, innerWidth - tip.offsetWidth - 8) + "px"; tip.style.top = Math.min(y, innerHeight - tip.offsetHeight - 8) + "px";
}
function hide() { tip.style.display = "none"; }
function panel(model) {
  const wrap = document.createElement("div"); wrap.className = "panel";
  const h = document.createElement("h2"); h.textContent = model; wrap.append(h);
  const svg = el("svg", {viewBox:`0 0 ${W} ${H}`, role:"img", "aria-label": model + " decode 每步耗时三组对比"}, wrap);
  const iw = W - M.l - M.r, ih = H - M.t - M.b, step = 20, ymax = Math.ceil(TOP * 1.18 / step) * step, Y = v => M.t + ih - v / ymax * ih;
  for (let v = 0; v <= ymax; v += step) {
    el("line", {x1:M.l, x2:W-M.r, y1:Y(v), y2:Y(v), class: v ? "grid-line" : "axis-line"}, svg);
    el("text", {x:M.l-6, y:Y(v)+4, "text-anchor":"end", class:"tick"}, svg).textContent = v;
  }
  const band = iw / CASES.length, bw = Math.min(30, band * 0.24), gap = 2;
  CASES.forEach((cs, i) => {
    const cx = M.l + band * (i + .5);
    SERIES.forEach((s, k) => {
      const d = DATA[model][cs][s], x = cx + (k - 1) * (bw + gap) - bw / 2, t = Y(d.median), r = 4;
      const g = el("g", {class:"bar", tabindex:0, role:"graphics-symbol", "aria-label": `${cs} ${NAMES[s]} ${d.median.toFixed(1)} ms`}, svg);
      el("rect", {class:"hit", x:x-gap, y:M.t, width:bw+gap*2, height:ih, fill:"transparent", rx:3}, g);
      const shape = `M${x},${Y(0)} V${t+r} Q${x},${t} ${x+r},${t} H${x+bw-r} Q${x+bw},${t} ${x+bw},${t+r} V${Y(0)} Z`;
      el("path", {class:"mark", fill:`var(--${s})`, d:shape}, g);
      if (HATCH[s]) el("path", {class:"texture", fill:`url(#${HATCH[s]})`, d:shape}, g);
      const mx = x + bw / 2;
      el("line", {class:"whisker", x1:mx, x2:mx, y1:Y(d.min), y2:Y(d.max)}, g);
      for (const v of [d.min, d.max]) el("line", {class:"whisker", x1:mx-3, x2:mx+3, y1:Y(v), y2:Y(v)}, g);
      if (LABELLED.has(s)) el("text", {x:mx, y:Y(d.max)-4, "text-anchor":"middle", class:"value"}, g).textContent = d.median.toFixed(1);
      if (i === 0) el("text", {x:mx, y:Y(0)-6, "text-anchor":"start", class:"series-name",
        transform:`rotate(-90 ${mx+3} ${Y(0)-6})`}, g).textContent = SHORT[s];
      g.addEventListener("pointermove", e => show(e, model, cs, s)); g.addEventListener("pointerleave", hide);
      g.addEventListener("focus", e => show(e, model, cs, s)); g.addEventListener("blur", hide);
    });
    el("text", {x:cx, y:H-M.b+16, "text-anchor":"middle", class:"tick"}, svg).textContent = cs;
  });
  el("text", {x:4, y:M.t-6, class:"tick"}, svg).textContent = "ms";
  return wrap;
}
function table(model) {
  const w = document.createElement("div"), h = document.createElement("h2"); h.style.fontSize = "13px";
  h.textContent = model + "（decode 每步 ms，中位数 [min, max]）"; w.append(h);
  const t = document.createElement("table"), hr = t.createTHead().insertRow();
  for (const s of ["场景", ...SERIES.map(k => NAMES[k]), "Peer ÷ LL+Graph"]) { const th = document.createElement("th"); th.textContent = s; hr.append(th); }
  const b = t.createTBody();
  for (const cs of CASES) { const d = DATA[model][cs], r = b.insertRow();
    for (const v of [cs, ...SERIES.map(s => `${d[s].median.toFixed(1)} [${d[s].min.toFixed(1)}, ${d[s].max.toFixed(1)}]`),
        (d.peer.median / d.llg.median).toFixed(2) + "×"]) r.insertCell().textContent = v; }
  w.append(t); return w;
}
const panels = document.getElementById("panels"), tables = document.getElementById("tables");
for (const m of Object.keys(DATA)) { panels.append(panel(m)); tables.append(table(m)); }
const root = document.querySelector(".viz-root"), tb = document.getElementById("tex");
tb.addEventListener("click", () => { const on = tb.getAttribute("aria-pressed") !== "true"; tb.setAttribute("aria-pressed", String(on));
  tb.textContent = on ? "纹理：开" : "纹理：关"; root.dataset.texture = on ? "on" : "off"; });
</script></body></html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    use_chinese_font()
    report = Path(a.report)
    data = summarize(report / "data" / "deploy-baseline-expC-ep4.json")
    title = "部署基线：decode 每步耗时，RouteEP Peer eager vs DeepEP LL（eager / CUDA Graph），EP4"
    plot_png(data, report / "deploy-baseline-ep4", title)
    write_table(data, report / "table-deploy-baseline-ep4.md")
    (report / "deploy-baseline-ep4.html").write_text(
        HTML.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, ensure_ascii=False))
        .replace("__CASES__", json.dumps(CASES)), encoding="utf-8")


if __name__ == "__main__":
    main()
