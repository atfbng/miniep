"""生成层级 microbenchmark 的交互式 HTML 图（三组“优化前/后”，ms/层随 Token 数变化）。

输入：报告目录 data/layer-*.json（benchmarks/dedup_layer_bench.py 的输出）。
输出：layer-latency.html，单文件，无外部依赖。

启动方式：
    python3 benchmarks/html_layer_report.py --report reports/h100-opt-20261028

图表约定：
    - 三张小图（small multiples），各自一条 y 轴（ms/层），x 轴为每 Rank Token 数（log2）。
    - 每张图只有两条线：“优化后”用 slot7 紫（实线），“优化前”用 slot5 品红（虚线，作为第二编码）。
      不复用相对速度图里已分配给 DeepEP 蓝 / NCCL 橙 / Peer 青的颜色，避免同一颜色在报告里有两种含义。
      该色对已用 dataviz 校验器检查：浅色 CVD ΔE 19.9、正常视觉 ΔE 28.1（品红对浅底 2.62 < 3:1，
      由终点数值标签与数据表兜底）；深色 CVD ΔE 16.0、正常视觉 ΔE 19.7、对比度全部 >= 3:1。
    - crosshair 吸附到最近的 Token 数，tooltip 同时列出前/后两个值与降幅；键盘可用左右键移动。
    - 只在每条线终点直接标数值；完整数值见下方数据表。名称一律用 textContent 写入。
"""
import argparse
import json
import statistics
from pathlib import Path

PANELS = [
    ("NCCL 变长路径：第 1 项前后", ("ep2-grouped_unfused", "", "优化前（未融合）"), ("ep2-grouped", "", "第 1 项后（融合）")),
    ("NCCL 固定容量路径：第 6 项前后", ("l6-commit", "-fixed", "9f9ff6f"), ("l6-new", "-fixed", "+第 6 项")),
    ("Peer/TMA 路径：移植前后", ("p9-commit", "", "9f9ff6f"), ("p9-new", "", "+Peer 移植")),
]


def series(data_dir: Path, prefix: str, variant: str) -> dict:
    runs = sorted(data_dir.glob(f"layer-{prefix}-r*{variant}.json"))
    if not runs:
        raise FileNotFoundError(f"no layer runs for {prefix}{variant}")
    by_n = {}
    for run in runs:
        for q in json.loads(run.read_text())["results"]:
            by_n.setdefault(q["tokens_per_rank"], []).append(q["ms_per_layer_median"])
    return {"runs": len(runs), "points": [[n, statistics.mean(v)] for n, v in sorted(by_n.items())]}


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>层级 microbenchmark：优化前后 ms/层</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ef;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --grid: #e4e3df; --axis: #c9c8c3; --cross: #b5b4ae;
  --after: #6250d6; --before: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #242423;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #94938b;
    --grid: #33332f; --axis: #4a4945; --cross: #5c5b56;
    --after: #9085e9; --before: #d55181;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #242423;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #94938b;
  --grid: #33332f; --axis: #4a4945; --cross: #5c5b56;
  --after: #9085e9; --before: #d55181;
}
body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.5 "Noto Sans SC", system-ui, sans-serif; padding: 24px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
.sub { color: var(--text-secondary); margin: 0 0 16px; font-size: 13px; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 24px; }
.panel h2 { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
.legend { display: flex; gap: 16px; font-size: 12px; color: var(--text-secondary); margin: 0 0 4px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend svg { width: 18px; height: 6px; flex: none; }
.plot { width: 100%; height: auto; display: block; overflow: visible; outline: none; }
.plot:focus .frame { stroke: var(--text-secondary); }
.grid-line { stroke: var(--grid); stroke-width: 1; }
.axis-line { stroke: var(--axis); stroke-width: 1; }
.tick { fill: var(--text-muted); font-size: 11px; }
.end { fill: var(--text-secondary); font-size: 11px; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.dot { stroke: var(--surface-1); stroke-width: 2; }
.crosshair { stroke: var(--cross); stroke-width: 1; display: none; }
.tooltip { position: fixed; pointer-events: none; background: var(--surface-2); color: var(--text-primary);
  border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; z-index: 10; min-width: 190px; }
.tooltip .h { color: var(--text-secondary); }
.tooltip .row { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); }
.tooltip .row b { color: var(--text-primary); font-weight: 600; margin-left: auto; font-size: 13px; }
.tooltip .k { width: 12px; height: 2px; display: inline-block; }
table { border-collapse: collapse; margin-top: 8px; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--grid); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 16px; }
details { margin-top: 20px; }
summary { cursor: pointer; color: var(--text-secondary); }
</style>
</head>
<body>
<div class="viz-root">
  <h1>层级 microbenchmark：优化前后每层耗时</h1>
  <p class="sub">EP2（2×H100），合成均匀路由，H=2048、F=768、E=128、K=8，BF16。纵轴 ms/层，越低越好；横轴为每 Rank Token 数（对数）。悬停或聚焦图后用左右键查看每个 Token 数下的数值。</p>
  <div class="panels" id="panels"></div>
  <details open><summary>数据表</summary><div id="tables"></div></details>
  <p class="note">统计：每轮连续执行 8 层，取最慢 Rank 的平均 ms/层；每配置 20 次重复取中位；图中为 2 轮的平均。</p>
  <div class="tooltip" id="tip" role="status" aria-live="polite"></div>
</div>
<script>
const PANELS = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const W = 400, H = 260, M = {t: 12, r: 60, b: 40, l: 40};
const tip = document.getElementById("tip");

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function niceMax(v) { const s = [0.5, 1, 2, 2.5, 5]; let p = 1;
  while (true) { for (const m of s) if (m * p >= v) return m * p; p *= 10; } }

function panel(p) {
  const wrap = document.createElement("div"); wrap.className = "panel";
  const h = document.createElement("h2"); h.textContent = p.title; wrap.appendChild(h);
  const legend = document.createElement("div"); legend.className = "legend";
  for (const [key, s] of [["before", p.before], ["after", p.after]]) {
    const span = document.createElement("span");
    const k = el("svg", {viewBox: "0 0 18 6"});
    el("line", {x1: 0, x2: 18, y1: 3, y2: 3, stroke: `var(--${key})`, "stroke-width": 2,
      "stroke-dasharray": key === "before" ? "4 3" : "none"}, k);
    const t = document.createElement("span"); t.textContent = s.name;
    span.append(k, t); legend.appendChild(span);
  }
  wrap.appendChild(legend);
  const svg = el("svg", {class: "plot", viewBox: `0 0 ${W} ${H}`, tabindex: 0, role: "img",
    "aria-label": p.title + "：优化前后 ms/层"}, wrap);
  const xsN = p.after.points.map(q => q[0]);
  const lx = Math.log2, x0 = lx(xsN[0]), x1 = lx(xsN[xsN.length - 1]);
  const ymax = niceMax(Math.max(...p.before.points.map(q => q[1]), ...p.after.points.map(q => q[1])) * 1.05);
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const X = n => M.l + (lx(n) - x0) / (x1 - x0) * iw;
  const Y = v => M.t + ih - v / ymax * ih;
  for (let i = 0; i <= 4; i++) {
    const v = ymax * i / 4;
    el("line", {x1: M.l, x2: M.l + iw, y1: Y(v), y2: Y(v), class: i ? "grid-line" : "axis-line"}, svg);
    el("text", {x: M.l - 6, y: Y(v) + 4, "text-anchor": "end", class: "tick"}, svg).textContent = +v.toFixed(2);
  }
  for (const n of xsN) {
    el("text", {x: X(n), y: H - M.b + 16, "text-anchor": "middle", class: "tick"}, svg).textContent = n;
  }
  const cross = el("line", {class: "crosshair", y1: M.t, y2: M.t + ih}, svg);
  for (const [key, s] of [["before", p.before], ["after", p.after]]) {
    el("path", {class: "line", stroke: `var(--${key})`, "stroke-dasharray": key === "before" ? "6 4" : "none",
      d: s.points.map((q, i) => (i ? "L" : "M") + X(q[0]) + "," + Y(q[1])).join("")}, svg);
    for (const q of s.points) el("circle", {class: "dot", cx: X(q[0]), cy: Y(q[1]), r: 4, fill: `var(--${key})`}, svg);
    // 只在终点直接标注数值。
    const last = s.points[s.points.length - 1];
    // 终点直接标注“数值 + 前/后”，系列身份不只靠颜色。
    el("text", {x: X(last[0]) + 8, y: Y(last[1]) + 4, class: "end"}, svg).textContent =
      last[1].toFixed(2) + (key === "after" ? " 后" : " 前");
  }
  // 整个绘图区是命中区域：crosshair 吸附到最近的 Token 数。
  const hit = el("rect", {x: M.l, y: M.t, width: iw, height: ih, fill: "transparent"}, svg);
  let idx = -1;
  function show(i, anchor) {
    idx = i; const n = xsN[i];
    cross.setAttribute("x1", X(n)); cross.setAttribute("x2", X(n)); cross.style.display = "inline";
    const b = p.before.points[i][1], a = p.after.points[i][1];
    tip.replaceChildren();
    const head = document.createElement("div"); head.className = "h";
    head.textContent = p.title.split("：")[0] + " · 每 Rank " + n + " Token"; tip.appendChild(head);
    for (const [key, s, v] of [["after", p.after, a], ["before", p.before, b]]) {
      const r = document.createElement("div"); r.className = "row";
      const k = document.createElement("i"); k.className = "k"; k.style.background = `var(--${key})`;
      const nm = document.createElement("span"); nm.textContent = s.name;
      const val = document.createElement("b"); val.textContent = v.toFixed(3) + " ms";
      r.append(k, nm, val); tip.appendChild(r);
    }
    const d = document.createElement("div"); d.className = "h";
    d.textContent = "变化 " + ((a / b - 1) * 100).toFixed(1) + "%（负数为更快）"; tip.appendChild(d);
    tip.style.display = "block";
    const box = svg.getBoundingClientRect(), sx = box.width / W;
    const px = anchor ? anchor.x : box.left + X(n) * sx + 12, py = anchor ? anchor.y : box.top + M.t * sx;
    tip.style.left = Math.min(px, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = Math.min(py, innerHeight - tip.offsetHeight - 8) + "px";
  }
  function hide() { cross.style.display = "none"; tip.style.display = "none"; }
  function nearest(clientX) {
    const box = svg.getBoundingClientRect(), vx = (clientX - box.left) * W / box.width;
    let best = 0; xsN.forEach((n, i) => { if (Math.abs(X(n) - vx) < Math.abs(X(xsN[best]) - vx)) best = i; });
    return best;
  }
  hit.addEventListener("pointermove", e => show(nearest(e.clientX), {x: e.clientX + 14, y: e.clientY + 14}));
  hit.addEventListener("pointerleave", hide);
  svg.addEventListener("focus", () => show(idx < 0 ? xsN.length - 1 : idx));
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", e => {
    if (e.key === "ArrowRight") { show(Math.min(xsN.length - 1, idx + 1)); e.preventDefault(); }
    if (e.key === "ArrowLeft") { show(Math.max(0, idx - 1)); e.preventDefault(); }
  });
  return wrap;
}

function table(p) {
  const wrap = document.createElement("div");
  const h = document.createElement("h2"); h.textContent = p.title; h.style.fontSize = "13px"; wrap.appendChild(h);
  const t = document.createElement("table");
  const head = t.createTHead().insertRow();
  for (const s of ["每 Rank Token", p.before.name + " (ms)", p.after.name + " (ms)", "变化"]) {
    const th = document.createElement("th"); th.textContent = s; head.appendChild(th);
  }
  const body = t.createTBody();
  p.after.points.forEach((q, i) => {
    const b = p.before.points[i][1], a = q[1], r = body.insertRow();
    for (const s of [q[0], b.toFixed(3), a.toFixed(3), ((a / b - 1) * 100).toFixed(1) + "%"]) r.insertCell().textContent = s;
  });
  wrap.appendChild(t);
  return wrap;
}

const panels = document.getElementById("panels"), tables = document.getElementById("tables");
for (const p of PANELS) { panels.appendChild(panel(p)); tables.appendChild(table(p)); }
</script>
</body>
</html>
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    report = Path(a.report)
    data = []
    for title, before, after in PANELS:
        b, af = series(report / "data", *before[:2]), series(report / "data", *after[:2])
        if [q[0] for q in b["points"]] != [q[0] for q in af["points"]]:
            raise ValueError(f"{title}: before/after token sets differ")
        data.append(dict(title=title, before=dict(name=before[2], **b), after=dict(name=after[2], **af)))
    (report / "layer-latency.html").write_text(
        TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")


if __name__ == "__main__":
    main()
