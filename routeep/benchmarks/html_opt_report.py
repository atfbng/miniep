"""生成 H100 优化报告的交互式 HTML 图（相对 DeepEP HT 的速度，分组柱状图 + 数据表）。

输入：报告目录 data/three-way-<label>.json（格式见 plot_opt_report.py）。
输出：relative-speed-<label>.html，单文件，无外部依赖，可直接用浏览器打开。

启动方式：
    python3 benchmarks/html_opt_report.py --report reports/h100-opt-20261028

图表约定：
    - DeepEP HT 是基准，画成 100% 参考线；每个场景两根柱：RouteEP NCCL 去重、RouteEP Peer/TMA。
    - 颜色按实现固定：NCCL=slot2 橙，Peer=slot3 青，DeepEP 参考线=slot1 蓝。浅色/深色两套色阶
      分别用 dataviz 校验器检查过（浅色 CVD ΔE 9.2；深色 CVD ΔE 9.4、对比度全部 >= 3:1）。
      浅色下青色对比度 2.74 < 3:1，由页面下方完整数据表兜底；图上只标注偏离 DeepEP
      超过 5 个百分点的柱，其余数值通过悬停/键盘聚焦和数据表读取。
    - 每根柱可悬停或键盘聚焦，tooltip 显示场景、实现、相对速度和三条路径的中位延迟。
      所有名称都通过 textContent 写入 DOM。
    - 纹理是可选通道（色盲、黑白打印、强制配色）：默认关闭，由页面开关、打印或 forced-colors
      触发；NCCL 45°、Peer 135° 斜线，线色为填充色的更深一档。
"""
import argparse
import json
import statistics
from pathlib import Path

CASES = ["decode_b1", "decode_b4", "decode_b32", "prefill_b1_long", "prefill_b4_short", "prefill_b4_long"]
MODELS = ["Qwen3-30B-A3B", "Qwen3.5-35B-A3B"]
LABELS = {"deepep": "DeepEP HT", "nccl": "RouteEP NCCL 去重", "peer": "RouteEP Peer/TMA 去重"}


def summarize(path: Path) -> dict:
    """{model: {case: {label: 中位 e2e ms, rel_<label>: 相对速度, n: 样本数}}}"""
    raw = json.loads(path.read_text())
    out = {}
    for model in MODELS:
        for case in CASES:
            g = raw[model][case]
            m = {k: statistics.median(q["e2e_ms"] for q in g[k]) for k in ("deepep", "nccl", "peer")}
            out.setdefault(model, {})[case] = dict(
                **m, rel_nccl=m["deepep"] / m["nccl"], rel_peer=m["deepep"] / m["peer"],
                n=min(len(g[k]) for k in g),
            )
    return out


TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --surface-2: #f3f2ef;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --grid: #e4e3df; --axis: #c9c8c3;
  --ref: #2a78d6; --nccl: #eb6834; --peer: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --surface-2: #242423;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #94938b;
    --grid: #33332f; --axis: #4a4945;
    --ref: #3987e5; --nccl: #d95926; --peer: #199e70;
  }
}
:root[data-theme="dark"] .viz-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --surface-2: #242423;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #94938b;
  --grid: #33332f; --axis: #4a4945;
  --ref: #3987e5; --nccl: #d95926; --peer: #199e70;
}
body { margin: 0; }
.viz-root { background: var(--surface-1); color: var(--text-primary);
  font: 14px/1.5 "Noto Sans SC", system-ui, sans-serif; padding: 24px; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
.sub { color: var(--text-secondary); margin: 0 0 16px; font-size: 13px; }
.legend { display: flex; gap: 20px; flex-wrap: wrap; margin: 0 0 12px; font-size: 13px; color: var(--text-secondary); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.key-rect { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
.key-line { width: 18px; height: 2px; display: inline-block; }
.panels { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 24px; }
.panel h2 { font-size: 14px; font-weight: 600; margin: 0 0 4px; }
.panel svg { width: 100%; height: auto; display: block; overflow: visible; }
/* 图例色块和纹理定义的 SVG 保持固定尺寸，不能继承面板 SVG 的 100% 宽度。 */
.legend svg { width: 12px; height: 12px; flex: none; display: inline-block; }
svg.defs { position: absolute; width: 0; height: 0; pointer-events: none; }
.grid-line { stroke: var(--grid); stroke-width: 1; }
.axis-line { stroke: var(--axis); stroke-width: 1; }
.tick { fill: var(--text-muted); font-size: 11px; }
.value { fill: var(--text-secondary); font-size: 11px; }
.series-name { fill: var(--text-primary); font-size: 10px; font-weight: 600; }
.bar { cursor: default; outline: none; }
.bar rect.mark { transition: opacity .1s; }
.bar:hover rect.mark, .bar:focus rect.mark { opacity: .8; }
.bar:focus rect.hit { stroke: var(--text-primary); stroke-width: 1.5; }
.ref-line { stroke: var(--ref); stroke-width: 2; }
.tooltip { position: fixed; pointer-events: none; background: var(--surface-2); color: var(--text-primary);
  border: 1px solid var(--grid); border-radius: 6px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); display: none; z-index: 10; min-width: 200px; }
.tooltip .v { font-size: 15px; font-weight: 600; }
.tooltip .row { display: flex; align-items: center; gap: 6px; color: var(--text-secondary); }
.tooltip .row b { color: var(--text-primary); font-weight: 600; margin-left: auto; }
.tooltip .k { width: 12px; height: 2px; display: inline-block; }
table { border-collapse: collapse; margin-top: 8px; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { padding: 4px 10px; border-bottom: 1px solid var(--grid); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 16px; }
/* 纹理是可选的辅助通道：默认关闭；开关打开、打印或系统强制配色时才显示。
   NCCL 用 45° 斜线、Peer 用 135° 斜线，线色是填充色本身的更深一档（同色调）。 */
.controls { display: flex; gap: 12px; margin: 0 0 12px; }
.controls button { font: inherit; font-size: 13px; color: var(--text-primary); background: var(--surface-2);
  border: 1px solid var(--grid); border-radius: 6px; padding: 4px 10px; cursor: pointer; }
.controls button[aria-pressed="true"] { border-color: var(--text-secondary); }
.texture { display: none; pointer-events: none; }
.hatch-nccl line { stroke: color-mix(in oklab, var(--nccl) 55%, black); }
.hatch-peer line { stroke: color-mix(in oklab, var(--peer) 55%, black); }
.viz-root[data-texture="on"] .texture { display: inline; }
@media print { .texture { display: inline; } .controls, .tooltip { display: none !important; } }
@media (forced-colors: active) {
  .texture { display: inline; }
  .hatch-nccl line, .hatch-peer line { stroke: CanvasText; }
  .bar .mark { fill: Canvas; stroke: CanvasText; stroke-width: 1; }
  .ref-line { stroke: Highlight; }
}
details { margin-top: 20px; }
summary { cursor: pointer; color: var(--text-secondary); }
</style>
</head>
<body>
<div class="viz-root">
  <h1>__TITLE__</h1>
  <p class="sub">EP4（TP2/DP2），4×H100 80GB；相对速度 = DeepEP HT 端到端中位延迟 ÷ 该路径中位延迟，越高越快，DeepEP = 100%。图上只标出偏离超过 5 个百分点的柱，悬停或 Tab 聚焦可看每根柱的数值，完整数值见下方数据表。</p>
  <div class="legend" aria-hidden="true">
    <span><i class="key-line" style="background:var(--ref)"></i>DeepEP HT（=100%）</span>
    <span><svg width="12" height="12"><rect width="12" height="12" rx="2" fill="var(--nccl)" class="mark-key"/><rect width="12" height="12" rx="2" fill="url(#hatch-nccl)" class="texture"/></svg>RouteEP NCCL 去重</span>
    <span><svg width="12" height="12"><rect width="12" height="12" rx="2" fill="var(--peer)" class="mark-key"/><rect width="12" height="12" rx="2" fill="url(#hatch-peer)" class="texture"/></svg>RouteEP Peer/TMA 去重</span>
  </div>
  <div class="controls">
    <button type="button" id="texture-toggle" aria-pressed="false">纹理：关</button>
  </div>
  <svg class="defs" width="0" height="0" aria-hidden="true" focusable="false">
    <defs>
      <pattern id="hatch-nccl" class="hatch-nccl" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke-width="2"/>
      </pattern>
      <pattern id="hatch-peer" class="hatch-peer" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(135)">
        <line x1="0" y1="0" x2="0" y2="6" stroke-width="2"/>
      </pattern>
    </defs>
  </svg>
  <div class="panels" id="panels"></div>
  <details open>
    <summary>数据表</summary>
    <div id="tables"></div>
  </details>
  <p class="note" id="note"></p>
  <div class="tooltip" id="tip" role="status" aria-live="polite"></div>
</div>
<script>
const DATA = __DATA__;
const CASES = __CASES__;
const NAMES = {deepep: "DeepEP HT", nccl: "RouteEP NCCL 去重", peer: "RouteEP Peer/TMA 去重"};
const NS = "http://www.w3.org/2000/svg";
const W = 560, H = 300, M = {t: 16, r: 8, b: 72, l: 40};
// 所有版本共用同一 y 上限（由生成脚本按全部数据算出），便于并排比较。
const YMAX = __YMAX__;
const tip = document.getElementById("tip");

function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}
function fmtMs(v) { return v >= 1000 ? Math.round(v).toLocaleString() + " ms" : v.toFixed(1) + " ms"; }

function showTip(evt, model, c, label) {
  const d = DATA[model][c];
  tip.replaceChildren();
  const head = document.createElement("div"); head.textContent = model + " · " + c;
  head.style.color = "var(--text-secondary)";
  const v = document.createElement("div"); v.className = "v";
  v.textContent = (d["rel_" + label] * 100).toFixed(1) + "%  " + NAMES[label];
  tip.append(head, v);
  for (const k of ["deepep", "nccl", "peer"]) {
    const r = document.createElement("div"); r.className = "row";
    const key = document.createElement("i"); key.className = "k";
    key.style.background = "var(--" + (k === "deepep" ? "ref" : k) + ")";
    const name = document.createElement("span"); name.textContent = NAMES[k];
    const val = document.createElement("b"); val.textContent = fmtMs(d[k]);
    r.append(key, name, val); tip.append(r);
  }
  const n = document.createElement("div"); n.style.color = "var(--text-muted)";
  n.textContent = "样本数 " + d.n + "（每个样本取 4 个 Rank 中最慢者）"; tip.append(n);
  tip.style.display = "block";
  let x, y;
  if (evt.type === "focus") { const b = evt.target.getBoundingClientRect(); x = b.right + 8; y = b.top; }
  else { x = evt.clientX + 14; y = evt.clientY + 14; }
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  tip.style.left = Math.min(x, innerWidth - tw - 8) + "px";
  tip.style.top = Math.min(y, innerHeight - th - 8) + "px";
}
function hideTip() { tip.style.display = "none"; }

function panel(model) {
  const wrap = document.createElement("div"); wrap.className = "panel";
  const h = document.createElement("h2"); h.textContent = model; wrap.appendChild(h);
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": model + " 两条 RouteEP 路径相对 DeepEP HT 的速度"}, wrap);
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const y = v => M.t + ih - v / YMAX * ih;
  for (let v = 0; v < YMAX; v += 20) {
    el("line", {x1: M.l, x2: W - M.r, y1: y(v), y2: y(v), class: v ? "grid-line" : "axis-line"}, svg);
    el("text", {x: M.l - 6, y: y(v) + 4, "text-anchor": "end", class: "tick"}, svg).textContent = v;
  }
  const band = iw / CASES.length, bw = Math.min(28, band * 0.32), gap = 2;
  CASES.forEach((c, i) => {
    const cx = M.l + band * (i + 0.5);
    ["nccl", "peer"].forEach((label, k) => {
      const v = DATA[model][c]["rel_" + label] * 100;
      const x = cx + (k === 0 ? -(bw + gap / 2) : gap / 2);
      const g = el("g", {class: "bar", tabindex: 0, role: "graphics-symbol",
        "aria-label": `${c} ${NAMES[label]} ${v.toFixed(1)}%`}, svg);
      // 命中区域比柱本身宽（含间隙），并覆盖到柱顶以上，便于悬停。
      el("rect", {class: "hit", x: x - gap, y: M.t, width: bw + gap * 2, height: ih, fill: "transparent", rx: 3}, g);
      const top = y(v), r = 4;
      // 只圆角数据端（顶部），底部贴基线保持直角。
      const shape = `M${x},${y(0)} V${top + r} Q${x},${top} ${x + r},${top} H${x + bw - r} Q${x + bw},${top} ${x + bw},${top + r} V${y(0)} Z`;
      el("path", {class: "mark", fill: `var(--${label})`, d: shape}, g);
      el("path", {class: "texture", fill: `url(#hatch-${label})`, d: shape}, g);
      // 选择性标注：只给偏离 DeepEP 超过 5 个百分点的柱写数值；其余靠悬停/聚焦与下方数据表。
      const labelled = Math.abs(v - 100) > 5;
      if (labelled)
        el("text", {x: x + bw / 2, y: top - 4, "text-anchor": "middle", class: "value"}, g).textContent = v.toFixed(0);
      // 第一组柱上方直接标出系列名（文字色），其余组按相同左右顺序排列。
      if (i === 0)
        el("text", {x: x + bw / 2, y: Math.min(top, y(100)) - (labelled ? 22 : 8), "text-anchor": "middle",
          class: "series-name"}, g).textContent = label === "nccl" ? "NCCL" : "Peer";
      g.addEventListener("pointermove", e => showTip(e, model, c, label));
      g.addEventListener("pointerleave", hideTip);
      g.addEventListener("focus", e => showTip(e, model, c, label));
      g.addEventListener("blur", hideTip);
    });
    const t = el("text", {x: cx, y: H - M.b + 16, "text-anchor": "end", class: "tick",
      transform: `rotate(-28 ${cx} ${H - M.b + 16})`}, svg);
    t.textContent = c;
  });
  // DeepEP HT 基准线最后画，压在柱子上方。
  el("line", {x1: M.l, x2: W - M.r, y1: y(100), y2: y(100), class: "ref-line"}, svg);
  return wrap;
}

function table(model) {
  const wrap = document.createElement("div");
  const h = document.createElement("h2"); h.textContent = model; h.style.fontSize = "13px"; wrap.appendChild(h);
  const t = document.createElement("table");
  const head = t.createTHead().insertRow();
  for (const s of ["场景", "DeepEP HT", "NCCL 去重", "Peer/TMA", "NCCL 相对", "Peer 相对", "样本"]) {
    const th = document.createElement("th"); th.textContent = s; head.appendChild(th);
  }
  const body = t.createTBody();
  for (const c of CASES) {
    const d = DATA[model][c], r = body.insertRow();
    for (const s of [c, fmtMs(d.deepep), fmtMs(d.nccl), fmtMs(d.peer),
        (d.rel_nccl * 100).toFixed(1) + "%", (d.rel_peer * 100).toFixed(1) + "%", d.n]) {
      r.insertCell().textContent = s;
    }
  }
  wrap.appendChild(t);
  return wrap;
}

const root = document.querySelector(".viz-root"), toggle = document.getElementById("texture-toggle");
toggle.addEventListener("click", () => {
  const on = toggle.getAttribute("aria-pressed") !== "true";
  toggle.setAttribute("aria-pressed", String(on));
  toggle.textContent = on ? "纹理：开" : "纹理：关";
  root.dataset.texture = on ? "on" : "off";
});
const panels = document.getElementById("panels"), tables = document.getElementById("tables");
for (const m of Object.keys(DATA)) { panels.appendChild(panel(m)); tables.appendChild(table(m)); }
document.getElementById("note").textContent = "__NOTE__";
</script>
</body>
</html>
"""


def render(path: Path, out: Path, title: str, note: str, ymax: int):
    data = summarize(path)
    html = (TEMPLATE.replace("__TITLE__", title)
            .replace("__DATA__", json.dumps(data, ensure_ascii=False))
            .replace("__CASES__", json.dumps(CASES))
            .replace("__YMAX__", str(ymax))
            .replace("__NOTE__", note.replace('"', "'")))
    out.write_text(html, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", required=True)
    a = p.parse_args()
    report = Path(a.report)
    titles = {"9f9ff6f": "基线 9f9ff6f：RouteEP 两条路径相对 DeepEP HT 的速度（Peer 未移植、无第 6 项）"}
    paths = sorted((report / "data").glob("three-way-*.json"))
    top = max(v * 100 for path in paths for m in summarize(path).values()
              for d in m.values() for v in (d["rel_nccl"], d["rel_peer"]))
    ymax = max(130, (int(top) // 10 + 3) * 10)
    for path in paths:
        label = path.stem.removeprefix("three-way-")
        render(path, report / f"relative-speed-{label}.html",
               titles.get(label, f"{label}：RouteEP 两条路径相对 DeepEP HT 的速度（第 6 项 + Peer 移植）"),
               "三条路径在同一台 4×H100 上交替运行（每个模型 3 个 trial，每 trial 3 种顺序轮换，"
               "预热 2 次、测量 3 次）；三条路径的自由生成 Token 完全一致。", ymax)


if __name__ == "__main__":
    main()
