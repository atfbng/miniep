"""把 deepep_latency_run.py 的多个运行目录汇总成 plot_scaling_lines.py 的输入 JSON。

运行目录命名：<任意前缀->?<模型目录名>-trial<t>-<label>，例如 decode-Qwen3-30B-A3B-trial0-ncclg。
label 决定图中的系列：ht / peer / nccl（eager），ncclg（RouteEP NCCL + Graph），llg64（DeepEP LL + Graph）。
只收 4 个 Rank 都写了 completed.json 的目录。

口径：
    decode 场景：稳态每步 ms = (最后一个 Token 时间 - 所有请求都出首 Token 的时间) / (步数 - 1)；
    prefill 场景：prefill 阶段 ms（samples.json 的 max_rank_ms[2]）；
    每个 (case, repetition) 取 4 个 Rank 中最慢者。
同时比较 eager 组之间、graph 组之间的生成 token 是否一致。

启动方式：
    python3 benchmarks/summarize_scaling.py results/sweep --output results/sweep-summary.json
"""
import argparse
import glob
import json
import os
import re
import statistics

MODELS = ("Qwen3-30B-A3B", "Qwen3.5-35B-A3B")


def parse_name(name: str):
    m = re.fullmatch(r"(?:.*?-)?(" + "|".join(map(re.escape, MODELS)) + r")-trial(\d+)-([a-z0-9]+)", name)
    return (m[1], m[2], m[3]) if m else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    data, tokens = {}, {}
    for d in sorted(glob.glob(os.path.join(a.root, "*"))):
        parsed = parse_name(os.path.basename(d))
        if not parsed or not os.path.isdir(d) or len(glob.glob(d + "/rank*/completed.json")) != 4:
            continue
        model, trial, label = parsed
        worst = {}
        for path in sorted(glob.glob(d + "/rank*/samples.json")):
            rank = path.split("/")[-2]
            for row in json.load(open(path)):
                key = (row["case"], row["repetition"])
                if row["case"].startswith("decode"):
                    q = row["request_metrics"]
                    steps = q[0]["output_tokens"]
                    first = max(x["first_token_ts"] for x in q)
                    last = max(x["last_token_ts"] for x in q)
                    value = (last - first) * 1000 / (steps - 1)
                else:
                    value = row["max_rank_ms"][2]
                worst[key] = max(worst.get(key, 0.0), value)
                tokens.setdefault((model, rank) + key, {})[(label, trial)] = row["token_ids"]
        for (case, rep), value in worst.items():
            data.setdefault(model, {}).setdefault(case, {}).setdefault(label, []).append(
                dict(trial=trial, rep=rep, value=value))
    with open(a.output, "w") as f:
        json.dump(data, f, indent=1)
    for model, cases in data.items():
        print("==", model)
        for case in sorted(cases):
            print(f"  {case:18s}", "  ".join(
                f"{l} {statistics.median(r['value'] for r in rows):7.1f} n={len(rows)}"
                for l, rows in sorted(cases[case].items())))
    for kind, labels in (("eager", {"ht", "peer", "nccl"}), ("graph", {"ncclg", "llg64"})):
        same = total = 0
        for runs in tokens.values():
            values = [json.dumps(v) for (l, _), v in runs.items() if l in labels]
            if len(values) > 1:
                total += 1
                same += len(set(values)) == 1
        print(f"token {kind} 组内一致: {same}/{total}")


if __name__ == "__main__":
    main()
