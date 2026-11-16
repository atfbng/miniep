"""Pair Dispatch sub-stages with the same repetition's slowest Rank."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path

PARTS = ("EXPERT_PACK", "RANK_PACK", "COUNT_ALLGATHER", "RECEIVE_LAYOUT")


def analyze(reports_path: Path) -> dict:
    """Attribute mean CUDA interval to matching subintervals, never separate maxima.

    Every trial has one sample per repetition per Rank. For each repetition,
    select the Rank with the largest parent Dispatch interval and take all
    four children from that same Rank. Their means plus the parent/child
    boundary remainder equal the mean parent interval. Uninstrumented control
    and CPU-metadata control retain their own repetition-wise max-Rank median.
    """
    reports = json.loads(reports_path.read_text())
    groups = {}
    for report in reports:
        path = reports_path.parent / report["report_path"]
        ep = report["config"]["ep_size"]
        rank_reports = [json.loads(path.with_name(path.stem + f"-rank{r}.json").read_text()) for r in range(ep)]
        data = [r["dispatch_breakdown"] for r in rank_reports]
        if any(d["correctness"] != "PASS" for d in data):
            raise ValueError("Dispatch variants do not match Reference")
        count = report["config"]["repeats"]
        paired = {name: [] for name in PARTS}
        host = {name: [] for name in PARTS}
        parent, residual = [], []
        for i in range(count):
            rank = max(range(ep), key=lambda r: data[r]["detailed"]["dispatch_ms"][i])
            chosen = data[rank]["detailed"]
            total = chosen["dispatch_ms"][i]
            parent.append(total)
            children = []
            for name in PARTS:
                value = chosen["parts_ms"][name][i]
                paired[name].append(value)
                host[name].append(chosen["host_parts_ms"][name][i])
                children.append(value)
            delta = total - sum(children)
            if delta < -1e-5:
                raise ValueError("nested Dispatch intervals exceed their parent")
            residual.append(max(0.0, delta))
        row = {
            "case": report["case"], "ep_size": ep, "trial": report["trial"],
            "parent_mean_ms": statistics.mean(parent),
            "parts_mean_ms": {name: statistics.mean(v) for name, v in paired.items()},
            "host_parts_mean_ms": {name: statistics.mean(v) for name, v in host.items()},
            "boundary_mean_ms": statistics.mean(residual),
            "control_median_ms": statistics.median(max(d["control"]["dispatch_ms"][i] for d in data) for i in range(count)),
            "cpu_metadata_median_ms": statistics.median(max(d["cpu_metadata"]["dispatch_ms"][i] for d in data) for i in range(count)),
            "scalar_reads_per_forward": {name: statistics.mean(d["scalar_reads"][name]["aten_item_calls_per_forward"] for d in data) for name in PARTS},
        }
        groups.setdefault((row["case"], ep), []).append(row)
    rows = []
    for (case, ep), trials in sorted(groups.items()):
        row = {"case": case, "ep_size": ep, "trials": len(trials)}
        for name in ("parent_mean_ms", "boundary_mean_ms"):
            row[name] = statistics.mean(t[name] for t in trials)
        for name in ("control_median_ms", "cpu_metadata_median_ms"):
            row[name] = statistics.median(t[name] for t in trials)
        for name in ("parts_mean_ms", "host_parts_mean_ms", "scalar_reads_per_forward"):
            row[name] = {part: statistics.mean(t[name][part] for t in trials) for part in PARTS}
        row["largest_part"] = max(PARTS, key=lambda p: row["parts_mean_ms"][p])
        row["largest_part_share"] = row["parts_mean_ms"][row["largest_part"]] / row["parent_mean_ms"]
        row["cpu_metadata_speedup"] = row["control_median_ms"] / row["cpu_metadata_median_ms"]
        row["dispatch_reduction_fraction"] = 1 - row["cpu_metadata_median_ms"] / row["control_median_ms"]
        rows.append(row)
    return {"attribution": "mean of matched critical-Rank parent and child CUDA Event intervals; controls use trial median of repetition-wise max Rank medians", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/scaling/reports.json"))
    parser.add_argument("--output", type=Path, default=Path("results/scaling/dispatch_summary.json"))
    args = parser.parse_args()
    result = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
