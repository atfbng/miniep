"""Inspect captured failure buffers without changing native kernels or thresholds.

Run on CPU against a failed run directory. Branches are joined by exact input
bytes, local expert and FP32 routing weight, since DeepEP deduplicates incoming
rows whereas Mini-EP expands Top-K. Unmatched rows fail diagnosis coverage.
Native FP32 down registers are not observable; captured post-weight branches
are the first directly comparable arithmetic boundary.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from validation.exact import exact_tensor_metrics


def row_bytes(x):
    return bytes(x.contiguous().view(torch.uint8).tolist())


def inspect_rank(path: Path) -> dict:
    q = torch.load(path, map_location="cpu", weights_only=True)
    native = q.get("native_expert_inputs", {})
    mini = q.get("mini_expert_inputs", {})
    report = dict(
        output=exact_tensor_metrics(q["mini_output"], q["native_output"]),
        available=list(q),
        native_fp32_registers="not observable; no claimed comparison",
    )
    if not native or not mini or "native_down_weighted" not in q:
        report["diagnosis"] = "native weighted branch capture unavailable"
        return report
    x = native["hidden_states"]
    ids = native["topk_ids"]
    weights = native["topk_weights"]
    branches = q["native_down_weighted"].reshape(ids.shape[0], ids.shape[1], -1)
    expert_map = native.get("expert_map")
    lookup = defaultdict(list)
    for row in range(ids.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[row, slot])
            if expert < 0:
                continue
            if expert_map is not None:
                expert = int(expert_map[expert])
            if expert < 0:
                continue
            key = (row_bytes(x[row]), expert, row_bytes(weights[row, slot].reshape(1)))
            lookup[key].append(branches[row, slot])
    observed = q["mini_weighted_branches_received_order"]
    first = None
    missing = 0
    compared = 0
    for i in range(mini["tokens"].shape[0]):
        expert = int(mini["local_ids"][i])
        if expert < 0:
            continue
        key = (
            row_bytes(mini["tokens"][i]),
            expert,
            row_bytes(mini["branch_weights"][i].reshape(1)),
        )
        matches = lookup.get(key, [])
        if not matches:
            missing += 1
            continue
        stats = exact_tensor_metrics(observed[i], matches[0])
        compared += 1
        if not stats["exact_equal"] and first is None:
            first = dict(received_row=i, local_expert=expert, **stats)
    report["weighted_branches"] = dict(
        compared=compared,
        unmatched=missing,
        first_difference=first,
        exact_equal=compared > 0 and missing == 0 and first is None,
    )
    report["diagnosis"] = (
        "weighted branch differs"
        if first
        else (
            "weighted branches match; inspect owner and final reduction"
            if missing == 0
            else "incomplete branch join; no arithmetic conclusion"
        )
    )
    # Reconstruct each source-token's owner partial from returned Top-K values.
    owner = q.get("native_owner_partials_received_order")
    parts = q.get("mini_owner_partials_replay")
    if owner is not None and parts is not None:
        native_rows = defaultdict(list)
        for i in range(x.shape[0]):
            native_rows[row_bytes(x[i])].append(owner[i])
        source = q["input"]
        rank_summary = json.loads((path.parent / "summary.json").read_text())
        ep_rank = rank_summary["ep_rank"]
        # On an owner rank the source input is only its own shard. Remote-source
        # rows live in other rank dumps and are inspected by the all-rank join.
        report["owner_capture"] = dict(
            native_rows=owner.shape[0],
            mini_sources=parts.shape[1],
            ep_rank=ep_rank,
            requires_cross_rank_join=True,
        )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    results = {}
    for path in sorted(Path(a.run).glob("rank*/first-failure/tensors.pt")):
        results[path.parent.parent.name] = inspect_rank(path)
    if not results:
        raise RuntimeError("no captured failed-layer tensors")
    Path(a.output).write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps({k: v["diagnosis"] for k, v in results.items()}))


if __name__ == "__main__":
    main()
