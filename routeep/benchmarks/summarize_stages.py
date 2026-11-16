"""Summarize the separate event-profile jobs, never primary latency samples.

CUDA Event intervals include dependencies and CPU launch gaps. They are not
exclusive GPU kernel times. Native DeepEP exposes whole-MoE intervals only.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics

def layer_index(name):
    match = re.search(r"\.layers\.(\d+)\.", name)
    if match is None:
        raise ValueError(f"unrecognized layer: {name}")
    return int(match[1])


def summarize(root: Path, mini: bool) -> dict:
    grouped = defaultdict(list)
    errors = []
    expected_shape = None
    for rank in range(4):
        path = root / f"rank{rank}"
        environment = json.loads((path / "environment.json").read_text())
        if environment["profile"] != "events":
            raise ValueError("stage summary requires the separate events job")
        shape = environment.get("model_parameters", {})
        layers = shape.get("num_hidden_layers", 48)
        hidden_size = shape.get("hidden_size", 2048)
        if not isinstance(layers, int) or layers < 1 or not isinstance(hidden_size, int) or hidden_size < 1:
            raise ValueError(f"invalid model shape: {shape}")
        if expected_shape is None:
            expected_shape = (layers, hidden_size)
        elif expected_shape != (layers, hidden_size):
            raise ValueError("profile ranks have different model shapes")
        selected = {0, layers // 2, layers - 1}
        cases = {q["name"]: q for q in json.loads((path / "manifest.json").read_text())}
        for sample in json.loads((path / "samples.json").read_text()):
            profile = sample["profiling"]
            if profile is None:
                raise ValueError("profiling records missing")
            case = cases[sample["case"]]
            steps = case["steps"]
            moe = defaultdict(list)
            for q in profile["moe"]:
                moe[layer_index(q["layer"])].append(q["milliseconds"])
            if set(moe) != set(range(layers)):
                raise ValueError(
                    f'incomplete MoE layers rank={rank} case={case["name"]}'
                )
            split = all(len(x) == steps for x in moe.values())
            if not split:
                errors.append(
                    f'cannot attribute model steps to prefill/decode rank={rank} case={case["name"]}'
                )
            measurements = {}

            def values(label, data):
                flat = [v for xs in data for v in xs]
                if not flat:
                    return
                if any(not math.isfinite(v) or v < 0 for v in flat):
                    raise ValueError("invalid event interval")
                measurements[("all_calls", label)] = statistics.mean(flat)
                if split:
                    measurements[("prefill", label)] = statistics.mean(
                        xs[0] for xs in data
                    )
                    if steps > 1:
                        measurements[("decode", label)] = statistics.mean(
                            v for xs in data for v in xs[1:]
                        )

            values(f"whole_moe_all{layers}", [moe[i] for i in range(layers)])
            values("whole_moe_selected3", [moe[i] for i in sorted(selected)])
            payloads = defaultdict(list)
            if mini:
                stages = defaultdict(lambda: defaultdict(list))
                for q in profile["stages"]:
                    stages[q["stage"]][layer_index(q["layer"])].append(
                        q["milliseconds"]
                    )
                expected = {
                    "ROUTER",
                    "DISPATCH",
                    "A2A_FORWARD",
                    "EXPERT",
                    "A2A_RETURN",
                    "COMBINE",
                }
                if shape.get("model_family") == "qwen3_5_moe":
                    expected.update(("SHARED_EXPERT", "SHARED_COMBINE"))
                if set(stages) != expected:
                    raise ValueError(f"incomplete stages: {set(stages)}")
                for stage, by_layer in stages.items():
                    if set(by_layer) != selected or any(
                        len(x) != len(moe[layer]) for layer, x in by_layer.items()
                    ):
                        raise ValueError("stage and whole-MoE coverage differ")
                    values(stage, [by_layer[i] for i in sorted(selected)])
                traffic = defaultdict(list)
                for q in profile["traffic"]:
                    traffic[layer_index(q["layer"])].append(q)
                if set(traffic) != selected or any(
                    len(x) != len(moe[layer]) for layer, x in traffic.items()
                ):
                    raise ValueError("incomplete traffic coverage")
                for layer in sorted(selected):
                    for index, q in enumerate(traffic[layer]):
                        if (
                            q["return_dtype"] != "torch.bfloat16"
                            or q["model_dtype"] != "torch.bfloat16"
                        ):
                            raise ValueError("wrong measured dtype")
                        ep = q["ep_rank"]
                        h = hidden_size
                        if q["return_send_bytes"] != q["recv_rows"] * h * 2:
                            raise ValueError("return byte count mismatch")
                        payload = dict(
                            forward_total=q["send_rows"] * (h * 2 + 8 + 4),
                            return_total=q["return_send_bytes"],
                            forward_cross_rank=(q["send_rows"] - q["send_splits"][ep])
                            * (h * 2 + 8 + 4),
                            return_cross_rank=(q["recv_rows"] - q["recv_splits"][ep])
                            * h
                            * 2,
                            count_total=0 if q["layout"] == "fixed" else 4 * 8,
                        )
                        for label, amount in payload.items():
                            payloads[("all_calls", label)].append(amount)
                            if split:
                                payloads[
                                    ("prefill" if index == 0 else "decode", label)
                                ].append(amount)
            grouped[(case["name"], sample["repetition"])].append(
                (rank, measurements, payloads, split)
            )
    intervals = defaultdict(list)
    traffic = defaultdict(list)
    for (case, rep), rows in grouped.items():
        if len(rows) != 4 or {x[0] for x in rows} != set(range(4)):
            raise ValueError("missing rank in profile repetition")
        keys = set(rows[0][1])
        if any(set(q[1]) != keys for q in rows):
            raise ValueError("phase attribution differs across ranks")
        for phase, label in keys:
            intervals[(case, phase, label)].append(
                max(q[1][phase, label] for q in rows)
            )
        pkeys = set(rows[0][2])
        if any(set(q[2]) != pkeys for q in rows):
            raise ValueError("payload coverage differs across ranks")
        for phase, label in pkeys:
            traffic[(case, phase, label)].append(
                max(statistics.mean(q[2][phase, label]) for q in rows)
            )
    result = {}
    for table, name in (
        (intervals, "milliseconds_per_layer"),
        (traffic, "payload_bytes_per_layer"),
    ):
        for (case, phase, label), samples in table.items():
            result.setdefault(case, {}).setdefault(phase, {}).setdefault(name, {})[
                label
            ] = dict(
                median=statistics.median(samples),
                minimum=min(samples),
                maximum=max(samples),
                repeats=len(samples),
            )
    return dict(
        cases=result,
        phase_attribution_errors=sorted(set(errors)),
        native_detailed_stages="not observed",
        selected_layers=sorted({0, expected_shape[0] // 2, expected_shape[0] - 1}),
        statistic="per-rank mean over observed layer calls, per-repetition maximum rank, then median across repetitions",
        payload_scope="layout payload including padding; cross_rank excludes self; no protocol overhead or native DeepEP bytes inferred",
        timing_scope="separate profiling only; intervals include dependencies and host submission gaps; no overlap conclusion",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--miniep", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = dict(native=summarize(a.native, False), miniep=summarize(a.miniep, True))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
