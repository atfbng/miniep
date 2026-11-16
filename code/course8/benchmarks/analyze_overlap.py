"""NCCL/GEMM interval unions within synchronized PIPELINE_PROFILE scopes."""

from __future__ import annotations
import argparse, json, sqlite3
from pathlib import Path


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersection_duration(a, b):
    i = j = total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    connection = sqlite3.connect(path)
    strings = dict(connection.execute("select id,value from StringIds"))
    scopes = []
    for start, end, text, textid, tid in connection.execute(
        "select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null"
    ):
        if (text or strings.get(textid)) == "PIPELINE_PROFILE":
            scopes.append((start, end, tid & ~((1 << 24) - 1)))
    if not scopes:
        raise ValueError("no synchronized PIPELINE_PROFILE NVTX scopes")
    rows = list(
        connection.execute(
            "select start,end,globalPid,demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
        )
    )
    result = []
    for start, end, pid in scopes:
        comm = []
        compute = []
        for a, b, process, nameid in rows:
            if process != pid or b <= start or a >= end:
                continue
            name = strings[nameid].lower()
            interval = (max(start, a), min(end, b))
            if "nccl" in name:
                comm.append(interval)
            # cuBLAS FP16 Hopper kernels can use nvjet_* names without "gemm".
            elif "gemm" in name or name.startswith("nvjet_"):
                compute.append(interval)
        comm, compute = merge_intervals(comm), merge_intervals(compute)
        communication = sum(b - a for a, b in comm)
        computation = sum(b - a for a, b in compute)
        overlap = intersection_duration(comm, compute)
        result.append(
            {
                "global_pid": pid,
                "scope_ms": (end - start) / 1e6,
                "communication_ms": communication / 1e6,
                "gemm_ms": computation / 1e6,
                "overlap_ms": overlap / 1e6,
                "exposed_communication_ms": (communication - overlap) / 1e6,
                "overlap_ratio": overlap / communication if communication else 0,
            }
        )
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.sqlite)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
