"""四 Rank 精确验收：任何缺失证据都不能被当成通过。

报告分别记录配置、源码身份、逐 Rank 数值和生成序列；只要有一个必需字段
缺失或不一致，整体结论就是失败。
"""

from collections import Counter, defaultdict
import json
from pathlib import Path


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path: Path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)


def references_match(actual: dict, expected: dict) -> bool:
    """把“计算/配置”与“观测器实现”分开冻结。

    两份完整 observer 源码清单都会保留。只修改测试侧统计代码时，可以复用独立的
    原始 MoERunner Reference；但模型权重、软件版本、启动选项、workload 和所有
    mini_ep 源码必须完全一致。
    """
    if "spec" not in actual or "spec" not in expected:
        return actual.get("signature") == expected.get("signature")

    def frozen(env):
        spec = dict(env["spec"])
        spec["source_sha256"] = {
            k: v for k, v in spec["source_sha256"].items() if k.startswith("mini_ep/")
        }
        return spec

    a, b = frozen(actual), frozen(expected)
    if a == b:
        return True
    audit = actual.get("native_reference_audit", {})
    if (
        actual.get("kind") != "candidate"
        or expected.get("kind") != "native"
        or not audit.get("passed")
        or not audit.get("native_runtime_digest")
        or audit.get("native_runtime_digest") != actual.get("native_runtime_digest")
        or audit.get("native_signatures", {}).get(str(actual.get("rank")))
        != expected.get("signature")
    ):
        return False
    # 审计复用 native 基线时，只允许未使用的 RouteEP/observer 源码哈希不同；
    # 模型、选项、版本、workload 和 seed 必须完全冻结，不设容差。
    a.pop("source_sha256")
    b.pop("source_sha256")
    return a == b


def output_map(path: Path, cases: list[dict], errors: list[str]) -> dict:
    expected = {
        (case["name"], i): case.get(
            "request_steps", [case["steps"]] * len(case["prompts"])
        )[i]
        for case in cases
        for i in range(len(case["prompts"]))
    }
    records = load(path, [])
    actual = {(q["case"], q["index"]): q["token_ids"] for q in records}
    if len(records) != len(actual):
        errors.append(f"duplicate output {path}")
    if set(actual) != set(expected) or any(
        len(actual.get(k, [])) != n for k, n in expected.items()
    ):
        errors.append(f"incomplete output {path}")
    return actual


def summarize_rank(root: Path, reference: Path | None = None) -> dict:
    errors = []
    env = load(root / "environment.json", {})
    cases = load(root / "manifest.json", [])
    completed = load(root / "completed.json", {})
    proof = load(root / "backend.json", {})
    kind = env.get("kind")
    if not cases:
        errors.append("missing workload manifest")
    if not env or not env.get("signature"):
        errors.append("missing frozen environment signature")
    if env.get("synthetic", False):
        errors.append("synthetic model cannot pass full acceptance")
    expected = (
        ("DeepEPHTAll2AllManager", "DeepEPHTPrepareAndFinalize")
        if env.get("backend") == "deepep_high_throughput"
        else None
    )
    if (
        expected is None
        or (proof.get("manager"), proof.get("prepare_finalize")) != expected
        or proof.get("experts") != "TritonExperts"
        or proof.get("ep_size") != env.get("world_size")
    ):
        errors.append("actual native backend not verified")
    if kind == "native" and proof.get("runner") != "MoERunner":
        errors.append("independent native Runner missing or replaced")
    if kind not in ("native", "candidate"):
        errors.append("missing run kind")
    rank = env.get("rank", -1)
    layers = env.get("layers", 48)
    expected_positions = {
        q["name"]: sum(q.get("request_steps", [q["steps"]] * len(q["prompts"])))
        for q in cases
    }
    expected_names = set(expected_positions)
    phases = (
        ("native",)
        if kind == "native"
        else ("native_check", "compare", "teacher", "free")
    )
    outputs = {}
    for phase in phases:
        if set(completed.get(phase, [])) != expected_names:
            errors.append(f"incomplete phase {phase}")
        outputs[phase] = output_map(root / phase / "outputs.json", cases, errors)
    if kind == "native":
        counts = defaultdict(Counter)
        for q in rows(root / "native-moe.jsonl"):
            counts[q["key"]][q["layer"]] += 1
        for case in expected_names:
            if len(counts[case]) != layers or len(set(counts[case].values())) != 1:
                errors.append(f"native layer coverage {case}")
        # 每个实际参与采样的逻辑位置都必须在全词表基线中存在，
        # 与引擎内部用了多少步无关。
        capture = load(root / "native-logits.json", {})
        if capture != expected_positions:
            errors.append("native logits coverage mismatch")
    elif kind == "candidate":
        if reference is None:
            errors.append("missing independent native reference")
        ref_cases = load(reference / "manifest.json", []) if reference else []
        if ref_cases != cases:
            errors.append("reference workload differs")
        ref_env = load(reference / "environment.json", {}) if reference else {}
        if not references_match(env, ref_env):
            errors.append("reference configuration/source differs")
        ref_outputs = (
            output_map(reference / "native/outputs.json", cases, errors)
            if reference
            else {}
        )
        for phase, actual in outputs.items():
            if actual != ref_outputs:
                errors.append(f"sequence mismatch {phase}")
        coverage = {
            phase: defaultdict(Counter) for phase in ("compare", "teacher", "free")
        }
        first = None
        for q in rows(root / f"moe-rank{rank}.jsonl"):
            phase = q.get("phase")
            if phase not in coverage:
                errors.append("unexpected MoE phase")
                continue
            coverage[phase][q["key"]][q["layer"]] += 1
            if (
                q.get("model_dtype") != "torch.bfloat16"
                or q.get("return_dtype") != q.get("model_dtype")
                or q.get("return_buffer_dtype") != q.get("model_dtype")
                or q.get("return_dtype_requested") != "auto"
                or q.get("preweighted") is not True
                or q.get("fuse_return_weight") is not True
            ):
                errors.append("candidate transport policy was not exercised")
            # 三种实现都执行同一条 DeepEP 精度规则（分支先舍入到 BF16，按 slot 顺序
            # FP32 累加，owner 舍入一次），只是标签不同：逐分支 NCCL 为 local_fp32，
            # NCCL 去重为 destination_fp32，Peer 去重为 slot_order_fp32_then_bf16。
            transport = q.get("transport", "nccl")
            expected_reduction = (
                {"slot_order_fp32_then_bf16"} if transport == "peer"
                else {"destination_fp32"} if q.get("deduplicated")
                else {"local_fp32"}
            )
            if q.get("owner_reduction") not in expected_reduction:
                errors.append("owner reduction differs from selected baseline")
            # Peer 直接写目标显存，没有 Host 侧回传行数；字节检查只适用于 NCCL A2A。
            if transport != "peer" and (
                q.get("return_send_bytes")
                != q.get("recv_rows", 0) * env.get("hidden_size", 2048) * 2
            ):
                errors.append("actual return buffer is not two bytes per value")
            if phase == "compare" and (
                not q.get("exact_equal", False) or not q.get("router_exact", False)
            ):
                if first is None:
                    first = {
                        k: q.get(k)
                        for k in (
                            "key",
                            "layer",
                            "first_difference",
                            "router_exact",
                            "max_abs",
                        )
                    }
        for phase, by_case in coverage.items():
            for case in expected_names:
                count = by_case[case]
                if len(count) != layers or len(set(count.values())) != 1:
                    errors.append(f"MoE coverage mismatch {phase} {case}")
        for label, file in [
            ("native_check", f"native-check-rank{rank}.jsonl"),
            ("teacher", f"logits-rank{rank}.jsonl"),
        ]:
            counts = Counter()
            seen = set()
            for q in rows(root / file):
                step = (q["key"], q["step"])
                if step in seen:
                    errors.append(f"duplicate logits {label}")
                seen.add(step)
                counts[q["key"]] += q["rows"]
                if not q.get("exact_equal", False) or not q.get("finite", False):
                    errors.append(f"logits mismatch {label}")
                if q.get("argmax_matches") != q.get("rows"):
                    errors.append(f"argmax mismatch {label}")
            if (
                any(counts[k] != v for k, v in expected_positions.items())
                or set(counts) - expected_names
            ):
                errors.append(f"logits position coverage {label}")
        if first:
            errors.append(f"first MoE mismatch: {first}")
    if (root / "first-failure").exists() or (root / "driver-failure.json").exists():
        errors.append("failure artifact present")
    # 空闲 DP Rank 也必须执行所有包含集合通信的模型步；
    # 有实际采样的 Decode 位置至少需要一次模型调用。
    for case in cases:
        if case["prompts"]:
            tables = (
                [counts]
                if kind == "native"
                else list(coverage.values()) if kind == "candidate" else []
            )
            for table in tables:
                if min(table[case["name"]].values(), default=0) < max(
                    case.get("request_steps", [case["steps"]])
                ):
                    errors.append(f'incomplete model steps {case["name"]}')
    model_calls = {
        phase: {case: dict(table[case]) for case in expected_names}
        for phase, table in (
            {"native": counts}
            if kind == "native"
            else coverage if kind == "candidate" else {}
        ).items()
    }
    expected_sequences = sum(len(q["prompts"]) for q in cases)
    return dict(
        passed=not errors,
        rank=rank,
        kind=kind,
        signature=env.get("signature"),
        errors=sorted(set(errors)),
        model_calls=model_calls,
        expected_sequences=expected_sequences,
        expected_positions=sum(expected_positions.values()),
    )


def summarize_run(
    root: Path, reference: Path | None = None, world_size: int = 4
) -> dict:
    reports = {}
    for rank in range(world_size):
        reports[str(rank)] = summarize_rank(
            root / f"rank{rank}", reference / f"rank{rank}" if reference else None
        )
    signatures = {q["signature"] for q in reports.values()}
    for rank, q in reports.items():
        if (
            q["rank"] != int(rank)
            or load(root / f"rank{rank}" / "environment.json", {}).get("world_size")
            != world_size
        ):
            q["passed"] = False
            q["errors"].append("rank/world-size identity differs")
    errors = []
    if len(signatures) != 1 or None in signatures:
        errors.append("rank signatures differ or missing")
    if reports:
        first = reports["0"]["model_calls"]
        if any(q["model_calls"] != first for q in reports.values()):
            errors.append("collective-bearing model call coverage differs across ranks")
    # TP 对等 Rank 的清单和序列必须完全一致；只有 TP leader
    # （全局 rank 0、2）参与唯一请求数统计。
    for a, b in ((i, i + 1) for i in range(0, world_size, 2)):
        for relative in (
            "manifest.json",
            "native/outputs.json",
            "native_check/outputs.json",
            "free/outputs.json",
        ):
            left = load(root / f"rank{a}" / relative)
            right = load(root / f"rank{b}" / relative)
            if left != right:
                errors.append(f"TP peers differ {a}/{b} {relative}")
    return dict(
        passed=not errors and all(q["passed"] for q in reports.values()),
        errors=errors,
        ranks=reports,
        request_sequences=sum(
            reports[str(i)]["expected_sequences"] for i in range(0, world_size, 2)
        ),
        generation_positions=sum(
            reports[str(i)]["expected_positions"] for i in range(0, world_size, 2)
        ),
    )
