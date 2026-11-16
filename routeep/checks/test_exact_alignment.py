"""零误差验收门的 CPU 回归；这些测试不证明真实 GPU 对齐。

覆盖 1 ULP、NaN/Inf、空张量、非连续输入、shape/dtype 差异，以及报告汇总
在缺失证据时必须失败的行为。
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from validation.exact import (
    exact_tensor_metrics,
    exact_logits_metrics,
    exact_tensor_flag,
)
from validation.alignment_report import summarize_rank, summarize_run, references_match


class ExactAcceptanceTests(unittest.TestCase):
    def test_single_ulp_is_failure_even_when_argmax_matches(self):
        a = torch.tensor([[1.0, 2.0]])
        b = a.clone()
        b[0, 0] = torch.nextafter(b[0, 0], torch.tensor(2.0))
        torch.testing.assert_close(a, b)
        q = exact_logits_metrics(a, b)
        self.assertFalse(q["passed"])
        self.assertEqual(q["argmax_matches"], 1)
        self.assertEqual(q["first_difference"]["index"], [0, 0])

    def test_contract_nonfinite_empty_and_noncontiguous(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            x = torch.tensor([value])
            self.assertFalse(exact_tensor_metrics(x, x)["exact_equal"])
            json.dumps(exact_tensor_metrics(x, x), allow_nan=False)
        self.assertFalse(
            exact_tensor_metrics(torch.ones(1), torch.ones(1).bfloat16())["exact_equal"]
        )
        x = torch.arange(12.0).reshape(3, 4)[:, ::2]
        self.assertTrue(exact_tensor_metrics(x, x.clone())["exact_equal"])
        self.assertTrue(exact_tensor_metrics(x[:0], x[:0])["exact_equal"])

    def test_device_predicate_matches_detailed_exact_check(self):
        for dtype in (torch.bfloat16, torch.float32, torch.int32):
            x = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
            for y in (x.clone(), x + 1, x[:1], x.float()):
                self.assertEqual(
                    bool(exact_tensor_flag(x, y)),
                    exact_tensor_metrics(x, y)["exact_equal"],
                )
        for value in (float("inf"), float("nan")):
            x = torch.tensor([value])
            self.assertFalse(bool(exact_tensor_flag(x, x)))
        x = torch.empty(0, 2)
        self.assertTrue(bool(exact_tensor_flag(x, x)))

    def test_reference_reuse_freezes_compute_but_records_observers_separately(self):
        from copy import deepcopy

        x = dict(
            signature="old",
            spec=dict(
                options={"backend": "native"},
                checkpoint="weights",
                versions={"vllm": "0.29.0"},
                source_sha256={
                    "mini_ep/model/moe.py": "compute",
                    "validation/exact.py": "observer-v1",
                },
            ),
        )
        y = deepcopy(x)
        y["signature"] = "new"
        y["spec"]["source_sha256"]["validation/exact.py"] = "observer-v2"
        self.assertTrue(references_match(x, y))
        y["spec"]["source_sha256"]["mini_ep/model/moe.py"] = "different-compute"
        self.assertFalse(references_match(x, y))
        y = deepcopy(x)
        y["spec"]["options"]["backend"] = "another-backend"
        self.assertFalse(references_match(x, y))

    def test_audited_native_reuse_still_freezes_every_inference_option(self):
        from copy import deepcopy

        native = dict(
            kind="native",
            rank=0,
            signature="golden",
            spec=dict(
                options={"dtype": "bfloat16"},
                versions={"vllm": "0.29.0"},
                checkpoint="weights",
                source_sha256={"mini_ep/model/moe.py": "unused-old"},
            ),
        )
        candidate = deepcopy(native)
        candidate.update(kind="candidate", signature="new")
        candidate["spec"]["source_sha256"]["mini_ep/model/moe.py"] = "new-compute"
        self.assertFalse(references_match(candidate, native))
        candidate.update(
            native_runtime_digest="binary-hash",
            native_reference_audit=dict(
                passed=True,
                native_runtime_digest="binary-hash",
                native_signatures={"0": "golden"},
                miniep_results_reusable=False,
            ),
        )
        self.assertTrue(references_match(candidate, native))
        changed = deepcopy(candidate)
        changed["spec"]["options"]["dtype"] = "float16"
        self.assertFalse(references_match(changed, native))
        changed = deepcopy(candidate)
        changed["native_runtime_digest"] = "changed-binary"
        self.assertFalse(references_match(changed, native))
        changed = deepcopy(candidate)
        changed["native_reference_audit"]["native_signatures"]["0"] = "another-golden"
        self.assertFalse(references_match(changed, native))

    def test_idle_peer_workloads_require_two_dp_replicas(self):
        from validation.alignment_cases import build_cases

        class Tokenizer:
            def apply_chat_template(self, messages, **kw):
                return messages[0]["content"]

            def encode(self, text, **kw):
                return list(range(len(text.split())))

        tokenizer = Tokenizer()
        single = build_cases(tokenizer, "boundaries", 0, "rotated", 32, 64, dp_size=1)
        double = build_cases(tokenizer, "boundaries", 0, "rotated", 32, 64, dp_size=2)
        self.assertFalse(any(c["name"].startswith("idle_dp") for c in single))
        self.assertEqual(
            {c["name"] for c in double if c["name"].startswith("idle_dp")},
            {"idle_dp0", "idle_dp1"},
        )
        self.assertIn("staggered_finish", {c["name"] for c in single})

    def fixture(self, root):
        def write(p, x):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(x))

        ref = root / "native"
        candidate = root / "candidate"
        cases = [dict(name="case", prompts=[{}], steps=2)]
        outputs = [dict(case="case", index=0, token_ids=[1, 2])]
        proof = dict(
            manager="DeepEPHTAll2AllManager",
            prepare_finalize="DeepEPHTPrepareAndFinalize",
            experts="TritonExperts",
            ep_size=4,
        )
        for folder, kind in ((ref, "native"), (candidate, "candidate")):
            write(
                folder / "environment.json",
                dict(
                    kind=kind,
                    signature="frozen",
                    rank=0,
                    layers=2,
                    hidden_size=3,
                    backend="deepep_high_throughput",
                    world_size=4,
                    synthetic=False,
                ),
            )
            write(folder / "manifest.json", cases)
            write(
                folder / "backend.json",
                dict(
                    proof,
                    runner="MoERunner" if kind == "native" else "DeepEPExactRunner",
                ),
            )
            phases = (
                ["native"]
                if kind == "native"
                else ["native_check", "compare", "teacher", "free"]
            )
            write(folder / "completed.json", {phase: ["case"] for phase in phases})
            for phase in phases:
                write(folder / phase / "outputs.json", outputs)
        records = [
            dict(key="case", layer=layer) for step in range(2) for layer in ("0", "1")
        ]
        (ref / "native-moe.jsonl").write_text(
            "".join(json.dumps(q) + "\n" for q in records)
        )
        write(ref / "native-logits.json", dict(case=2))
        records = [
            dict(
                key="case",
                phase=phase,
                layer=layer,
                exact_equal=True,
                router_exact=True,
                model_dtype="torch.bfloat16",
                return_dtype="torch.bfloat16",
                return_buffer_dtype="torch.bfloat16",
                return_dtype_requested="auto",
                preweighted=True,
                fuse_return_weight=True,
                owner_reduction="local_fp32",
                recv_rows=2,
                return_send_bytes=12,
            )
            for phase in ("compare", "teacher", "free")
            for step in range(2)
            for layer in ("0", "1")
        ]
        (candidate / "moe-rank0.jsonl").write_text(
            "".join(json.dumps(q) + "\n" for q in records)
        )
        for name in ("native-check", "logits"):
            (candidate / f"{name}-rank0.jsonl").write_text(
                "".join(
                    json.dumps(
                        dict(
                            key="case",
                            step=i,
                            rows=1,
                            finite=True,
                            exact_equal=True,
                            argmax_matches=1,
                        )
                    )
                    + "\n"
                    for i in range(2)
                )
            )
        return ref, candidate

    def test_missing_later_layer_execution_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, candidate = self.fixture(Path(tmp))
            self.assertTrue(summarize_rank(ref)["passed"])
            self.assertTrue(summarize_rank(candidate, ref)["passed"])
            path = candidate / "moe-rank0.jsonl"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:2] + lines[4:]) + "\n")
            self.assertFalse(summarize_rank(candidate, ref)["passed"])

    def test_explicit_bf16_diagnostic_does_not_prove_auto_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, candidate = self.fixture(Path(tmp))
            self.assertTrue(summarize_rank(candidate, ref)["passed"])
            path = candidate / "moe-rank0.jsonl"
            text = path.read_text()
            path.write_text(
                text.replace(
                    '"return_dtype_requested": "auto"',
                    '"return_dtype_requested": "torch.bfloat16"',
                    1,
                )
            )
            self.assertFalse(summarize_rank(candidate, ref)["passed"])

    def _rewrite_moe(self, candidate, **fields):
        path = candidate / "moe-rank0.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        for q in rows:
            q.update(fields)
        path.write_text("".join(json.dumps(q) + "\n" for q in rows))

    def test_owner_reduction_label_follows_transport(self):
        """三种实现的归约标签不同但规则相同；标签必须与实际 transport/去重匹配。"""
        cases = [
            (dict(owner_reduction="local_fp32"), True),
            (dict(owner_reduction="destination_fp32", deduplicated=True), True),
            (dict(owner_reduction="slot_order_fp32_then_bf16", deduplicated=True,
                  transport="peer", recv_rows=None, return_send_bytes=None), True),
            # 标签与路径不符时必须失败，不能因为放宽判定而接受任意值。
            (dict(owner_reduction="destination_fp32"), False),
            (dict(owner_reduction="local_fp32", deduplicated=True), False),
            (dict(owner_reduction="destination_fp32", deduplicated=True, transport="peer"), False),
        ]
        for fields, ok in cases:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as tmp:
                ref, candidate = self.fixture(Path(tmp))
                self._rewrite_moe(candidate, **fields)
                self.assertEqual(summarize_rank(candidate, ref)["passed"], ok)

    def test_nccl_return_bytes_still_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref, candidate = self.fixture(Path(tmp))
            self._rewrite_moe(candidate, owner_reduction="destination_fp32",
                              deduplicated=True, return_send_bytes=13)
            self.assertFalse(summarize_rank(candidate, ref)["passed"])

    def test_missing_rank_backend_fallback_and_failure_artifact_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref, candidate = self.fixture(root)
            self.assertFalse(summarize_run(root)["passed"])
            proof = json.loads((candidate / "backend.json").read_text())
            proof["manager"] = "AgRsAll2AllManager"
            (candidate / "backend.json").write_text(json.dumps(proof))
            self.assertFalse(summarize_rank(candidate, ref)["passed"])
            (candidate / "backend.json").write_text((ref / "backend.json").read_text())
            (candidate / "first-failure").mkdir()
            self.assertFalse(summarize_rank(candidate, ref)["passed"])


if __name__ == "__main__":
    unittest.main()
