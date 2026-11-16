"""Reject LL fallback and accidental low-precision dispatch before timing."""

from pathlib import Path
import sys
import os
import subprocess
import unittest
import json
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.deepep_native_backend import (
    validate_native_backend,
    install_buffer_init_memory_probe,
)


class NativeBackendTests(unittest.TestCase):
    def ll_proof(self):
        return dict(
            manager="DeepEPLLAll2AllManager",
            prepare_finalize="DeepEPLLPrepareAndFinalize",
            experts="BatchedTritonExperts",
            runner="MoERunner",
            tp_size=2,
            dp_size=2,
            ep_size=4,
            input_dtype="torch.bfloat16",
            use_fp8_dispatch=False,
        )

    def test_native_ll_bf16(self):
        validate_native_backend(self.ll_proof(), "deepep_low_latency")

    def test_fallback_and_changed_precision_fail(self):
        for name, value in [
            ("manager", "DeepEPHTAll2AllManager"),
            ("experts", "TritonExperts"),
            ("runner", "MiniEPRunner"),
            ("ep_size", 2),
            ("input_dtype", "torch.float16"),
            ("use_fp8_dispatch", True),
            ("use_fp8_dispatch", None),
        ]:
            with self.subTest(field=name, value=value):
                q = self.ll_proof()
                q[name] = value
                with self.assertRaises(RuntimeError):
                    validate_native_backend(q, "deepep_low_latency")

    def test_prior_ht_proof_and_unknown_backend(self):
        q = self.ll_proof()
        q.update(
            manager="DeepEPHTAll2AllManager",
            prepare_finalize="DeepEPHTPrepareAndFinalize",
            experts="TritonExperts",
        )
        q.pop("use_fp8_dispatch")
        q.pop("input_dtype")
        validate_native_backend(q, "deepep_high_throughput")
        with self.assertRaises(ValueError):
            validate_native_backend(q, "auto")

    def run_cli(self, backend, depth, world_size="0"):
        script = (
            Path(__file__).resolve().parents[1] / "benchmarks/deepep_latency_run.py"
        )
        command = [
            sys.executable,
            str(script),
            "--backend",
            backend,
            "--all2all-backend",
            "deepep_low_latency",
            "--model",
            "/unused",
            "--output",
            "/unused",
            "--checkpoint-manifest",
            "/unused",
            "--label",
            "guard",
            "--max-num-batched-tokens",
            "512",
            "--enable-chunked-prefill",
        ]
        return subprocess.run(
            command,
            env={
                **os.environ,
                "NVSHMEM_QP_DEPTH": str(depth),
                "WORLD_SIZE": world_size,
            },
            capture_output=True,
            text=True,
        )

    def test_queue_capacity_rejected_before_cuda_or_model_loading(self):
        result = self.run_cli("native", 1024)
        self.assertEqual(result.returncode, 2)
        self.assertIn("NVSHMEM_QP_DEPTH=1024 must be >= 1026", result.stderr)

    def test_sufficient_queue_reaches_torchrun_guard(self):
        result = self.run_cli("native", 2048)
        self.assertEqual(result.returncode, 2)
        self.assertIn("use torchrun with four ranks", result.stderr)

    def test_miniep_ll_combination_rejected_before_cuda(self):
        result = self.run_cli("miniep", 2048)
        self.assertEqual(result.returncode, 2)
        self.assertIn("native-only", result.stderr)

    def test_two_gpu_topology_is_explicit_not_a_four_gpu_pass(self):
        proof = self.ll_proof()
        proof.update(tp_size=1, ep_size=2)
        validate_native_backend(proof, "deepep_low_latency", (1, 2, 2))
        with self.assertRaises(RuntimeError):
            validate_native_backend(proof, "deepep_low_latency")

    def test_startup_cache_release_restores_original_constructor(self):
        state = dict(empty_calls=0, init_calls=0, reserved=9)

        def empty_cache():
            state["empty_calls"] += 1
            state["reserved"] = 3

        cuda = SimpleNamespace(
            mem_get_info=lambda: (100, 200),
            memory_allocated=lambda: 3,
            memory_reserved=lambda: state["reserved"],
            empty_cache=empty_cache,
        )

        class Buffer:
            def __init__(self, group=None, num_rdma_bytes=0):
                state["init_calls"] += 1

        original = Buffer.__init__
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules,
            {
                "torch": SimpleNamespace(cuda=cuda),
                "deep_ep": SimpleNamespace(Buffer=Buffer),
            },
        ), patch.dict(os.environ, {"MINIEP_BENCH_DIR": directory}):
            install_buffer_init_memory_probe()
            Buffer(num_rdma_bytes=64)
            self.assertIs(Buffer.__init__, original)
            Buffer(num_rdma_bytes=64)
            self.assertEqual(state["empty_calls"], 1)
            self.assertEqual(state["init_calls"], 2)
            record = json.loads(
                (Path(directory) / "buffer-init-memory.json").read_text()
            )
            self.assertTrue(record["completed"] and record["init_hook_removed"])
            self.assertEqual(
                record["before"]["allocated"],
                record["after_cache_release"]["allocated"],
            )
            self.assertEqual(record["before"]["reserved"], 9)
            self.assertEqual(record["after_cache_release"]["reserved"], 3)


if __name__ == "__main__":
    unittest.main()
