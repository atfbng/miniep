"""Unit and optional two-GPU integration tests for lesson 4."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from mini_ep.distributed import (
    equal_split_all_to_all,
    read_torchrun_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestTorchrunEnvironment(unittest.TestCase):
    def test_parses_rank_metadata(self) -> None:
        settings = read_torchrun_environment(
            {"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}
        )

        self.assertEqual(settings.global_rank, 3)
        self.assertEqual(settings.local_rank, 1)
        self.assertEqual(settings.world_size, 8)

    def test_missing_variable_points_to_torchrun(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "LOCAL_RANK.*torchrun"):
            read_torchrun_environment({"RANK": "0", "WORLD_SIZE": "2"})

    def test_rejects_invalid_rank_metadata(self) -> None:
        invalid_environments = (
            ({"RANK": "zero", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}, "RANK"),
            ({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "0"}, "WORLD_SIZE"),
            ({"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}, "RANK"),
            ({"RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "2"}, "LOCAL_RANK"),
        )
        for environment, message in invalid_environments:
            with self.subTest(environment=environment):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    read_torchrun_environment(environment)


class TestEqualSplitAllToAll(unittest.TestCase):
    def test_preserves_shape_dtype_device_and_uses_sync_collective(self) -> None:
        send_buffer = torch.arange(8, dtype=torch.int64).reshape(4, 2)

        def copy_collective(
            output: torch.Tensor,
            input_tensor: torch.Tensor,
            *,
            group: object,
            async_op: bool,
        ) -> None:
            self.assertIs(group, sentinel_group)
            self.assertFalse(async_op)
            output.copy_(input_tensor)

        sentinel_group = object()
        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
            mock.patch.object(dist, "all_to_all_single", side_effect=copy_collective),
        ):
            recv_buffer = equal_split_all_to_all(
                send_buffer,
                group=sentinel_group,  # type: ignore[arg-type]
            )

        self.assertEqual(recv_buffer.shape, send_buffer.shape)
        self.assertEqual(recv_buffer.dtype, send_buffer.dtype)
        self.assertEqual(recv_buffer.device, send_buffer.device)
        self.assertTrue(recv_buffer.is_contiguous())
        self.assertNotEqual(recv_buffer.data_ptr(), send_buffer.data_ptr())
        torch.testing.assert_close(recv_buffer, send_buffer)

    def test_rejects_uninitialized_scalar_strided_and_uneven_inputs(self) -> None:
        with mock.patch.object(dist, "is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                equal_split_all_to_all(torch.ones(2))

        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
        ):
            with self.assertRaisesRegex(ValueError, "scalar"):
                equal_split_all_to_all(torch.tensor(1.0))
            with self.assertRaisesRegex(ValueError, "contiguous"):
                equal_split_all_to_all(torch.ones(2, 4).transpose(0, 1))
            with self.assertRaisesRegex(ValueError, "divide evenly"):
                equal_split_all_to_all(torch.ones(3, 2))


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
    "two CUDA GPUs and NCCL are required",
)
class TestTwoGpuNcclIntegration(unittest.TestCase):
    def test_torchrun_example_checks_all_collective_values(self) -> None:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "test/demo.py",
            "--expected-world-size=2",
            "--timeout-seconds=60",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )

        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, msg=output)
        self.assertIn("global_rank=0", output)
        self.assertIn("global_rank=1", output)
        self.assertEqual(output.count("PASS"), 2, msg=output)
        self.assertIn("all_reduce=[3, 30]", output)
        self.assertIn("a2a_recv=[[0, 0], [1, 0]]", output)
        self.assertIn("a2a_recv=[[0, 1], [1, 1]]", output)


if __name__ == "__main__":
    unittest.main()
