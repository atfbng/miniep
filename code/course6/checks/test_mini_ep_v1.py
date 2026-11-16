"""Unit and optional GPU integration tests for lesson 6 (Mini-EP V1).

CPU tests cover placement, rank packing, receive layout and the All-to-All
wrappers with mocked collectives.  The same layout math is then exercised as a
pure-Tensor round trip, and finally as a real two-process run: the torchrun
integration tests below are the only ones that speak for NCCL and GPU
behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist

from mini_ep.distributed import (
    DistributedContext,
    ExpertParallelTopology,
    all_gather_expert_counts,
    build_receive_layout,
    combine_reference,
    compute_send_splits,
    dispatch_reference,
    pack_by_destination_rank,
    variable_split_all_to_all,
)
from mini_ep.model import SingleGPUMoE, distribute_single_gpu_moe


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestExpertParallelTopology(unittest.TestCase):
    def test_uniform_placement_maps_experts_to_ranks(self) -> None:
        rank0 = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)
        rank1 = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=1)
        expert_ids = torch.arange(4)

        self.assertEqual(rank0.experts_per_rank, 2)
        self.assertEqual(rank0.local_expert_ids, (0, 1))
        self.assertEqual(rank1.local_expert_ids, (2, 3))
        self.assertEqual(rank0.global_expert_id(1), 1)
        self.assertEqual(rank1.global_expert_id(1), 3)
        torch.testing.assert_close(
            rank0.destination_rank(expert_ids), torch.tensor([0, 0, 1, 1])
        )
        torch.testing.assert_close(
            rank1.destination_rank(expert_ids), torch.tensor([0, 0, 1, 1])
        )
        torch.testing.assert_close(
            rank0.local_expert_id(expert_ids), torch.tensor([0, 1, 0, 1])
        )

    def test_rejects_placement_that_does_not_divide_evenly(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_experts % ep_size"):
            ExpertParallelTopology(num_experts=4, ep_size=3, ep_rank=0)
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=2)
        with self.assertRaisesRegex(ValueError, "ep_size"):
            ExpertParallelTopology(num_experts=4, ep_size=0, ep_rank=0)
        with self.assertRaisesRegex(ValueError, "num_experts"):
            ExpertParallelTopology(num_experts=0, ep_size=1, ep_rank=0)

    def test_mapping_requires_integer_expert_ids(self) -> None:
        topology = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)

        with self.assertRaisesRegex(TypeError, "expert_ids"):
            topology.destination_rank(torch.tensor([0.0, 1.0]))
        with self.assertRaisesRegex(TypeError, "expert_ids"):
            topology.local_expert_id(torch.tensor([0.0, 1.0]))

    def test_from_context_uses_the_default_group_as_the_ep_group(self) -> None:
        context = DistributedContext(
            global_rank=1,
            local_rank=1,
            world_size=2,
            device=torch.device("cpu"),
            group=mock.MagicMock(),
        )

        topology = ExpertParallelTopology.from_context(context, num_experts=4)

        self.assertEqual(topology.ep_size, 2)
        self.assertEqual(topology.ep_rank, 1)
        self.assertEqual(topology.local_expert_ids, (2, 3))


class TestSendSplitsAndReceiveLayout(unittest.TestCase):
    # One rank's routing result: 16 assignments over 4 experts.
    LOCAL_COUNTS = torch.tensor([8, 3, 2, 3])
    # Counts gathered from both ranks: row s belongs to source rank s.
    ALL_COUNTS = torch.tensor([[8, 3, 2, 3], [5, 5, 3, 3]])

    def test_send_splits_group_experts_by_destination_rank(self) -> None:
        topology = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)

        send_splits = compute_send_splits(topology, self.LOCAL_COUNTS)

        # Experts 0 and 1 belong to rank 0, experts 2 and 3 to rank 1.
        torch.testing.assert_close(send_splits, torch.tensor([11, 5]))
        self.assertEqual(send_splits.dtype, torch.int64)
        self.assertEqual(int(send_splits.sum()), 16)

    def test_receive_layout_partitions_every_received_row(self) -> None:
        rank0 = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)

        layout = build_receive_layout(
            rank0,
            self.ALL_COUNTS,
            local_tokens_per_expert=self.LOCAL_COUNTS,
        )

        # Rank 0 receives 8 + 3 rows from itself and 5 + 5 rows from rank 1.
        torch.testing.assert_close(layout.recv_splits, torch.tensor([11, 10]))
        torch.testing.assert_close(layout.recv_offsets, torch.tensor([0, 11, 21]))
        torch.testing.assert_close(
            layout.local_counts,
            torch.tensor([[8, 3], [5, 5]]),
        )
        self.assertEqual(layout.num_recv_rows, 21)
        # Local expert 0 owns the leading block of each source group.
        torch.testing.assert_close(
            layout.rows_by_expert[0],
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 15]),
        )
        torch.testing.assert_close(
            layout.rows_by_expert[1],
            torch.tensor([8, 9, 10, 16, 17, 18, 19, 20]),
        )
        covered = sorted(
            int(row)
            for rows in layout.rows_by_expert
            for row in rows.tolist()
        )
        self.assertEqual(covered, list(range(21)))

    def test_receive_layout_for_the_other_rank(self) -> None:
        rank1 = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=1)

        layout = build_receive_layout(rank1, self.ALL_COUNTS)

        torch.testing.assert_close(layout.recv_splits, torch.tensor([5, 6]))
        torch.testing.assert_close(
            layout.local_counts,
            torch.tensor([[2, 3], [3, 3]]),
        )
        covered = sorted(
            int(row)
            for rows in layout.rows_by_expert
            for row in rows.tolist()
        )
        self.assertEqual(covered, list(range(11)))

    def test_rejects_inconsistent_counts(self) -> None:
        topology = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)

        with self.assertRaisesRegex(ValueError, "shape"):
            build_receive_layout(topology, self.ALL_COUNTS[:, :3])
        with self.assertRaisesRegex(TypeError, "int32 or int64"):
            build_receive_layout(topology, self.ALL_COUNTS.to(torch.float32))
        with self.assertRaisesRegex(ValueError, "negative"):
            build_receive_layout(topology, torch.tensor([[1, -1, 1, 1], [1, 1, 1, 1]]))
        with self.assertRaisesRegex(ValueError, "do not match the local routing"):
            build_receive_layout(
                topology,
                self.ALL_COUNTS,
                local_tokens_per_expert=torch.tensor([0, 0, 0, 16]),
            )


class TestVariableSplitAllToAll(unittest.TestCase):
    def test_allocates_receive_shape_and_forwards_splits(self) -> None:
        send_buffer = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        captured: dict[str, object] = {}

        def copy_collective(
            output: torch.Tensor,
            input_tensor: torch.Tensor,
            *,
            output_split_sizes: list[int],
            input_split_sizes: list[int],
            group: object,
            async_op: bool,
        ) -> None:
            captured["output_split_sizes"] = output_split_sizes
            captured["input_split_sizes"] = input_split_sizes
            captured["group"] = group
            captured["async_op"] = async_op
            output.zero_()
            output[: input_tensor.shape[0]].copy_(input_tensor)

        sentinel_group = object()
        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
            mock.patch.object(
                dist, "all_to_all_single", side_effect=copy_collective
            ),
        ):
            recv_buffer = variable_split_all_to_all(
                send_buffer,
                torch.tensor([3, 2]),
                [2, 3],
                group=sentinel_group,  # type: ignore[arg-type]
            )

        self.assertEqual(recv_buffer.shape, (5, 2))
        self.assertEqual(recv_buffer.dtype, send_buffer.dtype)
        self.assertEqual(recv_buffer.device, send_buffer.device)
        self.assertTrue(recv_buffer.is_contiguous())
        self.assertEqual(captured["input_split_sizes"], [3, 2])
        self.assertEqual(captured["output_split_sizes"], [2, 3])
        self.assertIs(captured["group"], sentinel_group)
        self.assertFalse(captured["async_op"])

    def test_rejects_invalid_splits_and_buffers(self) -> None:
        send_buffer = torch.zeros(4, 3)
        with mock.patch.object(dist, "is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                variable_split_all_to_all(send_buffer, [2, 2], [2, 2])

        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
        ):
            with self.assertRaisesRegex(ValueError, "sum to send_buffer"):
                variable_split_all_to_all(send_buffer, [3, 2], [2, 2])
            with self.assertRaisesRegex(ValueError, "one entry per"):
                variable_split_all_to_all(send_buffer, [2, 2, 2], [2, 2])
            with self.assertRaisesRegex(ValueError, "non-negative"):
                variable_split_all_to_all(send_buffer, [-1, 5], [2, 2])
            with self.assertRaisesRegex(TypeError, "int32 or int64"):
                variable_split_all_to_all(
                    send_buffer, torch.tensor([2.0, 2.0]), [2, 2]
                )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                variable_split_all_to_all(
                    torch.zeros(3, 4).transpose(0, 1), [2, 2], [2, 2]
                )


class TestPackByDestinationRank(unittest.TestCase):
    def test_packs_rows_by_destination_rank(self) -> None:
        tokens = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        expert_ids = torch.tensor([[0, 1], [2, 1], [2, 3]])
        routing_weights = torch.full((3, 2), 0.5)
        dispatch = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=4
        )
        topology = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)

        rank_pack = pack_by_destination_rank(dispatch, topology)

        # Expert-major order is already destination-major:
        # E0 | E1, E1 | E2, E2, E3.
        torch.testing.assert_close(
            rank_pack.send_splits, torch.tensor([3, 3])
        )
        torch.testing.assert_close(
            rank_pack.send_offsets, torch.tensor([0, 3, 6])
        )
        torch.testing.assert_close(
            rank_pack.packed_expert_ids, torch.tensor([0, 1, 1, 2, 2, 3])
        )
        # Uniform placement sorts rows that are already sorted, so the packing
        # permutation is the identity.
        torch.testing.assert_close(
            rank_pack.row_origin_index, torch.arange(6)
        )
        torch.testing.assert_close(
            rank_pack.packed_tokens[:3], dispatch.packed_tokens[:3]
        )
        self.assertEqual(rank_pack.num_send_rows, 6)
        self.assertTrue(rank_pack.packed_tokens.is_contiguous())

    def test_rejects_dispatch_from_a_different_expert_count(self) -> None:
        tokens = torch.zeros(2, 2)
        expert_ids = torch.tensor([[0, 1], [1, 2]])
        routing_weights = torch.full((2, 2), 0.5)
        dispatch = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts=3
        )
        topology = ExpertParallelTopology(num_experts=4, ep_size=2, ep_rank=0)

        with self.assertRaisesRegex(ValueError, "expert count"):
            pack_by_destination_rank(dispatch, topology)


class TestExpertCountExchange(unittest.TestCase):
    def test_gathers_counts_into_a_rank_by_expert_matrix(self) -> None:
        counts = torch.tensor([1, 2, 3, 4])
        captured: dict[str, torch.Tensor] = {}

        def gather_collective(
            output: torch.Tensor,
            input_tensor: torch.Tensor,
            *,
            group: object,
            async_op: bool,
        ) -> None:
            captured["output"] = output
            size = input_tensor.numel()
            output[:size] = input_tensor
            output[size:] = input_tensor + 10

        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
            mock.patch.object(
                dist, "all_gather_into_tensor", side_effect=gather_collective
            ),
        ):
            gathered = all_gather_expert_counts(counts)

        self.assertEqual(gathered.shape, (2, 4))
        self.assertEqual(gathered.dtype, torch.int64)
        torch.testing.assert_close(gathered[0], counts)
        torch.testing.assert_close(gathered[1], counts + 10)
        # The collective fills the flat receive buffer on every backend.
        self.assertEqual(captured["output"].shape, (8,))

    def test_rejects_non_vector_counts(self) -> None:
        with (
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=1),
        ):
            with self.assertRaisesRegex(ValueError, "shape \\[num_experts\\]"):
                all_gather_expert_counts(torch.zeros(2, 2))
            with self.assertRaisesRegex(TypeError, "int32 or int64"):
                all_gather_expert_counts(torch.zeros(2))


def _emulate_exchange(
    rank_buffers: list[torch.Tensor],
    rank_send_splits: list[torch.Tensor],
    destination_rank: int,
) -> torch.Tensor:
    """Copy the rows every rank sends to ``destination_rank``, in source order."""

    blocks = []
    for buffer, send_splits in zip(rank_buffers, rank_send_splits, strict=True):
        start = int(send_splits[:destination_rank].sum().item())
        count = int(send_splits[destination_rank].item())
        blocks.append(buffer[start : start + count])
    return torch.cat(blocks, dim=0)


class TestLayoutRoundTrip(unittest.TestCase):
    """Check the layout contract without any communication."""

    def test_skewed_routing_round_trip_matches_single_gpu_combine(self) -> None:
        torch.manual_seed(7)
        num_tokens, hidden_size, num_experts, top_k, ep_size = 6, 4, 4, 2, 2
        tokens = torch.randn(num_tokens, hidden_size)
        # Highly tilted routing: only experts 0, 1 and 3 receive assignments.
        expert_ids = torch.tensor(
            [[0, 0], [0, 1], [0, 0], [0, 3], [3, 3], [3, 3]]
        )
        routing_weights = torch.full((num_tokens, top_k), 0.5)

        dispatch = dispatch_reference(
            tokens, expert_ids, routing_weights, num_experts
        )

        def expert_function(expert_id: int, rows: torch.Tensor) -> torch.Tensor:
            return rows * float(expert_id + 1)

        # Single-GPU answer: every assignment runs on the rank that owns it.
        single_gpu_outputs = torch.empty_like(dispatch.packed_tokens)
        for expert_id in range(num_experts):
            start = int(dispatch.expert_offsets[expert_id].item())
            stop = int(dispatch.expert_offsets[expert_id + 1].item())
            single_gpu_outputs[start:stop] = expert_function(
                expert_id, dispatch.packed_tokens[start:stop]
            )
        expected = combine_reference(single_gpu_outputs, dispatch)

        # Distributed path: rank-pack, emulate the two All-to-All calls, run the
        # local experts, then unpermute and combine.
        topologies = [
            ExpertParallelTopology(num_experts, ep_size, ep_rank)
            for ep_rank in range(ep_size)
        ]
        rank_packs = [
            pack_by_destination_rank(dispatch, topology)
            for topology in topologies
        ]
        send_buffers = [pack.packed_tokens for pack in rank_packs]
        send_splits = [pack.send_splits for pack in rank_packs]
        # This simulation gives every rank the same routing counts, which keeps
        # every sent split consistent with the peers' receive splits.
        counts_per_source = (
            dispatch.tokens_per_expert.to(torch.int64).reshape(1, -1).repeat(
                ep_size, 1
            )
        )

        layouts = []
        expert_outputs = []
        for ep_rank, topology in enumerate(topologies):
            recv_tokens = _emulate_exchange(send_buffers, send_splits, ep_rank)
            layout = build_receive_layout(topology, counts_per_source)
            self.assertEqual(recv_tokens.shape[0], layout.num_recv_rows)

            outputs = torch.empty_like(recv_tokens)
            for local_expert_id, rows in enumerate(layout.rows_by_expert):
                global_id = topology.global_expert_id(local_expert_id)
                outputs.index_copy_(
                    0,
                    rows,
                    expert_function(global_id, recv_tokens.index_select(0, rows)),
                )
            layouts.append(layout)
            expert_outputs.append(outputs)

        # All-to-All #2 returns each block to the rank that produced it.  The
        # return buffer therefore repeats the block structure of the send
        # buffer, and scattering with row_origin_index restores the expert-major
        # order that Combine expects.
        round_trip = torch.empty_like(dispatch.packed_tokens)
        for ep_rank in range(ep_size):
            blocks = [
                expert_outputs[source][
                    int(layouts[source].recv_offsets[ep_rank].item()) :
                    int(layouts[source].recv_offsets[ep_rank + 1].item())
                ]
                for source in range(ep_size)
            ]
            returned = torch.cat(blocks, dim=0)
            self.assertEqual(returned.shape[0], rank_packs[ep_rank].num_send_rows)
            round_trip.index_copy_(
                0, rank_packs[ep_rank].row_origin_index, returned
            )

        actual = combine_reference(round_trip, dispatch)
        torch.testing.assert_close(actual, expected)


@unittest.skipIf(
    dist.is_initialized(),
    "a Process Group is already initialized in this process",
)
class TestExpertParallelMoEOnCpu(unittest.TestCase):
    """EP=1 data-path check on gloo.

    This test exercises Router -> Dispatch -> All-to-All -> Expert ->
    All-to-All -> Combine with a single rank, so it validates the data flow and
    the layout math.  It runs on gloo and is not evidence about NCCL or GPU
    behaviour; the torchrun integration tests below cover that.
    """

    @classmethod
    def setUpClass(cls) -> None:
        init_file = Path(tempfile.mkdtemp()) / "gloo_rendezvous"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            world_size=1,
            rank=0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_ep_one_matches_single_gpu_reference(self) -> None:
        torch.manual_seed(23)
        reference = SingleGPUMoE(
            hidden_size=8,
            expert_hidden_size=16,
            num_experts=4,
            top_k=2,
        )
        tokens = torch.randn(5, 8)
        ep_model = distribute_single_gpu_moe(reference, ep_size=1, ep_rank=0)

        # ``torch.no_grad()`` rather than ``torch.inference_mode()``: the
        # collectives write into preallocated buffers, and inference tensors
        # reject in-place updates from a CPU backend such as gloo.
        with torch.no_grad():
            expected = reference(tokens).hidden_states
            result = ep_model(tokens)

        torch.testing.assert_close(result.hidden_states, expected)
        torch.testing.assert_close(result.expert_ids, reference(tokens).expert_ids)
        torch.testing.assert_close(
            result.send_splits, torch.tensor([10])
        )
        torch.testing.assert_close(
            result.recv_splits, torch.tensor([10])
        )
        torch.testing.assert_close(
            result.local_expert_rows.sum(), torch.tensor(10)
        )

    def test_weight_transfer_keeps_local_experts_in_global_order(self) -> None:
        torch.manual_seed(29)
        reference = SingleGPUMoE(
            hidden_size=8, expert_hidden_size=16, num_experts=4, top_k=2
        )
        ep_model = distribute_single_gpu_moe(reference, ep_size=1, ep_rank=0)

        torch.testing.assert_close(
            ep_model.router.gate.weight, reference.router.gate.weight
        )
        for local_id, global_id in enumerate(ep_model.topology.local_expert_ids):
            torch.testing.assert_close(
                ep_model.experts[local_id].up_proj.weight,
                reference.experts[global_id].up_proj.weight,
            )


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and dist.is_nccl_available(),
    "two CUDA GPUs and NCCL are required",
)
class TestTwoGpuMiniEpV1Integration(unittest.TestCase):
    def run_example(self, *extra_arguments: str) -> str:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            "test/demo.py",
            "--expected-world-size=2",
            "--timeout-seconds=90",
            *extra_arguments,
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        output = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, msg=output)
        return output

    def test_random_routing_top2_fp32_matches_single_gpu(self) -> None:
        output = self.run_example("--scenario=random", "--top-k=2")

        self.assertEqual(output.count("PASS"), 2, msg=output)
        self.assertIn("send_splits=", output)
        self.assertIn("recv_splits=", output)
        self.assertIn("reference_check=max_abs_diff=", output)

    def test_skewed_routing_top1_fp16_matches_single_gpu(self) -> None:
        output = self.run_example(
            "--scenario=skewed",
            "--top-k=1",
            "--dtype=float16",
            "--tokens=8",
        )

        self.assertEqual(output.count("PASS"), 2, msg=output)
        # Skewed Top-1 routing leaves at least one local expert idle, which is
        # the empty-expert case that the equal-split lesson-4 path cannot
        # express.
        self.assertIn("local_expert_rows=[0,", output)

    def test_skewed_routing_top2_bfloat16_matches_single_gpu(self) -> None:
        output = self.run_example(
            "--scenario=skewed",
            "--top-k=2",
            "--dtype=bfloat16",
        )

        self.assertEqual(output.count("PASS"), 2, msg=output)

    def test_token_and_hidden_size_sweep_matches_single_gpu(self) -> None:
        configurations = (
            ("--tokens=16", "--hidden-size=32", "--expert-hidden-size=64"),
            ("--tokens=8", "--hidden-size=128", "--expert-hidden-size=256"),
            ("--tokens=4", "--hidden-size=64", "--expert-hidden-size=64"),
        )

        for configuration in configurations:
            with self.subTest(configuration=configuration):
                output = self.run_example("--top-k=2", *configuration)
                self.assertEqual(output.count("PASS"), 2, msg=output)


@unittest.skipUnless(
    torch.cuda.device_count() >= 1 and dist.is_nccl_available(),
    "a CUDA GPU and NCCL are required",
)
class TestSingleGpuEp1Integration(unittest.TestCase):
    def test_ep_size_one_matches_single_gpu_reference(self) -> None:
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=1",
            "test/demo.py",
            "--expected-world-size=1",
            "--timeout-seconds=90",
            "--tokens=8",
        ]
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            timeout=240,
            check=False,
        )
        output = completed.stdout + completed.stderr

        self.assertEqual(completed.returncode, 0, msg=output)
        self.assertEqual(output.count("PASS"), 1, msg=output)
        self.assertIn("ep_size=1", output)


if __name__ == "__main__":
    unittest.main()
