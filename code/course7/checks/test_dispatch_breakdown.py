"""The CPU-metadata control must preserve receive layout semantics."""
import unittest
import torch
from benchmarks.dispatch_breakdown import cpu_metadata_receive_layout
from mini_ep.distributed.topology import ExpertParallelTopology, build_receive_layout


class TestDispatchControl(unittest.TestCase):
    def test_source_blocks_and_empty_experts(self):
        for values in ([[2, 1, 3, 2], [1, 4, 1, 2]], [[0, 0, 2, 0], [1, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0]]):
            counts = torch.tensor(values, dtype=torch.int64)
            for rank in (0, 1):
                topology = ExpertParallelTopology(4, 2, rank)
                kwargs = {"local_tokens_per_expert": counts[rank]}
                expected = build_receive_layout(topology, counts, **kwargs)
                actual = cpu_metadata_receive_layout(topology, counts, **kwargs)
                for field in ("recv_splits", "recv_offsets", "per_source_local_expert_counts"):
                    torch.testing.assert_close(getattr(actual, field), getattr(expected, field))
                for actual_rows, expected_rows in zip(actual.local_expert_index, expected.local_expert_index):
                    torch.testing.assert_close(actual_rows, expected_rows)
                    self.assertEqual(actual_rows.dtype, torch.int64)
                    self.assertTrue(actual_rows.is_contiguous())

class TestPairedAttribution(unittest.TestCase):
    def test_children_follow_same_critical_rank(self):
        import json
        import tempfile
        from pathlib import Path
        from benchmarks.analyze_dispatch import analyze, PARTS
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            report = {"case": "small", "trial": 0, "report_path": "trial.json", "config": {"ep_size": 2, "repeats": 2}}
            (base / "reports.json").write_text(json.dumps([report]))
            children = [
                [[3,.3],[.2,.1],[.2,.1],[.2,.1]],
                [[.1,.2],[.3,3],[.1,.2],[.1,.2]],
            ]
            for rank in range(2):
                parent = [4,1] if rank==0 else [1,4]
                control = [2,.5] if rank==0 else [.5,2]
                parts = dict(zip(PARTS, children[rank]))
                detail = {"correctness":"PASS", "control":{"dispatch_ms":parent}, "cpu_metadata":{"dispatch_ms":control}, "detailed":{"dispatch_ms":parent,"parts_ms":parts,"host_parts_ms":parts}, "scalar_reads":{p:{"aten_item_calls_per_forward":0} for p in PARTS}}
                (base / f"trial-rank{rank}.json").write_text(json.dumps({"dispatch_breakdown":detail}))
            row = analyze(base / "reports.json")["rows"][0]
            self.assertEqual(row["parent_mean_ms"],4)
            self.assertAlmostEqual(row["parts_mean_ms"]["EXPERT_PACK"],1.6)
            self.assertAlmostEqual(row["parts_mean_ms"]["RANK_PACK"],1.6)
            self.assertAlmostEqual(row["boundary_mean_ms"],.4)
            self.assertEqual(row["cpu_metadata_speedup"],2)


if __name__ == "__main__":
    unittest.main()
