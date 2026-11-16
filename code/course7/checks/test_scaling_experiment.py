"""Validate experiment aggregation without requiring CUDA or NCCL."""
import unittest
from benchmarks.run_scaling_experiment import summarize


def report(ep, latency, trial=0):
    return {
        "case": "small", "trial": trial, "correctness": "PASS",
        "config": {"tokens": 32, "hidden_size": 16, "expert_hidden_size": 32, "num_experts": 8, "top_k": 2, "dtype": "float16", "seed": 0, "scenario": "uniform", "ep_size": ep, "repeats": 3},
        "routing_sha256": "fixed-route", "tokens_per_expert": [8] * 8,
        "rows_per_rank": [64 // ep] * ep,
        "max_rank_samples_ms": [latency] * 3, "total_latency_ms": latency,
        "communication_bytes": 0 if ep == 1 else 1024,
        "stage_max_rank_median_ms": {"EXPERT": latency / 2},
    }


class TestScalingSummary(unittest.TestCase):
    def test_trial_median_speedup_and_efficiency(self):
        rows = summarize([report(1, 4), report(1, 8, 1), report(2, 2), report(2, 4, 1)])["rows"]
        self.assertEqual(rows[0]["latency_ms"], 6)
        self.assertEqual(rows[1]["latency_ms"], 3)
        self.assertEqual(rows[1]["speedup"], 2)
        self.assertEqual(rows[1]["parallel_efficiency"], 1)
        self.assertAlmostEqual(rows[1]["tokens_per_second"], 32000 / 3)
        self.assertEqual(rows[1]["stages_ms"]["EXPERT"], 1.5)

    def test_changed_route_is_rejected(self):
        changed = report(2, 2)
        changed["routing_sha256"] = "different-route"
        with self.assertRaisesRegex(ValueError, "routing changed"):
            summarize([report(1, 4), changed])

    def test_missing_baseline_and_duplicate_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing EP=1"):
            summarize([report(2, 2)])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize([report(1, 4), report(1, 4)])

    def test_missing_rows_or_repetitions_are_rejected(self):
        bad = report(1, 4)
        bad["rows_per_rank"] = [63]
        with self.assertRaisesRegex(ValueError, "cover all branches"):
            summarize([bad])
        bad = report(1, 4)
        bad["max_rank_samples_ms"] = [4]
        with self.assertRaisesRegex(ValueError, "repetition count"):
            summarize([bad])


if __name__ == "__main__":
    unittest.main()
