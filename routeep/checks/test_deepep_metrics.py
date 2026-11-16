"""在 CPU 上检查 Benchmark 计时口径；不据此声称 GPU 性能。"""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.deepep_metrics import request_timing, batch_timing, distribution


class DeepEPMetricsTests(unittest.TestCase):
    def test_batch_span_keeps_scheduling_and_decode_boundaries(self):
        def metrics(start, first, last):
            return SimpleNamespace(
                scheduled_ts=start,
                first_token_ts=first,
                last_token_ts=last,
                num_generation_tokens=4,
                is_corrupted=False,
            )

        a = request_timing(metrics(10, 12, 15), 4)
        b = request_timing(metrics(11, 13, 16), 4)
        q = batch_timing([a, b])
        self.assertEqual(q["prefill_batch_ms"], 3000)
        self.assertEqual(q["decode_batch_ms"], 4000)
        self.assertAlmostEqual(q["decode_step_ms"], 4000 / 3)

    def test_prefill_only_has_no_decode_measurement(self):
        m = SimpleNamespace(
            scheduled_ts=10,
            first_token_ts=12,
            last_token_ts=12,
            num_generation_tokens=1,
            is_corrupted=False,
        )
        q = batch_timing([request_timing(m, 1)])
        self.assertIsNone(q["decode_batch_ms"])
        self.assertEqual(q["prefill_batch_ms"], 2000)
        for field, value in [
            ("scheduled_ts", 0),
            ("last_token_ts", 11),
            ("num_generation_tokens", 2),
            ("is_corrupted", True),
        ]:
            bad = SimpleNamespace(**vars(m))
            setattr(bad, field, value)
            with self.assertRaises(RuntimeError):
                request_timing(bad, 1)
        with self.assertRaises(RuntimeError):
            request_timing(None, 1)

    def test_distribution_preserves_samples(self):
        q = distribution([4.0, 1.0, 3.0])
        self.assertEqual(q["median"], 3.0)
        self.assertEqual(q["samples"], 3)
        with self.assertRaises(ValueError):
            distribution([float("nan")])


if __name__ == "__main__":
    unittest.main()
