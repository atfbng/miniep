"""Count and payload math used by the lesson-5 communication benchmark."""
import unittest

from benchmarks.benchmark_all_to_all import remote_payload_bytes, split_rows


class CommunicationMathTests(unittest.TestCase):
    def test_uniform_remainders_and_conservation(self):
        self.assertEqual(split_rows(5, 2, "uniform"), [3, 2])
        for size in (1, 2, 4):
            for rows in (0, 1, 7, 32):
                counts = split_rows(rows, size, "uniform")
                self.assertEqual(sum(counts), rows)
                self.assertLessEqual(max(counts) - min(counts), 1)

    def test_skew_and_single_rank(self):
        self.assertEqual(split_rows(5, 2, "imbalanced"), [4, 1])
        self.assertEqual(split_rows(7, 1, "imbalanced"), [7])
        self.assertEqual(split_rows(0, 4, "imbalanced"), [0, 0, 0, 0])
        self.assertEqual(sum(split_rows(17, 4, "imbalanced")), 17)

    def test_remote_edges_counted_once(self):
        self.assertEqual(remote_payload_bytes([[3, 2], [4, 1]], 8), 48)
        self.assertEqual(remote_payload_bytes([[5]], 512), 0)
        self.assertEqual(remote_payload_bytes([[3, 2], [3, 2]], 8), 40)
        self.assertEqual(remote_payload_bytes([[4, 1], [4, 1]], 8), 40)

    def test_invalid_split_configuration(self):
        for args in ((-1, 2, "uniform"), (5, 0, "uniform"), (5, 2, "unknown")):
            with self.assertRaises(ValueError):
                split_rows(*args)


if __name__ == "__main__":
    unittest.main()
