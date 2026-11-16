"""CPU checks for honest profiling scope and slowest-rank aggregation."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.summarize_stages import summarize


class StageSummaryTests(unittest.TestCase):
    def fixture(self, root, mini):
        for rank in range(4):
            path = root / f"rank{rank}"
            path.mkdir()
            (path / "environment.json").write_text(json.dumps(dict(profile="events")))
            (path / "manifest.json").write_text(
                json.dumps([dict(name="case", steps=2)])
            )
            moe = [
                dict(
                    layer=f"model.layers.{layer}.mlp.experts",
                    milliseconds=float(rank + 1),
                )
                for step in range(2)
                for layer in range(48)
            ]
            stages = []
            traffic = []
            if mini:
                for step in range(2):
                    for layer in (0, 24, 47):
                        name = f"model.layers.{layer}.mlp.experts"
                        for stage in (
                            "ROUTER",
                            "DISPATCH",
                            "A2A_FORWARD",
                            "EXPERT",
                            "A2A_RETURN",
                            "COMBINE",
                        ):
                            stages.append(
                                dict(
                                    layer=name,
                                    stage=stage,
                                    milliseconds=(rank + 1) / 10,
                                )
                            )
                        traffic.append(
                            dict(
                                layer=name,
                                ep_rank=rank,
                                model_dtype="torch.bfloat16",
                                return_dtype="torch.bfloat16",
                                send_rows=4,
                                recv_rows=4,
                                send_splits=[1] * 4,
                                recv_splits=[1] * 4,
                                return_send_bytes=4 * 2048 * 2,
                                layout="fixed",
                            )
                        )
            (path / "samples.json").write_text(
                json.dumps(
                    [
                        dict(
                            case="case",
                            repetition=0,
                            profiling=dict(moe=moe, stages=stages, traffic=traffic),
                        )
                    ]
                )
            )

    def test_maximum_rank_and_sampled_layer_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, True)
            q = summarize(root, True)
            phase = q["cases"]["case"]["decode"]
            self.assertEqual(
                phase["milliseconds_per_layer"]["whole_moe_all48"]["median"], 4.0
            )
            self.assertEqual(phase["milliseconds_per_layer"]["EXPERT"]["median"], 0.4)
            self.assertEqual(
                phase["payload_bytes_per_layer"]["return_cross_rank"]["median"],
                3 * 2048 * 2,
            )
            self.assertEqual(q["selected_layers"], [0, 24, 47])

    def test_native_has_no_invented_detailed_stages_or_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, False)
            q = summarize(root, False)
            for phase in q["cases"]["case"].values():
                self.assertEqual(
                    set(phase["milliseconds_per_layer"]),
                    {"whole_moe_all48", "whole_moe_selected3"},
                )
                self.assertNotIn("payload_bytes_per_layer", phase)

    def test_missing_rank_and_inconsistent_model_steps_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, False)
            path = root / "rank3/samples.json"
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                summarize(root, False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, False)
            path = root / "rank3/samples.json"
            q = json.loads(path.read_text())
            q[0]["profiling"]["moe"].pop()
            path.write_text(json.dumps(q))
            with self.assertRaises(ValueError):
                summarize(root, False)


if __name__ == "__main__":
    unittest.main()
