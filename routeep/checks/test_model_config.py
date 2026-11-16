"""Model-family selection and the shared/routed BF16 rounding boundary."""

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_ep.integration.config import model_metadata, engine_options
from validation.model_info import expected_layers, validated_model_shape


class ModelConfigTests(unittest.TestCase):
    @patch("mini_ep.integration.config.model_metadata", return_value={"model_family": "qwen3_moe", "num_hidden_layers": 48})
    def test_decode_graph_option(self, metadata):
        eager = engine_options("unused", dp_size=2, external=True)
        self.assertTrue(eager["enforce_eager"])
        self.assertNotIn("compilation_config", eager)
        graph = engine_options("unused", dp_size=2, external=True, cudagraph_capture_sizes=[8, 1, 8, 256])
        self.assertFalse(graph["enforce_eager"])
        self.assertEqual(graph["compilation_config"], dict(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 8, 256]))
        # 超过固定容量上限的 capture size 会让 decode 走变长路径，不能 capture。
        for sizes in ([], [0], [257]):
            with self.subTest(sizes=sizes), self.assertRaises(ValueError):
                engine_options("unused", dp_size=2, external=True, cudagraph_capture_sizes=sizes)

    def test_validated_model_shapes(self):
        cases = (
            ("qwen3_moe", 48, 128, 768, None),
            ("qwen3_5_moe", 40, 256, 512, 512),
        )
        for family, layers, experts, intermediate, shared in cases:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as name:
                text = dict(
                    num_hidden_layers=layers,
                    hidden_size=2048,
                    num_experts=experts,
                    num_experts_per_tok=8,
                    moe_intermediate_size=intermediate,
                    dtype="bfloat16",
                )
                if shared is not None:
                    text["shared_expert_intermediate_size"] = shared
                config = (
                    dict(model_type=family, text_config=text)
                    if shared is not None
                    else dict(model_type=family, **text)
                )
                path = Path(name) / "config.json"
                path.write_text(json.dumps(config))
                self.assertEqual(validated_model_shape(name)["num_hidden_layers"], layers)
                (config["text_config"] if shared is not None else config)[
                    "num_experts"
                ] += 1
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    validated_model_shape(name)

    def test_nested_text_config_and_coverage(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "config.json").write_text(
                json.dumps(
                    dict(
                        model_type="qwen3_5_moe", text_config=dict(num_hidden_layers=40)
                    )
                )
            )
            metadata = model_metadata(name)
            self.assertEqual(metadata["num_hidden_layers"], 40)
            (root / "environment.json").write_text(json.dumps(metadata))
            self.assertEqual(expected_layers(root), 40)
            (root / "environment.json").write_text(
                json.dumps(dict(model_family="qwen3_5_moe"))
            )
            with self.assertRaises(ValueError):
                expected_layers(root)


if __name__ == "__main__":
    unittest.main()
