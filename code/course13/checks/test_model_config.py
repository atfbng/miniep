"""Model-family selection and the shared/routed BF16 rounding boundary."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from mini_ep.integration.config import model_metadata
from validation.model_info import expected_layers


class ModelConfigTests(unittest.TestCase):
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
