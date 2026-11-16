"""Expected layer coverage comes from the recorded checkpoint configuration."""

import json
from pathlib import Path


def expected_layers(root: Path) -> int:
    path = root / "environment.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    # Reports predating multi-model support only covered 48-layer Qwen3.
    default = 48 if metadata.get("model_family", "qwen3_moe") == "qwen3_moe" else None
    count = metadata.get("num_hidden_layers", default)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"invalid recorded num_hidden_layers={count!r}")
    return count
