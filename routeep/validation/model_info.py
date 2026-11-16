"""从记录的 checkpoint 配置推导预期层覆盖。

只接受课程使用的两个 BF16 模型，避免用错模型后仍把报告标成通过。
"""

import json
from pathlib import Path


MODEL_SHAPES = {
    "qwen3_moe": dict(
        num_hidden_layers=48,
        hidden_size=2048,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
    ),
    "qwen3_5_moe": dict(
        num_hidden_layers=40,
        hidden_size=2048,
        num_experts=256,
        num_experts_per_tok=8,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
    ),
}


def validated_model_shape(model: str | Path) -> dict:
    """读取并校验 exact/latency 套件使用的两个 BF16 checkpoint 形状。"""
    config = json.loads((Path(model) / "config.json").read_text())
    family = config.get("model_type")
    expected = MODEL_SHAPES.get(family)
    if expected is None:
        raise ValueError(f"expected Qwen3-30B-A3B or Qwen3.5-35B-A3B, got {family!r}")
    text = config.get("text_config", config)
    if any(text.get(key) != value for key, value in expected.items()):
        actual = {key: text.get(key) for key in expected}
        raise ValueError(f"unexpected {family} checkpoint shape: {actual}; expected {expected}")
    dtype = text.get("dtype", text.get("torch_dtype"))
    if dtype not in (None, "bfloat16"):
        raise ValueError(f"expected BF16 checkpoint, got dtype={dtype!r}")
    return dict(model_family=family, **expected)


def expected_layers(root: Path) -> int:
    path = root / "environment.json"
    metadata = json.loads(path.read_text()) if path.exists() else {}
    # 早于多模型支持的报告只覆盖 48 层 Qwen3，因此旧报告默认按 48 层读取。
    default = 48 if metadata.get("model_family", "qwen3_moe") == "qwen3_moe" else None
    count = metadata.get("num_hidden_layers", default)
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError(f"invalid recorded num_hidden_layers={count!r}")
    return count
