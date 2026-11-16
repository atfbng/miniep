"""Validation setup is intentionally separate from production launch defaults."""

import os
from pathlib import Path
from mini_ep.integration.config import configure as configure_runtime


def configure(backend: str, output: str) -> None:
    if backend not in ("deepep_native", "deepep_exact"):
        raise ValueError(
            f"validation mode must be deepep_native/deepep_exact, got {backend!r}"
        )
    configure_runtime("native" if backend == "deepep_native" else "miniep")
    os.environ["MINIEP_VALIDATION_MODE"] = backend
    os.environ["VLLM_PLUGINS"] = "miniep_course13_validation"
    Path(output).mkdir(parents=True, exist_ok=True)
    os.environ["MINIEP_REPORT_DIR"] = str(Path(output).resolve())
