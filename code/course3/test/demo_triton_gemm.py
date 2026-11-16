#!/usr/bin/env python3
"""单 GPU Triton GEMM/MLP：先对照 PyTorch，再预热和计时。"""

from __future__ import annotations

import sys
from pathlib import Path

COURSE_ROOT = Path(__file__).resolve().parents[1]
if str(COURSE_ROOT) not in sys.path:
    sys.path.insert(0, str(COURSE_ROOT))

from benchmarks.benchmark_triton_gemm import main  # noqa: E402


if __name__ == "__main__":
    main()
