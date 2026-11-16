#!/usr/bin/env python3
"""Lesson 3 entry point for the GEMM and CUDA timing experiment."""

from __future__ import annotations

import sys
from pathlib import Path

# Keep the course example runnable before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.benchmark_gemm import main  # noqa: E402


if __name__ == "__main__":
    main()
