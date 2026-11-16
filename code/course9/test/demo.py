"""Correctness and steady-state benchmark for 09_triton_dispatch."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_kernels import main

if __name__ == "__main__":
    main("dispatch")
