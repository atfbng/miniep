"""EP=1/2/4 scaling on an identical global batch."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_ep import main

if __name__ == "__main__":
    main()
