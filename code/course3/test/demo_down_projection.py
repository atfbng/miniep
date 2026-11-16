"""单 GPU 对比下投影候选扩展；从本课目录运行。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.benchmark_down_projection import main

if __name__ == '__main__':
    main()
