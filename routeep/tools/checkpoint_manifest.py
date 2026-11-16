"""为本地模型目录生成 checkpoint 清单（文件大小、mtime、SHA256），供 benchmark 校验权重未被改动。

启动方式：
    python3 tools/checkpoint_manifest.py /models/Qwen3-30B-A3B results/manifests/Qwen3-30B-A3B-checkpoint.json

benchmarks/deepep_latency_run.py 与 benchmarks/peer_suite.py 通过 --checkpoint-manifest / --manifest-dir 读取该文件；
peer_suite 按 <manifest-dir>/<模型目录名>-checkpoint.json 查找。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "test"))
from deepep_alignment import checkpoint_manifest  # noqa: E402


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    model, output = Path(sys.argv[1]), Path(sys.argv[2])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(checkpoint_manifest(model), indent=2))
    print(output)


if __name__ == "__main__":
    main()
