"""只读审计：复用独立 stock-MoERunner 的 golden tensor。

规则：RouteEP 源码改动会使 RouteEP 结果失效；但从未执行的 RouteEP Kernel
不能使 stock native 基线失效。复用必须满足原生运行时文件未变、权重/选项/
workload 相同、原生类未被替换、原始覆盖完整。每个候选结果仍要在全部保存的
全词表 Tensor 上执行 native_check，之后才能比较 RouteEP。
"""

import hashlib
import importlib.util
import json
from pathlib import Path


def capture_runtime() -> dict:
    files = {}
    for package in ("vllm", "deep_ep"):
        root = Path(importlib.util.find_spec(package).origin).parent
        for path in sorted(root.rglob("*")):
            if path.is_file() and (path.suffix == ".py" or ".so" in path.name):
                stat = path.stat()
                with path.open("rb") as stream:
                    sha = hashlib.file_digest(stream, "sha256").hexdigest()
                files[str(path)] = dict(
                    sha256=sha, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns
                )
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(files=files, digest=digest)


def audit_native(root: Path, runtime: dict) -> dict:
    from validation.alignment_report import summarize_run

    proof = summarize_run(root, world_size=4)
    if not proof["passed"]:
        raise RuntimeError(f"incomplete native golden: {root}: {proof}")
    signatures = {}
    latest = max(q["mtime_ns"] for q in runtime["files"].values())
    for rank in range(4):
        path = root / f"rank{rank}" / "environment.json"
        env = json.loads(path.read_text())
        backend = json.loads((path.parent / "backend.json").read_text())
        if (
            env["kind"] != "native"
            or env["backend"] != "deepep_high_throughput"
            or backend["runner"] != "MoERunner"
            or env["world_size"] != 4
        ):
            raise RuntimeError(
                "only independent original four-rank DeepEP golden can be reused"
            )
        # 这些自管容器使用同一个固定不可变镜像；runtime manifest 记录所有
        # 源码/扩展文件，不能有文件比旧基线更新。修改过 native 代码后禁止使用
        # 这个复用门。
        if latest > path.stat().st_mtime_ns:
            raise RuntimeError(
                "native runtime newer than golden; run a fresh native baseline"
            )
        signatures[str(rank)] = env["signature"]
    return dict(
        passed=True,
        reference=str(root.resolve()),
        native_signatures=signatures,
        native_runtime_digest=runtime["digest"],
        native_runtime_latest_mtime_ns=latest,
        basis="original MoERunner; pinned immutable image; native files predate baseline; frozen config/model/workloads; full native_check required",
        miniep_results_reusable=False,
    )


def verify_runtime(manifest: dict) -> None:
    """控制器已完成全量哈希后，每个 Rank 再做一次廉价的不可变性检查。"""
    if not manifest.get("files") or not manifest.get("digest"):
        raise RuntimeError("missing native runtime fingerprint")
    for name, info in manifest["files"].items():
        stat = Path(name).stat()
        if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"native runtime changed after SHA capture: {name}")
