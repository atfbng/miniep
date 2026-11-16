"""Read-only audit for reusing independent stock-MoERunner golden tensors.

Mini-EP source changes invalidate Mini-EP results, but never executed Mini-EP
kernels cannot invalidate a stock native baseline. Require unchanged immutable
native runtime files, identical weights/options/workloads, original native
classes and full raw coverage. Every candidate still executes native_check on
all saved full-vocabulary tensors before comparing any Mini-EP result.
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
        # These owned containers use the same pinned immutable image. The
        # runtime manifest records every source/extension file; none is newer
        # than the old baseline. Never use this gate after patching native code.
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
    """Cheap per-rank immutability check after controller hashes all bytes."""
    if not manifest.get("files") or not manifest.get("digest"):
        raise RuntimeError("missing native runtime fingerprint")
    for name, info in manifest["files"].items():
        stat = Path(name).stat()
        if stat.st_size != info["bytes"] or stat.st_mtime_ns != info["mtime_ns"]:
            raise RuntimeError(f"native runtime changed after SHA capture: {name}")
