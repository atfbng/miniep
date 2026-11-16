"""启动临时服务，调用 HTTP 接口做 smoke test，最后关闭整个进程组。"""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--port", type=int, default=8133)
    p.add_argument("--dp-size", type=int, choices=[1, 2], default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--kv-cache-mib", type=int)
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    # 先确认端口可用，避免误测或误杀已经存在的服务。
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
    base = f"http://127.0.0.1:{a.port}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("serve.py")),
        "--model",
        a.model,
        "--port",
        str(a.port),
        "--dp-size",
        str(a.dp_size),
        "--gpu-memory-utilization",
        str(a.gpu_memory_utilization),
    ]
    if a.kv_cache_mib is not None:
        command += [
            "--kv-cache-mib",
            str(a.kv_cache_mib),
            "--max-model-len",
            "1024",
            "--max-batched-tokens",
            "2048",
        ]
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 600
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited {process.returncode}; see server.log"
                    )
                try:
                    with urllib.request.urlopen(
                        base + "/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("server startup exceeded 600s")
                time.sleep(2)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("service_check.py")),
                    "--url",
                    base,
                    "--output",
                    str(root / "requests.json"),
                ],
                check=True,
                timeout=300,
            )
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
            try:
                code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise RuntimeError("server did not shut down within 60s")
            (root / "shutdown.json").write_text(
                json.dumps(dict(exit_code=code, signal="SIGINT", forced_kill=False))
            )
    if (
        f"RouteEP TP=2 DP={a.dp_size} EP={2*a.dp_size}"
        not in (root / "server.log").read_text()
    ):
        raise RuntimeError("server log does not confirm RouteEP runner activation")
    print("PASS: temporary RouteEP server tested and stopped", flush=True)


if __name__ == "__main__":
    main()
