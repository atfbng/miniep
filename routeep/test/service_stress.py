"""对已启动的 RouteEP 服务做稳定性压测：持续混合各类请求，结束后确认服务仍健康。

覆盖的边界（每一类都会让 MoE 层看到不同的 Token 布局）：
    - 短请求、长 prompt（单副本 >256 Token，走 NCCL 变长路径或 Peer 路径）；
    - 并发数在 1..concurrency 间波动，制造 prefill/decode 混合批、单侧 DP 空闲（dummy batch）；
    - 流式请求中途断开（服务端 abort）；
    - 贪心请求重复发送，检查同一输入的输出是否稳定。
任何请求报错、超时或 /health 失败都记为失败；结束时打印汇总并以非零退出码报告失败。

启动方式（服务已用 test/serve.py 启动）：
    python3 test/service_stress.py --url http://127.0.0.1:8133 --duration 600 --output results/stress.json
"""

import argparse
import concurrent.futures
import json
import random
import threading
import time
import urllib.request
from pathlib import Path

WORDS = "expert routing token dispatch combine gradient tensor kernel memory stream".split()


def post(base: str, path: str, payload: dict, timeout: float):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def prompt_of(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


def one_request(base: str, kind: str, rng: random.Random, timeout: float) -> dict:
    """发送一个请求并返回结果；任何异常由调用方记为失败。"""
    start = time.time()
    if kind == "short":
        payload = dict(model="routeep", prompt=prompt_of(rng, rng.randint(3, 40)),
                       temperature=0, max_tokens=rng.randint(1, 64))
    elif kind == "long":
        # 约 600–1800 个 Token 的 prompt，保证单个 DP 副本超过固定容量 256。
        payload = dict(model="routeep", prompt=prompt_of(rng, rng.randint(600, 1800)),
                       temperature=0, max_tokens=rng.randint(1, 16))
    elif kind == "sampled":
        payload = dict(model="routeep", prompt=prompt_of(rng, rng.randint(5, 200)),
                       temperature=0.8, top_p=0.9, max_tokens=rng.randint(8, 128),
                       seed=rng.randint(0, 2**31))
    elif kind == "abort":
        # 流式读取几个分片后主动断开，服务端应 abort 该请求并继续服务其他请求。
        payload = dict(model="routeep", prompt=prompt_of(rng, rng.randint(5, 100)),
                       temperature=0, max_tokens=256, stream=True)
        with post(base, "/v1/completions", payload, timeout) as response:
            for i, _ in enumerate(response):
                if i >= 3:
                    break
        return dict(kind=kind, ms=(time.time() - start) * 1000)
    elif kind == "repeat":
        payload = dict(model="routeep", prompt="The capital of France is",
                       temperature=0, max_tokens=12)
    else:
        raise ValueError(kind)
    with post(base, "/v1/completions", payload, timeout) as response:
        result = json.load(response)
    choice = result["choices"][0]
    if result["usage"]["completion_tokens"] < 1:
        raise RuntimeError(f"empty completion: {result}")
    return dict(kind=kind, ms=(time.time() - start) * 1000, text=choice["text"],
                prompt_tokens=result["usage"]["prompt_tokens"],
                completion_tokens=result["usage"]["completion_tokens"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8133")
    p.add_argument("--duration", type=float, default=600, help="seconds")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    with urllib.request.urlopen(a.url + "/health", timeout=10) as response:
        assert response.status == 200
    kinds = ["short"] * 5 + ["long"] * 2 + ["sampled"] * 2 + ["abort", "repeat"]
    lock = threading.Lock()
    results, failures = [], []
    deadline = time.time() + a.duration

    def worker(index: int) -> None:
        rng = random.Random(a.seed * 1000 + index)
        while time.time() < deadline:
            # 每个 worker 随机停顿，使在途请求数持续波动，覆盖 DP 一侧空闲的批次。
            time.sleep(rng.choice([0, 0, 0, 0.05, 0.5, 2.0]))
            kind = rng.choice(kinds)
            try:
                item = one_request(a.url, kind, rng, a.timeout)
                with lock:
                    results.append(item)
            except Exception as exc:  # 记录后继续，最终统一判失败
                with lock:
                    failures.append(dict(kind=kind, error=f"{type(exc).__name__}: {exc}"[:300],
                                         at=time.time()))

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        list(pool.map(worker, range(a.concurrency)))
    with urllib.request.urlopen(a.url + "/health", timeout=10) as response:
        healthy = response.status == 200
    repeats = {r["text"] for r in results if r["kind"] == "repeat"}
    counts = {k: sum(r["kind"] == k for r in results) for k in sorted(set(kinds))}
    summary = dict(
        passed=healthy and not failures and len(repeats) <= 1,
        healthy_after=healthy, requests=len(results), failures=failures[:50],
        failure_count=len(failures), per_kind=counts,
        greedy_repeat_distinct_outputs=len(repeats),
        max_prompt_tokens=max((r.get("prompt_tokens", 0) for r in results), default=0),
        duration_s=a.duration, concurrency=a.concurrency,
    )
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(("PASS" if summary["passed"] else "FAIL"), json.dumps(
        {k: summary[k] for k in ("requests", "failure_count", "per_kind",
                                 "greedy_repeat_distinct_outputs", "max_prompt_tokens", "healthy_after")}))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
