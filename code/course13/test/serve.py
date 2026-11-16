"""Launch the normal vLLM OpenAI-compatible server with the course plugin."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
from mini_ep.integration.config import configure, model_metadata


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["native", "miniep"], default="miniep")
    p.add_argument("--dp-size", type=int, choices=[1, 2], default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--kv-cache-mib", type=int)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-batched-tokens", type=int, default=16384)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8133)
    a = p.parse_args()
    if a.backend == "native" and a.dp_size != 2:
        p.error(
            "native DeepEP requires TP=2/DP=2/EP=4; DP=1 NoDPEP is not a DeepEP control"
        )
    configure(a.backend)
    family = model_metadata(a.model)["model_family"]
    os.environ["MINIEP_MODEL_FAMILY"] = family
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        a.model,
        "--served-model-name",
        "miniep-qwen",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "2",
        "--enable-expert-parallel",
        "--enforce-eager",
        "--no-async-scheduling",
        "--seed",
        "13",
        "--max-model-len",
        str(a.max_model_len),
        "--max-num-seqs",
        "8",
        "--max-num-batched-tokens",
        str(a.max_batched_tokens),
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--gpu-memory-utilization",
        str(a.gpu_memory_utilization),
        "--all2all-backend",
        "deepep_high_throughput",
        "--moe-backend",
        "triton",
        "--distributed-executor-backend",
        "mp",
        "--host",
        a.host,
        "--port",
        str(a.port),
    ]
    if a.dp_size == 2:
        args += [
            "--data-parallel-size",
            "2",
            "--data-parallel-size-local",
            "2",
            "--data-parallel-backend",
            "mp",
        ]
    if a.kv_cache_mib is not None:
        args += ["--kv-cache-memory-bytes", str(a.kv_cache_mib * 1024 * 1024)]
    if family == "qwen3_5_moe":
        args += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
