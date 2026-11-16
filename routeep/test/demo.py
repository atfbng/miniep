"""四卡 TP2/DP2/EP4 生产推理示例，不依赖任何精度探针。

启动方式：
    torchrun --standalone --nproc-per-node=4 test/demo.py --model ... --output ...
"""

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mini_ep.integration.config import configure, engine_options


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--backend", choices=["native", "miniep"], default="miniep")
    p.add_argument("--output", required=True)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--prompt", default="请用一句话说明什么是混合专家模型。")
    a = p.parse_args()
    if int(os.environ.get("WORLD_SIZE", "0")) != 4:
        p.error("use torchrun --nproc-per-node=4")
    configure(a.backend)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    root = Path(a.output) / f"rank{rank}"
    root.mkdir(parents=True, exist_ok=False)
    from vllm import LLM, SamplingParams

    llm = LLM(**engine_options(a.model, dp_size=2, external=True))
    text = llm.get_tokenizer().apply_chat_template(
        [dict(role="user", content=a.prompt)],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts = [
        dict(
            prompt_token_ids=llm.get_tokenizer().encode(text, add_special_tokens=False)
        )
    ]
    llm.enqueue(
        prompts, SamplingParams(temperature=0, max_tokens=a.max_tokens), use_tqdm=False
    )
    outputs = llm.wait_for_completion(use_tqdm=False)
    records = [
        dict(text=q.outputs[0].text, token_ids=q.outputs[0].token_ids) for q in outputs
    ]
    (root / "outputs.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2)
    )
    print(json.dumps(dict(rank=rank, outputs=records), ensure_ascii=False), flush=True)
    from vllm.distributed import destroy_model_parallel, destroy_distributed_environment

    del llm
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
