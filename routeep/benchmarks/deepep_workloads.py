"""正确性检查和 DeepEP 计时共用的字面 Token workload。"""

PRESETS = (
    ("decode_b1", 1, 128, 128),
    ("decode_b4", 4, 128, 128),
    ("decode_b32", 32, 128, 128),
    ("prefill_b1_long", 1, 2048, 1),
    ("prefill_b4_short", 4, 128, 1),
    ("prefill_b4_long", 4, 2048, 1),
)


def build_workloads(tokenizer, dp_rank: int) -> list[dict]:
    """返回长度一致的批次，并在执行前保存字面 Token ID。

    TP 对等 Rank 共享输入；不同 DP 副本使用不同文本但保持相同形状。计时使用
    8192 Token 的调度预算和 32 个序列槽位，因此 batch 32 decode 与 batch 4
    长 prefill 都能在一次初始 Prefill 中容纳。
    """
    if dp_rank not in (0, 1):
        raise ValueError(f"expected DP rank 0/1, got {dp_rank}")
    result = []
    for name, batch, length, steps in PRESETS:
        prompts = []
        for index in range(batch):
            text = (
                f"Experiment {dp_rank}-{index}. "
                "Tokens travel to experts. We compare identical inference workloads. "
            ) * 300
            text += f"\nExplain expert parallelism for reader {dp_rank}-{index}."
            rendered = tokenizer.apply_chat_template(
                [dict(role="user", content=text)],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) < length:
                raise ValueError(f"tokenizer produced {len(ids)} tokens, need {length}")
            ids = ids[: length - 64] + ids[-64:]
            prompts.append(dict(prompt_token_ids=ids))
        result.append(
            dict(
                name=name,
                batch_size=batch,
                seq_len=length,
                prompts=prompts,
                steps=steps,
            )
        )
    return result
