"""Literal-token workloads shared by accuracy checks and DeepEP timings."""

PRESETS = (
    ("decode_b1", 1, 128, 128),
    ("decode_b4", 4, 128, 128),
    ("decode_b32", 32, 128, 128),
    ("prefill_b1_long", 1, 2048, 1),
    ("prefill_b4_short", 4, 128, 1),
    ("prefill_b4_long", 4, 2048, 1),
)


def build_workloads(tokenizer, dp_rank: int) -> list[dict]:
    """Return same-length batches; literal IDs are persisted before execution.

    TP peers share inputs; DP replicas receive different text at the same shape.
    Timing uses an 8192-token scheduling budget and 32 sequence slots. Batch 32
    decode and batch 4 long prefill therefore fit their initial prefill together.
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
