"""DP 边界共用 workload；即使某个 DP Rank 没有请求，也必须参与推理。

覆盖 balanced、uneven、空 Rank 等场景，确保集合通信不会因为某个 DP 副本
没有 Token 而顺序错位。
"""


def cases(tokenizer, dp_rank: int, steps: int) -> list[dict]:
    def prompt(text, length=None):
        rendered = tokenizer.apply_chat_template(
            [dict(role="user", content=text)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(rendered, add_special_tokens=False)
        if length is not None:
            ids = ids[: length - 64] + ids[-64:]
        return dict(prompt_token_ids=ids)

    long = (
        "Tokens travel to experts on different GPUs. We verify every numerical result. "
        * 160
    )
    a = prompt(
        "Explain expert parallelism in one sentence."
        if dp_rank == 0
        else "请解释什么是混合专家模型。"
    )
    b = prompt(
        "Compute 13 times 17."
        if dp_rank == 0
        else "Write a Python function that adds two numbers."
    )
    return [
        dict(name="balanced", prompts=[a, b], steps=steps),
        dict(
            name="uneven",
            prompts=[a] if dp_rank == 0 else [prompt(long + " Summarize.", 512), b],
            steps=max(1, steps // 2) if dp_rank == 0 else steps,
        ),
        dict(name="idle_dp0", prompts=[] if dp_rank == 0 else [a], steps=steps),
        dict(name="idle_dp1", prompts=[a] if dp_rank == 0 else [], steps=steps),
        dict(name="single_decode", prompts=[b], steps=steps),
        dict(
            name="staggered_finish",
            prompts=[a, b],
            steps=steps,
            request_steps=(
                [max(1, steps // 4), steps]
                if dp_rank == 0
                else [steps, max(1, steps // 2)]
            ),
        ),
        dict(
            name="wide_prefill",
            prompts=(
                [prompt(long + " Short.", 128), prompt(long + " Compare.", 768)]
                if dp_rank == 0
                else [prompt(long + " Medium.", 256), prompt(long + " 对比。", 1024)]
            ),
            steps=max(1, steps // 2),
        ),
        dict(
            name="long_decode",
            prompts=[a],
            steps=max(32, steps * 2),
        ),
        dict(
            name="long_prefill",
            prompts=(
                [prompt(long + " Explain the experiment.", 512), a]
                if dp_rank == 0
                else [prompt(long + " 请概括实验。", 768)]
            ),
            steps=steps,
        ),
        dict(
            name="max_prefill",
            prompts=[
                prompt(
                    long + (" Explain." if dp_rank == 0 else " 请解释。"),
                    2040,
                )
            ],
            steps=1,
        ),
    ]
