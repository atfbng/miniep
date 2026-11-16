"""为精确 DeepEP 验收固定 workload 清单。

同一份清单会被写成字面 Token ID，保证 native 和 RouteEP 使用完全相同的输入。
TP 对等 Rank 共享清单；DP 副本通过旋转交换短/长输入，覆盖不同调度边界。
"""

from validation.corpus import PROMPTS
from validation.dp_cases import cases as boundary_cases


def build_cases(
    tokenizer,
    suite: str,
    dp_rank: int,
    corpus: str,
    limit: int,
    steps: int,
    dp_size: int = 2,
) -> list[dict]:
    """返回带名称的批次，并保存字面 prompt ID 和预期生成长度。

    TP 对等 Rank 收到相同清单；DP 输入会轮换短/长两半。调度器补的 padding 或
    dummy 行不算真实请求，不能用来伪造“有输入”的通过结果。
    """
    if dp_size not in (1, 2) or not 0 <= dp_rank < dp_size:
        raise ValueError(
            f"expected DP=1/2 and dp_rank in group, got size={dp_size}, rank={dp_rank}"
        )
    if suite == "performance":
        from benchmarks.deepep_workloads import build_workloads

        return build_workloads(tokenizer, dp_rank)
    if suite == "boundaries":
        result = boundary_cases(tokenizer, dp_rank, steps)
        # DP=1 时没有对等副本维持推理；不能把一次 no-op 引擎调用
        # 标成“单边空闲”的分布式测试。
        return [
            case
            for case in result
            if dp_size > 1 or case["name"] not in ("idle_dp0", "idle_dp1")
        ]
    indices = [0, 1, 16, 17] if suite == "long" else list(range(limit))
    prompts = []
    filler = (
        "Experiment log: tokens are routed to experts. NCCL exchanges tokens between GPUs. "
        "We check reference values before measuring performance. "
    )
    for i in indices:
        source = (i + (16 if dp_rank == 1 and corpus == "rotated" else 0)) % len(
            PROMPTS
        )
        content = (
            PROMPTS[source] if source < 16 else filler * 180 + "\n" + PROMPTS[source]
        )
        text = tokenizer.apply_chat_template(
            [dict(role="user", content=content)],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tokenizer.encode(text, add_special_tokens=False)
        if source >= 16:
            ids = ids[:1984] + ids[-64:]
        prompts.append(dict(prompt_token_ids=ids))
    result = []
    for batch in (1, 4):
        for start in range(0, len(prompts), batch):
            result.append(
                dict(
                    name=f"b{batch}-i{start:02d}",
                    prompts=prompts[start : start + batch],
                    steps=steps,
                )
            )
    return result
