"""扩展 Peer/DeepEP 实验预先固定的请求矩阵。

batch size 指每个 DP 副本；记录语料、seed、字面输入 IDs 和长度。
禁用前缀缓存和 chunked prefill，避免把调度策略变化混入传输比较。"""
from __future__ import annotations
import json
from pathlib import Path

CORPORA = (
    'Explain expert parallel inference, routing weights, memory layouts, and how distributed GPU systems move tokens between experts. ',
    'Review this Python code and explain its correctness: def combine(parts):\n    total = 0\n    for part in parts:\n        total = total + part\n    return total\n',
    '请分析大模型推理中的通信与计算开销。Compare latency, throughput, numerical precision, GPU memory, and reproducibility. 给出可验证的分析步骤。',
)


def experiment_cases() -> list[dict]:
    """18个 prefill、7个 decode、2个混合长度场景；每 DP 最大输入32768。"""
    result=[]
    for batch in (1,4,8,16):
        for length in (128,512,2048):
            result.append(dict(name=f'prefill_b{batch}_s{length}',batch_size=batch,seq_len=length,steps=1))
    for length in (128,512,1024):
        result.append(dict(name=f'prefill_b32_s{length}',batch_size=32,seq_len=length,steps=1))
    for batch in (1,4,8):
        result.append(dict(name=f'prefill_b{batch}_s4096',batch_size=batch,seq_len=4096,steps=1))
    for batch in (1,4,8,16,32):
        result.append(dict(name=f'decode_b{batch}_s128',batch_size=batch,seq_len=128,steps=64))
    for batch in (1,16):
        result.append(dict(name=f'decode_b{batch}_s2048',batch_size=batch,seq_len=2048,steps=64))
    lengths=[64,128,256,512,1024,2048,4096,256]
    for steps in (1,64):
        result.append(dict(name=f'ragged_b8_o{steps}',batch_size=8,seq_len=None,lengths=lengths,steps=steps))
    return result


def build_extended_workloads(tokenizer, dp_rank: int, spec_path: str) -> list[dict]:
    """计时前校验 JSON 并生成确定的字面 Token 输入，保证不同后端用同一请求。"""
    spec=json.loads(Path(spec_path).read_text())
    seed=int(spec['seed']);corpus=int(spec['corpus'])
    if dp_rank not in (0,1) or corpus not in range(len(CORPORA)):
        raise ValueError('requires DP rank 0/1 and corpus 0/1/2')
    cases=spec['cases'];names=set();cache={};result=[]
    for case in cases:
        name=case['name'];batch=int(case['batch_size']);steps=int(case['steps'])
        lengths=case.get('lengths') or [int(case['seq_len'])]*batch
        if name in names or batch<1 or len(lengths)!=batch or min(lengths)<64 or steps<1:
            raise ValueError(f'invalid workload: {case}')
        names.add(name);prompts=[]
        for index,length in enumerate(lengths):
            if index not in cache:
                text=(f'Experiment seed={seed}, DP={dp_rank}, request={index}.\n'+CORPORA[corpus]*650+
                      f'\nGive a precise answer for reader {dp_rank}-{index}, experiment {seed}.')
                rendered=tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,
                              add_generation_prompt=True,enable_thinking=False)
                cache[index]=tokenizer.encode(rendered,add_special_tokens=False)
            ids=cache[index]
            if len(ids)<length:raise ValueError(f'prompt has {len(ids)} IDs, need {length}')
            # 保留模板末尾64个 ID，在固定输入长度的同时保留相同的生成提示尾部。
            selected=ids[:length-64]+ids[-64:]
            prompts.append(dict(prompt_token_ids=selected))
        result.append(dict(name=name,batch_size=batch,seq_len=case.get('seq_len'),
                           lengths=lengths,steps=steps,prompts=prompts,seed=seed,corpus=corpus))
    return result
