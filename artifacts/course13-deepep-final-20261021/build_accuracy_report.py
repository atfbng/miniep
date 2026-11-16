"""Re-audit complete raw logs from the frozen v6 numerical implementation.

Run inside the final owned GPU5 container, which sees /workspace and /experiment.
This postprocessor never runs inference or substitutes tolerant acceptance.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def load(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--course',type=Path,required=True)
    p.add_argument('--reader',type=Path,required=True)
    p.add_argument('--seed13',type=Path,required=True)
    p.add_argument('--gpu3',type=Path,required=True)
    p.add_argument('--empty',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('audited_reader',a.reader);reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    core={str(f.relative_to(a.course)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.course/'mini_ep').rglob('*.py')}
    definitions={
        'same-13-r1':(a.seed13/'candidate-same-13-r1','corpus','same',13,64),
        'rotated-13-r1':(a.seed13/'candidate-rotated-13-r1','corpus','rotated',13,64),
        'rotated-13-r2':(a.seed13/'candidate-rotated-13-r2','corpus','rotated',13,64),
        'rotated-13-r3':(a.seed13/'candidate-rotated-13-r3','corpus','rotated',13,64),
        'boundaries-13':(a.seed13/'candidate-boundaries-13','boundaries','rotated',13,64),
        'long-13':(a.seed13/'candidate-long-13','long','rotated',13,256),
        'rotated-37':(a.gpu3/'full-seed37/candidate-rotated-37','corpus','rotated',37,64),
        'rotated-123':(a.gpu3/'full-seed123/candidate-rotated-123','corpus','rotated',123,64),
        'performance-13':(a.gpu3/'performance-candidate','performance','rotated',13,128),
    }
    checkpoint=load(a.checkpoint)
    results={};errors=[]
    for name,(root,suite,corpus,seed,steps) in definitions.items():
        if not (root/'rank0/environment.json').exists():errors.append(f'not executed: {name}');continue
        env=load(root/'rank0/environment.json');command=env['command'];ref=Path(command[command.index('--reference')+1])
        proof=reader.summarize_run(root,ref,world_size=4)
        if not proof['passed']:errors.append(f'raw acceptance failed: {name}')
        expected_options=dict(dtype='bfloat16',tensor_parallel_size=2,data_parallel_size=2,
            enable_expert_parallel=True,enforce_eager=True,async_scheduling=False,
            all2all_backend='deepep_high_throughput',enable_prefix_caching=False,
            enable_chunked_prefill=False,enable_eplb=False,enable_dbo=False,quantization=None,
            max_model_len=2304,max_num_batched_tokens=8192 if suite=='performance' else 2304,
            max_num_seqs=32 if suite=='performance' else 4)
        counts={};policies=set();returns=set();teacher_rows=0;native_check_rows=0
        for rank in range(4):
            directory=root/f'rank{rank}';value=load(directory/'environment.json');frozen=value['spec']
            if any(frozen.get(k)!=v for k,v in dict(suite=suite,corpus=corpus,seed=seed,steps=steps,limit=32).items()):errors.append(f'wrong requested scope {name} rank={rank}')
            if any(frozen['options'].get(k)!=v for k,v in expected_options.items()):errors.append(f'wrong inference options {name} rank={rank}')
            recorded={k:v for k,v in frozen['source_sha256'].items() if k.startswith('mini_ep/')}
            if recorded!=core:errors.append(f'numerical source differs {name} rank={rank}')
            signature=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
            if signature!=value['signature']:errors.append(f'environment signature invalid {name} rank={rank}')
            if frozen.get('checkpoint')!=checkpoint['digest']:errors.append(f'checkpoint differs {name} rank={rank}')
            if frozen['versions']['vllm']!='0.29.0':errors.append(f'wrong vLLM version {name}')
            counts[rank]={phase:sum(sum(layers.values()) for layers in table.values()) for phase,table in proof['ranks'][str(rank)]['model_calls'].items()}
            for q in reader.rows(directory/f'moe-rank{rank}.jsonl'):
                policies.add(q.get('return_dtype_requested'));returns.add(q.get('return_buffer_dtype'))
            teacher_rows+=sum(q['rows'] for q in reader.rows(directory/f'logits-rank{rank}.jsonl'))
            native_check_rows+=sum(q['rows'] for q in reader.rows(directory/f'native-check-rank{rank}.jsonl'))
        if policies!={'auto'} or returns!={'torch.bfloat16'}:errors.append(f'default return policy not exercised {name}')
        results[name]=dict(passed=proof['passed'],root=str(root),native_reference=str(ref),
            unique_sequences=proof['request_sequences'],unique_generated_tokens=proof['generation_positions'],
            teacher_positions_including_tp=teacher_rows,native_check_positions_including_tp=native_check_rows,
            moe_calls_by_rank_and_phase=counts,actual_return_buffer_dtypes=sorted(returns),
            return_policy=sorted(policies),exact_values=True if proof['passed'] else False,
            full_acceptance=proof)
    repeats={}
    for repeat in (2,3):
        path=a.seed13/f'native-rotated-13-r{repeat}-repeat.json'
        if not path.exists():errors.append(f'missing native repeat {repeat}');continue
        value=load(path);repeats[str(repeat)]=value
        if not value.get('passed') or value.get('logits_positions_including_tp')!=16384:errors.append(f'native repeated baseline invalid {repeat}')
    empty={}
    for rank in range(4):
        path=a.empty/f'rank{rank}'
        if not (path/'completed.json').exists():errors.append(f'missing real-empty rank={rank}');continue
        value=load(path/'completed.json');records=load(path/'layers.json');empty[str(rank)]=value
        keys={(q['seed'],tuple(q['counts']),q['fixed_mode'],q['layer']) for q in records}
        expected={(seed,counts,mode,layer) for seed in (13,37) for counts in ((0,2),(2,0),(0,16),(16,0)) for mode in ('auto','off') for layer in range(48)}
        if (not value['passed'] or value['cases']!=768 or value['physical_empty_cases']!=384 or keys!=expected
                or len(records)!=len(keys) or any(not q['exact_equal'] or not q['router_exact'] or q['return_buffer_dtype']!='torch.bfloat16' or q['return_send_bytes']!=q['recv_rows']*2048*2 for q in records)):
            errors.append(f'incomplete or incorrect physical-empty layer API rank={rank}')
    result=dict(passed=not errors and len(results)==9,errors=errors,target='real Qwen3-30B-A3B BF16 / vLLM 0.29.0 DeepEP HT / TP2 DP2 EP4',
        numerical_source_sha256=core,checkpoint_digest=checkpoint["digest"],configurations=results,native_repeats=repeats,real_empty_layers=empty,
        precision='finite elementwise equality, exact route IDs/slot/FP32 weights, full-vocabulary teacher logits and all free tokens; no tolerant strict_passed used',
        unclaimed=['other models','full FP16 model','other DeepEP binaries/hardware/topologies','performance from precision-run timings'])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'],errors=errors,configurations=len(results))))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
