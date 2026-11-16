"""Compare completed before/final primary suites without mixing profiles."""
import argparse
import csv
import io
import json
from pathlib import Path


def load(p):return json.loads(p.read_text())


def devices(environment):
    rows=csv.DictReader(io.StringIO(environment['gpu_before']))
    return sorted((row[next(k for k in row if k.strip()=='uuid')].strip(),
                   row[next(k for k in row if k.strip()=='driver_version')].strip()) for row in rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('before-root','after-root','before-summary','after-summary','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();before=load(a.before_summary);after=load(a.after_summary)
    errors=[]
    if not before['passed'] or not after['passed']:errors.append('incomplete primary comparison')
    if set(before['cases'])!=set(after['cases']):errors.append('workload sets differ')
    for trial in range(3):
        for backend in ('native','miniep'):
            for rank in range(4):
                left=a.before_root/f'trial{trial}-{backend}/rank{rank}';right=a.after_root/f'trial{trial}-{backend}/rank{rank}'
                x,y=load(left/'environment.json'),load(right/'environment.json')
                if load(left/'manifest.json')!=load(right/'manifest.json'):errors.append(f'literal inputs differ trial={trial} backend={backend} rank={rank}')
                for key in ('options','versions','checkpoint_digest','model_parameters','cpu_quota','cuda_version','nccl_version','profile','warmup','repeats'):
                    if x.get(key)!=y.get(key):errors.append(f'{key} differs trial={trial} backend={backend} rank={rank}')
                if devices(x)!=devices(y):errors.append(f'GPU UUID/driver differs trial={trial} rank={rank}')
                tokens=lambda folder:[(q['case'],q['repetition'],q['token_ids']) for q in load(folder/'samples.json')]
                if tokens(left)!=tokens(right):errors.append(f'free sequences differ trial={trial} backend={backend} rank={rank}')
    cases={}
    for name in sorted(set(before['cases']) & set(after['cases'])):
        x,y=before['cases'][name],after['cases'][name]
        def change(first,last):return (last/first-1)*100
        result=dict(before_miniep_over_deepep=x['miniep_over_deepep'],final_miniep_over_deepep=y['miniep_over_deepep'],
            before_miniep_e2e_ms=x['miniep']['e2e_ms'],final_miniep_e2e_ms=y['miniep']['e2e_ms'],
            final_deepep_e2e_ms=y['native']['e2e_ms'],
            cleanup_e2e_median_change_percent=change(x['miniep']['e2e_ms']['median'],y['miniep']['e2e_ms']['median']),
            native_e2e_median_change_percent=change(x['native']['e2e_ms']['median'],y['native']['e2e_ms']['median']),
            cleanup_native_normalized_e2e_change_percent=change(x['miniep_over_deepep']['e2e_latency'],y['miniep_over_deepep']['e2e_latency']),
            final_prefill_ms={k:y[k]['prefill_batch_ms'] for k in ('native','miniep')},
            final_decode_step_ms={k:y[k]['decode_step_ms'] for k in ('native','miniep')},
            final_generated_tokens_per_second={k:y[k]['generated_tokens_per_second'] for k in ('native','miniep')},
            final_prefill_tokens_per_second={k:y[k]['prefill_tokens_per_second'] for k in ('native','miniep')})
        result['cleanup_e2e_p10_p90_overlap']=max(x['miniep']['e2e_ms']['p10'],y['miniep']['e2e_ms']['p10'])<=min(x['miniep']['e2e_ms']['p90'],y['miniep']['e2e_ms']['p90'])
        cases[name]=result
    output=dict(passed=not errors,errors=sorted(set(errors)),cases=cases,
        statistic='15 per-repetition maximum-rank primary samples per implementation/case, three independent engines, median and p10/p90; profiles excluded',
        interpretation='Measured median changes and overlap are descriptive; no confidence or causal speedup claim inferred from byte reduction.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=output['passed'],errors=output['errors'])))
    return 0 if output['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
