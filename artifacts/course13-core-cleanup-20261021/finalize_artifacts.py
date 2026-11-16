from pathlib import Path
import hashlib,json,tarfile,subprocess
ROOT=Path(__file__).resolve().parents[2];ART=Path(__file__).resolve().parent;COURSE=ROOT/'code/course13'
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
# Confirm the earlier evidence was preserved, rather than replaced by this run.
oldroot=ROOT/'artifacts/course13-deepep-final-20261021'
oldmanifest=load(oldroot/'archive-manifest.json')
for name,info in oldmanifest.items():
 p=oldroot/name
 assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
before_global=load(oldroot/'pre-promotion-source_map.json');after_global=load(ROOT/'code/source_map.json')
assert {k:v for k,v in before_global.items() if k!='13'}=={k:v for k,v in after_global.items() if k!='13'}
local=load(COURSE/'SOURCE_MAP.json')
assert local['files']==after_global['13']
for name,info in local['files'].items():assert sha(COURSE/name)==info['sha256'],name
results=load(ART/'results.json');assert results['passed'] and not results['errors']
audit=load(ART/'source-audit.json');assert audit['passed']
state=load(ART/'remote-raw/progress.json');assert state['status']=='passed'
failed=[r['name'] for r in state['jobs'] if r.get('exit_code')]
assert failed==['return-transport']
# Keep the failed argv/log as evidence; its corrected retry must pass.
assert next(r for r in state['jobs'] if r['name']=='return-transport-retry1')['exit_code']==0
with tarfile.open(ART/'final-course13.tgz','w:gz') as archive:
 for name in sorted(set(local['files'])|{'SOURCE_MAP.json'}):archive.add(COURSE/name,arcname=name)
(ART/'course13-source-map-entry.json').write_text(json.dumps(after_global['13'],ensure_ascii=False,indent=2)+'\n')
checks=dict(passed=True,core_python_files=24,core_lines_before=audit['source_lines_before'],core_lines_after=audit['source_lines_after'],h100_unit=dict(passed=42,skipped=0),local_unit=dict(passed=34,skipped=8),tp_dp_nccl=dict(ranks=4,cases_per_rank=72),return_transport_nccl=dict(ranks=4,cases_per_rank=432),exact_precision=results['accuracy'],performance_fairness_passed=True,performance=dict(independent_engines_per_version=2,samples_per_case_per_version=10,cases={k:v['e2e_ms']['median_change_percent'] for k,v in results['cases'].items()},all_available_metric_p10_p90_ranges_overlap=all(q['p10_p90_overlap'] for row in results['cases'].values() for q in row.values() if q is not None),scope='decode_b1, decode_b32, prefill_b4_long; no claim of universal zero overhead or stable speedup'),kernel_sources_byte_identical=True,validated_source_matches_current_core=True,static_isolation=True,compileall=True,git_diff_check=True,dual_source_maps_valid=True,other_course_map_entries_unchanged=True,prior_experiment_archives_sha256_unchanged=True,prior_archive_count=len(oldmanifest),resolved_tooling_invocation_failure='first return_transport call omitted --output; argparse rejected it before compute; corrected retry passed; both logs preserved',not_rerun=['prior nine-configuration precision matrix in full','fresh native DeepEP performance sweep','other model/dtype/topology/backend versions','Nsight timeline or overlap'])
(ART/'final-checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
# The archive stores exactly the downloaded bytes; the base64 is transport-only.
(ART/'remote-raw.tgz.b64').unlink()
files=['before.tgz','after-source.tgz','final-course13.tgz','remote-raw.tgz','source-audit.json','results.json','final-checks.json','cleanup-only.patch','local-unit.log','layout.json','course13-source-map-entry.json']
manifest={name:dict(bytes=(ART/name).stat().st_size,sha256=sha(ART/name)) for name in files}
(ART/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k not in ('exact_precision',)},ensure_ascii=False,indent=2))
