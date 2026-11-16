"""Narrow structural equivalence audit against the pre-cleanup source archive.

Only the enumerated dead fields/branch/wrapper and a direct-call alias are
normalized. Everything else, including all GPU kernels, must match exactly.
"""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]
COURSE=ROOT/'code/course13'
ART=Path(__file__).resolve().parent
counts={}
def note(name): counts[name]=counts.get(name,0)+1
class Normalize(ast.NodeTransformer):
 def __init__(self,path): self.path=path
 def visit_Expr(self,node):
  if isinstance(node.value,ast.Constant) and isinstance(node.value.value,str): return None
  return self.generic_visit(node)
 def visit_AnnAssign(self,node):
  if self.path=='mini_ep/model/router.py' and isinstance(node.target,ast.Name) and node.target.id=='tokens_per_expert':
   assert ast.unparse(node)=='tokens_per_expert: torch.Tensor | None = None'
   note('unused_router_count');return None
  return self.generic_visit(node)
 def visit_FunctionDef(self,node):
  if self.path=='mini_ep/integration/runner.py' and node.name=='_fixed_rows_per_rank':
   body=[x for x in node.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
   assert len(body)==1 and ast.unparse(body[0])=='return fixed_rows_per_rank(self.fixed_a2a_mode, sizes, top_k)'
   note('capacity_forwarder');return None
  return self.generic_visit(node)
 def visit_Assign(self,node):
  if self.path=='mini_ep/model/expert.py' and ast.unparse(node)=='compute = swiglu_triton':
   note('expert_alias');return None
  return self.generic_visit(node)
 def visit_If(self,node):
  if self.path=='mini_ep/integration/runner.py' and ast.unparse(node.test)=='not self.input_is_tp_sharded' and len(node.body)==1 and ast.unparse(node.body[0])=='shared_output = get_tp_group().all_reduce(shared_output)':
   assert not node.orelse
   note('unreachable_shared_reduce');return None
  return self.generic_visit(node)
 def visit_Call(self,node):
  if self.path=='mini_ep/integration/runner.py' and ast.unparse(node.func)=='self._fixed_rows_per_rank':
   assert not node.keywords and [ast.unparse(x) for x in node.args]==['ep_sizes','routing.expert_ids.shape[1]']
   node.func=ast.Name(id='fixed_rows_per_rank',ctx=ast.Load())
   node.args.insert(0,ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='fixed_a2a_mode',ctx=ast.Load()))
   note('capacity_call_inline')
  if self.path=='mini_ep/model/expert.py' and isinstance(node.func,ast.Name) and node.func.id=='compute':
   node.func.id='swiglu_triton';note('expert_direct_call')
  removed={'mini_ep/integration/runner.py':'router_logits_reused','mini_ep/model/moe.py':'fixed_pack_backend_requested'}.get(self.path)
  for kw in list(node.keywords):
   if removed is not None and kw.arg==removed:
    expected={'router_logits_reused':'self.gate is None and router_logits is not None','fixed_pack_backend_requested':'fixed_pack_backend'}[removed]
    assert ast.unparse(kw.value)==expected
    note(removed);node.keywords.remove(kw)
  return self.generic_visit(node)

rows=[]
runner_tree=ast.parse((COURSE/'mini_ep/integration/runner.py').read_text())
guards=[n for n in ast.walk(runner_tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='self._shared_experts is not None and (not self.input_is_tp_sharded)']
assert len(guards)==1 and len(guards[0].body)==1 and isinstance(guards[0].body[0],ast.Raise)
with tarfile.open(ART/'before.tgz') as archive:
 old={m.name:archive.extractfile(m).read() for m in archive if m.isfile()}
new={str(p.relative_to(COURSE)):p.read_bytes() for p in (COURSE/'mini_ep').rglob('*.py')}
assert set(old)==set(new)
for name in sorted(old):
 before=ast.dump(Normalize(name).visit(ast.parse(old[name])))
 after=ast.dump(Normalize(name).visit(ast.parse(new[name])))
 assert before==after,name
 rows.append(dict(file=name,before_sha256=hashlib.sha256(old[name]).hexdigest(),after_sha256=hashlib.sha256(new[name]).hexdigest(),byte_identical=old[name]==new[name],normalized_execution_ast_equal=True))
expected=['unused_router_count','capacity_forwarder','unreachable_shared_reduce','capacity_call_inline','router_logits_reused','expert_alias','expert_direct_call','fixed_pack_backend_requested']
assert counts==dict.fromkeys(expected,1),counts
report=dict(passed=True,changes=counts,files=rows,source_lines_before=sum(x.count(b'\n') for x in old.values()),source_lines_after=sum(x.count(b'\n') for x in new.values()),reason='Initialization rejects shared experts without input_is_tp_sharded; the removed shared TP reduce is unreachable on every accepted runner. All kernels and communication/rounding operations remain byte-identical. This audit proves the enumerated edits only; runtime checks and measurements are separate.')
(ART/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='files'}))
