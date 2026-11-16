from pathlib import Path
import base64,io,tarfile,json,hashlib
root=Path('/experiment/core-cleanup-20261021')
state=json.loads((root/'progress.json').read_text())
assert state['status']=='passed',state
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w:gz') as a:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.jsonl','.log'):
   a.add(p,arcname=p.relative_to(root))
print(base64.b64encode(buf.getvalue()).decode())
