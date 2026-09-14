"""Check source integrity, Python syntax and the packaged input index without executing models."""
from pathlib import Path
import ast,hashlib,json,sys
R=Path(__file__).resolve().parents[1]
errors=[];rows=json.loads((R/'environment/code_files.json').read_text())
for row in rows:
 p=R/row['path']
 if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:errors.append([row['path'],'hash mismatch']);continue
 if p.suffix=='.py':
  try:ast.parse(p.read_text(encoding='utf-8-sig'))
  except SyntaxError as e:errors.append([row['path'],str(e)])
 if p.suffix=='.ipynb':
  try:
   notebook=json.loads(p.read_text(encoding='utf-8'))
   for cell in notebook['cells']:
    if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
  except (SyntaxError,ValueError) as e:errors.append([row['path'],str(e)])
print(json.dumps({'source_files_checked':len(rows),'errors':errors},indent=2))
sys.exit(bool(errors))
