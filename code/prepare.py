"""Materialise a separate, hash-checked run directory; never alter the source data."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys
REPO=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def within(root,relative):
 p=(root/relative).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError('Path leaves root: '+relative)
 return p
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source-root',type=Path,required=True,help='External files arranged by input/required_files.json')
 p.add_argument('--destination',type=Path,help='New directory outside this GitHub repository')
 p.add_argument('--profile',choices=['figures','models','full'],default='full')
 p.add_argument('--check-only',action='store_true')
 a=p.parse_args();src=a.source_root.resolve()
 rows=json.loads((REPO/'input/required_files.json').read_text())
 selected=[r for r in rows if a.profile=='full' or a.profile in r['profiles']]
 failures=[]
 for i,r in enumerate(selected):
  f=within(src,r['path'])
  if not f.is_file():failures.append((r['path'],'missing'))
  elif f.stat().st_size!=r['bytes'] or digest(f)!=r['sha256']:failures.append((r['path'],'checksum mismatch'))
  if i%500==0:print('Checked external files',i,'/',len(selected),flush=True)
 if failures:print(json.dumps({'failures':failures},indent=2));return 2
 print('All',len(selected),'selected external files match their hashes.')
 if a.check_only:return 0
 if a.destination is None:p.error('--destination is required unless --check-only is used')
 dst=a.destination.resolve()
 if dst.exists() or dst.is_relative_to(REPO) or dst==src or src.is_relative_to(dst):p.error('Use a new directory outside the repository and source tree')
 dst.mkdir(parents=True)
 for r in json.loads((REPO/'environment/code_files.json').read_text()):
  if 'runtime_path' not in r:continue
  f=REPO/r['path'];assert digest(f)==r['sha256']
  out=within(dst,r['runtime_path']);out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,out)
 for r in selected:
  out=within(dst,r['path']);out.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(within(src,r['path']),out)
  if digest(out)!=r['sha256']:raise RuntimeError('Copied file verification failed: '+r['path'])
 (dst/'tools').mkdir(exist_ok=True)
 shutil.copy2(REPO/'code/notebook_runtime.py',dst/'tools/notebook_runtime.py')
 (dst/'RUN_WORKSPACE.json').write_text(json.dumps({'profile':a.profile,'input_count':len(selected),'reference_version':'2026-09-11','generated_results_copied':False},indent=2))
 print('Created run workspace:',dst)
 return 0
if __name__=='__main__':sys.exit(main())
