"""Explicit stage runner for a prepared workspace. No acquisition or uploads."""
from pathlib import Path
import argparse,json,subprocess,sys,os,hashlib,time,shutil
REPO=Path(__file__).resolve().parents[1]
P='submission_package_reference_style_20260910'
BASE='research_process/portable_reproduction_20260910'
CAP=BASE+'/capsule'
STAGES={
 'raw':[(CAP+'/run_raw_stage.py',[])],
 'seed':[(CAP+'/run_seed_stage.py',[])],
 'baci':[(CAP+'/run_baci_stage.py',[])],
 'historical':[(CAP+'/run_historical_stage.py',['--rebuilt-seed'])],
 'allocation':[(CAP+'/run_allocation_stage.py',[])],
 'routes':[(BASE+'/rebuild_routes_check.py',[])],
 'corridor-uncertainty':[(CAP+'/run_conditional_corridor.py',['--draws-per-design','250'])],
 'raw-uncertainty':[(CAP+'/run_raw_conditional.py',['--prepare']),(CAP+'/run_raw_conditional.py',['--start','1','--end','40'])],
 'no-propagation':[('research_process/reviewer_strengthening_20260910/historical.py',[])],
 'route-avoidance':[('research_process/reviewer_strengthening_20260910/routes.py',[])],
 'preferences':[('research_process/reviewer_strengthening_20260910/allocation.py',[])],
 'figures':[(P+'/reproduction/'+n,[]) for n in ['render_strengthened.ipynb','render_extended_data.ipynb','render_figure1_confirmed.ipynb']],
}
CONTRACTS={'run_historical_stage.py':'HISTORICAL_INPUT_MANIFEST.json','run_allocation_stage.py':'ALLOCATION_INPUT_MANIFEST.json','run_conditional_corridor.py':'CONDITIONAL_INPUT_MANIFEST.json','run_raw_conditional.py':'RAW_CONDITIONAL_INPUT_MANIFEST.json','run_precision_completion.py':'PRECISION_COMPLETION_INPUT_MANIFEST.json'}
OUTPUTS={'run_raw_stage.py':'outputs','run_seed_stage.py':'outputs/seed','run_baci_stage.py':'outputs/baci','run_historical_stage.py':'outputs/historical_rebuilt_seed','run_allocation_stage.py':'outputs/allocation'}
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rebind_generated_inputs(w,script):
 name=CONTRACTS.get(script)
 if not name:return
 cap=w/CAP;path=cap/name;rows=json.loads(path.read_text())
 receipts_path=w/'COMPLETED_OUTPUT_HASHES.json'
 receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else {}
 edits=[]
 for r in rows:
  if not r.get('file','').startswith('outputs/'):continue
  target=cap/r['file']
  if not target.is_file():raise RuntimeError('Run the prerequisite stage first: '+r['file'])
  h=digest(target);record=receipts.get(r['file'])
  if not record or record['sha256']!=h:raise RuntimeError('Generated input has no matching successful-run receipt: '+r['file'])
  if h!=r['sha256']:edits.append({'file':r['file'],'original_sha256':r['sha256'],'run_sha256':h});r['sha256']=h
 if edits:
  archive=cap/'original_contracts';archive.mkdir(exist_ok=True)
  if not (archive/name).exists():shutil.copy2(path,archive/name)
  path.write_text(json.dumps(rows,indent=2))
  log=w/'RUN_MANIFEST_REBINDINGS.json';history=json.loads(log.read_text()) if log.exists() else []
  history.append({'manifest':name,'reason':'New successful-run bytes, including gzip metadata; frozen source/fixture hashes and numerical tolerances unchanged','changes':edits})
  log.write_text(json.dumps(history,indent=2))
def record_outputs(w,script):
 relative=OUTPUTS.get(script)
 if not relative:return
 cap=w/CAP;folder=cap/relative
 files=folder.glob('*') if script=='run_raw_stage.py' else folder.rglob('*')
 path=w/'COMPLETED_OUTPUT_HASHES.json';records=json.loads(path.read_text()) if path.exists() else {}
 for f in files:
  if f.is_file():records[f.relative_to(cap).as_posix()]={'sha256':digest(f),'successful_script':script}
 path.write_text(json.dumps(records,indent=2))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=list(STAGES)+['seed-coordinates','allocation-uncertainty'])
 p.add_argument('--workspace',type=Path,required=True);p.add_argument('--python',default=sys.executable)
 p.add_argument('--node',default='node');p.add_argument('--pdfjs',type=Path);p.add_argument('--dry-run',action='store_true')
 a=p.parse_args();w=a.workspace.resolve()
 if not (w/'RUN_WORKSPACE.json').exists():p.error('Not a prepared workspace; use prepare_workspace.py first')
 if a.stage=='seed-coordinates':
  if not a.pdfjs:p.error('--pdfjs must point to pdfjs-dist/legacy/build/pdf.mjs (recorded version 5.6.205)')
  cmds=[[a.node,str(w/CAP/'extract_seed_coordinates.mjs'),str(a.pdfjs.resolve())]]
 else:
  jobs=STAGES.get(a.stage,[])
  if a.stage=='allocation-uncertainty':
   jobs=[(CAP+'/run_precision_completion.py',['--branch',branch,'--rho',str(rho),'--draw',str(draw)]) for branch in ['legacy','rebuilt'] for rho in [0.,.8] for draw in range(1,6)]
  cmds=[[a.python,'-I',str(w/'tools/notebook_runtime.py'),str(w/f),*args] if f.endswith('.ipynb') else [a.python,'-I',str(w/f),*args] for f,args in jobs]
 env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUTF8='1')
 for cmd in cmds:
  print(json.dumps(cmd),flush=True)
  if not a.dry_run:
   script=Path(cmd[2] if cmd[1]=='-I' else cmd[1]).name
   rebind_generated_inputs(w,script)
   subprocess.run(cmd,cwd=w,env=env,check=True)
   record_outputs(w,script)
if __name__=='__main__':main()
