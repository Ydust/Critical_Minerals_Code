"""Build an isolated allocation-draw entry by explicit, checked driver substitutions."""
from pathlib import Path
import ast,hashlib,json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CAP=HERE/'capsule';REF=ROOT/'research_process/unified_reanalysis_20260909'
files=[]
def record(p,role):files.append(dict(file=p.relative_to(CAP).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role=role))
def copy(src,rel,role):
    p=CAP/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);record(p,role)
def functions(src,names):
    code=src.read_text(encoding='utf-8');tree=ast.parse(code)
    return '\n\n'.join(ast.get_source_segment(code,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names)
copy(REF/'global_allocation_pilot.py','code/allocation_conditional_original.py','frozen draw driver')
copy(REF/'validate_results.py','code/allocation_validation_original.py','frozen independent audit')
copy(HERE/'harmonize_allocation_outputs.py','code/reporting_rules_original.py','versioned reporting implementation')
for name in ['OUTPUT_RULES_V1.md','OUTPUT_RULES_V2.md']:copy(HERE/name,'inputs/allocation_conditional/'+name,'existing reporting contract')
helpers='from pathlib import Path\nfrom types import SimpleNamespace\nimport numpy as np\nimport pandas as pd\nimport json\nHERE=Path(__file__).resolve().parents[1]\nu=SimpleNamespace(HERE=HERE)\nKEY=["base_year","metal","stage","importer_iso","scenario"]\nK=KEY\nEPS=1e-6\nVERSION=2\nroutes=pd.read_csv(HERE/"inputs/historical/routes.csv").rename(columns={"origin":"exporter_iso","dest":"importer_iso"})\n'
helpers+=functions(REF/'global_allocation_pilot.py',{'ar','perturb'})+'\n\n'
audit=functions(REF/'validate_results.py',{'audit'})
audit=audit.replace("'all_allocations.csv.gz'","'all_flows.csv.gz'").replace("u.HERE/'results/routes.csv'","HERE/'inputs/historical/routes.csv'").replace("(folder/'minerals').glob('*/profiles.csv.gz')","folder.glob('*/profiles.csv.gz')")
helpers+=audit+'\n\n'+functions(HERE/'harmonize_allocation_outputs.py',{'ranks','view'})+'\n'
hp=CAP/'code/allocation_conditional_helpers.py';hp.write_text(helpers,encoding='utf-8');record(hp,'extracted frozen perturbation/audit/reporting functions; path-only audit adaptations')
source=(HERE/'run_allocation_stage.py').read_text(encoding='utf-8')
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/allocation';OUT.mkdir(parents=True,exist_ok=True)","import argparse\nfor k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'\np=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True);p.add_argument('--rho',type=float,choices=[0.,.8],required=True);ARGS=p.parse_args();assert 1<=ARGS.draw<=5\nHERE=Path(__file__).resolve().parent;OUT=HERE/f'outputs/allocation_conditional/rho_{ARGS.rho:.1f}_draw_{ARGS.draw:03d}';OUT.mkdir(parents=True,exist_ok=True)\nassert not (OUT/'ALLOCATION_CONDITIONAL_QA.json').exists(),'Do not overwrite completed draws'")
replace("HERE/'ALLOCATION_INPUT_MANIFEST.json'","HERE/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json'")
replace("    sample=trade[trade.metal.eq('Nickel')&trade.year.eq(2018)]","    helper=load('allocation_conditional_helpers')\n    trade,seed,weights=helper.perturb(trade,seed,dict(infer.LOCAL_STAGE_WEIGHT),ARGS.draw,ARGS.rho)\n    infer.LOCAL_STAGE_WEIGHT.update(weights)\n    trade.to_csv(OUT/'trade_corridors.csv.gz',index=False);seed.to_csv(OUT/'production_seeds.csv.gz',index=False)\n    (OUT/'draw_parameters.json').write_text(json.dumps(dict(draw=ARGS.draw,rho=ARGS.rho,weights=weights),indent=2),encoding='utf-8')\n    sample=trade[trade.metal.eq('Nickel')&trade.year.eq(2018)]")
replace("order,.65,infer,lm,core.local.LANES)","order,weights['material'],infer,lm,core.local.LANES)")
replace("routes=core.local.load_routes();solver_plan=pd.read_csv(HERE/'inputs/allocation/solver_plan.csv').set_index(['base_year','metal','stage','scenario']).solver","routes=core.local.load_routes()")
replace("('integrated','integrated'),('without_origin','integrated')","('integrated','integrated')")
replace("solver='Clarabel' if scenario=='without_origin' else str(solver_plan.loc[(year,metal,stage,scenario)])","solver='Clarabel' # Frozen full-mineral conditional driver uses verified conic solve with unchanged fallback.")
source=source.split('    # All expected numerical results are opened only now')[0]+(HERE/'allocation_conditional_footer.txt').read_text(encoding='utf-8')
driver=CAP/'run_allocation_conditional.py';driver.write_text(source,encoding='utf-8');compile(source,str(driver),'exec');record(driver,'isolated driver derived with checked substitutions')
for folder in sorted((REF/'results/global_joint_allocation_pilot').glob('rho_*_draw_*')):
    for name in ['annual_indicators.csv.gz','all_cases.csv.gz','all_allocations.csv.gz','annual_allocation_summary.csv','complete.json']:
        copy(folder/name,f'expected/allocation_conditional/{folder.name}/{name}','comparison only')
for rel in ['outputs/baci/trade_corridors_rebuilt.csv.gz','outputs/seed/allocation_production_seed_rebuilt.csv.gz','outputs/baci/BACI_REPRODUCTION_QA.json','outputs/seed/SEED_REPRODUCTION_QA.json','inputs/historical/routes.csv']:
    if (CAP/rel).exists():record(CAP/rel,'independently rebuilt input or frozen route')
for item in json.loads((CAP/'ALLOCATION_INPUT_MANIFEST.json').read_text()):
    if item['file'].startswith('code/') and not any(x['file']==item['file'] for x in files):record(CAP/item['file'],'frozen numerical method')
(CAP/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print(f'Allocation conditional entry staged: {len(files)} files')
