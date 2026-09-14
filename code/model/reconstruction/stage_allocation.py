from pathlib import Path
import ast,json,hashlib,shutil
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CAP=HERE/'capsule'
F=ROOT/'research_process/nickel_scope_correction_20260908/frozen/reproduction/project_snapshot/modeling'
paths=[]
def copy(src,dest):
    p=CAP/dest;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);paths.append(p)
for rel in ['global_coupled_counterfactual/core.py','global_coupled_counterfactual/rolling_support.py','run_capacity_constrained_derisking_counterfactual.py']:
    copy(F/rel,'code/modeling/'+rel)
src=ROOT/'research_process/nickel_scope_correction_20260908/run_scope_reanalysis.py'
copy(src,'code/allocation_scope_original.py')
selected={'infer_inputs','fingerprints_dict','matched_quadratic','solve'}
body='\n\n'.join(ast.get_source_segment(src.read_text(encoding='utf-8'),n) for n in ast.parse(src.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name in selected)
header="import json,time\nimport numpy as np\nimport pandas as pd\nimport osqp\nfrom scipy.sparse import lil_matrix,csc_matrix,vstack,eye\nKEYS=['importer_iso','metal','stage']\nGROUP=KEYS+['year']\nEDGE=['metal','stage','exporter_iso','importer_iso','year']\n"
p=CAP/'code/allocation_scope.py';p.write_text(header+body+'\n',encoding='utf-8');paths.append(p)
src=ROOT/'research_process/unified_reanalysis_20260909/solver_verified.py';copy(src,'code/solver_verified_original.py')
fn=next(n for n in ast.parse(src.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='solve')
header="import json\nfrom types import SimpleNamespace\nimport allocation_scope\nimport numpy as np\nimport clarabel\nfrom scipy.sparse import csc_matrix,vstack,eye,triu,diags\nu=SimpleNamespace(s=allocation_scope)\nORIGINAL_SOLVE=allocation_scope.solve\n"
p=CAP/'code/allocation_verified.py';p.write_text(header+ast.get_source_segment(src.read_text(),fn)+'\n',encoding='utf-8');paths.append(p)
data=ROOT/'submission_package_aligned_20260910/supplementary_data'
for name in ['SD5_annual_indicators.csv.gz','SD5_allocation_cases.csv.gz','SD5_allocation_flows.csv.gz','SD6_origin_ablation_all_cases.csv.gz','SD6_origin_ablation_all_flows.csv.gz']:
    copy(data/name,'expected/'+name)
# Only solver identity is a declared execution input; no reference solution or objective enters the computation.
plan=pd.read_csv(data/'SD5_allocation_solver.csv.gz')[['base_year','metal','stage','scenario','solver']]
p=CAP/'inputs/allocation/solver_plan.csv';p.parent.mkdir(parents=True,exist_ok=True);plan.to_csv(p,index=False);paths.append(p)
copy(HERE/'run_allocation_stage.py','run_allocation_stage.py')
paths += [CAP/p for p in ['outputs/baci/trade_corridors_rebuilt.csv.gz','outputs/seed/allocation_production_seed_rebuilt.csv.gz','inputs/historical/routes.csv','code/infer_mine_origin_flows.py','code/longitudinal_mrci.py','code/cached_routes.py']]
manifest=[dict(file=p.relative_to(CAP).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='comparison_only' if p.parent.name=='expected' else 'input_or_method') for p in paths]
(CAP/'ALLOCATION_INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
install=json.loads((CAP/'SOLVER_INSTALL_REPORT.json').read_text(encoding='utf-8'))
lock=[f"{i['metadata']['name']}=={i['metadata']['version']} --hash=sha256:{i['download_info']['archive_info']['hashes']['sha256']}" for i in install['install']]
(CAP/'solver_requirements.lock').write_text('\n'.join(sorted(lock))+'\n',encoding='utf-8')
print('Allocation capsule staged; frozen function bodies retained verbatim.')
