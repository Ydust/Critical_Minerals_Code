from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; CAP=HERE/'capsule'
REF=ROOT/'research_process/unified_reanalysis_20260909'
files=[]
def record(p,role):
    files.append(dict(file=p.relative_to(CAP).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role=role))
def copy(src,rel,role):
    p=CAP/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,p);record(p,role)
for year in range(2012,2025):copy(ROOT/f'_raw/280461_{year}.json',f'inputs/raw_conditional/280461_{year}.json','Silicon raw response; shared reporter index only')
copy(REF/'raw_report_sensitivity_batch.py','code/raw_report_batch_original.py','frozen reference driver, not imported')
copy(HERE/'run_raw_conditional.py','run_raw_conditional.py','isolated driver')
for p in sorted((REF/'results/full_raw_reconstruction_audit').glob('*/observations.csv.gz')):
    copy(p,f'expected/raw_conditional/observations/{p.parent.name}.csv.gz','comparison only')
for folder in sorted((REF/'results/raw_report_sensitivity_batch').glob('rho_*_draw_*')):
    for name in ['annual_indicators.csv.gz','window_summaries.csv','complete.json']:
        copy(folder/name,f'expected/raw_conditional/{folder.name}/{name}','comparison only')
for p in sorted((CAP/'inputs/raw').glob('*.json')):record(p,'retained raw response or dictionary')
for rel in ['inputs/mapping.csv','code/build_model_inputs.py','code/reconstruct_dynamic_panel.py','code/conditional_engine.py','outputs/historical_trade_rebuilt.csv.gz','outputs/seed/historical_production_seed_rebuilt.csv.gz','outputs/historical_rebuilt_seed/annual.csv.gz','outputs/historical_rebuilt_seed/windows.csv.gz','outputs/historical_rebuilt_seed/HISTORICAL_REPRODUCTION_QA.json','inputs/historical/routes.csv']:
    record(CAP/rel,'rebuilt input or frozen method')
(CAP/'RAW_CONDITIONAL_INPUT_MANIFEST.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
print(f'Staged {len(files)} hashed files')
