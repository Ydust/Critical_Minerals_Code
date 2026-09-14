from pathlib import Path
import hashlib,json,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CAP=HERE/'capsule'
src=ROOT/'research_process/evidence_extension_20260908/run_historical_joint_sensitivity.py'
shutil.copy2(src,CAP/'code/conditional_engine.py')
shutil.copy2(HERE/'run_conditional_corridor.py',CAP/'run_conditional_corridor.py')
expected=CAP/'expected/corridor_500';expected.mkdir(exist_ok=True)
base=ROOT/'submission_package_aligned_20260910/supplementary_data/SD9_uncertainty/corridor_500'
for name in ['all_draw_summaries.csv.gz','draw_parameters.json']:shutil.copy2(base/name,expected/name)
files=[CAP/p for p in ['code/conditional_engine.py','run_conditional_corridor.py','outputs/historical_trade_rebuilt.csv.gz','outputs/seed/historical_production_seed_rebuilt.csv.gz','outputs/historical_rebuilt_seed/annual.csv.gz','outputs/historical_rebuilt_seed/windows.csv.gz','outputs/historical_rebuilt_seed/HISTORICAL_REPRODUCTION_QA.json','inputs/historical/routes.csv','expected/corridor_500/all_draw_summaries.csv.gz','expected/corridor_500/draw_parameters.json']]
records=[dict(file=p.relative_to(CAP).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),role='comparison_only' if 'expected' in p.parts else 'input_or_method') for p in files]
(CAP/'CONDITIONAL_INPUT_MANIFEST.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Conditional corridor stage staged')
