from pathlib import Path
import json,zipfile,shutil,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CAP=HERE/'capsule'
pre=json.loads((HERE/'SEED_SOURCE_PREFLIGHT.json').read_text())
with zipfile.ZipFile(ROOT/'manuscript_package_20260907_final/Reproduction_Code_and_Derived_Inputs.zip') as z:
    for item in pre['frozen_methods']:
        dest=CAP/'code/seed'/Path(item['member']).name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(z.read(item['member']))
    (CAP/'inputs/seed/nation_lat_lon.csv').parent.mkdir(parents=True,exist_ok=True)
    (CAP/'inputs/seed/nation_lat_lon.csv').write_bytes(z.read('reproduction/project_snapshot/route_template/inputs/nation_lat_lon.csv'))
    contract=json.loads(z.read('figure_source_tables/longitudinal_2012_2024/release_v3/longitudinal_run_manifest.json'))['inputs']
    (CAP/'inputs/seed/original_seed_input_contract.json').write_text(json.dumps({k:contract[k] for k in ['wmd_seed','reporters_reference','country_reference']},indent=2),encoding='utf-8')
for item in pre['source_files']:
    source=ROOT/item['file'];dest=CAP/'inputs/seed'/source.name;dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,dest)
shutil.copy2(CAP/'inputs/historical/production_seed.csv.gz',CAP/'expected/historical_production_seed.csv.gz')
shutil.copy2(ROOT/'submission_package_aligned_20260910/supplementary_data/SD5_production_seeds.csv.gz',CAP/'expected/SD5_production_seeds.csv.gz')
if (HERE/'run_seed_stage.py').exists():shutil.copy2(HERE/'run_seed_stage.py',CAP/'run_seed_stage.py')
if (HERE/'extract_seed_coordinates.mjs').exists():shutil.copy2(HERE/'extract_seed_coordinates.mjs',CAP/'extract_seed_coordinates.mjs')
files=list((CAP/'code/seed').glob('*.py'))+list((CAP/'inputs/seed').iterdir())+[CAP/'code/longitudinal_mrci.py',CAP/'code/infer_mine_origin_flows.py',CAP/'inputs/raw/_ref_Reporters.json',CAP/'expected/historical_production_seed.csv.gz',CAP/'expected/SD5_production_seeds.csv.gz']
if (CAP/'run_seed_stage.py').exists():files.append(CAP/'run_seed_stage.py')
if (CAP/'extract_seed_coordinates.mjs').exists():files.append(CAP/'extract_seed_coordinates.mjs')
manifest=[dict(file=p.relative_to(CAP).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,role='comparison_only' if p.parent.name=='expected' else 'source_or_method') for p in files]
for item in manifest:
    if item['file'].endswith('nation_lat_lon.csv'):item['role']='unused_diagnostic_reference_not_model_input'
(CAP/'SEED_INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Seed methods and raw files staged; no production values calculated.')
