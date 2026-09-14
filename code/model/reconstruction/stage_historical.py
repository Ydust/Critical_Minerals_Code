"""Stage frozen historical methods and declared inputs; no calculation here."""
from pathlib import Path
import hashlib, json, shutil, zipfile, io
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CAP = HERE / 'capsule'
FROZEN = ROOT / 'research_process/nickel_scope_correction_20260908/frozen/reproduction/project_snapshot'
records = []
def record(path, role, source):
    records.append(dict(file=str(path.relative_to(CAP)).replace('\\','/'), role=role,
                        source=source, bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
def copy(source, dest, role):
    target=CAP/dest; target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target); record(target,role,str(source.relative_to(ROOT)))

for name,rel in [('infer_mine_origin_flows','modeling/infer_mine_origin_flows.py'),
                 ('longitudinal_mrci','longitudinal_2012_2024/analysis/longitudinal_mrci.py')]:
    copy(FROZEN/rel,f'code/{name}.py','frozen_method')
copy(ROOT/'research_process/unified_reanalysis_20260909/cached_routes.py','code/cached_routes.py','equivalent_route_cache')
copy(ROOT/'research_process/unified_reanalysis_20260909/results/routes.csv','inputs/historical/routes.csv','fixed_route_input')
archive=ROOT/'manuscript_package_20260907_final/Reproduction_Code_and_Derived_Inputs.zip'
base='figure_source_tables/longitudinal_2012_2024/'
with zipfile.ZipFile(archive) as z:
    member=base+'release_v3/merged_wmd_seed_normalized.csv.gz'
    target=CAP/'inputs/historical/production_seed.csv.gz'
    target.write_bytes(z.read(member)); record(target,'fixed_derived_production_seed',str(archive.relative_to(ROOT))+'::'+member)
    # Only the declared window definitions are staged; no old numerical result is an input.
    member=base+'release_v3/mrci_all_windows_2012_2024.csv.gz'
    specs=pd.read_csv(io.BytesIO(z.read(member)),compression='gzip',usecols=['window_family','baseline_year','end_year']).drop_duplicates().sort_values(['window_family','baseline_year'])
    target=CAP/'inputs/historical/window_definitions.csv'; specs.to_csv(target,index=False)
    record(target,'frozen_temporal_design',str(archive.relative_to(ROOT))+'::'+member+' [design columns only]')
    print(specs.to_string(index=False))
for name in ['SD1_historical_annual.csv.gz','SD3_importer_origins.csv.gz','SD3_exporter_profiles.csv.gz','SD2_historical_windows.csv.gz','SD2_historical_summary.csv']:
    copy(ROOT/'submission_package_aligned_20260910/supplementary_data'/name,'expected/'+name,'comparison_only')
record(CAP/'outputs/historical_trade_rebuilt.csv.gz','stage_1_rebuilt_trade','independent raw-stage output')
record(CAP/'outputs/seed/historical_production_seed_rebuilt.csv.gz','stage_4_rebuilt_seed','raw WMD seed-stage output')
record(CAP/'outputs/seed/SEED_REPRODUCTION_QA.json','seed_stage_qa','raw WMD/BGS stage acceptance record')
copy(HERE/'run_historical_stage.py','run_historical_stage.py','portable_driver')
(CAP/'HISTORICAL_INPUT_MANIFEST.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Staged',len(records),'files')
