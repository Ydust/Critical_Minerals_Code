from pathlib import Path
import shutil,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]; CAP=HERE/'capsule'; records=[]
def copy(source,dest,role):
    target=CAP/dest;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    with target.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    records.append(dict(file=target.relative_to(CAP).as_posix(),source=source.relative_to(ROOT).as_posix(),role=role,sha256=digest,bytes=target.stat().st_size))
source=ROOT/'external_sources/baci_hs07_202601'
for name in ['BACI_HS07_V202601.zip','source_manifest.json','fig5_hs12_to_hs07_scope.csv']:
    copy(source/name,'inputs/baci/'+name,'raw_archive' if name.endswith('.zip') else 'source_metadata_or_scope')
copy(ROOT/'submission_package_aligned_20260910/supplementary_data/SD5_trade_corridors.csv.gz','expected/SD5_trade_corridors.csv.gz','comparison_only')
copy(HERE/'run_baci_stage.py','run_baci_stage.py','portable_driver')
(CAP/'BACI_INPUT_MANIFEST.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('BACI inputs staged',flush=True)
