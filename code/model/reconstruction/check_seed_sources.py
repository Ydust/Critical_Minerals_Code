"""Read-only source-file preflight, not a production-value reconstruction."""
from pathlib import Path
import hashlib,json,csv,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
lineage=ROOT/'submission_package_aligned_20260910/supplementary_data/SD10_provenance/wmd_seed_source_lineage.csv'
with lineage.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
expected={r['original_file']:r['original_sha256'] for r in rows}
records=[]
for name in ['wmd_2018.pdf','wmd_2021.pdf','wmd_2026_ch6_4_country_by_mineral.xlsx','fig5_history_audit_2008_2012/BGS_WMP_2008_2012.pdf','fig5_history_audit_2008_2012/WMD2016_archived_official.pdf']:
    p=ROOT/'external_sources'/name
    with p.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
    reference=expected.get(p.name)
    records.append(dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha,expected_lineage_sha256=reference,matches_lineage=sha==reference if reference else None))
    if reference: assert sha==reference,p.name
archive=ROOT/'manuscript_package_20260907_final/Reproduction_Code_and_Derived_Inputs.zip'
members=['reproduction/project_snapshot/longitudinal_2012_2024/wmd_parser/wmd_parser.py','reproduction/project_snapshot/longitudinal_2012_2024/wmd_parser/build_outputs.py','reproduction/project_snapshot/longitudinal_2012_2024/wmd_parser/bridge_shares.py','reproduction/project_snapshot/modeling/build_wmd_mine_origin_seed.py','reproduction/audit_bgs_wmd_overlap_2008_2012.py','reproduction/rebuild_fig5_inputs_2007_2024.py']
with zipfile.ZipFile(archive) as z:
    code=[dict(member=n,sha256=hashlib.sha256(z.read(n)).hexdigest()) for n in members]
report=dict(source_files=records,frozen_methods=code,wmd_lineage_rows=len(rows),wmd_lineage_files_verified=3,production_values_rebuilt=False,bgs_calibration_recomputed=False,scope='File availability and hash checks only; BGS files have no comparison hash in the WMD lineage table')
(HERE/'SEED_SOURCE_PREFLIGHT.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
