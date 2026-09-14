"""Post-run exact comparison of window membership counts; never model inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = HERE / 'capsule/outputs/conditional_corridor_d250'
EXPECTED = HERE / 'capsule/expected/corridor_membership'
SOURCE = ROOT / 'research_process/unified_reanalysis_20260909/results/historical_expanded_comtrade/joint_sensitivity'
KEYS = ['importer_iso', 'metal', 'stage', 'window_family', 'baseline_year', 'end_year']

def main():
    run = json.loads((OUT / 'CONDITIONAL_REPRODUCTION_QA.json').read_text())
    assert run['accepted'] and run['full_500_draw_run']
    EXPECTED.mkdir(exist_ok=True)
    comparisons = []
    for rho in ['0.0', '0.8']:
        name = f'membership_rho_{rho}.csv.gz'
        reference = EXPECTED / name
        shutil.copy2(SOURCE / name, reference)
        actual = OUT / name
        a = pd.read_csv(actual).set_index(KEYS).sort_index()
        e = pd.read_csv(reference).set_index(KEYS).sort_index()
        assert a.index.is_unique and e.index.is_unique
        same_keys = a.index.equals(e.index)
        same_columns = a.columns.equals(e.columns)
        differences = {c: int(a[c].ne(e[c]).sum()) for c in a.columns} if same_keys and same_columns else None
        comparisons.append(dict(rho=rho, rows=len(a), keys_exact=same_keys,
            columns_exact=same_columns, difference_counts=differences,
            reference_source=str(SOURCE / name), reference_role='post-computation process reference, not a model input',
            reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
            actual_sha256=hashlib.sha256(actual.read_bytes()).hexdigest()))
    accepted = all(c['keys_exact'] and c['columns_exact'] and not any(c['difference_counts'].values()) for c in comparisons)
    report = dict(accepted=accepted, comparisons=comparisons,
        scope='Exact per-window aggregate membership counts across 250 draws per temporal design; not stored per-draw classifications.',
        full_pipeline_complete=False)
    (OUT / 'MEMBERSHIP_REPRODUCTION_QA.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    assert accepted

if __name__ == '__main__':
    main()
