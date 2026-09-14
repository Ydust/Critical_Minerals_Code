"""Create compact audit tables from completed, preserved follow-up results."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/historical_joint_sensitivity'


def main():
    data=pd.read_csv(OUT/'window_summary_all_draws.csv')
    d=data[(data.draw>0)&(data.end_year=='pooled')&(data.metal=='all')].copy()
    d['point_gap']=d.transfer_share-d.substantive_share
    d['conditional_bound_gap']=d.transfer_lower_share-d.substantive_upper_share
    d['origin_unrestricted_bound_gap']=d.origin_free_transfer_lower_share-d.origin_free_substantive_upper_share
    rows=[]
    for (rho,family),group in d.groupby(['temporal_rho','window_family']):
        row=dict(temporal_rho=rho,window_family=family,draws=len(group))
        for metric in ['transfer_share','substantive_share','point_gap','conditional_bound_gap','origin_unrestricted_bound_gap','apparent_count']:
            for name,fn in [('min','min'),('median','median'),('max','max')]:
                row[metric+'_'+name]=float(getattr(group[metric],fn)())
        for metric in ['point_gap','conditional_bound_gap','origin_unrestricted_bound_gap']:
            row[metric+'_positive_draws']=int(group[metric].gt(0).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT/'pooled_sensitivity_ranges.csv',index=False)
    manifest=json.loads((OUT/'run_manifest.json').read_text())
    meta=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform(),
              original_run_manifest_sha256=hashlib.sha256((OUT/'run_manifest.json').read_bytes()).hexdigest(),
              actual_draws=manifest['actual_draws'],not_a_confidence_interval=True,
              scripts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')},
              output_files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in list(OUT.glob('*'))+list((ROOT/'external/facility_supply').glob('*.csv'))
                    +list((ROOT/'external/nickel_matte_coverage').glob('*.csv'))
                    +list((ROOT/'external/baci_nickel_intermediates').glob('*')) if p.is_file()})
    (ROOT/'results/followup_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()
