"""Paired amplitude comparison kept separate from the main draw ensemble."""
import json
from pathlib import Path
import pandas as pd
HERE=Path(__file__).resolve().parent
R=HERE/'results'
OUT=R/'raw_report_scale_stress'
rows=[]; weights={}; checks=[]
for sd,path in [(.1,R/'raw_report_sensitivity_pilot'),(.45,OUT/'sd_0.45'),(.75,OUT/'sd_0.75')]:
    d=pd.read_csv(path/'all_pilot_window_summaries.csv'); d['report_sd']=sd; rows.append(d)
    for marker in sorted(path.glob('*/complete.json')):
        m=json.loads(marker.read_text()); assert m['draw']==1
        weights.setdefault(m['rho'],m['weights']); assert weights[m['rho']]==m['weights']
    if sd>.1:
        c=json.loads((path/'independent_flow_spot_checks.json').read_text()); assert c['accepted']; checks.append(c)
data=pd.concat(rows,ignore_index=True)
data.to_csv(OUT/'paired_amplitude_comparison.csv',index=False)
pooled=data[data.window_family.eq('adjacent')&data.end_year.eq('pooled')&data.metal.eq('all')]
(OUT/'SUMMARY.json').write_text(json.dumps(dict(amplitudes=[.1,.45,.75],draws_per_amplitude_and_design=1,
    paired_weights_identical=True,independent_flow_spot_checks_passed=True,
    interpretation='One paired perturbation direction; diagnostic amplitude comparison, not calibrated uncertainty or confidence bounds',
    new_stress_runs=4,pooled_adjacent=pooled.to_dict('records')),indent=2),encoding='utf-8')
print(pooled[['report_sd','temporal_rho','transfer_share','substantive_share']].to_string(index=False))
