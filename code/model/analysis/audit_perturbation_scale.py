"""Observed mirror disagreement diagnostics, not an identified error model."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
RAW=HERE/'results/full_raw_reconstruction_audit'
OUT=HERE/'results/perturbation_scale_audit'

def weighted_quantile(x,w,q):
    order=np.argsort(x); x=x[order]; w=w[order]
    return float(x[np.searchsorted(np.cumsum(w),q*w.sum(),side='left')])

def summarize(d,label):
    both=d.mirror_status.eq('both'); x=d[both]
    gap=np.log1p(x.value_usd_importer_reported)-np.log1p(x.value_usd_exporter_reported)
    w=x.reconciled_value_usd.to_numpy(); g=gap.to_numpy()
    med=float(np.median(g))
    return dict(scope=label,positive_observed_edges=len(d),paired_edges=len(x),
        paired_value_share=float(x.reconciled_value_usd.sum()/d.reconciled_value_usd.sum()),
        paired_reporter_country_union=len(set(x.exporter_iso)|set(x.importer_iso)),
        signed_log_gap_median=med,absolute_log_gap_median=float(np.median(abs(g))),
        absolute_log_gap_p90=float(np.quantile(abs(g),.9)),
        value_weighted_absolute_log_gap_median=weighted_quantile(abs(g),w,.5),
        value_weighted_absolute_log_gap_p90=weighted_quantile(abs(g),w,.9),
        robust_gap_scale_1_4826_MAD=float(1.4826*np.median(abs(g-med))),
        fraction_abs_gap_above_0_1=float((abs(g)>.1).mean()),
        interpretation='Paired-report discrepancy contains systematic and random components; not an identified per-report error SD')

def main():
    OUT.mkdir(exist_ok=True)
    parts=[]
    cols=['RefYear','metal','stage','exporter_iso','importer_iso','mirror_status','reconciled_value_usd','value_usd_exporter_reported','value_usd_importer_reported']
    for path in sorted(RAW.glob('*/baseline.csv.gz')):
        d=pd.read_csv(path,usecols=cols)
        named=d.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&d.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
        parts.append(d[named&d.metal.ne('Silicon')])
    d=pd.concat(parts,ignore_index=True)
    rows=[summarize(d,'all')]
    for (metal,stage),x in d.groupby(['metal','stage']):
        if x.mirror_status.eq('both').any(): rows.append(summarize(x,metal+'|'+stage))
    pd.DataFrame(rows).to_csv(OUT/'mirror_disagreement_diagnostics.csv',index=False)
    report=dict(reference_raw_audit_accepted=json.loads((RAW/'SUMMARY.json').read_text())['accepted'],
        summary=rows[0],stage_groups=len(rows)-1,
        current_batch_report_log_sd=.1,calibrated=False,
        conclusion='Additional random draws cannot calibrate their input distributions. Mirror differences alone cannot separate reporting bias, correlated errors and true discrepancy; no per-report SD fitted here.')
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
