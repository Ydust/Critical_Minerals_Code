"""Recompute corrected-route bootstrap with original sample/seed conventions."""
import json
import sys
import run_unified as u
sys.path.insert(0,str(u.ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h
import pandas as pd
import numpy as np

if __name__=='__main__':
    _,lm,_=u.modules()
    annual,corridor,seed,oldlanes,oldwindows,hashes,records=h.load_inputs()
    path=u.ROUTES/'global_detector_audit/minerals_lanes_existing_support_geometric.csv'
    # Recompute all route columns, not only concentration, to keep the window
    # metadata (including dominant chokepoint labels) internally consistent.
    edges=corridor.rename(columns={'value':'reconstructed_value_usd'})
    routed=lm.route_stats(edges,annual,path)
    frame=annual.drop(columns=[c for c in routed if c not in lm.GROUP]).merge(routed,on=lm.GROUP,validate='one_to_one')
    cfg=lm.AnalysisConfig(bootstrap_reps=2000)
    windows=pd.concat([lm.build_window_panel(frame,list(zip(d.baseline_year,d.end_year)),fam,cfg)
        for fam,d in oldwindows[['window_family','baseline_year','end_year']].drop_duplicates().groupby('window_family')],ignore_index=True)
    summaries=lm.summarize_windows(windows,pd.DataFrame(),.025,'local_full_evidence_endpoint','none',cfg)
    out=u.OUT/'historical_corrected_routes_old_scope'
    summaries.to_csv(out/'frozen_method_bootstrap_summary.csv',index=False)
    windows.to_csv(out/'corrected_full_windows.csv.gz',index=False)
    old=pd.read_csv(out/'reference_summary.csv')
    p=summaries.query('window_family == "adjacent" and aggregation_scope == "pooled_family"').iloc[0]
    r=old.query('window_family == "adjacent" and end_year == "pooled" and metal == "all"').iloc[0]
    assert abs(p.risk_transfer_value_share_of_apparent-r.transfer_share)<1e-12
    assert abs(p.substantive_value_share_of_apparent-r.substantive_share)<1e-12
    manifest=dict(replicates=2000,random_seed=cfg.random_seed,sample_mode='local_full_evidence_endpoint',
        primary_output='frozen_method_bootstrap_summary.csv',
        earlier_output='country_cluster_intervals.csv uses an independent random seed/implementation as a numerical sensitivity check, not a second primary inference',
        script_sha256=u.s.sha(__file__),frozen_analysis_sha256=u.s.sha(lm.__file__))
    (out/'bootstrap_reporting_contract.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(p.filter(regex='share|bootstrap').to_string(),flush=True)
