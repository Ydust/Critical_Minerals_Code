"""Cumulative Monte Carlo diagnostics, deliberately not statistical CIs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
R=HERE/'results'
OUT=R/'joint_batch_diagnostics'

def checkpoints(frame,keys,metrics,steps,design):
    rows=[]
    for label,g in frame.groupby(keys,dropna=False,sort=True):
        if not isinstance(label,tuple): label=(label,)
        previous=None
        for n in steps:
            x=g[g.draw.le(n)]
            if len(x)!=n or x.draw.nunique()!=n: continue
            for metric in metrics:
                values=x[metric].to_numpy()
                row=dict(design=design,**dict(zip(keys,label)),draws=n,metric=metric,
                    median=float(np.median(values)),p05=float(np.quantile(values,.05)),p95=float(np.quantile(values,.95)),
                    minimum=float(np.min(values)),maximum=float(np.max(values)),
                    finite_draws=int(np.isfinite(values).sum()))
                rows.append(row)
    result=pd.DataFrame(rows)
    if len(result):
        group=keys+['metric']
        result=result.sort_values(group+['draws'])
        for c in ['median','p05','p95']:
            result[c+'_change_pp']=100*result.groupby(group,dropna=False)[c].diff()
        result['maximum_endpoint_change_pp']=result[['p05_change_pp','p95_change_pp']].abs().max(axis=1)
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--require-complete',action='store_true'); args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    alloc=[]; raw=[]; audits=[]; changed=[]; fingerprints={}; weights=[]; mineral_means=[]; ranks=[]
    for marker in sorted((R/'global_joint_allocation_pilot').glob('*/complete.json')):
        m=json.loads(marker.read_text()); assert m['validation']['accepted']
        if m['draw']>5: continue
        folder=marker.parent
        assert len(list((folder/'minerals').glob('*/allocation/*_complete.json')))==m['validation']['optimisation_groups']
        d=pd.read_csv(folder/'annual_allocation_summary.csv')
        d['draw']=m['draw']; d['rho']=m['rho']; alloc.append(d)
        audits.append(dict(draw=m['draw'],rho=m['rho'],**m['validation']))
        cases=pd.read_csv(folder/'all_cases.csv.gz')
        baseline=cases[cases.scenario.eq('baseline')]
        ranking=baseline.groupby('metal',as_index=False).agg(years=('base_year','nunique'),value_usd=('baseline_value_usd','sum'))
        ranking=ranking[ranking.years.eq(13)].sort_values('value_usd',ascending=False)
        ranking['rank']=np.arange(1,len(ranking)+1); ranking['draw']=m['draw']; ranking['rho']=m['rho']; ranks.append(ranking)
        x=cases[cases.scenario.ne('baseline')].copy()
        cols=['delta_direct_hhi','delta_origin_hhi','delta_top_chokepoint_share']
        for c in cols: x[c+'_weighted']=x[c]*x.baseline_value_usd
        yearly=x.groupby(['metal','base_year','scenario'],as_index=False)[['baseline_value_usd']+[c+'_weighted' for c in cols]].sum()
        for c in cols: yearly[c]=yearly[c+'_weighted']/yearly.baseline_value_usd
        means=yearly.groupby(['metal','scenario'],as_index=False)[cols].mean()
        means['draw']=m['draw']; means['rho']=m['rho']; mineral_means.append(means)
        weights.extend(dict(branch='allocation',draw=m['draw'],rho=m['rho'],stage=k,weight=v) for k,v in m['stage_weights'].items())
        fingerprints[str(marker.relative_to(HERE))]=hashlib.sha256(marker.read_bytes()).hexdigest()
    for marker in sorted((R/'raw_report_sensitivity_batch').glob('*/complete.json')):
        m=json.loads(marker.read_text()); assert m['max_conservation_error']<1e-10
        if m['draw']>20: continue
        d=pd.read_csv(marker.parent/'window_summaries.csv'); assert d.draw.eq(m['draw']).all()
        assert d.temporal_rho.eq(m['rho']).all(); raw.append(d)
        changed.append(m)
        weights.extend(dict(branch='raw_report',draw=m['draw'],rho=m['rho'],stage=k,weight=v) for k,v in m['weights'].items())
        fingerprints[str(marker.relative_to(HERE))]=hashlib.sha256(marker.read_bytes()).hexdigest()
    complete=len(audits)==10 and len(changed)==40
    if args.require_complete: assert complete,(len(audits),len(changed))
    a=pd.concat(alloc,ignore_index=True); h=pd.concat(raw,ignore_index=True)
    assert not a.duplicated(['draw','rho','base_year','scenario']).any()
    assert not h.duplicated(['draw','temporal_rho','window_family','end_year','metal']).any()
    a.to_csv(OUT/'all_allocation_annual_summaries.csv',index=False)
    h.to_csv(OUT/'all_raw_report_window_summaries.csv',index=False)
    mm=pd.concat(mineral_means,ignore_index=True); mm.to_csv(OUT/'mineral_equal_year_tradeoff_means.csv',index=False)
    pd.concat(ranks,ignore_index=True).to_csv(OUT/'mineral_complete_coverage_value_ranks.csv',index=False)
    reference_cases=pd.read_csv(R/'all_cases.csv.gz'); reference_cases=reference_cases[reference_cases.scenario.eq('baseline')]
    ref_ranks=reference_cases.groupby('metal',as_index=False).agg(years=('base_year','nunique'),value_usd=('baseline_value_usd','sum'))
    ref_ranks=ref_ranks[ref_ranks.years.eq(13)].sort_values('value_usd',ascending=False)
    ref_ranks['rank']=np.arange(1,len(ref_ranks)+1); ref_ranks.to_csv(OUT/'reference_mineral_value_ranks.csv',index=False)
    tradeoff=mm.pivot(index=['draw','rho','metal'],columns='scenario',values=['delta_direct_hhi','delta_origin_hhi','delta_top_chokepoint_share'])
    delta=pd.DataFrame(index=tradeoff.index)
    for c in ['delta_direct_hhi','delta_origin_hhi','delta_top_chokepoint_share']:
        delta[c+'_integrated_minus_direct']=tradeoff[(c,'integrated')]-tradeoff[(c,'direct_partner')]
    delta['maritime_improves_relative_to_direct']=delta.delta_top_chokepoint_share_integrated_minus_direct.lt(0)
    delta['supplier_reduction_smaller']=delta.delta_direct_hhi_integrated_minus_direct.gt(0)
    delta['origin_reduction_smaller']=delta.delta_origin_hhi_integrated_minus_direct.gt(0)
    delta=delta.reset_index(); delta['reference_top_four']=delta.metal.isin(ref_ranks.head(4).metal)
    delta.to_csv(OUT/'mineral_paired_tradeoff_checks.csv',index=False)
    weight_frame=pd.DataFrame(weights)
    weight_frame['at_zero']=weight_frame.weight.eq(0); weight_frame['at_one']=weight_frame.weight.eq(1)
    weight_frame.to_csv(OUT/'stage_weight_draws.csv',index=False)
    weight_frame.groupby(['branch','rho','stage'],as_index=False).agg(draws=('draw','size'),at_zero=('at_zero','sum'),at_one=('at_one','sum'),
        mean=('weight','mean'),minimum=('weight','min'),maximum=('weight','max')).to_csv(OUT/'stage_weight_boundary_counts.csv',index=False)
    am=['material_joint_value_share','risk_transfer_value_share']
    hm=['transfer_share','substantive_share','transfer_lower_share','substantive_upper_share']
    ac=checkpoints(a[a.scenario.ne('baseline')],['rho','base_year','scenario'],am,[2,3,5],'allocation')
    hc=checkpoints(h[h.metal.eq('all')],['temporal_rho','window_family','end_year'],hm,[5,10,20],'raw_report')
    ac.to_csv(OUT/'allocation_cumulative_diagnostics.csv',index=False)
    hc.to_csv(OUT/'historical_cumulative_diagnostics.csv',index=False)
    pool=h[h.metal.eq('all')&h.end_year.eq('pooled')].copy()
    pool['nominal_ordering']=pool.transfer_share>pool.substantive_share
    pool['conditional_route_bound_ordering']=pool.transfer_lower_share>pool.substantive_upper_share
    pool['free_origin_bound_ordering']=pool.origin_free_transfer_lower_share>pool.origin_free_substantive_upper_share
    order=pool.groupby(['temporal_rho','window_family'],as_index=False).agg(draws=('draw','size'),
        nominal_ordering_count=('nominal_ordering','sum'),conditional_route_bound_ordering_count=('conditional_route_bound_ordering','sum'),
        free_origin_bound_ordering_count=('free_origin_bound_ordering','sum'))
    order.to_csv(OUT/'historical_ordering_counts.csv',index=False)
    def warnings(d):
        if d.empty: return {}
        x=d[d.maximum_endpoint_change_pp.notna()]
        latest=x[x.draws.eq(d.draws.max())]
        return dict(comparisons=len(x),endpoint_changes_above_one_pp=int(x.maximum_endpoint_change_pp.gt(1).sum()),
                    maximum_endpoint_change_pp=float(x.maximum_endpoint_change_pp.max()) if len(x) else None,
                    latest_checkpoint_draws=int(d.draws.max()),latest_comparisons=len(latest),
                    latest_changes_above_one_pp=int(latest.maximum_endpoint_change_pp.gt(1).sum()),
                    latest_maximum_endpoint_change_pp=float(latest.maximum_endpoint_change_pp.max()) if len(latest) else None)
    info=dict(allocation_draws=len(audits),raw_report_draws=len(changed),planned_batch_complete=complete,
        allocation_actual_optimisations=sum(x['actual_optimisations'] for x in audits),
        allocation_unit_year_range=[min(x['unit_years'] for x in audits),max(x['unit_years'] for x in audits)],
        all_allocation_draws_independently_accepted=True,
        max_raw_conservation_error=max(x['max_conservation_error'] for x in changed),
        allocation_checkpoints=warnings(ac),historical_checkpoints=warnings(hc),
        endpoint_interpretation='Empirical sensitivity percentiles under uncalibrated input laws; not confidence intervals',
        calibrated_input_distribution=False,stable_tail_estimates_established=False,
        completion_record_sha256=fingerprints,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'SUMMARY.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='completion_record_sha256'},indent=2))

if __name__=='__main__': main()
