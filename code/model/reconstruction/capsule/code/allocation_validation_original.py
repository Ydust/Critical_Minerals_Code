"""Independent arithmetic checks of saved allocation outputs and release gates."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import run_unified as u
import numpy as np
import pandas as pd

KEY=['base_year','metal','stage','importer_iso','scenario']

def audit(folder, trade_input=None):
    cases=pd.read_csv(folder/'all_cases.csv.gz')
    alloc=pd.read_csv(folder/'all_allocations.csv.gz')
    solvers=pd.read_csv(folder/'all_solver.csv.gz')
    assert not cases.duplicated(KEY).any()
    assert solvers.success.all() and cases.solver_feasible.all()
    metrics=['kkt_stationarity','kkt_complementarity','dual_sign_error','max_abs_demand_share_error','max_scaled_inequality_violation','max_bound_violation']
    maxima={k:float(solvers[k].max()) if k in solvers else 0. for k in metrics}
    assert maxima['max_abs_demand_share_error']<=2e-6
    assert maxima['max_scaled_inequality_violation']<=2e-7
    assert maxima['max_bound_violation']<=2e-8
    assert max(maxima[k] for k in metrics[:3])<2e-7
    saved=cases.set_index(KEY)
    routes=pd.read_csv(u.HERE/'results/routes.csv').fillna({'chokepoints':''})
    lookup={(r.metal,r.origin,r.dest):str(r.chokepoints).split('|') if r.mode=='sea' and r.chokepoints else [] for r in routes.itertuples()}
    fp={}
    for p in (folder/'minerals').glob('*/profiles.csv.gz'):
        for k,d in pd.read_csv(p).groupby(['year','metal','stage','exporter_iso'],sort=False):
            fp[k]=dict(zip(d.inferred_mine_origin_iso,d.share))
    errors={k:0. for k in ['demand_share','direct_hhi','origin_hhi','top_chokepoint_share','route_coverage_share','origin_coverage_share','value_balance_relative']}
    unresolved=0
    for key,d in alloc.groupby(KEY,sort=False):
        y,m,stage,imp,scenario=key
        origin={}; choke={}; route_cover=0.; origin_cover=0.
        for r in d.itertuples():
            mapping=fp.get((y,m,stage,r.exporter_iso))
            if mapping is None:
                # Reproduce the existing model convention, but count and expose it.
                mapping={'UNRESOLVED':1.}
                unresolved+=int(r.scenario_share>1e-12)
            else: origin_cover+=r.scenario_share
            total=sum(mapping.values())
            for o,p in mapping.items(): origin[o]=origin.get(o,0.)+r.scenario_share*p/total
            route=lookup.get((m,r.exporter_iso,imp))
            if route is not None:
                route_cover+=r.scenario_share
                for c in route: choke[c]=choke.get(c,0.)+r.scenario_share
        expected=dict(direct_hhi=float((d.scenario_share**2).sum()),origin_hhi=sum(x*x for x in origin.values()),
                      top_chokepoint_share=max(choke.values(),default=0.),route_coverage_share=route_cover,origin_coverage_share=origin_cover)
        row=saved.loc[key]
        errors['demand_share']=max(errors['demand_share'],abs(d.scenario_share.sum()-1))
        errors['value_balance_relative']=max(errors['value_balance_relative'],abs(d.scenario_value_usd.sum()/row.baseline_value_usd-1))
        for k,val in expected.items(): errors[k]=max(errors[k],abs(val-row[k]))
    assert max(errors.values())<2e-6, errors
    assert max(errors[k] for k in errors if k not in ['demand_share','value_balance_relative'])<1e-8
    cap=alloc.groupby(['base_year','metal','stage','scenario','exporter_iso']).agg(
        allocated=('scenario_value_usd','sum'),outside=('outside_universe_commitment_usd','max'),ceiling=('export_envelope_value_usd','max'))
    cap['relative_excess']=np.maximum(cap.allocated+cap.outside-cap.ceiling,0)/np.maximum(cap.ceiling,1)
    assert cap.relative_excess.max()<2e-7
    # Independently rebuild historical capacity and outside-cohort commitments
    # from the trade input, instead of trusting repeated allocation metadata.
    trade=pd.read_csv(u.HERE/'results/trade_corridors.csv.gz') if trade_input is None else trade_input.copy()
    if trade_input is None and (folder/'complete.json').exists():
        meta=json.loads((folder/'complete.json').read_text())
        trade=trade[trade.metal.eq('Nickel')].reset_index(drop=True)
        sup=pd.MultiIndex.from_frame(trade[['stage','exporter_iso']]).factorize(sort=True)[0]
        lane=pd.MultiIndex.from_frame(trade[['stage','exporter_iso','importer_iso']]).factorize(sort=True)[0]
        rng=np.random.default_rng(202609090+meta['draw'])
        def ar(n):
            z=rng.standard_normal((n,18))
            for j in range(1,18): z[:,j]=meta['rho']*z[:,j-1]+np.sqrt(1-meta['rho']**2)*z[:,j]
            return z
        sx=ar(int(sup.max()+1)); lx=ar(int(lane.max()+1)); y=trade.year.to_numpy()-2007
        trade.reconstructed_value_usd*=np.exp(.1*(np.sqrt(.5)*sx[sup,y]+np.sqrt(.5)*lx[lane,y]))
    totals=trade.groupby(['metal','stage','exporter_iso','year']).reconstructed_value_usd.sum()
    rebuilt=[]
    for year in sorted(cases.base_year.unique()):
        table=totals.reset_index()
        table=table[table.year.between(year-5,year)]
        peaks=table.groupby(['metal','stage','exporter_iso']).reconstructed_value_usd.max()
        current=table[table.year.eq(year)].set_index(['metal','stage','exporter_iso']).reconstructed_value_usd
        subset=alloc[alloc.base_year.eq(year)]
        inside=subset.groupby(['metal','stage','scenario','exporter_iso']).baseline_value_usd.sum()
        for (m,stage,scenario,exp),row in cap.loc[year].iterrows():
            expected_ceiling=1.1*peaks.get((m,stage,exp),0.)
            expected_outside=max(current.get((m,stage,exp),0.)-inside.loc[(m,stage,scenario,exp)],0.)
            rebuilt.append(dict(year=int(year),metal=m,stage=stage,scenario=scenario,exporter=exp,
                ceiling_relative_error=abs(row.ceiling-expected_ceiling)/max(expected_ceiling,1.),
                outside_relative_error=abs(row.outside-expected_outside)/max(current.get((m,stage,exp),0.),1.)))
    rebuilt=pd.DataFrame(rebuilt)
    historical_errors={k:float(rebuilt[k].max()) for k in ['ceiling_relative_error','outside_relative_error']}
    assert max(historical_errors.values())<1e-9,historical_errors
    rebuilt.to_csv(folder/'independent_historical_capacity_inputs.csv.gz',index=False)
    cap.to_csv(folder/'independent_capacity_audit.csv.gz')
    d=cases.delta_direct_hhi; o=cases.delta_origin_hhi; c=cases.delta_top_chokepoint_share
    assert np.array_equal(cases.risk_transfer,(d<=-.025)&((o>=.025)|(c>=.025)))
    assert np.array_equal(cases.material_joint_reduction,(d<=-.025)&(o<=-.025)&(c<=-.025))
    pair=cases[cases.scenario.ne('baseline')].pivot(index=KEY[:-1],columns='scenario',values='objective_weight_direct')
    assert np.allclose(pair.direct_partner,pair.integrated,atol=1e-14,rtol=0)
    report=dict(accepted=True,case_rows=len(cases),unit_years=len(cases)//3,
        years=sorted(cases.base_year.unique().tolist()),minerals=int(cases.metal.nunique()),
        optimisation_groups=len(solvers)//3,actual_optimisations=int(solvers.scenario.ne('baseline').sum()),
        solver_checks=maxima,independent_metric_errors=errors,
        max_relative_capacity_excess=float(cap.relative_excess.max()),positive_unresolved_origin_allocations=unresolved,
        historical_capacity_input_errors=historical_errors,
        allocation_classification='three indicators; NOT the historical attribution-index-qualified class')
    (folder/'independent_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--pilot',action='store_true'); args=ap.parse_args()
    if args.pilot:
        reports=[]
        for marker in sorted((u.OUT/'nickel_joint_allocation_pilot_conic').glob('*/complete.json')):
            reports.append(dict(folder=marker.parent.name,**audit(marker.parent)))
        (u.OUT/'nickel_joint_allocation_pilot_conic/validation_summary.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    else: audit(u.OUT)
