"""Outer bounds for unassigned route mass in fixed saved allocations.

No reoptimisation against adversarial routes, and no relaxation of origin priors.
For known chokepoint maximum C and unmatched share u, C_true is in [C,min(1,C+u)].
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
R=HERE/'results'
OUT=R/'allocation_unmatched_route_bounds'
KEY=['base_year','metal','stage','importer_iso']

def calculate(folder,draw,rho):
    d=pd.read_csv(folder/'all_cases.csv.gz')
    base=d[d.scenario.eq('baseline')][KEY+['top_chokepoint_share','route_coverage_share']]
    base=base.rename(columns={'top_chokepoint_share':'C0','route_coverage_share':'coverage0'})
    x=d[d.scenario.ne('baseline')].merge(base,on=KEY,validate='many_to_one')
    u0=(1-x.coverage0).clip(0,1); u1=(1-x.route_coverage_share).clip(0,1)
    x['route_delta_lower']=x.top_chokepoint_share-np.minimum(1,x.C0+u0)
    x['route_delta_upper']=np.minimum(1,x.top_chokepoint_share+u1)-x.C0
    direct=x.delta_direct_hhi.le(-.025); origin_improves=x.delta_origin_hhi.le(-.025); origin_worsens=x.delta_origin_hhi.ge(.025)
    x['joint_lower']=direct&origin_improves&x.route_delta_upper.le(-.025)
    x['joint_upper']=direct&origin_improves&x.route_delta_lower.le(-.025)
    x['transfer_lower']=direct&(origin_worsens|x.route_delta_lower.ge(.025))
    x['transfer_upper']=direct&(origin_worsens|x.route_delta_upper.ge(.025))
    assert (x.joint_lower<=x.material_joint_reduction).all() and (x.material_joint_reduction<=x.joint_upper).all()
    assert (x.transfer_lower<=x.risk_transfer).all() and (x.risk_transfer<=x.transfer_upper).all()
    rows=[]
    for (year,scenario),g in x.groupby(['base_year','scenario']):
        w=g.baseline_value_usd
        row=dict(draw=draw,rho=rho,base_year=int(year),scenario=scenario,units=len(g),baseline_value_usd=float(w.sum()),
            baseline_unmatched_share=float(np.average((1-g.coverage0).clip(0,1),weights=w)),
            allocated_unmatched_share=float(np.average((1-g.route_coverage_share).clip(0,1),weights=w)),
            maximum_unit_route_coverage_change=float(abs(g.route_coverage_share-g.coverage0).max()),
            minimum_allocated_route_coverage=float(g.route_coverage_share.min()))
        for c in ['joint_lower','joint_upper','transfer_lower','transfer_upper','material_joint_reduction','risk_transfer']:
            row[c+'_value_share']=float(np.average(g[c],weights=w))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True)
    results=[calculate(R,0,-1.)]
    for marker in sorted((R/'global_joint_allocation_pilot').glob('*/complete.json')):
        m=json.loads(marker.read_text()); assert m['validation']['accepted']
        results.append(calculate(marker.parent,m['draw'],m['rho']))
    data=pd.concat(results,ignore_index=True)
    data.to_csv(OUT/'annual_bounds.csv',index=False)
    p=data.pivot(index=['draw','rho','base_year'],columns='scenario',values=['transfer_lower_value_share','transfer_upper_value_share','joint_lower_value_share','joint_upper_value_share'])
    comparison=pd.DataFrame(index=p.index)
    comparison['transfer_comparison_margin']=p[('transfer_lower_value_share','direct_partner')]-p[('transfer_upper_value_share','integrated')]
    comparison['joint_comparison_margin']=p[('joint_lower_value_share','integrated')]-p[('joint_upper_value_share','direct_partner')]
    comparison['guaranteed_lower_integrated_transfer']=comparison.transfer_comparison_margin.gt(0)
    comparison['guaranteed_higher_integrated_joint_reduction']=comparison.joint_comparison_margin.gt(0)
    comparison.reset_index().to_csv(OUT/'objective_comparison_bounds.csv',index=False)
    (OUT/'SCOPE.json').write_text(json.dumps(dict(draws=len(results)-1,reference_included=True,
        route_bounds='Unmatched route mass can add between zero and its entire share to any named bottleneck',
        fixed=['allocation shares','inferred origin mixtures','matched-route incidence'],
        interval_type='Conservative identification outer bounds, not confidence intervals',
        adversarial_route_reoptimisation=False,full_unknown_origin_bounds=False),indent=2),encoding='utf-8')
    print(data[(data.draw==0)&(data.base_year==2024)].to_string(index=False))

if __name__=='__main__': main()
