"""Separate measurement, cohort and reoptimization effects of route extension."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from build_routes import ROOT, OUT, SCOPE, FROZEN, prepare, sr, sha, scope
from audit_geometric_incidence import robust_labels
import pandas as pd
import numpy as np

GEOMETRIC=OUT/'minerals_lanes_nickel_extended_geometric.csv'
COVERAGE=OUT/'minerals_lanes_nickel_extended.csv'
YEARS=list(range(2012,2025))
CONFIGS={
    'original3_extended':('original3',GEOMETRIC,YEARS,False,False),
    'expanded6_extended':('expanded6',GEOMETRIC,YEARS,False,False),
    'expanded6_fixed_cohort':('expanded6',GEOMETRIC,YEARS,True,False),
    'expanded6_coverage_only':('expanded6',COVERAGE,[2024],False,False),
    'expanded6_reverse_extended':('expanded6_reverse',GEOMETRIC,[2024],False,False),
    'expanded6_w0_extended':('expanded6_w0',GEOMETRIC,[2024],False,False),
    'expanded6_w1_extended':('expanded6_w1',GEOMETRIC,[2024],False,False),
    'expanded6_halifax_endpoint':('expanded6',GEOMETRIC,[2019,2022,2024],False,True),
}


def modules():
    frozen=SCOPE.parent/'frozen/reproduction/project_snapshot'
    lm=scope.load_module('nickel_route_lm',frozen/'longitudinal_2012_2024/analysis/longitudinal_mrci.py')
    sys.path.insert(0,str(frozen/'modeling/global_coupled_counterfactual'))
    import core
    core.local.FIRST_DATA_YEAR=2007
    core.local.FIRST_IDENTIFIABLE_BASE_YEAR=2012
    scope.OUT=OUT
    return lm,core


def port_scenario():
    rt,template,assets=prepare()
    ports=sr.setup_P('networkx')
    selected={data['port']:(node,data) for node,data in ports.nodes(data=True) if data['port'] in ['CAHAL','CUMOA']}
    assert len(selected)==2
    start,end=selected['CUMOA'][0],selected['CAHAL'][0]
    result=sr.searoute(start,end,units='km',speed_knot=24,append_orig_dest=False,restrictions=[rt.Passage.northwest],
        include_ports=False,port_params={},return_passages=False,algorithm=None,backend='networkx')
    vertices=result.geometry['coordinates']
    labels=robust_labels(vertices,rt.load_chokepoints(template/'inputs/chokepoint_boxes.csv'))
    table=pd.read_csv(GEOMETRIC).fillna({'chokepoints':''})
    row=table.metal.eq('Nickel')&table.origin.eq('CUB')&table.dest.eq('CAN')
    assert row.sum()==1
    before=table.loc[row,'chokepoints'].iloc[0]
    table.loc[row,'chokepoints']='|'.join(labels)
    path=OUT/'material_stage_halifax_endpoint_routes.csv'
    table.to_csv(path,index=False)
    manifest=dict(material_stage_only=True,country_pair=['CUB','CAN'],port_origin=selected['CUMOA'][1],port_dest=selected['CAHAL'][1],
        old_labels=before,new_labels=labels,route_vertices=vertices,route_length_km=result.properties['length'],
        route_port_database_sha256=sha(assets/'data/ports_dict.py'),
        dated_operator_evidence=['Q4 2019','Q1 2022'],scenario_years=[2019,2022,2024],
        interpretation='Entire material-stage national corridor assigned to named facility endpoints as a sensitivity endpoint; actual national flow fraction unknown; 2024 is an extrapolated scenario, not an observed yearly route.')
    (OUT/'halifax_endpoint_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return path,tuple(labels)


def route_annual(annual,trade,lm,path,material_path=None):
    direct_cols=[c for c in annual if c in scope.GROUP or c.startswith('direct_')]
    direct=annual[direct_cols]
    if material_path is None:
        routed=lm.route_stats(trade[trade.year.ge(2008)],direct,path)
    else:
        parts=[]
        for stage,g in trade[trade.year.ge(2008)].groupby('stage'):
            parts.append(lm.route_stats(g,direct[direct.stage.eq(stage)],material_path if stage=='material' else path))
        routed=pd.concat(parts,ignore_index=True)
    old=[c for c in routed if c not in scope.GROUP]
    result=annual.drop(columns=old,errors='ignore').merge(routed,on=scope.GROUP,validate='one_to_one')
    assert len(result)==len(annual)
    for col in ['direct_hhi','origin_hhi','direct_total_value_usd','origin_total_value_usd']:
        a=annual.set_index(scope.GROUP)[col].sort_index()
        b=result.set_index(scope.GROUP)[col].sort_index()
        pd.testing.assert_series_equal(a,b)
    return result


def evaluate_old_allocations(lm,core):
    """No optimizer is called. Re-evaluate the frozen old decisions on new geometry."""
    dest=OUT/'expanded6_fixed_allocations/allocation'
    dest.mkdir(parents=True,exist_ok=True)
    source=SCOPE/'expanded6'
    trade=pd.read_csv(source/'trade_corridors.csv.gz')
    oldannual=pd.read_csv(source/'annual_indicators.csv.gz')
    fps=scope.fingerprints_dict(pd.read_csv(source/'exporter_origin_profiles.csv.gz'))
    allocations=pd.read_csv(source/'allocation/all_allocations.csv.gz')
    share_lookup=allocations.set_index(['base_year','stage','importer_iso','scenario','exporter_iso']).scenario_share
    core.local.LANES=GEOMETRIC
    routes=core.local.load_routes()
    cases=[]
    for year in YEARS:
        universe=core.local.rolling_universe(oldannual,year)
        baseline,capacity,pools=core.local.year_trade_components(trade,year)
        for _,units in universe.groupby(['metal','stage']):
            model=core.build_group_model(year,units,baseline,capacity,pools,fps[year],routes)
            for unit in model.units:
                for scenario in ['baseline','direct_partner','integrated']:
                    prefix=(year,unit.context.stage,unit.context.importer_iso,scenario)
                    shares=np.array([share_lookup.get(prefix+(exp,),0.) for exp in unit.context.exporters])
                    assert abs(shares.sum()-1)<2e-6
                    frozen_error=float(np.abs(shares[unit.frozen_exporter_indices]-unit.baseline_shares[unit.frozen_exporter_indices]).max(initial=0.))
                    case=core.case_from_shares(unit,shares,year,scenario,frozen_error<=2e-8,'Re-evaluation only; no new optimum claimed',0)
                    case['optimizer_run']=False
                    case['new_frozen_supplier_max_share_error']=frozen_error
                    case['decision_basis']='frozen_old_route_solution'
                    if scenario=='direct_partner':
                        case['objective']='matched_normalized_direct'
                        case['objective_weight_direct']=core.integrated_weights_for_unit(unit)[0]
                    cases.append(case)
    cases=pd.DataFrame(cases)
    cases.to_csv(dest/'all_cases.csv.gz',index=False)
    allocations.to_csv(dest/'all_allocations.csv.gz',index=False)
    core.CF.summarize_cases(cases,['base_year','scenario','scenario_label']).to_csv(dest/'annual_summary.csv',index=False)
    core.CF.summarize_cases(cases,['base_year','stage','scenario','scenario_label']).to_csv(dest/'annual_stage_summary.csv',index=False)
    core.global_capacity_audit(allocations).to_csv(dest/'capacity_audit.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--variants',default=','.join(CONFIGS))
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    lm,core=modules()
    material_path,material_labels=port_scenario()
    original_select=core.local.rolling_universe
    original_context=core.local.build_context
    inventory=[]
    configs={}
    for name in args.variants.split(','):
        source,route_path,years,fixed,port=CONFIGS[name]
        folder=SCOPE/source
        trade=pd.read_csv(folder/'trade_corridors.csv.gz')
        oldannual=pd.read_csv(folder/'annual_indicators.csv.gz')
        fps=pd.read_csv(folder/'exporter_origin_profiles.csv.gz')
        annual=route_annual(oldannual,trade,lm,route_path,material_path if port else None)
        dest=OUT/name
        dest.mkdir(parents=True,exist_ok=True)
        annual.to_csv(dest/'annual_indicators.csv.gz',index=False)
        windows,summary=scope.historical(annual)
        windows.to_csv(dest/'historical_three_indicator_windows.csv.gz',index=False)
        summary.to_csv(dest/'historical_three_indicator_summary.csv',index=False)
        core.local.LANES=route_path
        core.local.rolling_universe=(lambda _,y,ref=oldannual: original_select(ref,y)) if fixed else original_select
        if port:
            def context(unit,baseline,capacity,pools,fingerprints,routes):
                if unit.stage=='material':
                    routes=dict(routes)
                    routes[('Nickel','CUB','CAN')]=material_labels
                return original_context(unit,baseline,capacity,pools,fingerprints,routes)
            core.local.build_context=context
        else:
            core.local.build_context=original_context
        for year in years:
            selected=core.local.rolling_universe(annual,year)
            for stage in ['all']+sorted(annual.stage.unique()):
                all_current=annual[annual.year.eq(year)]
                s=selected
                if stage!='all':
                    all_current=all_current[all_current.stage.eq(stage)]
                    s=s[s.stage.eq(stage)]
                total=all_current.direct_total_value_usd.sum()
                included=s.direct_total_value_usd_end.sum()
                inventory.append(dict(variant=name,year=year,stage=stage,all_units=len(all_current),selected_units=len(s),all_value_usd=total,selected_value_usd=included,selected_value_fraction=included/total if total else np.nan))
        run_years=([2024]+[y for y in years if y!=2024]) if 2024 in years else years
        scope.run_allocation(name,trade,annual,fps,run_years,core,args.resume)
        configs[name]=dict(origin_source_variant=source,route_file=str(route_path.relative_to(OUT)),route_sha256=sha(route_path),years=years,
            cohort='old_route_cohort' if fixed else 'unchanged_rules_rescreened',material_stage_port_alternative=port)
        core.local.rolling_universe=original_select
        core.local.build_context=original_context
    if set(args.variants.split(','))==set(CONFIGS):
        evaluate_old_allocations(lm,core)
    pd.DataFrame(inventory).to_csv(OUT/'allocation_selected_coverage.csv',index=False)
    (OUT/'analysis_manifest.json').write_text(json.dumps(dict(variants=configs,code_sha256=sha(__file__),scope_driver_sha256=sha(scope.__file__),
        input_scope='BACI HS07 V202601 nickel only; original Comtrade manuscript pipeline not replaced',
        uncertainty='deterministic conditional reanalysis; no sampling intervals',status='candidate outputs not promoted'),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
