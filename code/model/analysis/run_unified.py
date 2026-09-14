"""Isolated all-mineral BACI integration and matched allocation rerun."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'research_process/nickel_scope_correction_20260908'))
import run_scope_reanalysis as s
import numpy as np
import pandas as pd

OUT = HERE/'results'
ROUTES = ROOT/'research_process/nickel_route_extension_20260908/results'
FROZEN = ROOT/'research_process/nickel_scope_correction_20260908/frozen/reproduction/project_snapshot'

def modules():
    infer = s.load_module('unified_infer', FROZEN/'modeling/infer_mine_origin_flows.py')
    lm = s.load_module('unified_lm', FROZEN/'longitudinal_2012_2024/analysis/longitudinal_mrci.py')
    sys.path.insert(0, str(FROZEN/'modeling/global_coupled_counterfactual'))
    import core
    core.local.FIRST_DATA_YEAR = 2007
    core.local.FIRST_IDENTIFIABLE_BASE_YEAR = 2012
    core.local.LANES = OUT/'routes.csv'
    return infer, lm, core

def read_archive(member):
    with zipfile.ZipFile(s.ARCHIVE) as z:
        return pd.read_csv(z.open(s.IP+member), compression='gzip')

def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [s.ARCHIVE, s.PRODUCTS, ROUTES/'global_detector_audit/minerals_lanes_existing_support_geometric.csv',
             ROUTES/'minerals_lanes_nickel_extended_geometric.csv']
    contract = {'inputs': {str(p.relative_to(ROOT)): s.sha(p) for p in paths},
                'driver_sha256': s.sha(__file__), 'scope_driver_sha256': s.sha(s.__file__)}
    for p in [FROZEN/'modeling/infer_mine_origin_flows.py', FROZEN/'modeling/global_coupled_counterfactual/core.py',
              FROZEN/'modeling/global_coupled_counterfactual/rolling_support.py',
              FROZEN/'modeling/run_capacity_constrained_derisking_counterfactual.py',
              FROZEN/'longitudinal_2012_2024/analysis/longitudinal_mrci.py']:
        contract['inputs'][str(p.relative_to(ROOT))] = s.sha(p)
    p = OUT/'input_contract.json'
    if p.exists():
        assert json.loads(p.read_text()) == contract, 'Changed contract: start a separate output branch'
    else:
        p.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    requirements = [dict(hs6=code, year=y, global_raw_file=str(ROOT/f'_raw/{code}_{y}.json'),
                         available=(ROOT/f'_raw/{code}_{y}.json').is_file())
                    for code in [750110,750120,283324] for y in range(2012,2025)]
    pd.DataFrame(requirements).to_csv(OUT/'comtrade_expansion_input_gate.csv', index=False)
    global_routes = pd.read_csv(paths[2]).fillna({'chokepoints':''})
    nickel_routes = pd.read_csv(paths[3]).fillna({'chokepoints':''})
    combined = pd.concat([global_routes[global_routes.metal.ne('Nickel')], nickel_routes[nickel_routes.metal.eq('Nickel')]], ignore_index=True)
    assert not combined.duplicated(['metal','origin','dest']).any()
    combined.to_csv(OUT/'routes.csv', index=False)
    old_trade = read_archive('baci_hs07_trade_panel_2007_2024.csv.gz')
    raw = pd.read_csv(s.PRODUCTS)
    nickel, count = s.trade_inputs(raw, list(s.STAGE))
    valid = raw.hs07_code.isin(s.STAGE) & raw.trade_value_usd.gt(0)
    valid &= raw.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False) & raw.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
    valid &= raw.exporter_iso.ne(raw.importer_iso)
    raw.loc[~valid].to_csv(OUT/'nickel_excluded_product_records.csv.gz', index=False)
    trade = pd.concat([old_trade[old_trade.metal.ne('Nickel')][s.EDGE+['reconstructed_value_usd']], nickel], ignore_index=True)
    assert not trade.duplicated(s.EDGE).any()
    trade.to_csv(OUT/'trade_corridors.csv.gz', index=False)
    seed = read_archive('mine_origin_seed_bgs_wmd_2008_2024.csv.gz')
    seed.to_csv(OUT/'production_seeds.csv.gz', index=False)
    return trade, seed, old_trade

def build(trade, seed, infer, lm):
    old_annual = read_archive('risk_indicator_panel_baci_2008_2024.csv.gz')
    regression = []
    for metal in sorted(seed.metal.unique()):
        folder = OUT/'minerals'/metal.replace('/','_')
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'inputs_complete.json').exists():
            regression.extend(json.loads((folder/'inputs_complete.json').read_text())['regression'])
            continue
        t = trade[trade.metal.eq(metal)]
        order = ['ore','material','compound','metal'] if metal=='Nickel' else sorted(t.stage.unique(), key=lambda x: infer.STAGE_ORDER[x])
        print(f'Infer {metal}: {len(t)} corridors; order={order}', flush=True)
        annual, fps, mass = s.infer_inputs(t, seed[seed.metal.eq(metal)], order, .65, infer, lm, OUT/'routes.csv')
        assert mass.max_relative_value_error.max() < 1e-12
        records=[]
        if metal!='Nickel':
            a=annual.set_index(s.GROUP).sort_index()
            b=old_annual[old_annual.metal.eq(metal)].set_index(s.GROUP).sort_index()
            assert a.index.equals(b.index), metal
            for c in ['direct_total_value_usd','direct_hhi','origin_hhi','origin_coverage_ratio']:
                error=float((abs(a[c]-b[c])/np.maximum(abs(b[c]),1.)).max())
                assert error < 1e-11, (metal,c,error)
                records.append(dict(metal=metal,field=c,max_normalized_error=error))
        else:
            ref=pd.read_csv(ROUTES/'expanded6_extended/annual_indicators.csv.gz').set_index(s.GROUP).sort_index()
            a=annual.set_index(s.GROUP).sort_index()
            assert a.index.equals(ref.index)
            for c in ['direct_hhi','origin_hhi','route_coverage_share','top_chokepoint_share']:
                error=float(abs(a[c]-ref[c]).max())
                assert error < 1e-11,(metal,c,error)
                records.append(dict(metal=metal,field=c,max_normalized_error=error))
        annual.to_csv(folder/'annual.csv.gz',index=False)
        fps.to_csv(folder/'profiles.csv.gz',index=False)
        mass.to_csv(folder/'conservation.csv',index=False)
        (folder/'inputs_complete.json').write_text(json.dumps({'regression':records}),encoding='utf-8')
        regression.extend(records)
    pd.DataFrame(regression).to_csv(OUT/'reference_regression.csv',index=False)
    annual=pd.concat([pd.read_csv(p) for p in sorted((OUT/'minerals').glob('*/annual.csv.gz'))],ignore_index=True)
    annual.to_csv(OUT/'annual_indicators.csv.gz',index=False)
    return annual

def allocate(trade, annual, core, years):
    routes=core.local.load_routes()
    for metal, data in annual.groupby('metal', sort=True):
        folder=OUT/'minerals'/metal.replace('/','_')
        fps=s.fingerprints_dict(pd.read_csv(folder/'profiles.csv.gz'))
        t=trade[trade.metal.eq(metal)]
        dest=folder/'allocation'
        dest.mkdir(exist_ok=True)
        for year in years:
            units=core.local.rolling_universe(data,year)
            baseline,capacity,pools=core.local.year_trade_components(t,year)
            for stage, group in units.groupby('stage',sort=True):
                token=f'{year}_{stage}'
                marker=dest/f'{token}_complete.json'
                if marker.exists(): continue
                model=core.build_group_model(year,group,baseline,capacity,pools,fps[year],routes)
                print(f'Allocate {metal}/{token}: {len(model.units)} units, {model.variable_count} variables',flush=True)
                cases, allocations, audits=[],[],[]
                cap=model.exporter_capacity.set_index('exporter_iso')
                for scenario,objective in [('baseline',None),('direct_partner','matched_direct'),('integrated','integrated')]:
                    tick=time.monotonic()
                    if objective is None:
                        vector=model.start.copy()
                        audit=dict(success=True,message='Reconstructed BACI baseline',iterations=0,**core.constraint_violations(model,vector))
                    else:
                        vector,audit=s.solve(core,model,objective)
                        if audit['success']: vector=core.set_route_epigraph_to_realized(model,vector)
                    audits.append(dict(base_year=year,metal=metal,stage=stage,scenario=scenario,elapsed_seconds=time.monotonic()-tick,**audit))
                    if not audit['success']:
                        pd.DataFrame(audits).to_csv(dest/f'{token}_failed.csv',index=False)
                        raise RuntimeError(f'Unverified solver: {metal}/{token}/{scenario}')
                    for unit in model.units:
                        shares=unit.baseline_shares.copy()
                        shares[unit.active_exporter_indices]=vector[unit.variable_indices]
                        case=core.case_from_shares(unit,shares,year,scenario,True,audit['message'],audit['iterations'])
                        if scenario=='direct_partner':
                            case['objective']='matched_normalized_direct'
                            case['objective_weight_direct']=core.integrated_weights_for_unit(unit)[0]
                        cases.append(case)
                        for j,exp in enumerate(unit.context.exporters):
                            if shares[j]<=0 and unit.context.baseline_value[j]<=0: continue
                            c=cap.loc[exp]
                            allocations.append(dict(base_year=year,metal=metal,stage=stage,importer_iso=unit.context.importer_iso,scenario=scenario,exporter_iso=exp,
                                baseline_value_usd=float(unit.context.baseline_value[j]),scenario_value_usd=float(shares[j]*unit.demand),scenario_share=float(shares[j]),
                                current_global_value_usd=float(c.current_global_value_usd),outside_universe_commitment_usd=float(c.outside_universe_commitment_usd),export_envelope_value_usd=float(c.envelope_value_usd)))
                pd.DataFrame(cases).to_csv(dest/f'{token}_cases.csv.gz',index=False)
                pd.DataFrame(allocations).to_csv(dest/f'{token}_allocations.csv.gz',index=False)
                pd.DataFrame(audits).to_csv(dest/f'{token}_solver.csv',index=False)
                marker.write_text(json.dumps(dict(year=year,metal=metal,stage=stage)),encoding='utf-8')
    frames={name:pd.concat([pd.read_csv(p) for p in sorted((OUT/'minerals').glob(f'*/allocation/*_{name}.{ext}'))],ignore_index=True)
            for name,ext in [('cases','csv.gz'),('allocations','csv.gz'),('solver','csv')]}
    assert not frames['cases'].duplicated(['base_year','metal','stage','importer_iso','scenario']).any()
    for name,df in frames.items(): df.to_csv(OUT/f'all_{name}.csv.gz',index=False)
    core.global_capacity_audit(frames['allocations']).to_csv(OUT/'capacity_audit.csv',index=False)
    core.CF.summarize_cases(frames['cases'],['base_year','scenario','scenario_label']).to_csv(OUT/'annual_allocation_summary.csv',index=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--years',default='2024')
    ap.add_argument('--inputs-only',action='store_true')
    args=ap.parse_args()
    trade,seed,_=setup()
    infer,lm,core=modules()
    annual=build(trade,seed,infer,lm)
    if not args.inputs_only: allocate(trade,annual,core,[int(y) for y in args.years.split(',')])
    print('Requested reference stage complete; full-paper release gates remain open.',flush=True)
