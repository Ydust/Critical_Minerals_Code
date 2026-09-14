from pathlib import Path
import os,sys,json,importlib.util,time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CAP=ROOT/'research_process/portable_reproduction_20260910/capsule'
OUT=HERE/'results/allocation';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(CAP/'code'))
import allocation_scope as scope
import allocation_verified as verified
spec=importlib.util.spec_from_file_location('core',CAP/'code/modeling/global_coupled_counterfactual/core.py')
core=importlib.util.module_from_spec(spec);sys.modules['core']=core;spec.loader.exec_module(core)
core.local.LANES=CAP/'inputs/historical/routes.csv'
METALS=['Copper','Aluminium','PGM','Nickel']

def main():
    trade=pd.read_csv(CAP/'outputs/baci/trade_corridors_rebuilt.csv.gz')
    annual=pd.read_csv(CAP/'outputs/allocation/annual_indicators.csv.gz')
    cohort=core.local.rolling_universe(annual,2024)
    routes=core.local.load_routes();original=core.integrated_weights_for_unit
    # Multiplier sweep keeps the original direct coefficient, including no-route units.
    configs=[('reference',1.,1.,.025,.10),('direct_only',0.,0.,.025,.10)]
    configs += [(f'origin_{v:g}',v,1.,.025,.10) for v in [0,.25,.5,2,4]]
    configs += [(f'route_{v:g}',1.,v,.025,.10) for v in [.25,.5,2,4]]
    configs += [(f'floor_{v:g}',1.,1.,v,.10) for v in [.01,.05]]
    configs += [(f'headroom_{v:g}',1.,1.,.025,v) for v in [0,.25]]
    configs += [(f'headroom_{v:g}_direct',0.,0.,.025,v) for v in [0,.25]]
    rows=[];audits=[];caps=[]
    expected=pd.read_csv(CAP/'outputs/allocation/all_cases.csv.gz')
    plan=pd.read_csv(CAP/'inputs/allocation/solver_plan.csv').set_index(['base_year','metal','stage','scenario']).solver
    coverage=[]
    for metal in METALS:
        t=trade[trade.metal.eq(metal)];units=cohort[cohort.metal.eq(metal)]
        current=annual[annual.year.eq(2024)&annual.metal.eq(metal)]
        base=current[(current.direct_total_value_usd>=1e7)&(current.origin_coverage_ratio>=.8)&(current.route_coverage_share>=.5)]
        coverage.append(dict(metal=metal,rolling_units=len(units),baseline_eligible_units=len(base),rolling_value_usd=float(units.direct_total_value_usd_end.sum()),baseline_eligible_value_usd=float(base.direct_total_value_usd.sum())))
        profiles=scope.fingerprints_dict(pd.read_csv(CAP/f'outputs/allocation/{metal}/profiles.csv.gz'))[2024]
        baseline,capacity,pools=core.local.year_trade_components(t,2024)
        for stage,group in units.groupby('stage',sort=True):
            for name,om,rm,floor,headroom in configs:
                token=f'{metal}_{stage}_{name}';saved=OUT/(token+'_cases.csv')
                if saved.exists():
                    rows.extend(pd.read_csv(saved).to_dict('records'))
                    audits.append(json.loads((OUT/(token+'_audit.json')).read_text()))
                    caps.extend(pd.read_csv(OUT/(token+'_capacity.csv')).to_dict('records'));continue
                core.MAIN_HEADROOM=headroom;core.ROUTE_ZERO_BASELINE_EPSILON_C=floor
                core.integrated_weights_for_unit=lambda u:(original(u)[0],original(u)[1]*om,original(u)[2]*rm)
                model=core.build_group_model(2024,group,baseline,capacity,pools,profiles,routes)
                tick=time.monotonic()
                # Reproduce the reference using its recorded solver, not a substituted solver.
                solver=scope.solve if name=='reference' and not str(plan.loc[(2024,metal,stage,'integrated')]).startswith('Clarabel') else verified.solve
                vector,audit=solver(core,model,'integrated')
                audit.update(metal=metal,stage=stage,configuration=name,elapsed_seconds=time.monotonic()-tick)
                (OUT/(token+'_audit.json')).write_text(json.dumps(audit,indent=2))
                if not audit['success']:raise RuntimeError('Strict solve failure '+token)
                vector=core.set_route_epigraph_to_realized(model,vector)
                cr=[];used={}
                for unit in model.units:
                    shares=unit.baseline_shares.copy();shares[unit.active_exporter_indices]=vector[unit.variable_indices]
                    row=core.case_from_shares(unit,shares,2024,'integrated',True,audit['message'],audit['iterations'])
                    row.update(configuration=name,origin_multiplier=om,route_multiplier=rm,floor=floor)
                    cr.append(row)
                    for exp,share in zip(unit.context.exporters,shares):used[exp]=used.get(exp,0.)+float(share*unit.demand)
                ca=model.exporter_capacity.copy();ca['selected_value_usd']=ca.exporter_iso.map(used).fillna(0)
                ca['total_value_usd']=ca.selected_value_usd+ca.outside_universe_commitment_usd
                ca['relative_excess']=np.maximum(ca.total_value_usd-ca.envelope_value_usd,0)/np.maximum(ca.envelope_value_usd,1)
                ca['binding_envelope']=(ca.envelope_value_usd-ca.total_value_usd)<=np.maximum(1,ca.envelope_value_usd*2e-6)
                ca['configuration']=name
                assert ca.relative_excess.max()<2e-6
                if name=='reference':
                    got=pd.DataFrame(cr).set_index('importer_iso');exp=expected[(expected.base_year==2024)&expected.metal.eq(metal)&expected.stage.eq(stage)&expected.scenario.eq('integrated')].set_index('importer_iso')
                    cols=['direct_hhi','origin_hhi','top_chokepoint_share']
                    err=float(abs(got[cols]-exp[cols]).max().max());audit['baseline_metric_regression_error']=err
                    assert err<1e-6,(token,err)
                pd.DataFrame(cr).to_csv(saved,index=False);ca.to_csv(OUT/(token+'_capacity.csv'),index=False)
                (OUT/(token+'_audit.json')).write_text(json.dumps(audit,indent=2))
                rows.extend(cr);caps.extend(ca.to_dict('records'));audits.append(audit)
                print(token,'accepted',round(time.monotonic()-tick,2),flush=True)
    data=pd.DataFrame(rows);data.to_csv(OUT/'all_cases.csv.gz',index=False)
    pd.DataFrame(audits).to_csv(OUT/'solver_audit.csv',index=False)
    pd.DataFrame(caps).to_csv(OUT/'capacity_audit.csv.gz',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'cohort_coverage.csv',index=False)
    summary=[]
    for (metal,config),d in data.groupby(['metal','configuration']):
        w=d.baseline_value_usd
        row=dict(metal=metal,configuration=config,units=len(d),baseline_value_usd=float(w.sum()))
        for c in ['delta_direct_hhi','delta_origin_hhi','delta_top_chokepoint_share','reallocation_share_of_baseline','risk_transfer','material_joint_reduction']:
            row[c]=float(np.average(d[c].astype(float),weights=w))
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/'mineral_summary.csv',index=False)
    (OUT/'QA.json').write_text(json.dumps(dict(accepted=True,solves=len(audits),minerals=METALS,base_year=2024,configurations=len(configs),scope='Targeted latest-baseline preference and design diagnostics, not complete Pareto frontier or calibrated uncertainty'),indent=2))
if __name__=='__main__':main()
