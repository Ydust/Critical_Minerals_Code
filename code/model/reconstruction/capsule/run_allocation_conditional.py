from pathlib import Path
import sys,os,json,hashlib,importlib.util,time,importlib.metadata
import argparse
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
p=argparse.ArgumentParser();p.add_argument('--draw',type=int,required=True);p.add_argument('--rho',type=float,choices=[0.,.8],required=True);ARGS=p.parse_args();assert 1<=ARGS.draw<=5
HERE=Path(__file__).resolve().parent;OUT=HERE/f'outputs/allocation_conditional/rho_{ARGS.rho:.1f}_draw_{ARGS.draw:03d}';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'ALLOCATION_CONDITIONAL_QA.json').exists(),'Do not overwrite completed draws'
allowed=[HERE.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()];blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    raw=os.fsdecode(args[0])
    if raw.lower() in ('nul','\\\\.\\nul'):return
    p=Path(raw).resolve()
    if not any(p.is_relative_to(r) for r in allowed):
        blocked.append(str(p));raise PermissionError('Outside capsule/runtime: '+str(p))
sys.addaudithook(guard)
import numpy as np
import pandas as pd
def load(name,relative=None):
    spec=importlib.util.spec_from_file_location(name,HERE/'code'/(relative or name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
KEY=['base_year','metal','stage','importer_iso','scenario']
def compare(a,e,keys,tol,columns=None,ignore=()):
    assert not a.duplicated(keys).any() and not e.duplicated(keys).any()
    a=a.set_index(keys).sort_index();e=e.set_index(keys).sort_index();common=a.index.intersection(e.index)
    report=dict(rows=len(a),expected_rows=len(e),missing_keys=len(e.index.difference(a.index)),extra_keys=len(a.index.difference(e.index)),errors={},mismatches={})
    cols=columns or list(e.columns)
    for c in cols:
        if c in ignore:continue
        if c not in a:report['mismatches'][c]='missing_column';continue
        x=a.loc[common,c];y=e.loc[common,c]
        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
            x=pd.to_numeric(x,errors='coerce');bad=int((x.isna()!=y.isna()).sum());v=(abs(x-y)/np.maximum(abs(y),1)).max();report['errors'][c]=float(v) if pd.notna(v) else 0.
        else:bad=int((x.fillna('').astype(str)!=y.fillna('').astype(str)).sum())
        if bad:report['mismatches'][c]=bad
    report['accepted']=not report['missing_keys'] and not report['extra_keys'] and not report['mismatches'] and all(v<tol for v in report['errors'].values())
    return report
def flow_check(a,e):
    keys=KEY+['exporter_iso'];assert not a.duplicated(keys).any() and not e.duplicated(keys).any()
    x=a.set_index(keys).scenario_share;y=e.set_index(keys).scenario_share
    ix=x.index.union(y.index);err=float(abs(x.reindex(ix,fill_value=0)-y.reindex(ix,fill_value=0)).max())
    return dict(accepted=err<1e-6,max_absolute_share_error=err,extra_sparse_keys=len(x.index.difference(y.index)),missing_sparse_keys=len(y.index.difference(x.index)))
def main():
    start=time.monotonic()
    manifest=json.loads((HERE/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json').read_text())
    for i in manifest:assert hashlib.sha256((HERE/i['file']).read_bytes()).hexdigest()==i['sha256'],i['file']
    infer=load('infer_mine_origin_flows');lm=load('longitudinal_mrci');cache=load('cached_routes');scope=load('allocation_scope');verified=load('allocation_verified')
    core=load('core','modeling/global_coupled_counterfactual/core.py')
    core.local.FIRST_DATA_YEAR=2007;core.local.FIRST_IDENTIFIABLE_BASE_YEAR=2012;core.local.LANES=HERE/'inputs/historical/routes.csv'
    trade=pd.read_csv(HERE/'outputs/baci/trade_corridors_rebuilt.csv.gz');seed=pd.read_csv(HERE/'outputs/seed/allocation_production_seed_rebuilt.csv.gz')
    helper=load('allocation_conditional_helpers')
    trade,seed,weights=helper.perturb(trade,seed,dict(infer.LOCAL_STAGE_WEIGHT),ARGS.draw,ARGS.rho)
    infer.LOCAL_STAGE_WEIGHT.update(weights)
    trade.to_csv(OUT/'trade_corridors.csv.gz',index=False);seed.to_csv(OUT/'production_seeds.csv.gz',index=False)
    (OUT/'draw_parameters.json').write_text(json.dumps(dict(draw=ARGS.draw,rho=ARGS.rho,weights=weights),indent=2),encoding='utf-8')
    sample=trade[trade.metal.eq('Nickel')&trade.year.eq(2018)]
    cache_test=cache.install(lm,core.local.LANES,sample,lm.concentration_stats(sample,'exporter_iso','reconstructed_value_usd','direct'))
    annuals=[];profiles={}
    for metal in sorted(seed.metal.unique()):
        t=trade[trade.metal.eq(metal)]
        order=['ore','material','compound','metal'] if metal=='Nickel' else sorted(t.stage.unique(),key=lambda x:infer.STAGE_ORDER[x])
        annual,fp,mass=scope.infer_inputs(t,seed[seed.metal.eq(metal)],order,weights['material'],infer,lm,core.local.LANES)
        folder=OUT/metal.replace('/','_');folder.mkdir(exist_ok=True)
        annual.to_csv(folder/'annual.csv.gz',index=False);fp.to_csv(folder/'profiles.csv.gz',index=False);mass.to_csv(folder/'conservation.csv',index=False)
        annuals.append(annual);profiles[metal]=scope.fingerprints_dict(fp)
        print('Rebuilt allocation attribution: '+metal,flush=True)
    annual=pd.concat(annuals,ignore_index=True);annual.to_csv(OUT/'annual_indicators.csv.gz',index=False)
    cohorts={y:core.local.rolling_universe(annual,y) for y in range(2012,2025)}
    routes=core.local.load_routes()
    original_weights=core.integrated_weights_for_unit
    def no_origin(unit):
        d,o,c=original_weights(unit);return d,0.,c
    allcases=[];allflows=[];allaudits=[];count=0
    for metal in sorted(profiles):
        t=trade[trade.metal.eq(metal)]
        for year in [2024]+list(range(2012,2024)):
            units=cohorts[year];units=units[units.metal.eq(metal)]
            baseline,capacity,pools=core.local.year_trade_components(t,year)
            for stage,group in units.groupby('stage',sort=True):
                model=core.build_group_model(year,group,baseline,capacity,pools,profiles[metal][year],routes)
                cap=model.exporter_capacity.set_index('exporter_iso');cases=[];flows=[];audits=[];integrated_vector=None
                for scenario,obj in [('baseline',None),('direct_partner','matched_direct'),('integrated','integrated')]:
                    core.integrated_weights_for_unit=no_origin if scenario=='without_origin' else original_weights
                    tick=time.monotonic()
                    if obj is None:vector=model.start.copy();audit=dict(success=True,message='Reconstructed BACI baseline',iterations=0,**core.constraint_violations(model,vector))
                    else:
                        solver='Clarabel' # Frozen full-mineral conditional driver uses verified conic solve with unchanged fallback.
                        vector,audit=(verified.solve if solver.startswith('Clarabel') else scope.solve)(core,model,obj)
                        if audit['success']:vector=core.set_route_epigraph_to_realized(model,vector)
                    if not audit['success']:
                        (OUT/'FAILED_SOLVE.json').write_text(json.dumps(dict(metal=metal,year=year,stage=stage,scenario=scenario,**audit),indent=2));raise RuntimeError('Strict solver gate failed')
                    if scenario=='integrated':integrated_vector=vector.copy()
                    if scenario=='without_origin':
                        fun=core.objective_and_gradient(model,'integrated')[0]
                        new=float(fun(vector));ref=float(fun(integrated_vector));assert new<=ref+2e-7
                        audit.update(ablation_objective=new,reference_under_ablation_objective=ref)
                    audits.append(dict(base_year=year,metal=metal,stage=stage,scenario=scenario,elapsed_seconds=time.monotonic()-tick,**audit))
                    for unit in model.units:
                        shares=unit.baseline_shares.copy();shares[unit.active_exporter_indices]=vector[unit.variable_indices]
                        case=core.case_from_shares(unit,shares,year,'integrated' if scenario=='without_origin' else scenario,True,audit['message'],audit['iterations'])
                        if scenario=='direct_partner':case.update(objective='matched_normalized_direct',objective_weight_direct=original_weights(unit)[0])
                        if scenario=='without_origin':case.update(scenario=scenario,scenario_label='Direct and maritime objective; original coefficients retained',objective='integrated_without_origin')
                        cases.append(case)
                        for j,exp in enumerate(unit.context.exporters):
                            if shares[j]<=0 and unit.context.baseline_value[j]<=0:continue
                            c=cap.loc[exp];flows.append(dict(base_year=year,metal=metal,stage=stage,importer_iso=unit.context.importer_iso,scenario=scenario,exporter_iso=exp,baseline_value_usd=float(unit.context.baseline_value[j]),scenario_value_usd=float(shares[j]*unit.demand),scenario_share=float(shares[j]),current_global_value_usd=float(c.current_global_value_usd),outside_universe_commitment_usd=float(c.outside_universe_commitment_usd),export_envelope_value_usd=float(c.envelope_value_usd)))
                core.integrated_weights_for_unit=original_weights
                folder=OUT/metal.replace('/','_');token=f'{year}_{stage}'
                pd.DataFrame(cases).to_csv(folder/(token+'_cases.csv.gz'),index=False);pd.DataFrame(flows).to_csv(folder/(token+'_flows.csv.gz'),index=False)
                pd.DataFrame(audits).to_csv(folder/(token+'_solver.csv'),index=False)
                allcases.extend(cases);allflows.extend(flows);allaudits.extend(audits);count+=1
        print(f'Allocation and ablation complete: {metal}; {count} groups',flush=True)
    cases=pd.DataFrame(allcases);flows=pd.DataFrame(allflows);audits=pd.DataFrame(allaudits)
    for name,frame in [('cases',cases),('flows',flows),('solver',audits)]:frame.to_csv(OUT/('all_'+name+'.csv.gz'),index=False)
    capacity_audit=core.global_capacity_audit(flows);capacity_audit.to_csv(OUT/'capacity_audit.csv',index=False)
    maincases=cases[cases.scenario.ne('without_origin')];abl=cases[cases.scenario.eq('without_origin')]
    core.CF.summarize_cases(maincases,['base_year','scenario','scenario_label']).to_csv(OUT/'annual_summary.csv',index=False)
    # All expected numerical results are opened after independently generated outputs.
    physical=helper.audit(OUT,trade_input=trade)
    expected=HERE/'expected/allocation_conditional'/OUT.name
    ec=pd.read_csv(expected/'all_cases.csv.gz');ef=pd.read_csv(expected/'all_allocations.csv.gz')
    checks=dict(annual=compare(annual,pd.read_csv(expected/'annual_indicators.csv.gz'),lm.GROUP,1e-10),
        baseline=compare(maincases[maincases.scenario.eq('baseline')],ec[ec.scenario.eq('baseline')],KEY,1e-10,ignore=['solver_message','solver_iterations']),
        cases=compare(maincases,ec,KEY,1e-6,ignore=['solver_message','solver_iterations']),
        flows=flow_check(flows,ef),annual_summary=compare(pd.read_csv(OUT/'annual_summary.csv'),pd.read_csv(expected/'annual_allocation_summary.csv'),['base_year','scenario'],1e-6))
    meta=json.loads((expected/'complete.json').read_text());weights_exact=weights==meta['stage_weights']
    a=helper.view(maincases,flows);e=helper.view(ec,ef);assert a.index.equals(e.index)
    a.to_csv(OUT/'reporting_v2_recomputed.csv.gz');e.to_csv(OUT/'reporting_v2_reference.csv.gz')
    labels=['supplier_status','supplier_label','supplier_candidates','chokepoint_status','chokepoint_label','chokepoint_candidates','reallocation_status']
    mismatches={c:int(a[c].ne(e[c]).sum()) for c in labels}
    amount_error=float(((a.reallocation_value_usd-e.reallocation_value_usd).abs()/e.baseline_value_usd).max())
    identities=[float(abs(x.reallocation_value_usd/x.baseline_value_usd-x.reallocation_share_of_baseline).max()) for x in [a,e]]
    numeric={c:v for c,v in checks['cases']['errors'].items() if c!='reallocation_value_usd'}
    categorical={c:v for c,v in checks['cases']['mismatches'].items() if c not in ['top_exporter_iso','top_chokepoint']}
    reporting=not any(mismatches.values()) and amount_error<1e-6 and max(identities)<1e-10 and not categorical and max(numeric.values())<1e-6
    accepted=reporting and weights_exact and physical['accepted'] and all(c['accepted'] for k,c in checks.items() if k!='cases')
    report=dict(accepted=bool(accepted),draw=ARGS.draw,rho=ARGS.rho,original_strict_regression_passed=all(c['accepted'] for c in checks.values()),reporting_v2_equivalence_passed=bool(reporting),reporting_label_mismatches=mismatches,amount_difference_as_baseline_share=amount_error,amount_share_identity_errors=identities,weights_exact=weights_exact,checks=checks,independent_validation=physical,groups=count,unit_years=int(maincases.scenario.eq('baseline').sum()),elapsed_seconds=time.monotonic()-start,blocked_external_reads=blocked,full_pipeline_complete=False,interpretation='Conditional input reoptimisation, not calibrated confidence intervals or independent observations',input_manifest_sha256=hashlib.sha256((HERE/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json').read_bytes()).hexdigest())
    (OUT/'INPUT_MANIFEST.json').write_bytes((HERE/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json').read_bytes())
    (OUT/'ALLOCATION_CONDITIONAL_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['checks','independent_validation']},indent=2),flush=True)
    assert accepted,report
if __name__=='__main__':main()
