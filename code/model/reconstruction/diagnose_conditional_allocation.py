"""Locate threshold failures and cross-evaluate saved solutions on rebuilt problems.

Diagnostic only: never replaces either solution or changes acceptance thresholds.
"""
from pathlib import Path
import importlib.util,sys,json,hashlib
CAP=Path(__file__).resolve().parent/'capsule'
SCRIPT_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('allocation_run',CAP/'run_allocation_stage.py');a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
import numpy as np
import pandas as pd
def main():
    scope=a.load('allocation_scope');core=a.load('core','modeling/global_coupled_counterfactual/core.py');core.local.FIRST_DATA_YEAR=2007;core.local.FIRST_IDENTIFIABLE_BASE_YEAR=2012;core.local.LANES=CAP/'inputs/historical/routes.csv';routes=core.local.load_routes()
    reports=[]
    for path in sorted((CAP/'outputs/allocation_conditional').glob('rho_*/ALLOCATION_CONDITIONAL_QA.json')):
        qa=json.loads(path.read_text())
        if qa['accepted']:continue
        folder=path.parent;dest=folder/'diagnostics';dest.mkdir(exist_ok=True);expected=CAP/'expected/allocation_conditional'/folder.name
        ac=pd.read_csv(folder/'all_cases.csv.gz').set_index(a.KEY).sort_index();ec=pd.read_csv(expected/'all_cases.csv.gz').set_index(a.KEY).sort_index();assert ac.index.equals(ec.index)
        af=pd.read_csv(folder/'all_flows.csv.gz');ef=pd.read_csv(expected/'all_allocations.csv.gz');differences=[]
        for c in qa['checks']['cases']['errors']:
            if c=='reallocation_value_usd':norm=ec.baseline_value_usd
            else:norm=np.maximum(abs(ec[c]),1.)
            delta=(abs(ac[c]-ec[c])/norm)
            for key in delta[delta>=1e-6].index:
                differences.append(dict(zip(a.KEY,key),field=c,recomputed=float(ac.loc[key,c]),reference=float(ec.loc[key,c]),absolute_difference=float(abs(ac.loc[key,c]-ec.loc[key,c])),normalized_difference=float(delta.loc[key])))
        details=pd.DataFrame(differences,columns=a.KEY+['field','recomputed','reference','absolute_difference','normalized_difference']);details.to_csv(dest/'numeric_case_differences.csv',index=False)
        fk=a.KEY+['exporter_iso'];x=af.set_index(fk).scenario_share;y=ef.set_index(fk).scenario_share;ix=x.index.union(y.index);d=(x.reindex(ix,fill_value=0)-y.reindex(ix,fill_value=0)).abs();fd=d[d>=1e-6].rename('absolute_share_difference').reset_index();fd.to_csv(dest/'flow_share_differences.csv',index=False)
        rv=pd.read_csv(folder/'reporting_v2_recomputed.csv.gz').set_index(a.KEY).sort_index();ev=pd.read_csv(folder/'reporting_v2_reference.csv.gz').set_index(a.KEY).sort_index()
        labels=['supplier_status','supplier_label','supplier_candidates','chokepoint_status','chokepoint_label','chokepoint_candidates','reallocation_status']
        mask=rv[labels].fillna('').ne(ev[labels].fillna('')).any(axis=1)
        label_diff=rv.loc[mask,labels].join(ev.loc[mask,labels],lsuffix='_recomputed',rsuffix='_reference').reset_index();label_diff.to_csv(dest/'reporting_label_differences.csv',index=False)
        groups=pd.concat([details[['base_year','metal','stage','scenario']],fd[['base_year','metal','stage','scenario']],label_diff[['base_year','metal','stage','scenario']]],ignore_index=True).drop_duplicates()
        trade=pd.read_csv(folder/'trade_corridors.csv.gz');annual=pd.read_csv(folder/'annual_indicators.csv.gz');checks=[]
        for row in groups.itertuples(index=False):
            year,metal,stage,scenario=row;cohort=core.local.rolling_universe(annual,year);cohort=cohort[cohort.metal.eq(metal)&cohort.stage.eq(stage)]
            t=trade[trade.metal.eq(metal)];baseline,capacity,pools=core.local.year_trade_components(t,year);profiles=scope.fingerprints_dict(pd.read_csv(folder/metal.replace('/','_')/'profiles.csv.gz'))
            model=core.build_group_model(year,cohort,baseline,capacity,pools,profiles[year],routes)
            vectors=[]
            for flow in [af,ef]:
                part=flow[flow.base_year.eq(year)&flow.metal.eq(metal)&flow.stage.eq(stage)&flow.scenario.eq(scenario)].set_index(['importer_iso','exporter_iso']).scenario_share
                vector=model.start.copy()
                for unit in model.units:
                    for j,var in zip(unit.active_exporter_indices,unit.variable_indices):vector[var]=part.get((unit.context.importer_iso,unit.context.exporters[j]),0.)
                vectors.append(core.set_route_epigraph_to_realized(model,vector))
            obj='matched_direct' if scenario=='direct_partner' else 'integrated';h,c=scope.matched_quadratic(core,model,obj);v,w=vectors;delta=v-w
            fv=float(.5*v@(h@v)+c@v);fw=float(.5*w@(h@w)+c@w)
            checks.append(dict(base_year=int(year),metal=metal,stage=stage,scenario=scenario,recomputed_objective=fv,reference_objective_on_rebuilt_model=fw,objective_difference=fv-fw,max_vector_difference=float(abs(delta).max()),quadratic_difference_energy=float(.5*delta@(h@delta)),recomputed_constraints=core.constraint_violations(model,v),reference_constraints_on_rebuilt_model=core.constraint_violations(model,w)))
        report=dict(draw=qa['draw'],rho=qa['rho'],failed_numeric_case_fields=len(details),failed_flow_shares=len(fd),groups=checks,classification_mismatches={k:v for k,v in qa['checks']['cases']['mismatches'].items() if k not in ['top_exporter_iso','top_chokepoint']},original_acceptance_unchanged=qa['accepted'],threshold_unchanged=1e-6,diagnosis_scope='Cross-evaluation of saved solutions, not a replacement solve or a new acceptance criterion',script_sha256=SCRIPT_SHA)
        report['reporting_label_difference_rows']=len(label_diff)
        (dest/'DIAGNOSIS.json').write_text(json.dumps(report,indent=2),encoding='utf-8');reports.append(report)
    print(json.dumps(reports,indent=2))
if __name__=='__main__':main()
