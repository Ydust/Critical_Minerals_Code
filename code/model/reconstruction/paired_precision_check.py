"""Paired tighter solver test on the six previously diagnosed problems.

Only solver stopping tolerances change (1e-11 to 1e-13); acceptance stays 1e-6.
Legacy and rebuilt inputs are kept in distinct branches. No old solution is input.
"""
from pathlib import Path
import importlib.util,sys,json,hashlib,types
CAP=Path(__file__).resolve().parent/'capsule';SCRIPT_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('allocation_run',CAP/'run_allocation_stage.py');a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)
import numpy as np
import pandas as pd
def main():
    out=CAP/'outputs/paired_precision';out.mkdir(exist_ok=True)
    scope=a.load('allocation_scope');infer=a.load('infer_mine_origin_flows');lm=a.load('longitudinal_mrci');cache=a.load('cached_routes');helper=a.load('allocation_conditional_helpers')
    code=(CAP/'code/allocation_verified.py').read_text();assert code.count('=1e-11')==3
    strict=types.ModuleType('precision_solver');exec(compile(code.replace('=1e-11','=1e-13'),'<precision_solver>','exec'),strict.__dict__)
    core=a.load('core','modeling/global_coupled_counterfactual/core.py');core.local.FIRST_DATA_YEAR=2007;core.local.FIRST_IDENTIFIABLE_BASE_YEAR=2012;core.local.LANES=CAP/'inputs/historical/routes.csv';routes=core.local.load_routes()
    oldtrade=pd.read_csv(CAP/'expected/SD5_trade_corridors.csv.gz');oldseed=pd.read_csv(CAP/'expected/SD5_production_seeds.csv.gz');baseweights=dict(infer.LOCAL_STAGE_WEIGHT)
    checks=[];hashes={}
    def read(p):hashes[str(p.relative_to(CAP))]=hashlib.sha256(p.read_bytes()).hexdigest();return pd.read_csv(p)
    for path in sorted((CAP/'outputs/allocation_conditional').glob('rho_*/diagnostics/DIAGNOSIS.json')):
        diag=json.loads(path.read_text());draw=diag['draw'];rho=diag['rho'];folder=path.parents[1];expected=CAP/'expected/allocation_conditional'/folder.name
        ot,os,w=helper.perturb(oldtrade,oldseed,baseweights,draw,rho);nt=read(folder/'trade_corridors.csv.gz');ns=read(folder/'production_seeds.csv.gz')
        annuals=[read(expected/'annual_indicators.csv.gz'),read(folder/'annual_indicators.csv.gz')];models={};profiles={}
        for g in diag['groups']:
            year,metal,stage,scenario=[g[k] for k in ['base_year','metal','stage','scenario']];rows=[];vectors=[];audits=[]
            for branch,t,q,annual in zip(['legacy','rebuilt'],[ot,nt],[os,ns],annuals):
                infer.LOCAL_STAGE_WEIGHT.update(w);part=t[t.metal.eq(metal)];order=['ore','material','compound','metal'] if metal=='Nickel' else sorted(part.stage.unique(),key=lambda s:infer.STAGE_ORDER[s])
                cache.install(lm,core.local.LANES,part[part.year.eq(2018)],lm.concentration_stats(part[part.year.eq(2018)],'exporter_iso','reconstructed_value_usd','direct'))
                if (branch,metal) not in profiles:
                    _,fp,mass=scope.infer_inputs(part,q[q.metal.eq(metal)],order,w['material'],infer,lm,core.local.LANES);profiles[(branch,metal)]=scope.fingerprints_dict(fp)
                cohort=core.local.rolling_universe(annual,year);cohort=cohort[cohort.metal.eq(metal)&cohort.stage.eq(stage)];baseline,capacity,pools=core.local.year_trade_components(part,year)
                model=core.build_group_model(year,cohort,baseline,capacity,pools,profiles[(branch,metal)][year],routes);obj='matched_direct' if scenario=='direct_partner' else 'integrated'
                vector,audit=strict.solve(core,model,obj);assert audit['success'] and audit.get('solver','').startswith('Clarabel'),audit
                vector=core.set_route_epigraph_to_realized(model,vector);vectors.append(vector);audits.append(audit);flow=[]
                for unit in model.units:
                    shares=unit.baseline_shares.copy();shares[unit.active_exporter_indices]=vector[unit.variable_indices]
                    for j,exp in enumerate(unit.context.exporters):flow.append(dict(importer_iso=unit.context.importer_iso,exporter_iso=exp,share=shares[j]))
                frame=pd.DataFrame(flow).set_index(['importer_iso','exporter_iso']).sort_index();rows.append(frame)
                frame.to_csv(out/f'{folder.name}_{year}_{metal.replace("/","_")}_{stage}_{branch}.csv.gz')
            assert rows[0].index.equals(rows[1].index);error=float(abs(rows[0].share-rows[1].share).max())
            checks.append(dict(draw=draw,rho=rho,year=year,metal=metal,stage=stage,scenario=scenario,share_error=error,accepted=error<1e-6,audits=audits));print(f'Paired precision: {draw}/{rho}/{year}/{metal}/{stage}: {error:.4g}',flush=True)
    report=dict(accepted=all(x['accepted'] for x in checks),checks=checks,stopping_tolerance=1e-13,share_acceptance_unchanged=1e-6,original_failures_preserved=True,full_batch_precision_rerun=False,script_sha256=SCRIPT_SHA,input_hashes=hashes)
    (out/'PAIRED_PRECISION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['checks','input_hashes']},indent=2));assert report['accepted']
if __name__=='__main__':main()
