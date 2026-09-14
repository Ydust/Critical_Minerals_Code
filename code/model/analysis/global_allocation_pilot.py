"""Resumable full-mineral conditional joint reoptimisation execution pilot."""
import argparse
import json
import time
import run_unified as u
import solver_verified as v
import cached_routes
import validate_results
import numpy as np
import pandas as pd

REF = u.OUT
OUT = REF/'global_joint_allocation_pilot'

def ar(rng,n,rho):
    z=rng.standard_normal((n,18))
    for j in range(1,18): z[:,j]=rho*z[:,j-1]+np.sqrt(1-rho*rho)*z[:,j]
    return z

def perturb(trade,seed,weights,draw,rho):
    # Shared supplier-country shocks span minerals; residual shocks are corridor specific.
    supplier=pd.MultiIndex.from_frame(trade[['stage','exporter_iso']]).factorize(sort=True)[0]
    lane=pd.MultiIndex.from_frame(trade[['metal','stage','exporter_iso','importer_iso']]).factorize(sort=True)[0]
    producer=pd.Index(seed.mine_origin_iso).factorize(sort=True)[0]
    rng=np.random.default_rng(202609091000+draw)
    x=ar(rng,int(supplier.max()+1),rho)
    y=ar(rng,int(lane.max()+1),rho)
    p=ar(rng,int(producer.max()+1),rho)
    w={k:float(np.clip(value+rng.uniform(-.2,.2),0,1)) for k,value in weights.items()}
    t=trade.copy(); q=seed.copy()
    ti=t.year.to_numpy()-2007; qi=q.year.to_numpy()-2007
    t.reconstructed_value_usd*=np.exp(.1*(np.sqrt(.5)*x[supplier,ti]+np.sqrt(.5)*y[lane,ti]))
    q.production*=np.exp(.1*p[producer,qi])
    return t,q,w

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--draws-per-design',type=int,default=1)
    args=ap.parse_args(); assert args.draws_per_design>0
    OUT.mkdir(exist_ok=True)
    paths=[REF/'trade_corridors.csv.gz',REF/'production_seeds.csv.gz',REF/'routes.csv',REF/'input_contract.json',
           u.Path(__file__),u.Path(u.__file__),u.Path(u.s.__file__),u.Path(v.__file__),u.Path(cached_routes.__file__),u.Path(validate_results.__file__)]
    contract={str(p.relative_to(u.ROOT)):u.s.sha(p) for p in paths}
    reference=json.loads((REF/'input_contract.json').read_text())
    for path,digest in reference['inputs'].items(): assert u.s.sha(u.ROOT/path)==digest
    assert u.s.sha(u.__file__)==reference['driver_sha256']
    saved=OUT/'contract.json'
    if saved.exists(): assert json.loads(saved.read_text())==contract,'Use a new branch for changed code or input'
    else: saved.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    trade=pd.read_csv(REF/'trade_corridors.csv.gz'); seed=pd.read_csv(REF/'production_seeds.csv.gz')
    infer,lm,core=u.modules(); weights=dict(infer.LOCAL_STAGE_WEIGHT)
    original_select=core.local.rolling_universe
    u.s.solve=v.solve
    check=trade[trade.metal.eq('Nickel')&trade.year.eq(2018)]
    check_direct=lm.concentration_stats(check,'exporter_iso','reconstructed_value_usd','direct')
    cache_audit=cached_routes.install(lm,REF/'routes.csv',check,check_direct)
    protocol=dict(scope='All 23 supported minerals; 2012–2024 allocations with 2007–2024 BACI histories',
        pilot_only=True,rhos=[0.,.8],random_seed_base=202609091000,trade_log_sd=.1,production_log_sd=.1,
        stage_weight_uniform_halfwidth=.2,trade_covariance='Half variance shared across minerals by supplier-country/stage, half mineral-corridor specific',
        production_covariance='Common country shock across minerals; country processes independent',
        weight_covariance='One stage-weight draw shared across all minerals and years',
        distributions='Uncalibrated median-preserving lognormal perturbations; not confidence intervals',
        fixed=['positive trade and mine support','matched route geometry','BACI source reconciliation'],
        recomputed=['origins','cohort','route trade weights','six-year export ceilings','outside-cohort commitments','both optimisation objectives'],
        route_cache_regression=cache_audit,full_raw_data_uncertainty=False)
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for draw in range(1,args.draws_per_design+1):
        for rho in [0.,.8]:
            dest=OUT/f'rho_{rho:.1f}_draw_{draw:03d}'; dest.mkdir(exist_ok=True)
            if (dest/'complete.json').exists(): continue
            tick=time.monotonic(); t,q,w=perturb(trade,seed,weights,draw,rho)
            print(f'Global draw {draw}, rho={rho}: reinference',flush=True)
            infer.LOCAL_STAGE_WEIGHT.update(w)
            if not (dest/'draw_inputs.json').exists():
                t.to_csv(dest/'trade_corridors.csv.gz',index=False); q.to_csv(dest/'production_seeds.csv.gz',index=False)
                (dest/'draw_inputs.json').write_text(json.dumps(dict(draw=draw,rho=rho,weights=w,
                    trade_sha256=u.s.sha(dest/'trade_corridors.csv.gz'),seed_sha256=u.s.sha(dest/'production_seeds.csv.gz'))),encoding='utf-8')
            else:
                meta=json.loads((dest/'draw_inputs.json').read_text()); assert meta['weights']==w
                for filename,key in [('trade_corridors.csv.gz','trade_sha256'),('production_seeds.csv.gz','seed_sha256')]:
                    assert u.s.sha(dest/filename)==meta[key]
            for metal in sorted(seed.metal.unique()):
                folder=dest/'minerals'/metal.replace('/','_'); folder.mkdir(parents=True,exist_ok=True)
                if (folder/'inputs_complete.json').exists(): continue
                part=t[t.metal.eq(metal)]
                order=['ore','material','compound','metal'] if metal=='Nickel' else sorted(part.stage.unique(),key=lambda s:infer.STAGE_ORDER[s])
                annual,fp,mass=u.s.infer_inputs(part,q[q.metal.eq(metal)],order,w['material'],infer,lm,REF/'routes.csv')
                assert mass.max_relative_value_error.max()<1e-12
                annual.to_csv(folder/'annual.csv.gz',index=False); fp.to_csv(folder/'profiles.csv.gz',index=False)
                mass.to_csv(folder/'conservation.csv',index=False)
                (folder/'inputs_complete.json').write_text(json.dumps(dict(stage_order=order)),encoding='utf-8')
                print(f'Global draw {draw}, rho={rho}: {metal} origin inputs complete',flush=True)
            annual=pd.concat([pd.read_csv(p) for p in sorted((dest/'minerals').glob('*/annual.csv.gz'))],ignore_index=True)
            annual.to_csv(dest/'annual_indicators.csv.gz',index=False)
            cohorts={year:original_select(annual,year) for year in range(2012,2025)}
            def selected(frame,year): return cohorts[year][cohorts[year].metal.isin(frame.metal.unique())].copy()
            core.local.rolling_universe=selected
            u.OUT=dest
            u.allocate(t,annual,core,list(range(2012,2025)))
            report=validate_results.audit(dest,trade_input=t)
            assert report['minerals']==23 and report['years']==list(range(2012,2025))
            (dest/'complete.json').write_text(json.dumps(dict(draw=draw,rho=rho,elapsed_seconds=time.monotonic()-tick,
                validation=report,stage_weights=w)),encoding='utf-8')
            print(f'Global draw {draw}, rho={rho}: independently validated',flush=True)
    summary=[]
    for marker in sorted(OUT.glob('*/complete.json')):
        meta=json.loads(marker.read_text()); frame=pd.read_csv(marker.parent/'annual_allocation_summary.csv')
        frame.insert(0,'draw',meta['draw']); frame.insert(1,'rho',meta['rho']); summary.append(frame)
    pd.concat(summary,ignore_index=True).to_csv(OUT/'all_pilot_annual_summaries.csv',index=False)
    print(f'{len(summary)} full-mineral pilot draws complete. Not uncertainty intervals.',flush=True)

if __name__=='__main__': main()
