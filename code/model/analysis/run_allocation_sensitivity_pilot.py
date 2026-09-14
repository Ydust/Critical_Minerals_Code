"""20 nickel joint-input allocation pilots; not a full-mineral uncertainty release."""
from pathlib import Path
import hashlib
import json
import time
import run_unified as u
import numpy as np
import pandas as pd

REF=u.OUT
PILOT=REF/'nickel_joint_allocation_pilot'

def ar(rng,n,rho):
    z=rng.standard_normal((n,18))
    for j in range(1,18): z[:,j]=rho*z[:,j-1]+np.sqrt(1-rho*rho)*z[:,j]
    return z

def main():
    PILOT.mkdir(parents=True,exist_ok=True)
    trade=pd.read_csv(REF/'trade_corridors.csv.gz').query('metal == "Nickel"').reset_index(drop=True)
    seed=pd.read_csv(REF/'production_seeds.csv.gz').query('metal == "Nickel"').reset_index(drop=True)
    infer,lm,core=u.modules()
    weights=dict(infer.LOCAL_STAGE_WEIGHT)
    supplier=pd.MultiIndex.from_frame(trade[['stage','exporter_iso']]).factorize(sort=True)[0]
    lane=pd.MultiIndex.from_frame(trade[['stage','exporter_iso','importer_iso']]).factorize(sort=True)[0]
    producer=pd.MultiIndex.from_frame(seed[['metal','mine_origin_iso']]).factorize(sort=True)[0]
    years=trade.year.to_numpy()-2007
    py=seed.year.to_numpy()-2007
    contract=dict(driver_sha256=u.s.sha(__file__),unified_driver_sha256=u.s.sha(u.__file__),
                  reference_contract_sha256=u.s.sha(REF/'input_contract.json'),
                  trade_sha256=u.s.sha(REF/'trade_corridors.csv.gz'),seed_sha256=u.s.sha(REF/'production_seeds.csv.gz'),route_sha256=u.s.sha(REF/'routes.csv'))
    saved=PILOT/'contract.json'
    if saved.exists(): assert json.loads(saved.read_text())==contract
    else: saved.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    protocol=dict(scope='Nickel six-product BACI only; all 2012–2024 allocation years',draws_per_temporal_design=10,rhos=[0.,.8],
        random_seed_base=202609090,trade_log_sd=.1,production_log_sd=.1,stage_weight_uniform_halfwidth=.2,
        covariance='Half variance shared by stage-supplier and half corridor-specific; stationary annual AR(1)',
        distribution_status='Uncalibrated sensitivity assumptions; pilot output is not a confidence interval',
        rerun=['2007–2024 trade values','2008–2024 production seeds','stage weights','recursive origins','route trade weights','cohort screens','historical export ceilings','joint optimisation'],
        excluded=['raw BACI reconciliation','unseen trading partners','new mine countries','route geometry draws','actual voyages','non-nickel minerals'])
    (PILOT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    for rho in [0.,.8]:
        for draw in range(1,11):
            name=f'rho_{rho:.1f}_draw_{draw:02d}'
            dest=PILOT/name
            marker=dest/'complete.json'
            if marker.exists(): continue
            dest.mkdir(exist_ok=True)
            rng=np.random.default_rng(202609090+draw)
            x=ar(rng,int(supplier.max()+1),rho)
            y=ar(rng,int(lane.max()+1),rho)
            p=ar(rng,int(producer.max()+1),rho)
            w={key:float(np.clip(v+rng.uniform(-.2,.2),0,1)) for key,v in weights.items()}
            t=trade.copy()
            t.reconstructed_value_usd*=np.exp(.1*(np.sqrt(.5)*x[supplier,years]+np.sqrt(.5)*y[lane,years]))
            q=seed.copy()
            q.production*=np.exp(.1*p[producer,py])
            infer.LOCAL_STAGE_WEIGHT.update(w)
            tick=time.monotonic()
            print(f'Joint allocation pilot {name}: reinference begins',flush=True)
            annual,fps,mass=u.s.infer_inputs(t,q,['ore','material','compound','metal'],w['material'],infer,lm,REF/'routes.csv')
            folder=dest/'minerals/Nickel'
            folder.mkdir(parents=True,exist_ok=True)
            fps.to_csv(folder/'profiles.csv.gz',index=False)
            annual.to_csv(folder/'annual.csv.gz',index=False)
            mass.to_csv(folder/'conservation.csv',index=False)
            u.OUT=dest
            u.allocate(t,annual,core,list(range(2012,2025)))
            marker.write_text(json.dumps(dict(draw=draw,rho=rho,stage_weights=w,elapsed_seconds=time.monotonic()-tick,
                max_conservation_error=float(mass.max_relative_value_error.max()),years=list(range(2012,2025)))),encoding='utf-8')
    parts=[]
    for marker in sorted(PILOT.glob('*/complete.json')):
        m=json.loads(marker.read_text())
        d=pd.read_csv(marker.parent/'annual_allocation_summary.csv')
        d.insert(0,'draw',m['draw']); d.insert(1,'rho',m['rho'])
        parts.append(d)
    pd.concat(parts,ignore_index=True).to_csv(PILOT/'all_pilot_annual_summaries.csv',index=False)
    print('20 paired-design nickel allocation pilots complete; not full-mineral uncertainty.',flush=True)

if __name__=='__main__': main()
