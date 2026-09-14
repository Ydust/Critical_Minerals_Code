"""Source-report to classification joint pipeline pilot; not calibrated inference."""
import argparse
import hashlib
import json
import sys
import time
import run_unified as u
import build_expanded_comtrade as bld
import numpy as np
import pandas as pd
sys.path.insert(0,str(u.ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h

REF=u.HERE/'results/historical_expanded_comtrade'
RAW=u.HERE/'results/full_raw_reconstruction_audit'
OUT=u.HERE/'results/raw_report_sensitivity_batch'

def corridor_frame(trade):
    trade=trade.copy()
    trade['quality']=trade.data_quality_weight.fillna(.5).clip(0,1)
    s=trade.observation_sigma_log
    trade['sigma']=s.where(np.isfinite(s)&s.gt(0),.75)
    c=trade.groupby(h.KEY+['exporter_iso'],as_index=False).agg(value=('reconstructed_value_usd','sum'),
        k=('CmdCode','size'),quality=('quality','mean'),sigma=('sigma','mean'))
    c['draw_sigma']=0. # reports already perturbed: no second corridor perturbation
    return c

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--draws-per-design',type=int,default=1); ap.add_argument('--report-sd',type=float,choices=[.1,.45,.75],required=True)
    args=ap.parse_args(); assert args.draws_per_design>0
    global OUT
    OUT=u.HERE/f'results/raw_report_scale_stress/sd_{args.report_sd:.2f}'
    OUT.mkdir(parents=True,exist_ok=True)
    assert json.loads((RAW/'SUMMARY.json').read_text())['accepted']
    sources=sorted(RAW.glob('*/observations.csv.gz'))
    contract={str(p.relative_to(u.ROOT)):u.s.sha(p) for p in sources+[u.Path(__file__),u.Path(h.__file__),u.HERE/'BATCH_PROTOCOL_20260909.md',u.HERE/'SCALE_STRESS_PROTOCOL.md',u.HERE/'frozen/build_model_inputs.py',u.HERE/'frozen/reconstruct_dynamic_panel.py',u.HERE/'results/routes.csv',u.s.ARCHIVE,RAW/'SUMMARY.json',REF/'annual_indicators.csv.gz',REF/'all_windows.csv.gz']}
    saved=OUT/'contract.json'
    if saved.exists(): assert json.loads(saved.read_text())==contract
    else: saved.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    b,r=bld.load_frozen()
    annual=pd.read_csv(REF/'annual_indicators.csv.gz'); windows=pd.read_csv(REF/'all_windows.csv.gz')
    _,_,seed,_,_,seed_hashes,_=h.load_inputs(); routes=pd.read_csv(u.HERE/'results/routes.csv')
    h.ORDER=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    reference=pd.read_csv(REF/'analytical_trade_panel.csv.gz',dtype={'CmdCode':str})
    blocks,pair,sizes,chokes=h.prepare(annual,corridor_frame(reference),seed,routes,windows)
    matrix,err=h.recompute(blocks,len(annual))
    regression={c:float(np.nanmax(abs(matrix[:,j]-annual[c])/np.maximum(abs(annual[c]),1))) for j,c in enumerate(h.FIELDS)}
    assert max(regression.values())<1e-10
    key=['CmdCode','exporter_iso','importer_iso','year']
    expected_keys=reference.set_index(key).sort_index().index
    reference_quality=reference.set_index(key).sort_index()
    reporter_codes=sorted(set().union(*(set(pd.read_csv(p,usecols=['ReporterCode']).ReporterCode) for p in sources)))
    reporter_index={v:i for i,v in enumerate(reporter_codes)}
    protocol=dict(requested_draws=2*args.draws_per_design,temporal_rhos=[0.,.8],source_report_log_sd=args.report_sd,shared_reporter_variance_fraction=.5,
        residual='HS-corridor-reporting-role process; stationary annual AR(1); same innovations across rho designs',
        production_log_sd=.1,stage_weight_halfwidth=.2,origin_seed_hashes=seed_hashes,
        baseline_regression=regression,source_layer='Normalized source observations reproduced from all 663 original responses',
        recomputed=['mirror reconciliation','mirror-gap quality weights','quality tiers','short-gap interpolation','uncertainty proxies','recursive origins','route value weights','classification screens'],
        fixed=['source presence and positive support','quantity and estimated-quantity flags','matched route geometry'],
        interpretation='Uncalibrated conditional pipeline pilot, not confidence intervals or independent mine-source identification',
        allocation_branch='BACI allocation uses a separate source family and is not rerun from Comtrade draws')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    all_rows=[]
    for draw,rho in [(draw,rho) for draw in range(1,args.draws_per_design+1) for rho in [0.,.8]]:
        dest=OUT/f'rho_{rho:.1f}_draw_{draw:03d}'; dest.mkdir(exist_ok=True)
        marker=dest/'complete.json'
        if marker.exists():
            all_rows.extend(pd.read_csv(dest/'window_summaries.csv').to_dict('records')); continue
        start=time.monotonic()
        shared=h.ar_errors(np.random.default_rng((202609092000+draw)),len(reporter_codes),rho)
        panels=[]
        for source in sources:
            name=source.parent.name
            out=dest/name; out.mkdir(exist_ok=True)
            p=out/'reconstructed.csv.gz'
            if not p.exists():
                obs=pd.read_csv(source,dtype={'CmdCode':str})
                # Stable mineral seed avoids dependence on source iteration order.
                code=int(hashlib.sha256(name.encode()).hexdigest()[:8],16)
                rng=np.random.default_rng((202609092000+draw)+code)
                lane=pd.MultiIndex.from_frame(obs[['CmdCode','exporter_iso','importer_iso','reporter_role']]).factorize(sort=True)[0]
                residual=h.ar_errors(rng,int(lane.max()+1),rho)
                year=obs.RefYear.to_numpy()-2012
                z=np.sqrt(.5)*shared[obs.ReporterCode.map(reporter_index).to_numpy(),year]+np.sqrt(.5)*residual[lane,year]
                obs.value_usd*=np.exp(args.report_sd*z)
                mirror=b.reconcile_mirrors(obs)
                baseline=mirror[mirror.model_eligible].copy()
                baseline.to_csv(out/'baseline.csv.gz',index=False)
                r.BASELINE=out/'baseline.csv.gz'
                panel=r.reconstruct_panel(r.prepare_observed())
                assert panel.year.between(2012,2024).all() and panel.next_observed_year.le(2024).all()
                panel.to_csv(p,index=False)
            else: panel=pd.read_csv(p,dtype={'CmdCode':str})
            panels.append(panel)
            print(f'Raw report batch draw={draw}, rho={rho}: {name} reconstructed',flush=True)
        trade=pd.concat(panels,ignore_index=True)
        named=trade.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&trade.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
        trade=trade[named&trade.metal.isin(annual.metal.unique())].copy()
        indexed=trade.set_index(key).sort_index()
        assert indexed.index.is_unique and indexed.index.equals(expected_keys)
        changed={c:int((abs(indexed[c]-reference_quality[c])>1e-12).sum()) for c in ['reconstructed_value_usd','data_quality_weight','observation_sigma_log']}
        blocks,pair,sizes,chokes=h.prepare(annual,corridor_frame(trade),seed,routes,windows)
        # Trade is already perturbed/reconstructed. Only seeds and weights are
        # perturbed here; zero supplier/lane noise prevents double counting.
        rng=np.random.default_rng((202609093000+draw))
        inp=dict(supplier=np.zeros((sizes['supplier'],13)),lane=np.zeros((sizes['lane'],13)),
            producer=h.ar_errors(rng,sizes['producer'],rho),
            weights={k:float(np.clip(w+rng.uniform(-.2,.2),0,1)) for k,w in h.WEIGHTS.items()})
        matrix,err=h.recompute(blocks,len(annual),inp)
        assert err<1e-10 and np.isfinite(matrix).all()
        frame=annual[h.KEY].copy()
        for j,c in enumerate(h.FIELDS): frame[c]=matrix[:,j]
        if draw==1 and args.report_sd==.1:
            pilot=pd.read_csv(u.HERE/f'results/raw_report_sensitivity_pilot/rho_{rho:.1f}_draw_001/annual_indicators.csv.gz')
            pd.testing.assert_frame_equal(frame,pilot,check_exact=False,rtol=1e-12,atol=1e-12)
        frame.to_csv(dest/'annual_indicators.csv.gz',index=False)
        cats=h.classify(matrix,pair); rows=h.summarize(cats,list(h.selections(windows)),draw,rho)
        pd.DataFrame(rows).to_csv(dest/'window_summaries.csv',index=False); all_rows.extend(rows)
        marker.write_text(json.dumps(dict(report_sd=args.report_sd,rho=rho,draw=draw,unit_years=len(annual),product_rows=len(trade),
            max_conservation_error=err,changed_product_records=changed,weights=inp['weights'],
            elapsed_seconds=time.monotonic()-start)),encoding='utf-8')
        print(f'Raw report batch draw={draw}, rho={rho}: classification complete',flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'all_pilot_window_summaries.csv',index=False)
    print('Requested source-report batch complete; assess numerical stability separately.',flush=True)

if __name__=='__main__': main()
