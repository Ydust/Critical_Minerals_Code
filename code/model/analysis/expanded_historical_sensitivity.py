"""Joint conditional draws on corrected, expanded Comtrade historical inputs."""
import json
import sys
import time
import run_unified as u
import pandas as pd
import numpy as np
sys.path.insert(0,str(u.ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h

if __name__=='__main__':
    root=u.OUT/'historical_expanded_comtrade'
    out=root/'joint_sensitivity'; out.mkdir(parents=True,exist_ok=True)
    annual=pd.read_csv(root/'annual_indicators.csv.gz')
    trade=pd.read_csv(root/'analytical_trade_panel.csv.gz',dtype={'CmdCode':str})
    windows=pd.read_csv(root/'all_windows.csv.gz')
    _,_,seed,_,_,hashes,_=h.load_inputs()
    trade['quality']=trade.data_quality_weight.fillna(.5).clip(0,1)
    sigma=trade.observation_sigma_log
    trade['sigma']=sigma.where(np.isfinite(sigma)&sigma.gt(0),.75)
    trade['variance']=(trade.reconstructed_value_usd*trade.sigma)**2
    corridor=trade.groupby(h.KEY+['exporter_iso'],as_index=False).agg(value=('reconstructed_value_usd','sum'),k=('CmdCode','size'),
        quality=('quality','mean'),sigma=('sigma','mean'),variance=('variance','sum'))
    corridor['draw_sigma']=(np.sqrt(corridor.variance)/corridor.value).clip(.05,1.5)
    # Explicit order removes the material/compound tie exposed by nickel expansion.
    h.ORDER=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    blocks,pair,sizes,chokes=h.prepare(annual,corridor,seed,pd.read_csv(u.OUT/'routes.csv'),windows)
    baseline,err=h.recompute(blocks,len(annual))
    errors={c:float(np.nanmax(abs(baseline[:,j]-annual[c])/np.maximum(abs(annual[c]),1))) for j,c in enumerate(h.FIELDS)}
    assert max(errors.values())<1e-10,errors
    cats=h.classify(baseline,pair)
    selected=windows.three_layer_evidence_eligible&windows.apparent_derisking
    assert np.array_equal(cats['apparent'],selected)
    assert np.array_equal(cats['transfer'],selected&windows.classification.isin(['mine_origin_transfer','route_transfer','multiple_risk_transfer']))
    assert np.array_equal(cats['substantive'],selected&windows.classification.eq('substantive_derisking'))
    masks=list(h.selections(windows)); rows=h.summarize(cats,masks,0,-1.)
    params=[]; start=time.monotonic()
    for rho in [0.,.8]:
        membership={k:np.zeros(len(windows),int) for k in ['eligible','apparent','transfer','substantive']}
        for draw in range(1,251):
            inp=h.draw_inputs(sizes,draw,rho)
            matrix,error=h.recompute(blocks,len(annual),inp)
            assert error<1e-10
            err=max(err,error)
            c=h.classify(matrix,pair)
            for k in membership: membership[k]+=c[k]
            rows.extend(h.summarize(c,masks,draw,rho))
            params.append(dict(draw=draw,rho=rho,weights=inp['weights']))
            if draw%50==0: print(f'Expanded historical rho={rho} draw={draw}/250, elapsed={time.monotonic()-start:.1f}s',flush=True)
        counts=windows[['importer_iso','metal','stage','window_family','baseline_year','end_year']].copy()
        for k,n in membership.items(): counts[k+'_draw_count']=n
        counts['draws']=250
        counts.to_csv(out/f'membership_rho_{rho:.1f}.csv.gz',index=False)
    pd.DataFrame(rows).to_csv(out/'all_draw_summaries.csv.gz',index=False)
    (out/'draw_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    info=dict(draws=500,unit_years=len(annual),windows=len(windows),corridor_years=len(corridor),baseline_regression=errors,
        max_conservation_error=err,stage_order=h.ORDER,input_sizes=sizes,origin_seed_hashes=hashes,
        input_sha256={str(p.relative_to(u.ROOT)):u.s.sha(p) for p in [root/'annual_indicators.csv.gz',root/'analytical_trade_panel.csv.gz',root/'all_windows.csv.gz',u.OUT/'routes.csv']},
        code_sha256=u.s.sha(__file__),engine_sha256=u.s.sha(h.__file__),
        interpretation='Uncalibrated conditional joint sensitivity, not confidence intervals. Trade/origin support and matched route geometry fixed; unmatched routes bounded.',
        raw_mirror_reconstruction_each_draw=False,global_allocation_each_draw=False,manuscript_updated=False)
    (out/'run_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('Expanded corrected historical conditional sensitivity complete.',flush=True)
