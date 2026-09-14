from pathlib import Path
import os, sys, json, ast, inspect
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CAP=ROOT/'research_process/portable_reproduction_20260910/capsule'
sys.path.insert(0,str(ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h
h.ORDER=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
OUT=HERE/'results/historical';OUT.mkdir(parents=True,exist_ok=True)
KEY=['importer_iso','metal','stage']

def inputs():
    annual=pd.read_csv(CAP/'outputs/historical_rebuilt_seed/annual.csv.gz')
    windows=pd.read_csv(CAP/'outputs/historical_rebuilt_seed/windows.csv.gz')
    trade=pd.read_csv(CAP/'outputs/historical_trade_rebuilt.csv.gz',dtype={'CmdCode':str})
    seed=pd.read_csv(CAP/'outputs/seed/historical_production_seed_rebuilt.csv.gz')
    trade['quality']=trade.data_quality_weight.fillna(.5).clip(0,1)
    trade['sigma']=trade.observation_sigma_log.where(np.isfinite(trade.observation_sigma_log)&trade.observation_sigma_log.gt(0),.75)
    trade['variance']=(trade.reconstructed_value_usd*trade.sigma)**2
    c=trade.groupby(h.KEY+['exporter_iso'],as_index=False).agg(value=('reconstructed_value_usd','sum'),k=('CmdCode','size'),quality=('quality','mean'),sigma=('sigma','mean'),variance=('variance','sum'))
    c['draw_sigma']=(np.sqrt(c.variance)/c.value).clip(.05,1.5)
    routes=pd.read_csv(CAP/'inputs/historical/routes.csv')
    blocks,pair,_,chokes=h.prepare(annual,c,seed,routes,windows)
    for b in blocks:
        t=c[(c.metal==b['metal'])&(c.year==b['year'])]
        b['countries']=np.array(sorted(set(t.exporter_iso)|set(t.importer_iso)|set(b['origin'])))
    return annual,windows,c,blocks,pair,chokes

def main():
    annual,windows,c,blocks,pair,chokes=inputs()
    cohort=pd.read_csv(ROOT/'research_process/unified_reanalysis_20260909/results/structural_integration/fixed_endpoint_cohort.csv')[KEY]
    keys=set(cohort.itertuples(index=False,name=None))
    src=inspect.getsource(h.recompute)
    assert src.count('            proposed = w*mix[good]+(1-w)*shares')==1
    src=src.replace('            proposed = w*mix[good]+(1-w)*shares','            capture(b,s,exporter_mix,value)\n            proposed = w*mix[good]+(1-w)*shares')
    summary=[];countries=[];sets={};errors={}
    reference_events=pd.read_csv(ROOT/'submission_package_verified_20260910/source_data/Figure3_events.csv')
    refkeys=set(reference_events[KEY+['year']].itertuples(index=False,name=None))
    for variant in ['reference','no_propagation']:
        cells=[];roles=[]
        def capture(b,s,pi,value):
            cs=b['countries'];orig=np.array(b['origin']);stage=s['stage'];year=b['year'];metal=b['metal']
            mask=np.array([(x,metal,stage) in keys for x in cs[s['imp']]])
            if not mask.any():return
            exp=cs[s['exp'][mask]];v=value[mask];amount=v[:,None]*pi[s['exp'][mask]]
            cells.append(dict(year=year,metal=metal,stage=stage,value_usd=float(v.sum()),mismatch_usd=float(amount[exp[:,None]!=orig[None,:]].sum())))
            direct=pd.Series(v,index=exp).groupby(level=0).sum();origin=pd.Series(amount.sum(axis=0),index=orig)
            for iso in sorted(set(direct.index)|set(origin.index)):
                roles.append(dict(year=year,iso=iso,direct_usd=float(direct.get(iso,0)),origin_usd=float(origin.get(iso,0))))
        ns=dict(h.__dict__);ns['capture']=capture
        code=src if variant=='reference' else src.replace('            mix[good[keep]] = proposed[keep]/norm[keep,None]','            pass  # No between-stage state propagation; initialization is retained.')
        exec(compile(code,str(__file__),'exec'),ns)
        matrix,cons=ns['recompute'](blocks,len(annual));assert cons<1e-10
        if variant=='reference':
            errors={f:float(np.nanmax(abs(matrix[:,j]-annual[f].to_numpy())/np.maximum(1,abs(annual[f].to_numpy())))) for j,f in enumerate(h.FIELDS)}
            assert max(errors.values())<1e-10,errors
        delta=matrix[pair[1]]-matrix[pair[0]]
        e=windows.copy();e['delta_origin_hhi']=delta[:,2];e['delta_attribution_uncertainty_index']=delta[:,3]
        mask=e.window_family.eq('adjacent')&e.year.ge(2021)&e.analysis_eligible&e.apparent_derisking&(delta[:,2]>=.025)
        events=e[mask].merge(cohort,on=KEY,validate='many_to_one')
        sets[variant]=set(events[KEY+['year']].itertuples(index=False,name=None))
        if variant=='reference':assert sets[variant]==refkeys
        folder=OUT/variant;folder.mkdir(exist_ok=True)
        events.to_csv(folder/'origin_events.csv',index=False)
        fixed=e.merge(reference_events[KEY+['year']],on=KEY+['year']).query("window_family == 'adjacent'")
        fixed.to_csv(folder/'fixed_reference_events.csv',index=False)
        pd.DataFrame(matrix,columns=h.FIELDS).to_csv(folder/'indicator_matrix.csv.gz',index=False)
        cells=pd.DataFrame(cells);cells.to_csv(folder/'mismatch_cells.csv',index=False)
        role=pd.DataFrame(roles).groupby(['year','iso'],as_index=False)[['direct_usd','origin_usd']].sum()
        role['difference_pp']=100*(role.origin_usd-role.direct_usd)/role.year.map(role.groupby('year').direct_usd.sum())
        role['variant']=variant;countries.append(role)
        for year,d in cells.groupby('year'):
            summary.append(dict(variant=variant,year=int(year),mismatch_pct=100*d.mismatch_usd.sum()/d.value_usd.sum(),redistribution_pp=.5*role[role.year.eq(year)].difference_pp.abs().sum(),origin_events=int(events.year.eq(year).sum()),conservation_error=cons))
        print(variant,'events',len(events),'conservation',cons,flush=True)
    pd.DataFrame(summary).to_csv(OUT/'structure_summary.csv',index=False)
    pd.concat(countries).to_csv(OUT/'country_comparison.csv',index=False)
    overlap=[dict(variant=k,events=len(s),intersection=len(s&refkeys),jaccard=len(s&refkeys)/len(s|refkeys)) for k,s in sets.items()]
    pd.DataFrame(overlap).to_csv(OUT/'event_overlap.csv',index=False)
    (OUT/'QA.json').write_text(json.dumps(dict(accepted=True,baseline_errors=errors,overlap=overlap,scope='No-propagation diagnostic, not physical-source validation'),indent=2))
    print(pd.DataFrame(summary).query('year == 2024').to_string(index=False),flush=True)
if __name__=='__main__':main()
