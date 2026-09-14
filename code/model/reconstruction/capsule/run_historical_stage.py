"""Portable historical recomputation from independently rebuilt Comtrade data."""
from pathlib import Path
import sys, os, json, hashlib, importlib.util, time, argparse
parser=argparse.ArgumentParser();parser.add_argument('--rebuilt-seed',action='store_true');ARGS=parser.parse_args()
HERE=Path(__file__).resolve().parent
OUT=HERE/('outputs/historical_rebuilt_seed' if ARGS.rebuilt_seed else 'outputs/historical'); OUT.mkdir(parents=True,exist_ok=True)
allowed=[HERE.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)): return
    raw=os.fsdecode(args[0])
    if raw.lower() in ('nul', '\\\\.\\nul'): return
    p=Path(raw).resolve()
    if not any(p.is_relative_to(r) for r in allowed):
        blocked.append(str(p)); raise PermissionError('Outside capsule/runtime: '+str(p))
sys.addaudithook(guard)
import numpy as np
import pandas as pd

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'code'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec); sys.modules[name]=mod
    spec.loader.exec_module(mod); return mod

def compare(actual, name, keys):
    expected=pd.read_csv(HERE/'expected'/name)
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any(), name
    a=actual.set_index(keys).sort_index(); e=expected.set_index(keys).sort_index()
    missing=e.index.difference(a.index); extra=a.index.difference(e.index)
    result=dict(rows=len(a),expected_rows=len(e),missing_keys=len(missing),extra_keys=len(extra),missing_columns=sorted(set(e.columns)-set(a.columns)),extra_columns=sorted(set(a.columns)-set(e.columns)),errors={},mismatches={})
    common=e.index.intersection(a.index)
    for c in e.columns.intersection(a.columns):
        x=a.loc[common,c]; y=e.loc[common,c]
        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
            x=pd.to_numeric(x,errors='coerce')
            mismatch=int((x.isna()!=y.isna()).sum())
            finite=x.notna()&y.notna()
            err=(abs(x[finite]-y[finite])/np.maximum(abs(y[finite]),1)).max()
            result['errors'][c]=float(err) if pd.notna(err) else 0.
        else:
            mismatch=int((x.fillna('').astype(str)!=y.fillna('').astype(str)).sum())
        if mismatch: result['mismatches'][c]=mismatch
    result['accepted']=not any([len(missing),len(extra),len(result['missing_columns']),len(result['extra_columns']),len(result['mismatches'])]) and all(v<1e-10 for v in result['errors'].values())
    return result

def main():
    start=time.monotonic()
    for item in json.loads((HERE/'HISTORICAL_INPUT_MANIFEST.json').read_text()):
        assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256'],item['file']
    infer=load('infer_mine_origin_flows'); lm=load('longitudinal_mrci'); cache=load('cached_routes')
    trade=pd.read_csv(HERE/'outputs/historical_trade_rebuilt.csv.gz',dtype={'CmdCode':str})
    seed_path=HERE/('outputs/seed/historical_production_seed_rebuilt.csv.gz' if ARGS.rebuilt_seed else 'inputs/historical/production_seed.csv.gz')
    if ARGS.rebuilt_seed:assert json.loads((HERE/'outputs/seed/SEED_REPRODUCTION_QA.json').read_text())['accepted']
    seed=pd.read_csv(seed_path)
    routes=HERE/'inputs/historical/routes.csv'
    key=['importer_iso','metal','stage','year']
    sample=trade[trade.metal.eq('Nickel')&trade.year.eq(2018)].groupby(key+['exporter_iso'],as_index=False).reconstructed_value_usd.sum()
    cache_test=cache.install(lm,routes,sample,lm.concentration_stats(sample,'exporter_iso','reconstructed_value_usd','direct'))
    production=infer.build_production_mix(seed)
    annuals=[]; origins_all=[]; profiles=[]
    for metal in sorted(production.metal.unique()):
        for year in range(2012,2025):
            p=production[production.year.eq(year)&production.metal.eq(metal)]
            world=dict(zip(p.inferred_mine_origin_iso,p.origin_share))
            if not world: continue
            mix={country:{country:1.} for country in world}
            current=trade[trade.year.eq(year)&trade.metal.eq(metal)]
            order=['ore','material','compound','metal'] if metal=='Nickel' else sorted(current.stage.unique(),key=lambda x:infer.STAGE_ORDER[x])
            for stage in order:
                t=current[current.stage.eq(stage)]
                if t.empty: continue
                flows,mix=infer.infer_year_stage(t,mix,world,stage)
                origins=flows.groupby(key+['inferred_mine_origin_iso'],as_index=False).attributed_value_usd.sum()
                origins_all.append(origins)
                stats=lm.concentration_stats(origins,'inferred_mine_origin_iso','attributed_value_usd','origin')
                uncertainty=lm.attribution_stats_from_corridors(lm._uncertainty_by_corridor(flows))
                annuals.append(lm.build_annual_risk_panel(t,stats,uncertainty,routes))
                f=flows.groupby(['year','metal','stage','exporter_iso','inferred_mine_origin_iso'],as_index=False).attributed_value_usd.sum()
                f['share']=f.attributed_value_usd/f.groupby(['year','metal','stage','exporter_iso']).attributed_value_usd.transform('sum')
                profiles.append(f)
        print('Attribution complete: '+metal,flush=True)
    annual=pd.concat(annuals,ignore_index=True)
    direct=trade[trade.metal.isin(annual.metal.unique())].groupby(key+['exporter_iso'],as_index=False).reconstructed_value_usd.sum()
    routed=lm.route_stats(direct,annual,routes)
    annual=annual.drop(columns=[c for c in routed if c not in lm.GROUP]).merge(routed,on=lm.GROUP,validate='one_to_one').sort_values(key).reset_index(drop=True)
    origins=pd.concat(origins_all,ignore_index=True); profile=pd.concat(profiles,ignore_index=True)
    cfg=lm.AnalysisConfig(bootstrap_reps=2000)
    design=pd.read_csv(HERE/'inputs/historical/window_definitions.csv')
    windows=pd.concat([lm.build_window_panel(annual,list(zip(d.baseline_year,d.end_year)),fam,cfg) for fam,d in design.groupby('window_family')],ignore_index=True)
    for name,frame in [('annual',annual),('importer_origins',origins),('exporter_profiles',profile),('windows',windows)]:
        frame.to_csv(OUT/(name+'.csv.gz'),index=False)
    print('Historical panels written; running 2,000-replicate frozen bootstrap',flush=True)
    summary=lm.summarize_windows(windows,pd.DataFrame(),.025,'local_full_evidence_endpoint','none',cfg)
    summary.to_csv(OUT/'summary.csv',index=False)
    # Numeric expected results are accessed only after all generated results exist.
    checks={}
    for frame,name,keys in [(annual,'SD1_historical_annual.csv.gz',key),(origins,'SD3_importer_origins.csv.gz',key+['inferred_mine_origin_iso']),(profile,'SD3_exporter_profiles.csv.gz',['year','metal','stage','exporter_iso','inferred_mine_origin_iso']),(windows,'SD2_historical_windows.csv.gz',['importer_iso','metal','stage','window_family','baseline_year','end_year']),(summary,'SD2_historical_summary.csv',['analysis_id'])]:
        checks[name]=compare(frame,name,keys)
    report=dict(stage='historical_attribution_routes_classification_bootstrap',accepted=all(c['accepted'] for c in checks.values()),checks=checks,route_cache_test=cache_test,blocked_external_reads=blocked,python=sys.version,numpy=np.__version__,pandas=pd.__version__,isolated_mode=bool(sys.flags.isolated),bootstrap_reps=cfg.bootstrap_reps,random_seed=cfg.random_seed,elapsed_seconds=time.monotonic()-start,full_pipeline_complete=False,limitations=['Fixed derived production seeds and static route inputs; not independent source validation','Bundled dependencies; not pristine installation','Allocation and conditional uncertainty not rerun in this stage'])
    report['independently_rebuilt_production_seed']=ARGS.rebuilt_seed
    report['seed_input_sha256']=hashlib.sha256(seed_path.read_bytes()).hexdigest()
    if ARGS.rebuilt_seed:report['limitations'][0]='Production seeds rebuilt from raw documents; static routes and mine-origin model remain assumptions, not observed shipment origins'
    (OUT/'HISTORICAL_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2),flush=True)
    assert report['accepted'],{k:v for k,v in checks.items() if not v['accepted']}

if __name__=='__main__': main()
