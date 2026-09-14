from pathlib import Path
import os,sys,json,hashlib,importlib.util,argparse,time
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--draws-per-design',type=int,default=250);args=parser.parse_args();N=args.draws_per_design;assert 1<=N<=250
OUT=HERE/f'outputs/conditional_corridor_d{N:03d}';OUT.mkdir(exist_ok=True)
allowed=[HERE.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()];blocked=[]
def guard(event,items):
    if event!='open' or not items or not isinstance(items[0],(str,bytes,os.PathLike)):return
    raw=os.fsdecode(items[0])
    if raw.lower() in ('nul','\\\\.\\nul'):return
    p=Path(raw).resolve()
    if not any(p.is_relative_to(x) for x in allowed):blocked.append(str(p));raise PermissionError(str(p))
sys.addaudithook(guard)
import pandas as pd
import numpy as np
def main():
    start=time.monotonic()
    (OUT/'INPUT_MANIFEST.json').write_bytes((HERE/'CONDITIONAL_INPUT_MANIFEST.json').read_bytes())
    for item in json.loads((HERE/'CONDITIONAL_INPUT_MANIFEST.json').read_text()):assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256']
    qa=json.loads((HERE/'outputs/historical_rebuilt_seed/HISTORICAL_REPRODUCTION_QA.json').read_text());assert qa['accepted'] and qa['independently_rebuilt_production_seed']
    spec=importlib.util.spec_from_file_location('conditional_engine',HERE/'code/conditional_engine.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    annual=pd.read_csv(HERE/'outputs/historical_rebuilt_seed/annual.csv.gz');windows=pd.read_csv(HERE/'outputs/historical_rebuilt_seed/windows.csv.gz')
    trade=pd.read_csv(HERE/'outputs/historical_trade_rebuilt.csv.gz',dtype={'CmdCode':str});seed=pd.read_csv(HERE/'outputs/seed/historical_production_seed_rebuilt.csv.gz')
    trade['quality']=trade.data_quality_weight.fillna(.5).clip(0,1);s=trade.observation_sigma_log;trade['sigma']=s.where(np.isfinite(s)&s.gt(0),.75);trade['variance']=(trade.reconstructed_value_usd*trade.sigma)**2
    corridor=trade.groupby(h.KEY+['exporter_iso'],as_index=False).agg(value=('reconstructed_value_usd','sum'),k=('CmdCode','size'),quality=('quality','mean'),sigma=('sigma','mean'),variance=('variance','sum'))
    corridor['draw_sigma']=(np.sqrt(corridor.variance)/corridor.value).clip(.05,1.5)
    h.ORDER=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    blocks,pair,sizes,chokes=h.prepare(annual,corridor,seed,pd.read_csv(HERE/'inputs/historical/routes.csv'),windows)
    matrix,cons=h.recompute(blocks,len(annual));errors={c:float(np.nanmax(abs(matrix[:,j]-annual[c])/np.maximum(abs(annual[c]),1))) for j,c in enumerate(h.FIELDS)};assert max(errors.values())<1e-10
    cats=h.classify(matrix,pair);selected=windows.three_layer_evidence_eligible&windows.apparent_derisking
    assert np.array_equal(cats['apparent'],selected)
    assert np.array_equal(cats['transfer'],selected&windows.classification.isin(['mine_origin_transfer','route_transfer','multiple_risk_transfer']))
    assert np.array_equal(cats['substantive'],selected&windows.classification.eq('substantive_derisking'))
    masks=list(h.selections(windows));rows=h.summarize(cats,masks,0,-1.);parameters=[];timings=[]
    for rho in [0.,.8]:
        membership={k:np.zeros(len(windows),int) for k in ['eligible','apparent','transfer','substantive']}
        for draw in range(1,N+1):
            tick=time.monotonic();inp=h.draw_inputs(sizes,draw,rho);matrix,error=h.recompute(blocks,len(annual),inp);assert error<1e-10;cons=max(cons,error)
            c=h.classify(matrix,pair)
            for k in membership:membership[k]+=c[k]
            rows.extend(h.summarize(c,masks,draw,rho));parameters.append(dict(draw=draw,rho=rho,weights=inp['weights']));timings.append(time.monotonic()-tick)
            if draw%10==0 or draw==N:print(f'rho={rho} draw={draw}/{N}; elapsed={time.monotonic()-start:.1f}s',flush=True)
        counts=windows[['importer_iso','metal','stage','window_family','baseline_year','end_year']].copy()
        for k,n in membership.items():counts[k+'_draw_count']=n
        counts['draws']=N;counts.to_csv(OUT/f'membership_rho_{rho:.1f}.csv.gz',index=False)
    actual=pd.DataFrame(rows);actual.to_csv(OUT/'all_draw_summaries.csv.gz',index=False);(OUT/'draw_parameters.json').write_text(json.dumps(parameters,indent=2),encoding='utf-8')
    expected=pd.read_csv(HERE/'expected/corridor_500/all_draw_summaries.csv.gz');expected=expected[expected.draw.le(N)]
    keys=['draw','temporal_rho','window_family','end_year','metal']
    for d in [actual,expected]:d['end_year']=d.end_year.astype(str)
    a=actual.set_index(keys).sort_index();e=expected.set_index(keys).sort_index();assert a.index.equals(e.index)
    diff={c:float((abs(a[c]-e[c])/np.maximum(abs(e[c]),1)).max()) for c in a.columns}
    missingness=all(np.array_equal(a[c].isna(),e[c].isna()) for c in a.columns)
    refparams=json.loads((HERE/'expected/corridor_500/draw_parameters.json').read_text());refparams=[x for x in refparams if x['draw']<=N]
    accepted=missingness and max(diff.values())<1e-10 and parameters==refparams and all(diff[c]==0 for c in ['eligible_count','apparent_count'])
    report=dict(accepted=bool(accepted),draws=2*N,full_500_draw_run=N==250,summary_rows=len(a),baseline_errors=errors,comparison_errors=diff,draw_parameters_exact=parameters==refparams,max_conservation_error=cons,median_draw_seconds=float(np.median(timings)),elapsed_seconds=time.monotonic()-start,blocked_external_reads=blocked,independent_raw_trade_and_seed_inputs=True,interpretation='Uncalibrated conditional sensitivity, not confidence intervals or independent observations',raw_report_draws_rerun=False,allocation_draws_rerun=False,full_pipeline_complete=False)
    (OUT/'CONDITIONAL_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2));assert accepted,report
if __name__=='__main__':main()
