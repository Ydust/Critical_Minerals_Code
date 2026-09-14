"""Raw-response reconstruction followed by frozen report-layer conditional draws."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,time,argparse
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/raw_conditional';OUT.mkdir(exist_ok=True)
allowed=[HERE,Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()];blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    raw=os.fsdecode(args[0])
    if raw.lower() in ('nul','\\\\.\\nul'):return
    p=Path(raw).resolve()
    if not any(p.is_relative_to(x) for x in allowed):blocked.append(str(p));raise PermissionError(str(p))
sys.addaudithook(guard)
import numpy as np
import pandas as pd
def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'code'/f'{name}.py');m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify_manifest():
    for x in json.loads((HERE/'RAW_CONDITIONAL_INPUT_MANIFEST.json').read_text()):assert sha(HERE/x['file'])==x['sha256'],x['file']
def compare(a,e,keys,ignore=()):
    a=a.drop(columns=list(ignore),errors='ignore').set_index(keys).sort_index();e=e.drop(columns=list(ignore),errors='ignore').set_index(keys).sort_index()
    assert a.index.is_unique and e.index.is_unique and a.index.equals(e.index)
    assert set(a.columns)==set(e.columns)
    errors={};mismatches={}
    for c in e:
        x,y=a[c],e[c]
        if pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y):
            mismatches[c]=int(x.isna().ne(y.isna()).sum());v=(abs(x-y)/np.maximum(abs(y),1)).max();errors[c]=float(v) if pd.notna(v) else 0.
        else:mismatches[c]=int(x.fillna('').astype(str).ne(y.fillna('').astype(str)).sum())
    return dict(accepted=not any(mismatches.values()) and max(errors.values(),default=0)<1e-10,rows=len(a),errors=errors,mismatches=mismatches)
class Sources:
    def __init__(self,paths):self.paths=paths
    def glob(self,pattern):assert pattern=='*.json';return self.paths
def prepare():
    verify_manifest();b=load('build_model_inputs');b.TRADE_RAW=HERE/'inputs/raw'
    mapping=pd.read_csv(HERE/'inputs/mapping.csv',dtype={'CmdCode':str})
    mapping=pd.concat([mapping,pd.DataFrame([dict(CmdCode='280461',metal='Silicon',stage='metal')])],ignore_index=True)
    checks=[];records=[];reporters=set();folder=OUT/'observations';folder.mkdir(exist_ok=True)
    for metal,d in mapping.groupby('metal',sort=True):
        mm={x.CmdCode:(x.metal,x.stage,x.CmdCode) for x in d.itertuples()};b.load_mineral_map=lambda:mm
        b.RAW=Sources([HERE/f'inputs/{"raw_conditional" if code=="280461" else "raw"}/{code}_{year}.json' for code in mm for year in range(2012,2025)])
        obs,stats=b.read_raw_observations();name=metal.replace('/','_');p=folder/f'{name}.csv.gz'
        obs.to_csv(p,index=False,compression={'method':'gzip','compresslevel':1})
        actual=pd.read_csv(p,dtype={'CmdCode':str});reporters.update(actual.ReporterCode.unique())
        expected=pd.read_csv(HERE/f'expected/raw_conditional/observations/{name}.csv.gz',dtype={'CmdCode':str})
        # HS text labels are presentation metadata; all raw values, roles, keys and flags are compared.
        check=compare(actual,expected,['CmdCode','exporter_iso','importer_iso','RefYear','reporter_role'],ignore=['hs_label'])
        check['metal']=metal;checks.append(check);assert check['accepted'],check
        records.append(dict(file=p.relative_to(HERE).as_posix(),sha256=sha(p)))
        print(f'Raw observations rebuilt and checked: {metal}, {len(actual)}',flush=True)
    report=dict(accepted=all(c['accepted'] for c in checks),raw_files=663,checks=checks,files=records,reporter_codes=sorted(int(x) for x in reporters),blocked_external_reads=blocked,manifest_sha256=sha(HERE/'RAW_CONDITIONAL_INPUT_MANIFEST.json'),label_policy='hs_label uses HS code; excluded only from metadata equality, not numerical/role/flag checks')
    (OUT/'OBSERVATION_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
def corridor(trade,h):
    trade=trade.copy();trade['quality']=trade.data_quality_weight.fillna(.5).clip(0,1);s=trade.observation_sigma_log;trade['sigma']=s.where(np.isfinite(s)&s.gt(0),.75)
    c=trade.groupby(h.KEY+['exporter_iso'],as_index=False).agg(value=('reconstructed_value_usd','sum'),k=('CmdCode','size'),quality=('quality','mean'),sigma=('sigma','mean'));c['draw_sigma']=0.;return c
def run(start,end):
    verify_manifest();qa=json.loads((OUT/'OBSERVATION_REPRODUCTION_QA.json').read_text());assert qa['accepted'] and qa['manifest_sha256']==sha(HERE/'RAW_CONDITIONAL_INPUT_MANIFEST.json')
    for rec in qa['files']:assert sha(HERE/rec['file'])==rec['sha256']
    assert json.loads((HERE/'outputs/historical_rebuilt_seed/HISTORICAL_REPRODUCTION_QA.json').read_text())['accepted']
    b=load('build_model_inputs');r=load('reconstruct_dynamic_panel');h=load('conditional_engine');h.ORDER=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    annual=pd.read_csv(HERE/'outputs/historical_rebuilt_seed/annual.csv.gz');windows=pd.read_csv(HERE/'outputs/historical_rebuilt_seed/windows.csv.gz');seed=pd.read_csv(HERE/'outputs/seed/historical_production_seed_rebuilt.csv.gz');routes=pd.read_csv(HERE/'inputs/historical/routes.csv')
    reference=pd.read_csv(HERE/'outputs/historical_trade_rebuilt.csv.gz',dtype={'CmdCode':str});key=['CmdCode','exporter_iso','importer_iso','year'];ref=reference.set_index(key).sort_index()
    sources={Path(x['file']).name[:-7]:pd.read_csv(HERE/x['file'],dtype={'CmdCode':str}) for x in qa['files']};reporter_index={v:i for i,v in enumerate(qa['reporter_codes'])}
    blocks,pair,sizes,_=h.prepare(annual,corridor(reference,h),seed,routes,windows);base,_=h.recompute(blocks,len(annual));assert np.max(np.abs(base-annual[h.FIELDS].to_numpy())/np.maximum(abs(annual[h.FIELDS].to_numpy()),1))<1e-10
    for draw in range(start,end+1):
      for rho in [0.,.8]:
        tick=time.monotonic();dest=OUT/f'rho_{rho:.1f}_draw_{draw:03d}';dest.mkdir(exist_ok=True)
        assert not (dest/'REPRODUCTION_QA.json').exists(),'No automatic skip or overwrite of completed draws'
        shared=h.ar_errors(np.random.default_rng(202609092000+draw),len(reporter_index),rho);panels=[]
        for name,original in sources.items():
            obs=original.copy();code=int(hashlib.sha256(name.encode()).hexdigest()[:8],16);rng=np.random.default_rng(202609092000+draw+code)
            lane=pd.MultiIndex.from_frame(obs[['CmdCode','exporter_iso','importer_iso','reporter_role']]).factorize(sort=True)[0];residual=h.ar_errors(rng,int(lane.max()+1),rho);year=obs.RefYear.to_numpy()-2012
            z=np.sqrt(.5)*shared[obs.ReporterCode.map(reporter_index).to_numpy(),year]+np.sqrt(.5)*residual[lane,year];obs.value_usd*=np.exp(.1*z)
            mirror=b.reconcile_mirrors(obs);folder=dest/name;folder.mkdir(exist_ok=True);r.BASELINE=folder/'baseline.csv.gz';mirror[mirror.model_eligible].to_csv(r.BASELINE,index=False,compression={'method':'gzip','compresslevel':1})
            panel=r.reconstruct_panel(r.prepare_observed());assert panel.year.between(2012,2024).all() and panel.next_observed_year.le(2024).all()
            panel.to_csv(folder/'reconstructed.csv.gz',index=False,compression={'method':'gzip','compresslevel':1});panels.append(panel)
        trade=pd.concat(panels,ignore_index=True);named=trade.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&trade.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False);trade=trade[named&trade.metal.isin(annual.metal.unique())].copy();indexed=trade.set_index(key).sort_index();assert indexed.index.is_unique and indexed.index.equals(ref.index)
        changed={c:int((abs(indexed[c]-ref[c])>1e-12).sum()) for c in ['reconstructed_value_usd','data_quality_weight','observation_sigma_log']}
        blocks,pair,sizes,_=h.prepare(annual,corridor(trade,h),seed,routes,windows);rng=np.random.default_rng(202609093000+draw)
        inp=dict(supplier=np.zeros((sizes['supplier'],13)),lane=np.zeros((sizes['lane'],13)),producer=h.ar_errors(rng,sizes['producer'],rho),weights={k:float(np.clip(w+rng.uniform(-.2,.2),0,1)) for k,w in h.WEIGHTS.items()})
        matrix,err=h.recompute(blocks,len(annual),inp);assert err<1e-10 and np.isfinite(matrix).all();frame=annual[h.KEY].copy()
        for j,c in enumerate(h.FIELDS):frame[c]=matrix[:,j]
        frame.to_csv(dest/'annual_indicators.csv.gz',index=False);cats=h.classify(matrix,pair);summary=pd.DataFrame(h.summarize(cats,list(h.selections(windows)),draw,rho));summary.to_csv(dest/'window_summaries.csv',index=False)
        # Comparison-only expected data are opened after both outputs are saved.
        expected=HERE/'expected/raw_conditional'/dest.name;ea=pd.read_csv(expected/'annual_indicators.csv.gz');es=pd.read_csv(expected/'window_summaries.csv');meta=json.loads((expected/'complete.json').read_text())
        ac=compare(frame,ea,h.KEY)
        for f in [summary,es]:f['end_year']=f.end_year.astype(str)
        sc=compare(summary,es,['draw','temporal_rho','window_family','end_year','metal'])
        counts_exact=all(sc['errors'][c]==0 for c in ['eligible_count','apparent_count'])
        report=dict(accepted=ac['accepted'] and sc['accepted'] and counts_exact and inp['weights']==meta['weights'] and changed==meta['changed_product_records'],draw=draw,rho=rho,annual=ac,summary=sc,counts_exact=counts_exact,weights=inp['weights'],weights_exact=inp['weights']==meta['weights'],changed_product_records=changed,changed_counts_exact=changed==meta['changed_product_records'],max_conservation_error=err,elapsed_seconds=time.monotonic()-tick,blocked_external_reads=blocked,input_manifest_sha256=sha(HERE/'RAW_CONDITIONAL_INPUT_MANIFEST.json'))
        (dest/'REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(f'Draw {draw}, rho={rho}: accepted={report["accepted"]}, {report["elapsed_seconds"]:.1f}s',flush=True);assert report['accepted'],report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--start',type=int,default=1);p.add_argument('--end',type=int,default=1);a=p.parse_args()
    if a.prepare:prepare()
    else:assert 1<=a.start<=a.end<=20;run(a.start,a.end)
