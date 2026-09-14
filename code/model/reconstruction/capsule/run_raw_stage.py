from pathlib import Path
import sys,os,json,hashlib,importlib.util,time
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
OUT.mkdir(exist_ok=True)
# -I excludes user site packages and PYTHONPATH. This guard also prevents
# accidental access to the original project while imported methods execute.
runtime=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
allowed=[HERE.resolve(),*runtime]
blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    p=Path(os.fsdecode(args[0])).resolve()
    if str(p).lower() in ['nul','\\.\nul']:return
    if not any(p.is_relative_to(r) for r in allowed):
        blocked.append(str(p));raise PermissionError('Outside reproduction capsule/runtime: '+str(p))
sys.addaudithook(guard)
import numpy as np
import pandas as pd

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'code'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod;spec.loader.exec_module(mod);return mod

class RawFiles:
    def __init__(self,paths):self.paths=paths
    def glob(self,pattern):
        assert pattern=='*.json'
        return self.paths

def main():
    start=time.monotonic()
    manifest=json.loads((HERE/'INPUT_MANIFEST.json').read_text())
    for item in manifest:
        assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256'],item['file']
    b=load('build_model_inputs');r=load('reconstruct_dynamic_panel')
    b.TRADE_RAW=HERE/'inputs/raw'
    mapping=pd.read_csv(HERE/'inputs/mapping.csv',dtype={'CmdCode':str})
    rebuilt=[];audits=[]
    for metal,d in mapping.groupby('metal',sort=True):
        folder=OUT/metal.replace('/','_');folder.mkdir(exist_ok=True)
        mm={row.CmdCode:(row.metal,row.stage,row.CmdCode) for row in d.itertuples()}
        b.load_mineral_map=lambda:mm
        b.RAW=RawFiles([HERE/f'inputs/raw/{code}_{year}.json' for code in mm for year in range(2012,2025)])
        print('Reconstruct '+metal,flush=True)
        obs,stats=b.read_raw_observations()
        mirror=b.reconcile_mirrors(obs)
        baseline=mirror[mirror.model_eligible]
        baseline.to_csv(folder/'baseline.csv.gz',index=False)
        r.BASELINE=folder/'baseline.csv.gz'
        actual=r.reconstruct_panel(r.prepare_observed())
        stats.to_csv(folder/'raw_audit.csv',index=False)
        actual.to_csv(folder/'reconstructed.csv.gz',index=False)
        rebuilt.append(actual)
        audits.append({'metal':metal,'raw_files':len(mm)*13,'rebuilt_rows':len(actual)})
    panel=pd.concat(rebuilt,ignore_index=True)
    named=panel.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&panel.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
    panel[~named].to_csv(OUT/'excluded_non_country.csv.gz',index=False)
    actual=panel[named].copy()
    actual.to_csv(OUT/'historical_trade_rebuilt.csv.gz',index=False)
    # The expected data are first opened here, after writing the rebuilt panel.
    expected=pd.read_csv(HERE/'expected/SD1_historical_trade.csv.gz',dtype={'CmdCode':str})
    keys=['CmdCode','exporter_iso','importer_iso','year']
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    a=actual.set_index(keys).sort_index();e=expected.set_index(keys).sort_index()
    missing=e.index.difference(a.index);extra=a.index.difference(e.index);common=e.index.intersection(a.index)
    fields=['reconstructed_value_usd','data_quality_weight','observation_sigma_log']
    errors={c:float((abs(a.loc[common,c]-e.loc[common,c])/np.maximum(abs(e.loc[common,c]),1)).max()) for c in fields}
    accepted=not len(missing) and not len(extra) and all(v<1e-10 for v in errors.values())
    report={'stage':'raw_comtrade_to_supported_trade','accepted':accepted,'rebuilt_rows':len(actual),'expected_rows':len(expected),'missing_keys':len(missing),'extra_keys':len(extra),'normalised_errors':errors,'mineral_audits':audits,'blocked_external_reads':blocked,'python':sys.version,'executable':sys.executable,'isolated_mode':bool(sys.flags.isolated),'numpy':np.__version__,'pandas':pd.__version__,'elapsed_seconds':time.monotonic()-start,'full_pipeline_complete':False,'environment_scope':'isolated process and file access; bundled dependencies, not pristine dependency installation'}
    (OUT/'RAW_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='mineral_audits'},indent=2),flush=True)
    assert accepted,report

if __name__=='__main__':main()
