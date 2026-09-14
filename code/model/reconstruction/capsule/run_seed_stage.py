"""Raw-source WMD and BGS reconstruction, preserving frozen methods."""
from pathlib import Path
import sys,os,json,hashlib,importlib.util,time,gzip
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/seed';OUT.mkdir(parents=True,exist_ok=True)
allowed=[HERE.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()];blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    raw=os.fsdecode(args[0])
    if raw.lower() in ('nul','\\\\.\\nul'):return
    p=Path(raw).resolve()
    if not any(p.is_relative_to(r) for r in allowed):
        blocked.append(str(p));raise PermissionError('Outside capsule/runtime: '+str(p))
sys.addaudithook(guard)
import pandas as pd
import numpy as np
import pdfplumber
import openpyxl
def load(name,folder='seed'):
    path=HERE/'code'/folder/(name+'.py')
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def coordinates(path,dest,config):
    with pdfplumber.open(path) as pdf,gzip.open(dest,'wt',encoding='utf-8') as f:
        for num in range(config.first_page,config.last_page+1):
            page=pdf.pages[num-1];words=page.extract_words(use_text_flow=False,keep_blank_chars=True)
            for i,w in enumerate(words):
                f.write(json.dumps(dict(record_type='text',page=num,item_index=i,x=float(w['x0']),y=float(page.height-w['bottom']),width=float(w['x1']-w['x0']),height=float(w['bottom']-w['top']),text=w['text']))+'\n')
def compare(actual,name):
    expected=pd.read_csv(HERE/'expected'/name)
    keys=['metal','year','mine_origin_iso','commodity_label']
    # Same commodity-level keys; neither cross-mineral units nor component records are pooled away.
    assert not actual.duplicated(keys).any() and not expected.duplicated(keys).any()
    a=actual.set_index(keys).sort_index();e=expected.set_index(keys).sort_index();common=a.index.intersection(e.index)
    missing=e.index.difference(a.index);extra=a.index.difference(e.index)
    errors={}; mismatches={}
    for c in e.columns.intersection(a.columns):
        x=a.loc[common,c];y=e.loc[common,c]
        if pd.api.types.is_numeric_dtype(y):
            x=pd.to_numeric(x,errors='coerce');bad=int((x.isna()!=y.isna()).sum());v=(abs(x-y)/np.maximum(abs(y),1)).max();errors[c]=float(v) if pd.notna(v) else 0.
        else:bad=int((x.fillna('').astype(str)!=y.fillna('').astype(str)).sum())
        if bad:mismatches[c]=bad
    result=dict(rows=len(a),expected_rows=len(e),missing_keys=len(missing),extra_keys=len(extra),missing_columns=sorted(set(e.columns)-set(a.columns)),extra_columns=sorted(set(a.columns)-set(e.columns)),errors=errors,mismatches=mismatches)
    result['values_accepted']=not len(missing) and not len(extra) and errors.get('production',1)<1e-10
    result['all_fields_accepted']=result['values_accepted'] and not mismatches and not result['missing_columns'] and not result['extra_columns'] and all(v<1e-10 for v in errors.values())
    return result
def main():
    start=time.monotonic()
    prior=OUT/'SEED_REPRODUCTION_QA.json'
    if prior.exists():
        previous=prior.read_bytes();digest=hashlib.sha256(previous).hexdigest()[:12]
        (OUT/('previous_attempt_'+digest+'.json')).write_bytes(previous)
    for item in json.loads((HERE/'SEED_INPUT_MANIFEST.json').read_text()):assert hashlib.sha256((HERE/item['file']).read_bytes()).hexdigest()==item['sha256'],item['file']
    parser=load('wmd_parser');load('bridge_shares');build=load('build_outputs');latest=load('build_wmd_mine_origin_seed');lm=load('longitudinal_mrci','');infer=load('infer_mine_origin_flows','')
    records=[];parser_audits=[]
    extraction=json.loads((OUT/'coordinate_extraction.json').read_text())
    assert extraction['script_sha256']==hashlib.sha256((HERE/'extract_seed_coordinates.mjs').read_bytes()).hexdigest()
    for item in extraction['records']:
        assert item['source_sha256']==hashlib.sha256((HERE/'inputs/seed'/('wmd_'+item['edition']+'.pdf')).read_bytes()).hexdigest()
        assert item['coordinate_sha256']==hashlib.sha256((OUT/item['coordinate_file']).read_bytes()).hexdigest()
    for edition in ['WMD2018','WMD2021']:
        cfg=parser.EDITIONS[edition];dest=OUT/(edition+'_coordinates.jsonl.gz')
        result=parser.parse_coordinates(dest,cfg)
        rows=build.with_dataset(result.records)
        pd.DataFrame(rows).to_csv(OUT/(edition+'_all_parsed.csv.gz'),index=False)
        parser_audits.extend(result.table_audit)
        records.extend(r for r in rows if r['commodity_label'] in parser.MODEL_METAL_BY_COMMODITY and (edition!='WMD2018' or r['year']<=2014))
        print(edition,'parsed',len(rows),'warnings',len(result.warnings),flush=True)
    historical=pd.DataFrame(records).reindex(columns=build.LONG_FIELDS)
    historical.to_csv(OUT/'wmd_2012_2019_mine_origin_seed.csv',index=False)
    latest.WMD_64=HERE/'inputs/seed/wmd_2026_ch6_4_country_by_mineral.xlsx'
    recent=latest.build_mine_origin_seed();recent.to_csv(OUT/'wmd_mine_origin_seed_2020_2024.csv',index=False)
    contract=json.loads((HERE/'inputs/seed/original_seed_input_contract.json').read_text())
    assert contract['country_reference'] is None
    assert hashlib.sha256((HERE/'inputs/raw/_ref_Reporters.json').read_bytes()).hexdigest()==contract['reporters_reference']['sha256']
    historical['input_seed_file']=Path(contract['wmd_seed'][0]['path']).name;recent['input_seed_file']=Path(contract['wmd_seed'][1]['path']).name
    mapped,coverage,gaps=lm.normalize_wmd_seed(pd.concat([historical,recent],ignore_index=True),HERE/'inputs/raw/_ref_Reporters.json',None,2012,2024)
    mapped.to_csv(OUT/'historical_production_seed_rebuilt.csv.gz',index=False)
    coverage.to_csv(OUT/'wmd_coverage.csv',index=False);gaps.to_csv(OUT/'wmd_mapping_gaps.csv',index=False)
    (OUT/'wmd_table_audit.json').write_text(json.dumps(parser_audits,indent=2),encoding='utf-8')
    print('Historical seed mapped:',len(mapped),flush=True)
    audit=load('audit_bgs_wmd_overlap_2008_2012')
    audit.BGS_PDF=HERE/'inputs/seed/BGS_WMP_2008_2012.pdf';audit.WMD_PDF=HERE/'inputs/seed/WMD2016_archived_official.pdf'
    audit.OUTPUT_DIR=OUT/'bgs_audit';audit.OUTPUT_DIR.mkdir(exist_ok=True)
    audit.WMD_COORDINATES=audit.OUTPUT_DIR/'WMD2016_coordinates.jsonl.gz'
    audit.WMD2018_SEED=OUT/'wmd_2012_2019_mine_origin_seed.csv'
    # Force raw PDF re-extraction on each run; do not let frozen cache existence skip it.
    cfg=parser.EditionConfig('WMD2016',(2010,2011,2012,2013,2014),94,137)
    coordinates(audit.WMD_PDF,audit.WMD_COORDINATES,cfg)
    gates_passed=audit.write_outputs()
    bgs=pd.read_csv(audit.output_path('bgs_calibrated_mine_production_2008_2012.csv.gz'))
    bgs=bgs[bgs.year.between(2008,2011)].copy()
    infer.REPORTERS=HERE/'inputs/raw/_ref_Reporters.json';infer.NATION_LIST=HERE/'inputs/seed/optional_country_reference_not_supplied.csv'
    assert not infer.NATION_LIST.exists()
    proxy=infer.map_wmd_countries(bgs.rename(columns={'source_country_labels':'mine_origin_country_wmd'}))
    bgs['mine_origin_iso']=proxy.mine_origin_iso;bgs['production']=pd.to_numeric(bgs.calibrated_production,errors='coerce');bgs['metal']=bgs.model_metal
    bgs['source_edition']='BGS2008-2012 calibrated to WMD2018 2012';bgs['source_dataset']='BGS World Mineral Production 2008-2012';bgs['source_url']='https://www2.bgs.ac.uk/mineralsuk/statistics/worldStatistics.html';bgs['source_confidence']=.5
    mapping_share=float(bgs.loc[bgs.mine_origin_iso.notna(),'production'].sum()/bgs.production.sum())
    common=['metal','year','mine_origin_iso','production','commodity_label','measurement_basis','source_edition','source_dataset','source_url','source_confidence']
    early=bgs[bgs.mine_origin_iso.notna()&bgs.production.gt(0)][common].copy();early['source_component']='calibrated_BGS'
    wmd=mapped[common].copy();wmd['source_component']='WMD'
    full=pd.concat([early,wmd],ignore_index=True)
    full=full.groupby([c for c in full if c!='production'],as_index=False,dropna=False,observed=True).production.sum().sort_values(['year','metal','mine_origin_iso']).reset_index(drop=True)
    full.to_csv(OUT/'allocation_production_seed_rebuilt.csv.gz',index=False)
    checks={name:compare(frame,name) for frame,name in [(mapped,'historical_production_seed.csv.gz'),(full,'SD5_production_seeds.csv.gz')]}
    report=dict(stage='raw_WMD_BGS_production_seeds',checks=checks,accepted=all(c['all_fields_accepted'] for c in checks.values()) and bool(gates_passed) and mapping_share>=.95,bgs_gates_passed=bool(gates_passed),bgs_mapping_production_share=mapping_share,blocked_external_reads=blocked,python=sys.version,pandas=pd.__version__,numpy=np.__version__,pdfplumber=pdfplumber.__version__,openpyxl=openpyxl.__version__,isolated_mode=bool(sys.flags.isolated),elapsed_seconds=time.monotonic()-start,full_pipeline_complete=False)
    (OUT/'SEED_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)
    assert report['accepted'],report
if __name__=='__main__':main()
