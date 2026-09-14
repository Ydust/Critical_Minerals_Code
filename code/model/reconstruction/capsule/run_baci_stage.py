"""Rebuild all accepted BACI trade corridors directly from the full archive."""
from pathlib import Path
import sys,os,json,hashlib,zipfile,io,time
HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs/baci';OUT.mkdir(parents=True,exist_ok=True)
allowed=[HERE.resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()];blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
    raw=os.fsdecode(args[0])
    if raw.lower() in ('nul','\\\\.\\nul'):return
    p=Path(raw).resolve()
    if not any(p.is_relative_to(r) for r in allowed):
        blocked.append(str(p));raise PermissionError('Outside capsule/runtime: '+str(p))
sys.addaudithook(guard)
import numpy as np
import pandas as pd
EDGE=['metal','stage','exporter_iso','importer_iso','year']
def main():
    start=time.monotonic()
    manifest=json.loads((HERE/'BACI_INPUT_MANIFEST.json').read_text())
    for item in manifest:
        with (HERE/item['file']).open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==item['sha256'],item['file']
    source=HERE/'inputs/baci'
    original=json.loads((source/'source_manifest.json').read_text())
    assert next(i['sha256'] for i in manifest if i['role']=='raw_archive')==original['archive']['sha256']
    scope=pd.read_csv(source/'fig5_hs12_to_hs07_scope.csv')[['hs07_code','metal','stage']].drop_duplicates()
    # Accepted additions, declared in the scope-correction method, not inferred from expected rows.
    scope=pd.concat([scope,pd.DataFrame([dict(hs07_code=750110,metal='Nickel',stage='material'),dict(hs07_code=750120,metal='Nickel',stage='material'),dict(hs07_code=283324,metal='Nickel',stage='compound')])],ignore_index=True)
    assert not scope.hs07_code.duplicated().any()
    scope.to_csv(OUT/'effective_scope.csv',index=False)
    pieces=[];audits=[]
    with zipfile.ZipFile(source/'BACI_HS07_V202601.zip') as z:
        countries=pd.read_csv(io.BytesIO(z.read('country_codes_V202601.csv')))
        assert not countries.country_code.duplicated().any()
        iso=countries.set_index('country_code').country_iso3
        products=pd.read_csv(io.BytesIO(z.read('product_codes_HS07_V202601.csv')))
        assert set(scope.hs07_code)<=set(products.code)
        for entry in original['annual_members']:
            raw=z.read(entry['filename'])
            assert hashlib.sha256(raw).hexdigest()==entry['decompressed_sha256']
            selected=[];scanned=0
            for part in pd.read_csv(io.BytesIO(raw),chunksize=500000,usecols=['t','i','j','k','v','q']):
                scanned+=len(part);selected.append(part[part.k.isin(scope.hs07_code)].copy())
            d=pd.concat(selected,ignore_index=True)
            assert not d.duplicated(['t','i','j','k']).any()
            d=d.merge(scope,left_on='k',right_on='hs07_code',validate='many_to_one')
            d['exporter_iso']=d.i.map(iso);d['importer_iso']=d.j.map(iso)
            d['reconstructed_value_usd']=d.v*1000.;d=d.rename(columns={'t':'year'})
            valid=d.metal.ne('Silicon')&d.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&d.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)&d.exporter_iso.ne(d.importer_iso)&d.reconstructed_value_usd.gt(0)
            d['exclusion_reason']=''
            d.loc[d.metal.eq('Silicon'),'exclusion_reason']='no_production_seed'
            d.loc[~(d.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&d.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)),'exclusion_reason']='non_country_or_unmapped'
            d.loc[d.exporter_iso.eq(d.importer_iso),'exclusion_reason']='same_country'
            d.loc[~d.reconstructed_value_usd.gt(0),'exclusion_reason']='nonpositive_or_missing_value'
            year=int(d.year.iloc[0]);assert d.year.eq(year).all()
            d.to_csv(OUT/f'selected_products_{year}.csv.gz',index=False)
            corridor=d.loc[valid].groupby(EDGE,as_index=False).reconstructed_value_usd.sum()
            pieces.append(corridor)
            audits.append(dict(year=year,raw_rows_scanned=scanned,selected_rows=len(d),retained_product_rows=int(valid.sum()),excluded_rows=int((~valid).sum()),corridors=len(corridor),source_member=entry['filename'],source_sha256=entry['decompressed_sha256']))
            print(f'BACI {year}: {scanned:,} raw rows, {len(corridor):,} corridors',flush=True)
    trade=pd.concat(pieces,ignore_index=True).sort_values(EDGE).reset_index(drop=True)
    assert set(trade.year)==set(range(2007,2025)) and trade.metal.nunique()==23
    assert not trade.duplicated(EDGE).any()
    trade.to_csv(OUT/'trade_corridors_rebuilt.csv.gz',index=False)
    pd.DataFrame(audits).to_csv(OUT/'annual_source_audit.csv',index=False)
    # Read expected numeric values only after rebuilt data have been written.
    expected=pd.read_csv(HERE/'expected/SD5_trade_corridors.csv.gz')
    assert not expected.duplicated(EDGE).any()
    a=trade.set_index(EDGE).sort_index();e=expected.set_index(EDGE).sort_index()
    missing=e.index.difference(a.index);extra=a.index.difference(e.index);common=a.index.intersection(e.index)
    error=float((abs(a.loc[common,'reconstructed_value_usd']-e.loc[common,'reconstructed_value_usd'])/np.maximum(abs(e.loc[common,'reconstructed_value_usd']),1)).max())
    accepted=not len(missing) and not len(extra) and error<1e-10 and set(a.columns)==set(e.columns)
    report=dict(stage='raw_baci_to_all_mineral_corridors',accepted=accepted,rows=len(a),expected_rows=len(e),missing_keys=len(missing),extra_keys=len(extra),max_normalized_error=error,raw_rows_scanned=sum(i['raw_rows_scanned'] for i in audits),selected_product_rows=sum(i['selected_rows'] for i in audits),excluded_product_rows=sum(i['excluded_rows'] for i in audits),years=18,minerals=23,scope_codes=len(scope),source_archive_sha256=original['archive']['sha256'],blocked_external_reads=blocked,isolated_mode=bool(sys.flags.isolated),python=sys.version,numpy=np.__version__,pandas=pd.__version__,elapsed_seconds=time.monotonic()-start,full_pipeline_complete=False,production_seed_rebuilt=False,allocation_rerun=False)
    (OUT/'BACI_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    assert accepted,report
if __name__=='__main__':main()
