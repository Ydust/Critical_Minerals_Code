"""Extract omitted nickel intermediates from the preserved complete BACI archive.

This does not modify the adopted model, infer metal content, or label BACI as an
independent mine-origin observation. All raw source archives remain untouched.
"""
from pathlib import Path
import hashlib
import io
import json
import time
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'external_sources/baci_hs07_202601'
OUT=ROOT/'external/baci_nickel_intermediates'
CODES=[260400,282540,750210,750110,750120,283324]
ADDED={750110,750120,283324}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((SOURCE/'source_manifest.json').read_text())
    started=time.perf_counter()
    archive=SOURCE/'BACI_HS07_V202601.zip'
    with archive.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==manifest['archive']['sha256']
    frames=[]
    checked=[]
    with zipfile.ZipFile(archive) as z:
        products=pd.read_csv(io.BytesIO(z.read('product_codes_HS07_V202601.csv')))
        countries=pd.read_csv(io.BytesIO(z.read('country_codes_V202601.csv')))
        descriptions=products[products.code.isin(CODES)].copy()
        assert len(descriptions)==len(CODES)
        descriptions['scope_status']=descriptions.code.map(lambda k:'omitted_candidate' if k in ADDED else 'released_product')
        descriptions.to_csv(OUT/'nickel_product_scope.csv',index=False)
        # One source member at a time, no full raw archive extraction on disk.
        for entry in manifest['annual_members']:
            name=entry['filename']
            raw=z.read(name)
            member_sha=hashlib.sha256(raw).hexdigest()
            assert member_sha==entry['decompressed_sha256'],name
            selected=[]
            with io.BytesIO(raw) as stream:
                for part in pd.read_csv(stream,chunksize=500000):
                    selected.append(part[part.k.isin(CODES)].copy())
            frame=pd.concat(selected,ignore_index=True)
            assert not frame.duplicated(['t','i','j','k']).any()
            assert frame.v.gt(0).all()
            frames.append(frame)
            checked.append(dict(member=name,sha256=member_sha,selected_rows=len(frame),
                                omitted_rows=int(frame.k.isin(ADDED).sum())))
            print(name,len(frame),f'{time.perf_counter()-started:.1f}s',flush=True)
    rawtable=pd.concat(frames,ignore_index=True)
    assert set(rawtable.t)==set(range(2007,2025))
    rawtable.to_csv(OUT/'baci_nickel_six_products_2007_2024_raw.csv.gz',index=False)
    iso=countries.set_index('country_code').country_iso3
    data=rawtable.rename(columns={'t':'year','i':'exporter_code','j':'importer_code','k':'hs07_code','v':'value_thousand_usd','q':'quantity_tonnes'}).copy()
    data['exporter_iso']=data.exporter_code.map(iso)
    data['importer_iso']=data.importer_code.map(iso)
    assert data[['exporter_iso','importer_iso']].notna().all().all()
    data['trade_value_usd']=data.value_thousand_usd*1000
    data['reported_product_mass_kg']=data.quantity_tonnes*1000
    data['scope_status']=data.hs07_code.map(lambda k:'omitted_candidate' if k in ADDED else 'released_product')
    data['source_version']='BACI_HS07_V202601'
    data.to_csv(OUT/'baci_nickel_six_products_2007_2024_prepared.csv.gz',index=False)
    annual=data.groupby(['year','hs07_code','scope_status'],as_index=False).agg(
        bilateral_records=('trade_value_usd','size'),trade_value_usd=('trade_value_usd','sum'),
        product_mass_kg=('reported_product_mass_kg',lambda s:s.sum(min_count=1)),
        missing_mass_records=('reported_product_mass_kg',lambda s:s.isna().sum()))
    annual.to_csv(OUT/'global_product_coverage_by_year.csv',index=False)
    fin=data[data.importer_iso.eq('FIN')].copy()
    fin['russian_partner_value_usd']=fin.trade_value_usd.where(fin.exporter_iso.eq('RUS'),0)
    summary=fin.groupby(['year','hs07_code'],as_index=False).agg(total_usd=('trade_value_usd','sum'),
                  russian_partner_usd=('russian_partner_value_usd','sum'))
    summary['russian_partner_share']=summary.russian_partner_usd/summary.total_usd
    summary.to_csv(OUT/'finland_product_composition.csv',index=False)
    check=dict(source_url=manifest['source_url'],source_version=manifest['version'],license=manifest['license'],
        archive_sha256=digest,source_archive_unchanged=True,years=sorted(map(int,data.year.unique())),
        records=len(data),omitted_records=int(data.hs07_code.isin(ADDED).sum()),
        historical_2012_2024_omitted_records=int((data.year.ge(2012)&data.hs07_code.isin(ADDED)).sum()),
        selected_products=CODES,omitted_products=sorted(ADDED),checked_members=checked,
        missing_country_mappings=0,duplicate_keys=0,
        units=dict(raw_value='1000 USD',prepared_value='USD',raw_quantity='metric tonnes',prepared_quantity='kg of product NOT kg of nickel'),
        new_mine_origin_observations=False,manuscript_model_updated=False,
        purpose='Product-completeness audit and candidate inputs for matched reruns; no fitted stage weights',
        seconds=time.perf_counter()-started)
    (OUT/'extraction_manifest.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in check.items() if k!='checked_members'},indent=2))


if __name__=='__main__':
    main()
