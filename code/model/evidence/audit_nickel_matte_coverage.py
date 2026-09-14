"""Acquire a narrowly scoped public Comtrade preview and audit omitted nickel matte.

The Finland-reported import records share the manuscript's trade source family;
they are a product-coverage audit, not independent mine-origin validation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'external/nickel_matte_coverage'
PACKAGE=ROOT.parents[1]/'manuscript_package_20260907_final'
ENDPOINT='https://comtradeapi.un.org/public/v1/preview/C/A/HS'


def retrieve(year):
    params=dict(period=year,reporterCode=246,cmdCode='750110',flowCode='M',
                partner2Code=0,customsCode='C00',motCode=0,maxRecords=500,
                includeDesc='true')
    url=ENDPOINT+'?'+urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchDataAudit/1.0'}),timeout=45) as response:
        raw=response.read()
    data=json.loads(raw)
    assert isinstance(data.get('data'),list), data
    assert len(data['data'])<500,'Preview ceiling reached; cannot treat as complete'
    path=OUT/f'finland_imports_750110_{year}.json'
    path.write_bytes(raw)
    return dict(year=year,url=url,file=path.name,sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),returned_rows=len(data['data']),
                retrieved_utc=datetime.now(timezone.utc).isoformat())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--download',action='store_true')
    ap.add_argument('--first-year',type=int,default=2012)
    ap.add_argument('--last-year',type=int,default=2024)
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'source_manifest.json'
    records=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    if args.download:
        for year in range(args.first_year,args.last_year+1):
            if any(r['year']==year for r in records):
                continue
            record=retrieve(year)
            records.append(record)
            manifest_path.write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(record,flush=True)
            time.sleep(2)
    parts=[]
    for record in records:
        raw=(OUT/record['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
        part=pd.DataFrame(json.loads(raw)['data'])
        if part.empty:
            continue
        assert (part.refYear==record['year']).all()
        assert (part.reporterCode==246).all() and (part.flowCode=='M').all()
        assert (part.cmdCode.astype(str)=='750110').all()
        assert (part.partner2Code==0).all() and (part.customsCode=='C00').all() and (part.motCode==0).all()
        part['source_url']=record['url']
        parts.append(part)
    data=pd.concat(parts,ignore_index=True)
    assert not data.duplicated(['refYear','partnerCode']).any()
    data.to_csv(OUT/'finland_nickel_matte_imports_reported.csv',index=False)
    with zipfile.ZipFile(PACKAGE/'Reproduction_Code_and_Derived_Inputs.zip') as z:
        member='figure_source_tables/longitudinal_2012_2024/input_snapshot/dynamic_reconstruction_panel.csv.gz'
        raw=z.read(member)
    trade=pd.read_csv(io.BytesIO(raw),compression='gzip',usecols=['CmdCode','metal','stage','year','importer_iso','exporter_iso','reconstructed_value_usd'])
    nickel=trade[trade.metal.eq('Nickel') & trade.year.between(2012,2024)]
    assert '750110' not in set(nickel.CmdCode.astype(str))
    scope=nickel[nickel.importer_iso.eq('FIN')].groupby('year').reconstructed_value_usd.sum()
    rows=[]
    for year,d in data.groupby('refYear'):
        world=d[d.partnerCode.eq(0)]
        rus=d[d.partnerCode.eq(643)]
        assert len(world)==1 and len(rus)<=1
        worldvalue=float(world.primaryValue.iloc[0])
        # Absence denotes zero contribution in this returned partner partition,
        # not a finding of zero physical Russian-origin material.
        rusvalue=float(rus.primaryValue.iloc[0]) if len(rus) else 0.
        partners=float(d.loc[d.partnerCode.ne(0),'primaryValue'].sum())
        assert abs(partners-worldvalue)/worldvalue<1e-6,'Partner partition does not reconcile to World'
        rows.append(dict(year=int(year),reported_matte_world_import_usd=worldvalue,
            reported_matte_russia_import_usd=rusvalue,russia_share_reported_matte_import=rusvalue/worldvalue,
            russia_partner_row_present=bool(len(rus)),
            world_isReported=bool(world.isReported.iloc[0]),world_isAggregate=bool(world.isAggregate.iloc[0]),
            partner_sum_usd=partners,partner_world_relative_gap=(partners-worldvalue)/worldvalue,
            released_selected_nickel_import_usd=float(scope.get(year,float('nan'))),
            omitted_matte_to_selected_nickel_value_ratio=worldvalue/float(scope.get(year,float('nan'))),
            source_url=world.source_url.iloc[0]))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'nickel_matte_omission_summary.csv',index=False)
    audit=dict(released_nickel_codes=sorted(map(str,nickel.CmdCode.unique())),
               omitted_code='750110',label='Nickel mattes',
               source_role='same_source_family_product_coverage_audit_not_independent_origin_validation',
               released_input_member=member,released_input_sha256=hashlib.sha256(raw).hexdigest(),
               years=sorted(map(int,data.refYear.unique())),records=len(data),
               max_abs_partner_world_relative_gap=float(summary.partner_world_relative_gap.abs().max()),
               caveats=['Reporter imports versus released mirror-reconciled selected products have different constructions.',
                        'Values across process stages are not physical material balance and must not be added as nickel content.',
                        'No automatic insertion into the mine-origin recursion or allocation model.'],
               manuscript_changed=False)
    (OUT/'coverage_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(summary.drop(columns='source_url').to_string(index=False))


if __name__=='__main__':
    main()
