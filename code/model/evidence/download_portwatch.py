"""Acquire public aggregate AIS-derived chokepoint data without local uploads."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.parse
import urllib.request
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'external/portwatch'
SERVICE='https://services9.arcgis.com/weJ1QsnbMYJlCHdG/ArcGIS/rest/services/Daily_Chokepoints_Data/FeatureServer/0'
ITEM='https://www.arcgis.com/sharing/rest/content/items/3da2b9ca97684916b75c4013f95d18ab'


def get(url,params,name):
    url=url+'?'+urllib.parse.urlencode(dict(f='json',**params))
    with urllib.request.urlopen(url,timeout=30) as response:
        raw=response.read()
    data=json.loads(raw)
    if 'error' in data:
        raise RuntimeError(data['error'])
    (OUT/name).write_bytes(raw)
    return data,dict(url=url,file=name,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    for url,name in [(SERVICE,'layer_metadata.json'),(ITEM,'item_metadata.json')]:
        result,record=get(url,{},name)
        records.append(record)
        if name=='item_metadata.json':
            print(json.dumps({k:result.get(k) for k in ['title','owner','access','licenseInfo','description']},ensure_ascii=False),flush=True)
    names,record=get(SERVICE+'/query',dict(where='year >= 2019 AND year <= 2024',outFields='portid,portname',returnDistinctValues='true',returnGeometry='false'),'chokepoint_names.json')
    records.append(record)
    ports=pd.DataFrame([f['attributes'] for f in names['features']])
    print(ports.to_string(index=False),flush=True)
    # Retain the paper's eight chokepoints, identified by observed provider names.
    selected=ports[ports.portname.str.contains('Suez|Panama|Malacca|Dover|Gibraltar|Hormuz|Mandeb|Good Hope',case=False,regex=True)]
    if len(selected)!=8:
        raise RuntimeError(f'Expected 8 matched provider names, found {len(selected)}; inspect names before proceeding')
    ids=','.join("'"+x.replace("'","''")+"'" for x in selected.portid)
    where=f'year >= 2019 AND year <= 2024 AND portid IN ({ids})'
    count,record=get(SERVICE+'/query',dict(where=where,returnCountOnly='true'),'query_count.json')
    records.append(record)
    expected=int(count['count'])
    def page(offset):
        return get(SERVICE+'/query',dict(where=where,outFields='*',orderByFields='ObjectId',
                                         returnGeometry='false',resultOffset=offset,resultRecordCount=1000),
                   f'page_{offset:06d}.json')
    parts=[]
    with ThreadPoolExecutor(max_workers=4) as executor:
        for data,record in executor.map(page,range(0,expected,1000)):
            parts.extend(x['attributes'] for x in data['features'])
            records.append(record)
            print(f'received {len(parts)}/{expected}',flush=True)
    frame=pd.DataFrame(parts)
    assert len(frame)==expected and not frame.duplicated(['portid','date']).any()
    assert set(frame.year)==set(range(2019,2025))
    assert frame.ObjectId.nunique()==expected
    assert frame['n_total'].notna().all() and (frame['n_total']>=0).all()
    assert (frame.n_cargo+frame.n_tanker==frame.n_total).all()
    frame.sort_values(['portid','year','month','day']).to_csv(OUT/'portwatch_daily_2019_2024.csv.gz',index=False)
    annual=frame.groupby(['portid','portname','year'],as_index=False).agg(days=('date','nunique'),
            total_transits=('n_total','sum'),cargo_transits=('n_cargo','sum'),dry_bulk_transits=('n_dry_bulk','sum'),
            container_transits=('n_container','sum'))
    annual['expected_days']=annual.year.map(lambda year:366 if year%4==0 else 365)
    assert (annual.days==annual.expected_days).all(), 'Incomplete years must not be silently compared'
    for key in ['total_transits','cargo_transits','dry_bulk_transits','container_transits']:
        annual[key+'_daily_mean']=annual[key]/annual.days
        annual[key+'_daily_mean_yoy_pct']=annual.groupby('portid')[key+'_daily_mean'].pct_change()*100
    annual.to_csv(OUT/'portwatch_annual_2019_2024.csv',index=False)
    manifest=dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),expected_rows=expected,actual_rows=len(frame),
                  coverage='2019-2024; eight manuscript chokepoints',files=records,checks_passed=True,
                  role='independent aggregate vessel-activity evidence, not observed mineral-specific routes',
                  mineral_fields_present=False,commodity_linkage_available=False,
                  manuscript_model_inputs_changed=False)
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(annual[annual.year==2024][['portname','total_transits_daily_mean_yoy_pct','dry_bulk_transits_daily_mean_yoy_pct']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
