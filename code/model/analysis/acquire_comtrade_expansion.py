"""Public Comtrade acquisition with explicit cap detection and subdivisions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import time
import urllib.parse
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'external/comtrade_expansion'
ENDPOINT='https://comtradeapi.un.org/public/v1/preview/C/A/HS'

def query(code,year,flow='M,X',reporters=None):
    suffix=reporters or 'all'
    if len(suffix)>90: suffix='group_'+hashlib.sha256(suffix.encode()).hexdigest()[:16]
    token=f'{code}_{year}_{flow.replace(",", "-")}_{suffix}'
    path=OUT/'queries'/f'{token}.json'
    meta=path.with_suffix('.metadata.json')
    params=dict(period=year,cmdCode=str(code),flowCode=flow,partner2Code=0,customsCode='C00',motCode=0,maxRecords=500,includeDesc='true')
    if reporters is not None: params['reporterCode']=reporters
    url=ENDPOINT+'?'+urllib.parse.urlencode(params)
    if path.exists():
        raw=path.read_bytes()
        record=json.loads(meta.read_text())
        assert hashlib.sha256(raw).hexdigest()==record['sha256']
    else:
        last=None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchDataAudit/1.0'}),timeout=45) as response:
                    raw=response.read()
                payload=json.loads(raw)
                assert isinstance(payload.get('data'),list) and not payload.get('error'),payload
                break
            except Exception as exc:
                last=exc
                time.sleep(2*(attempt+1))
        else: raise RuntimeError(f'{token}: {last}')
        path.write_bytes(raw)
        record=dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),retrieved_utc=datetime.now(timezone.utc).isoformat())
        meta.write_text(json.dumps(record,indent=2),encoding='utf-8')
        time.sleep(.5)
    data=json.loads(raw)['data']
    assert all(int(r['refYear'])==year and str(r['cmdCode'])==str(code) and r['flowCode'] in flow.split(',') for r in data)
    assert all(r.get('partner2Code')==0 and r.get('customsCode')=='C00' and r.get('motCode')==0 for r in data)
    if reporters is not None: assert all(int(r['reporterCode']) in set(map(int,reporters.split(','))) for r in data)
    record.update(query_file=str(path.relative_to(OUT)),rows=len(data),ceiling_reached=len(data)>=500)
    return data,record

def main():
    (OUT/'queries').mkdir(parents=True,exist_ok=True)
    (OUT/'complete_products').mkdir(exist_ok=True)
    refs=json.loads((ROOT/'_raw/_ref_Reporters.json').read_text(encoding='utf-8-sig'))
    refs=refs.get('results',refs) if isinstance(refs,dict) else refs
    codes=sorted({int(r['id']) if 'id' in r else int(r['reporterCode']) for r in refs if not r.get('isGroup')})
    records=[]
    for code in [750110,750120,283324]:
        for year in range(2012,2025):
            data,meta=query(code,year)
            trace=[meta]
            if meta['ceiling_reached']:
                parts=[]
                for flow in ['M','X']:
                    d,m=query(code,year,flow)
                    trace.append(m)
                    if m['ceiling_reached']:
                        split=[]
                        def collect(group):
                            q,r=query(code,year,flow,','.join(map(str,group)))
                            trace.append(r)
                            if r['ceiling_reached']:
                                assert len(group)>1,'Even one reporter capped: partner-level acquisition required'
                                mid=len(group)//2
                                return collect(group[:mid])+collect(group[mid:])
                            return q
                        for k in range(0,len(codes),100):
                            split.extend(collect(codes[k:k+100]))
                        d=split
                    parts.extend(d)
                data=parts
            keys=[(r['reporterCode'],r['partnerCode'],r['flowCode'],r['classificationCode']) for r in data]
            assert len(keys)==len(set(keys)),(code,year,'duplicate source rows')
            dest=OUT/'complete_products'/f'{code}_{year}.json'
            encoded=json.dumps(data,ensure_ascii=False,separators=(',',':')).encode('utf-8')
            if dest.exists(): assert dest.read_bytes()==encoded
            else: dest.write_bytes(encoded)
            records.append(dict(code=code,year=year,rows=len(data),reporters=len({r['reporterCode'] for r in data}),
                file=str(dest.relative_to(OUT)),sha256=hashlib.sha256(encoded).hexdigest(),queries=trace,
                admission='query complete below detected cap; source-vintage, harmonisation and mirror reconstruction checks pending'))
            (OUT/'source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(f'Acquired {code}/{year}: {len(data)} rows, {len(trace)} queries',flush=True)
    print('All 39 product-year queries acquired; not yet admitted into reconstructed history.',flush=True)

if __name__=='__main__': main()
