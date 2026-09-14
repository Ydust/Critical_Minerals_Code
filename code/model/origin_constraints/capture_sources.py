"""Archive dated public source documents for a scoped provenance ledger."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import urllib.request
import argparse

HERE=Path(__file__).resolve().parent
OUT=HERE/'external'
SOURCES={
    'harjavalta_2018_html':'https://ar2018.nornickel.com/en/business-overview/performance/finland.html',
    'harjavalta_2018_pdf':'https://ar2018.nornickel.com/pdf/ar/en/business-overview_performance_finland.pdf',
    'finland_usgs_2017_2018':'https://pubs.usgs.gov/myb/vol3/2017-18/myb3-2017-18-finland.pdf',
    'sherritt_2024_results':'https://sherritt.com/sherritt-reports-fourth-quarter-and-full-year-2024-results-strong-operational-performance-at-metals-and-power-provides-guidance-for-2025/',
    'census_ch75_2007':'https://www.census.gov/foreign-trade/schedules/b/2007/c75.html',
    'census_ch75_2018':'https://www.census.gov/foreign-trade/schedules/b/2018/c75.html',
    'census_ch75_2017':'https://www.census.gov/foreign-trade/schedules/b/2017/c75.html',
}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',default=','.join(SOURCES))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    previous=OUT/'source_manifest.json'
    rows=[]
    if previous.exists():
        old=json.loads(previous.read_text())
        rows=list(old)
    for name in args.sources.split(','):
        url=SOURCES[name]
        suffix='.pdf' if url.endswith('.pdf') else '.html'
        path=OUT/(name+suffix)
        if path.exists():
            record=next(r for r in rows if r['source_id']==name and r['status']=='archived')
            assert hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256']
            continue
        row=dict(source_id=name,url=url,retrieved_utc=datetime.now(timezone.utc).isoformat())
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'ResearchDataAudit/1.0'})
            with urllib.request.urlopen(req,timeout=45) as response:
                raw=response.read()
                row['resolved_url']=response.url
            if suffix=='.pdf':
                assert raw.startswith(b'%PDF')
            path.write_bytes(raw)
            row.update(status='archived',file=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        except Exception as exc:
            row.update(status='unavailable',error=str(exc))
        if any(r['source_id']==name for r in rows):
            attempts=OUT/(name+'_previous_attempts.json')
            history=json.loads(attempts.read_text()) if attempts.exists() else []
            history.extend(r for r in rows if r['source_id']==name)
            attempts.write_text(json.dumps(history,indent=2),encoding='utf-8')
            rows=[r for r in rows if r['source_id']!=name]
        rows.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'source_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
