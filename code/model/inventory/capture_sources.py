from pathlib import Path
from datetime import datetime,timezone
import urllib.request
import hashlib
import json

HERE=Path(__file__).resolve().parent
SOURCES={
 'revenue_2018':'https://ar2018.nornickel.com/en/financial-overview/financial-performance/revenue.html',
 'statements_2018':'https://ar2018.nornickel.com/en/financial-overview/ifrs/statements.html',
 'cost_of_sales_2018':'https://ar2018.nornickel.com/en/financial-overview/financial-performance/cost-of-implementation.html',
}
def main():
 out=HERE/'external';out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for name,url in SOURCES.items():
  p=out/(name+'.html')
  if p.exists(): raise RuntimeError('Do not overwrite archived sources')
  req=urllib.request.Request(url,headers={'User-Agent':'ResearchDataAudit/1.0'})
  with urllib.request.urlopen(req,timeout=45) as response:
   raw=response.read();resolved=response.url
  p.write_bytes(raw)
  row=dict(source_id=name,url=url,resolved_url=resolved,file=p.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),retrieved_utc=datetime.now(timezone.utc).isoformat())
  rows.append(row);print(json.dumps(row),flush=True)
 (out/'source_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
if __name__=='__main__': main()
