"""Public, unauthenticated completeness probe; no credential access."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import urllib.parse
import urllib.request

OUT=Path(__file__).resolve().parent/'external/comtrade_public_probe'

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    params=dict(period=2024,cmdCode='750110',flowCode='M,X',partner2Code=0,customsCode='C00',motCode=0,maxRecords=500,includeDesc='true')
    url='https://comtradeapi.un.org/public/v1/preview/C/A/HS?'+urllib.parse.urlencode(params)
    path=OUT/'global_750110_2024.json'
    if path.exists():
        raw=path.read_bytes()
    else:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchDataAudit/1.0'}),timeout=45) as response:
            raw=response.read()
        path.write_bytes(raw)
    data=json.loads(raw)
    rows=data.get('data',[])
    report=dict(url=url,sha256=hashlib.sha256(raw).hexdigest(),checked_utc=datetime.now(timezone.utc).isoformat(),
        metadata={k:v for k,v in data.items() if k!='data'},returned_rows=len(rows),
        reporter_count=len({r.get('reporterCode') for r in rows}),flow_codes=sorted({r.get('flowCode','') for r in rows}),
        ceiling_reached=len(rows)>=500,admitted_to_main_analysis=False)
    (OUT/'probe_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
