"""Archive public operator evidence with bounded date and estimand metadata."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
SOURCES=[
    ('sherritt_q4_2019','https://sherritt.com/strong-operational-performance-drives-sherritts-q4-2019-results/','Q4 2019','Halifax entry port; trucking to Fort Saskatchewan during rail strike'),
    ('sherritt_q1_2022','https://sherritt.com/higher-nickel-and-cobalt-prices-drive-sherritts-strong-first-quarter-results/','Q1 2022','Mixed-sulphide feed delayed on rail from Halifax to Fort Saskatchewan'),
    ('sherritt_current_operations','https://sherritt.com/operations/metals/','Undated page accessed 2026-09-08','Moa mixed sulphides via Halifax and rail; current schematic, not an annual port-share panel'),
]

def get(item):
    key,url,period,claim=item
    out=HERE/'external'
    out.mkdir(parents=True,exist_ok=True)
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 research source verification'})
    with urllib.request.urlopen(request,timeout=45) as response:
        data=response.read(15_000_000)
        resolved=response.url
    assert b'Halifax' in data and len(data)<15_000_000
    path=out/(key+'.html')
    path.write_bytes(data)
    return dict(source_id=key,url=url,resolved_url=resolved,accessed_utc=datetime.now(timezone.utc).isoformat(),path=path.name,
                sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),evidence_period=period,claim=claim,
                boundary='Operator evidence for a named facility chain; national HS corridor share and intervening years not identified.')

if __name__=='__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        records=list(pool.map(get,SOURCES))
    (HERE/'external/source_manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
