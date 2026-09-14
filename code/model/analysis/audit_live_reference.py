"""Verify public-query subdivision coverage against current official references."""
import json
import urllib.request
from datetime import datetime,timezone
import run_unified as u

if __name__=='__main__':
    folder=u.HERE/'external/comtrade_reference_audit'; folder.mkdir(parents=True,exist_ok=True)
    results={}
    for name in ['Reporters','partnerAreas']:
        url=f'https://comtradeapi.un.org/files/v1/app/reference/{name}.json'
        path=folder/f'{name}.json'
        if not path.exists():
            with urllib.request.urlopen(url,timeout=45) as r: raw=r.read()
            path.write_bytes(raw)
        new=json.loads(path.read_text(encoding='utf-8-sig'))
        old=json.loads((u.ROOT/f'_raw/_ref_{name}.json').read_text(encoding='utf-8-sig'))
        def rows(x): return x.get('results',[]) if isinstance(x,dict) else x
        old={int(r['id']):r for r in rows(old) if not r.get('isGroup')}
        new={int(r['id']):r for r in rows(new) if not r.get('isGroup')}
        iso='reporterCodeIsoAlpha3' if name=='Reporters' else 'PartnerCodeIsoAlpha3'
        changes=[dict(code=k,old=old[k].get(iso),new=new[k].get(iso)) for k in old.keys()&new.keys() if old[k].get(iso)!=new[k].get(iso)]
        results[name]=dict(url=url,current_sha256=u.s.sha(path),old_sha256=u.s.sha(u.ROOT/f'_raw/_ref_{name}.json'),
            new_nongroup_codes=sorted(new.keys()-old.keys()),old_only_codes=sorted(old.keys()-new.keys()),iso_changes=changes)
    results['checked_utc']=datetime.now(timezone.utc).isoformat()
    (folder/'audit.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))
    assert not results['Reporters']['new_nongroup_codes'],'Public split requests need additional reporter codes'
    assert not results['Reporters']['iso_changes'] and not results['partnerAreas']['iso_changes'],'Crosswalk changed: review required'
