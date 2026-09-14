"""Rebuild every historical product from retained raw responses, not derived inputs."""
import json
import build_expanded_comtrade as bld
import run_unified as u
import numpy as np
import pandas as pd

OUT=u.HERE/'results/full_raw_reconstruction_audit'

def main():
    OUT.mkdir(exist_ok=True)
    reference_path=u.HERE/'results/historical_expanded_comtrade/reconstructed_all_endpoints.csv.gz'
    reference=pd.read_csv(reference_path,dtype={'CmdCode':str})
    manifest=json.loads((bld.EXTERNAL/'source_manifest.json').read_text())
    expanded={(str(m['code']),int(m['year'])):bld.EXTERNAL/m['file'] for m in manifest}
    for m in manifest: assert u.s.sha(bld.EXTERNAL/m['file'])==m['sha256']
    b,r=bld.load_frozen(); bld.OUT=OUT
    key=['CmdCode','exporter_iso','importer_iso','year']
    reports=[]
    for metal,expected in reference.groupby('metal',sort=True):
        name=metal.replace('/','_'); marker=OUT/name/'audit.json'
        mapping={code:(metal,d.stage.iloc[0],d.hs_label.iloc[0]) for code,d in expected.groupby('CmdCode')}
        paths=[expanded.get((code,year),u.ROOT/f'_raw/{code}_{year}.json') for code in mapping for year in range(2012,2025)]
        contract={str(p.relative_to(u.ROOT)):u.s.sha(p) for p in paths+[reference_path,u.Path(__file__),u.Path(b.__file__),u.Path(r.__file__)]}
        if marker.exists():
            report=json.loads(marker.read_text()); assert report['contract']==contract; reports.append(report); continue
        print(f'Raw reconstruction audit: {metal}, {len(paths)} product-years',flush=True)
        actual=bld.reconstruct(paths,mapping,b,r,name)
        a=actual.set_index(key).sort_index(); e=expected.set_index(key).sort_index()
        missing=e.index.difference(a.index); extra=a.index.difference(e.index)
        common=a.index.intersection(e.index)
        fields=['reconstructed_value_usd','data_quality_weight','observation_sigma_log']
        errors={c:float((abs(a.loc[common,c]-e.loc[common,c])/np.maximum(abs(e.loc[common,c]),1)).max()) for c in fields}
        accepted=not len(missing) and not len(extra) and max(errors.values())<1e-10
        report=dict(metal=metal,raw_product_year_files=len(paths),reference_rows=len(e),rebuilt_rows=len(a),missing_rows=len(missing),extra_rows=len(extra),
            errors=errors,accepted=accepted,contract=contract)
        marker.write_text(json.dumps(report,indent=2),encoding='utf-8'); reports.append(report)
        if not accepted:
            e.loc[missing].to_csv(marker.parent/'unreproduced_reference.csv.gz')
            a.loc[extra].to_csv(marker.parent/'extra_rebuilt.csv.gz')
        print(f'Raw reconstruction audit: {metal}, accepted={accepted}',flush=True)
    summary=dict(minerals=len(reports),raw_product_year_files=sum(x['raw_product_year_files'] for x in reports),
        rebuilt_rows=sum(x['rebuilt_rows'] for x in reports),accepted=all(x['accepted'] for x in reports),
        failures=[{k:v for k,v in x.items() if k!='contract'} for x in reports if not x['accepted']],
        max_regression_error=max(max(x['errors'].values()) for x in reports),
        distinction='Reference reconstruction audit only, not repeated raw-data uncertainty propagation')
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)
    assert summary['accepted'],summary

if __name__=='__main__': main()
