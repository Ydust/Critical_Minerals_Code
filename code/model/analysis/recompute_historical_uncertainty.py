"""Corrected-route, old-product-scope uncertainty; not expanded Comtrade."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[k]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h
import numpy as np
import pandas as pd

def main():
    out=HERE/'results/historical_corrected_routes_old_scope'
    out.mkdir(parents=True,exist_ok=True)
    path=ROOT/'research_process/nickel_route_extension_20260908/results/global_detector_audit/minerals_lanes_existing_support_geometric.csv'
    annual,corridor,seed,oldlanes,windows,hashes,records=h.load_inputs()
    lanes=pd.read_csv(path)
    blocks,pair,sizes,_=h.prepare(annual,corridor,seed,lanes,windows)
    matrix,err=h.recompute(blocks,len(annual))
    ref=pd.read_csv(ROOT/'research_process/nickel_route_extension_20260908/results/global_detector_audit/annual_detector_comparison.csv.gz')
    assert annual[h.KEY].equals(ref[h.KEY])
    assert np.allclose(matrix,ref[h.FIELDS],equal_nan=True,rtol=1e-12,atol=1e-12)
    cats=h.classify(matrix,pair)
    masks=list(h.selections(windows))
    rows=h.summarize(cats,masks,0,-1.)
    pd.DataFrame(rows).to_csv(out/'reference_summary.csv',index=False)
    # Country clusters preserve all repeated mineral/stage/windows within country.
    bootstrap=[]
    rng=np.random.default_rng(20260909)
    draw_records=[]
    for family,year,metal,m in masks:
        if metal!='all': continue
        eligible=m&cats['eligible']
        countries=sorted(windows.loc[eligible,'importer_iso'].unique())
        index={c:i for i,c in enumerate(countries)}
        selected=m&cats['apparent']
        idx=windows.loc[selected,'importer_iso'].map(index).to_numpy(dtype=int)
        v=cats['value'][selected]
        amounts=np.column_stack([np.bincount(idx,weights=v,minlength=len(countries)),
            np.bincount(idx,weights=v*cats['transfer'][selected],minlength=len(countries)),
            np.bincount(idx,weights=v*cats['substantive'][selected],minlength=len(countries))])
        freq=rng.multinomial(len(countries),np.full(len(countries),1/len(countries)),size=2000)
        totals=freq@amounts
        ratios=np.divide(totals[:,1:],totals[:,0,None],out=np.full((2000,2),np.nan),where=totals[:,0,None]>0)
        estimates=np.column_stack([ratios,ratios[:,0]-ratios[:,1]])
        for j,name in enumerate(['transfer','substantive','difference']):
            lo,hi=np.nanquantile(estimates[:,j],[.025,.975])
            bootstrap.append(dict(window_family=family,end_year=year,metric=name,country_clusters=len(countries),
                replicates=2000,undefined_denominator_draws=int(np.isnan(estimates[:,j]).sum()),lower=lo,upper=hi))
        draw_records.extend(dict(window_family=family,end_year=year,replicate=i+1,transfer=estimates[i,0],substantive=estimates[i,1],difference=estimates[i,2]) for i in range(2000))
    pd.DataFrame(bootstrap).to_csv(out/'country_cluster_intervals.csv',index=False)
    pd.DataFrame(draw_records).to_csv(out/'country_cluster_replicates.csv.gz',index=False)
    started=time.monotonic()
    params=[]
    for rho in [0.,.8]:
        for draw in range(1,251):
            inputs=h.draw_inputs(sizes,draw,rho)
            sample,error=h.recompute(blocks,len(annual),inputs)
            assert error<1e-10
            err=max(err,error)
            rows.extend(h.summarize(h.classify(sample,pair),masks,draw,rho))
            params.append(dict(draw=draw,rho=rho,stage_weights=inputs['weights']))
            if draw%25==0: print(f'Corrected historical sensitivity rho={rho} {draw}/250, elapsed={time.monotonic()-started:.1f}s',flush=True)
    results=pd.DataFrame(rows)
    results.to_csv(out/'conditional_draw_summary.csv.gz',index=False)
    (out/'draw_parameters.json').write_text(json.dumps(params,indent=2),encoding='utf-8')
    previous=pd.read_csv(h.OUT/'window_summary_all_draws.csv')
    key=['draw','temporal_rho','window_family','end_year','metal']
    previous.end_year=previous.end_year.astype(str)
    results.end_year=results.end_year.astype(str)
    comparison=results.merge(previous,on=key,suffixes=('_corrected','_old'),validate='one_to_one')
    comparison.to_csv(out/'paired_detector_uncertainty_comparison.csv.gz',index=False)
    manifest=dict(scope='Original Comtrade product basket; corrected existing route support only',unit_years=len(annual),windows=len(windows),
        conditional_draws=500,bootstrap_replicates_per_comparison=2000,maximum_conservation_error=err,
        route_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_hashes=hashes,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        engine_sha256=hashlib.sha256(Path(h.__file__).read_bytes()).hexdigest(),
        expanded_nickel_comtrade=False,raw_mirror_reconstruction_per_draw=False,allocation_per_draw=False,
        conditional_draws_are_confidence_intervals=False,route_incidence_fixed=True,unknown_routes_adversarial_bounds=True,
        manuscript_updated=False)
    (out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Corrected old-scope historical uncertainty complete.',flush=True)

if __name__=='__main__': main()
