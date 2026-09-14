"""Origin-unrestricted outer bounds on the frozen historical comparison set."""
from pathlib import Path
import hashlib
import json
import io
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT.parents[1] / 'manuscript_package_20260907_final/Reproduction_Code_and_Derived_Inputs.zip'
MEMBER = 'figure_source_tables/longitudinal_2012_2024/release_v3/mrci_all_windows_2012_2024.csv.gz'
OUT = ROOT / 'results'
TAU = 0.025
REPS = 2000
SEED = 20260908


def bounds(data, missing):
    c0 = data.top_chokepoint_share_previous.to_numpy(float)
    c1 = data.top_chokepoint_share.to_numpy(float)
    if missing:
        r0 = data.route_coverage_share_previous.to_numpy(float)
        r1 = data.route_coverage_share.to_numpy(float)
        low = c1 - np.minimum(1., c0 + 1. - r0)
        high = np.minimum(1., c1 + 1. - r1) - c0
    else:
        low = high = c1 - c0
    if np.any(low > high + 1e-12):
        raise AssertionError('Invalid exposure bounds')
    return low >= TAU, low <= -TAU, low, high


def summarise(data, label, missing, bootstrap=False):
    transfer, substantive, low, high = bounds(data, missing)
    value = data.transition_value_usd.to_numpy(float)
    denom = value.sum()
    lb, ub = value @ transfer / denom, value @ substantive / denom
    row = dict(sample=label,route_treatment='arbitrary_unmatched' if missing else 'released_static',
               unit_windows=len(data),countries=data.importer_iso.nunique(),denominator_usd=denom,
               transfer_lower_pct=100*lb,substantive_upper_pct=100*ub,
               transfer_minus_substantive_lower_pp=100*(lb-ub))
    if bootstrap:
        clusters = pd.DataFrame({'country':data.importer_iso.to_numpy(),'den':value,
                                 't':value*transfer,'s':value*substantive}).groupby('country',sort=True).sum().to_numpy()
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0,len(clusters),size=(REPS,len(clusters)))
        totals = clusters[idx].sum(axis=1)
        rates = totals[:,1:]/totals[:,0,None]
        draws = np.column_stack([rates, rates[:,0]-rates[:,1]])*100
        for i,key in enumerate(['transfer_lower_pct','substantive_upper_pct','transfer_minus_substantive_lower_pp']):
            row[key+'_bootstrap_p025'],row[key+'_bootstrap_p975']=np.quantile(draws[:,i],[.025,.975])
        pd.DataFrame(draws,columns=['transfer_lower_pct','substantive_upper_pct','lower_gap_pp']).to_csv(
            OUT / f'bootstrap_{label}_{row["route_treatment"]}.csv',index=False)
    return row


def main():
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        source_bytes = archive.read(MEMBER)
    full = pd.read_csv(io.BytesIO(source_bytes), compression='gzip')
    data = full[(full.window_family=='adjacent') & full.three_layer_evidence_eligible & full.apparent_derisking].copy()
    assert len(data)==5172
    keys=['importer_iso','metal','stage','baseline_year','end_year']
    assert not data.duplicated(keys).any()
    assert set(data.end_year)==set(range(2013,2025))
    value=data.transition_value_usd.to_numpy(float)
    transfer=data.classification.isin(['mine_origin_transfer','route_transfer','multiple_risk_transfer']).to_numpy()
    substantive=data.classification.eq('substantive_derisking').to_numpy()
    reference=[100*value@transfer/value.sum(),100*value@substantive/value.sum()]
    assert np.allclose(reference,[38.50462533951954,8.035707486710696],rtol=0,atol=1e-10)
    rows=[]
    for missing in [False,True]:
        rows.append(summarise(data,'pooled',missing,True))
        for year,part in data.groupby('end_year',sort=True):
            rows.append(summarise(part,str(year),missing,True))
        for mineral,part in data.groupby('metal',sort=True):
            rows.append(summarise(part,'mineral_'+mineral.replace('/','-'),missing))
        for cutoff in [.8,.9,.95,1.]:
            part=data[(data.route_coverage_share>=cutoff-1e-12)&(data.route_coverage_share_previous>=cutoff-1e-12)]
            if len(part):
                rows.append(summarise(part,f'coverage_{cutoff:.2f}',missing,True))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'origin_unrestricted_bounds.csv',index=False)
    detail=data[keys+['transition_value_usd','delta_direct_hhi','top_chokepoint_share_previous',
                     'top_chokepoint_share','route_coverage_share_previous','route_coverage_share']].copy()
    for missing,label in [(False,'released_static'),(True,'arbitrary_unmatched')]:
        t,s,lo,hi=bounds(data,missing)
        detail[label+'_transfer_sufficient']=t
        detail[label+'_substantive_necessary']=s
        detail[label+'_delta_c_lower']=lo
        detail[label+'_delta_c_upper']=hi
    detail.to_csv(OUT/'origin_unrestricted_unit_windows.csv.gz',index=False)
    # Regression checks on logical implications using the released classification.
    sure,possible,_,_=bounds(data,False)
    assert np.all(~sure | transfer)
    assert np.all(~substantive | possible)
    assert (summary.transfer_lower_pct>=0).all() and (summary.substantive_upper_pct<=100+1e-10).all()
    metadata=dict(input_archive=str(ARCHIVE),input_member=MEMBER,input_sha256=hashlib.sha256(source_bytes).hexdigest(),
                  source_release='20260908_focused_logic_revision',threshold=TAU,bootstrap_replicates=REPS,
                  seed=SEED,reference_transfer_pct=reference[0],reference_substantive_pct=reference[1],
                  independent_unit='demand country; all years, minerals and stages retained per sampled country',
                  scope='fixed released sample; arbitrary origins and attribution index; matched routes fixed',
                  intervals='percentile sampling intervals of bound endpoints; not full physical uncertainty',
                  checks_passed=True,full_pipeline_uncertainty_executed=False)
    (OUT/'identification_manifest.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(summary[summary['sample'].eq('pooled')].to_string(index=False))
    print('checks_passed=True')


if __name__=='__main__':
    main()
