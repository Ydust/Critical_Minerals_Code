"""Conditional, shared-input sensitivity of all historical comparisons.

Recomputes direct concentration, recursive origins, attribution index, route
weights and sample screens from perturbed reconstructed corridors. This is NOT
raw-trade reconstruction, commodity-voyage validation, or an allocation rerun.
All new perturbation distributions are explicit sensitivity assumptions.
"""
from pathlib import Path
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import argparse
import hashlib
import io
import json
import time
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT.parents[1] / 'manuscript_package_20260907_final'
OUT = ROOT / 'results/historical_joint_sensitivity'
BASE = 'figure_source_tables/longitudinal_2012_2024/'
KEY = ['importer_iso', 'metal', 'stage', 'year']
ORDER = dict(ore=0, mineral=0, material=1, compound=1, metal=2, alloy=3, magnet=3)
WEIGHTS = dict(ore=.95, mineral=.95, material=.65, compound=.55, metal=.45, alloy=.25, magnet=.20)
FIELDS = ['direct_total_value_usd', 'direct_hhi', 'origin_hhi',
          'attribution_uncertainty_index', 'route_coverage_share', 'top_chokepoint_share']


def load_inputs():
    manifest = {}
    with zipfile.ZipFile(PACKAGE / 'Reproduction_Code_and_Derived_Inputs.zip') as z:
        def read(name, **kwargs):
            raw = z.read(BASE + name)
            manifest[name] = hashlib.sha256(raw).hexdigest()
            return pd.read_csv(io.BytesIO(raw), compression='gzip' if name.endswith('.gz') else None, **kwargs)
        trade = read('input_snapshot/dynamic_reconstruction_panel.csv.gz', usecols=[
            'CmdCode', 'metal', 'stage', 'exporter_iso', 'importer_iso', 'year',
            'reconstructed_value_usd', 'data_quality_weight', 'observation_sigma_log'])
        seed = read('release_v3/merged_wmd_seed_normalized.csv.gz')
        annual = read('release_v3/longitudinal_risk_indicator_panel_2012_2024.csv.gz')
        lanes = read('input_snapshot/minerals_lanes.csv')
        windows = read('release_v3/mrci_all_windows_2012_2024.csv.gz')
    trade = trade[trade.year.between(2012, 2024) & trade.reconstructed_value_usd.gt(0)].copy()
    trade['quality'] = pd.to_numeric(trade.data_quality_weight, errors='coerce').fillna(.5).clip(0, 1)
    sigma = pd.to_numeric(trade.observation_sigma_log, errors='coerce')
    trade['sigma'] = sigma.where(np.isfinite(sigma) & (sigma > 0), .75)
    trade['variance'] = (trade.reconstructed_value_usd * trade.sigma)**2
    corridor = trade.groupby(KEY + ['exporter_iso'], as_index=False, observed=True).agg(
        value=('reconstructed_value_usd', 'sum'), k=('CmdCode', 'size'),
        quality=('quality', 'mean'), sigma=('sigma', 'mean'), variance=('variance', 'sum'))
    corridor['draw_sigma'] = (np.sqrt(corridor.variance)/corridor.value).clip(.05, 1.5)
    return annual, corridor, seed, lanes, windows, manifest, len(trade)


def prepare(annual, corridor, seed, lanes, windows):
    unit_index = {tuple(k): i for i, k in enumerate(annual[KEY].itertuples(index=False, name=None))}
    assert len(unit_index) == len(annual)
    # Same union-of-labels treatment as the released route function.
    route_lookup = {}
    for key, d in lanes.groupby(['metal', 'origin', 'dest'], sort=False):
        mode = 'sea' if d['mode'].astype(str).eq('sea').any() else str(d['mode'].dropna().iloc[0]) if d['mode'].notna().any() else ''
        labels = {x.strip() for s in d.chokepoints.dropna().astype(str) for x in s.split('|') if x.strip()}
        route_lookup[key] = (mode, labels)
    chokes = sorted({x for mode, labels in route_lookup.values() if mode == 'sea' for x in labels})
    supplier_keys = sorted(set(zip(corridor.metal, corridor.stage, corridor.exporter_iso)))
    lane_keys = sorted(set(zip(corridor.metal, corridor.stage, corridor.exporter_iso, corridor.importer_iso)))
    producer_keys = sorted(set(zip(seed.metal, seed.mine_origin_iso)))
    sup_index = {k:i for i,k in enumerate(supplier_keys)}
    lane_index = {k:i for i,k in enumerate(lane_keys)}
    prod_index = {k:i for i,k in enumerate(producer_keys)}
    blocks = []
    for (metal, year), sub in corridor.groupby(['metal', 'year'], sort=True, observed=True):
        prod = seed[(seed.metal == metal) & (seed.year == year) & (seed.production > 0)]
        p = prod.groupby('mine_origin_iso').production.sum()
        origins = list(p.index)
        countries = sorted(set(sub.exporter_iso) | set(sub.importer_iso) | set(origins))
        ci = {c:i for i,c in enumerate(countries)}
        block = dict(metal=metal, year=int(year), n=len(countries), origin=origins,
                     origin_i=np.array([ci[c] for c in origins], int), production=p.to_numpy(),
                     production_i=np.array([prod_index[(metal,c)] for c in origins], int), stages=[])
        for stage in sorted(sub.stage.unique(), key=lambda s:ORDER.get(s,9)):
            d = sub[sub.stage.eq(stage)]
            imp = d.importer_iso.map(ci).to_numpy()
            exp = d.exporter_iso.map(ci).to_numpy()
            outputs = np.unique(imp)
            unit = np.array([unit_index[(countries[i], metal, stage, int(year))] for i in outputs])
            matched = np.array([(metal,e,i) in route_lookup for e,i in zip(d.exporter_iso,d.importer_iso)])
            incidence = np.array([[c in route_lookup.get((metal,e,i),('',set()))[1]
                         and route_lookup.get((metal,e,i),('',set()))[0]=='sea' for c in chokes]
                         for e,i in zip(d.exporter_iso,d.importer_iso)], dtype=float)
            block['stages'].append(dict(stage=stage, imp=imp, exp=exp, outputs=outputs, unit=unit,
                value=d.value.to_numpy(), k=d.k.to_numpy(), quality=d.quality.to_numpy(),
                sigma=d.sigma.to_numpy(), draw_sigma=d.draw_sigma.to_numpy(), matched=matched,
                incidence=incidence,
                supplier_i=np.array([sup_index[(metal,stage,e)] for e in d.exporter_iso]),
                lane_i=np.array([lane_index[(metal,stage,e,i)] for e,i in zip(d.exporter_iso,d.importer_iso)])))
        blocks.append(block)
    lo = np.array([unit_index[(r.importer_iso,r.metal,r.stage,r.baseline_year)] for r in windows.itertuples()])
    hi = np.array([unit_index[(r.importer_iso,r.metal,r.stage,r.end_year)] for r in windows.itertuples()])
    return blocks, (lo,hi), dict(supplier=len(supplier_keys), lane=len(lane_keys), producer=len(producer_keys)), chokes


def ar_errors(rng, n, rho):
    noise = rng.standard_normal((n,13))
    for year in range(1,13):
        noise[:,year] = rho*noise[:,year-1] + np.sqrt(1-rho*rho)*noise[:,year]
    return noise


def draw_inputs(sizes, number, rho):
    # Shared seed across temporal designs enables a paired comparison.
    rng = np.random.default_rng(20260908 + number)
    supplier = ar_errors(rng, sizes['supplier'], rho)
    lane = ar_errors(rng, sizes['lane'], rho)
    producer = ar_errors(rng, sizes['producer'], rho)
    weights = {stage: float(np.clip(w+rng.uniform(-.2,.2),0,1)) for stage,w in WEIGHTS.items()}
    return dict(supplier=supplier, lane=lane, producer=producer, weights=weights)


def recompute(blocks, units, inputs=None):
    out = np.full((units,len(FIELDS)), np.nan)
    conservation = 0.
    for b in blocks:
        n, year = b['n'], b['year']-2012
        p = b['production'].copy()
        if inputs is not None and len(p):
            p *= np.exp(.1*inputs['producer'][b['production_i'],year])
        g = p/p.sum() if len(p) else np.empty(0)
        mix = np.zeros((n,len(p)))
        mix[b['origin_i'], np.arange(len(p))] = 1
        for s in b['stages']:
            value = s['value']
            w = (WEIGHTS if inputs is None else inputs['weights'])[s['stage']]
            if inputs is not None:
                z = np.sqrt(.5)*inputs['supplier'][s['supplier_i'],year] + np.sqrt(.5)*inputs['lane'][s['lane_i'],year]
                # The sigma convention follows the existing conditional analysis.
                # No clipping of z or silent dropping of positive-support lanes.
                value = np.maximum(np.expm1(np.log1p(value)+s['draw_sigma']*z),1e-12)
            total = np.bincount(s['imp'],weights=value,minlength=n)
            good = s['outputs']
            hhi = np.bincount(s['imp'],weights=value**2,minlength=n)[good]/total[good]**2
            cover = np.bincount(s['imp'],weights=value*s['matched'],minlength=n)[good]/total[good]
            choke = np.zeros((n,s['incidence'].shape[1]))
            np.add.at(choke,s['imp'],value[:,None]*s['incidence'])
            cstar = choke.max(axis=1)[good]/total[good]
            out[s['unit'],0] = total[good]
            out[s['unit'],1] = hhi
            out[s['unit'],4] = cover
            out[s['unit'],5] = cstar
            if not len(p):
                continue
            exporter_mix = np.where((mix.sum(axis=1)>0)[:,None], w*mix+(1-w)*g, g)
            edge = np.zeros((n,n))
            np.add.at(edge,(s['imp'],s['exp']),value)
            amounts = edge@exporter_mix
            shares = amounts[good]/total[good,None]
            conservation = max(conservation,float(np.abs(shares.sum(axis=1)-1).max()))
            out[s['unit'],2] = (shares**2).sum(axis=1)
            pi = exporter_mix[s['exp']]
            support = (pi>0).sum(axis=1)
            ent = -np.sum(np.where(pi>0,pi*np.log(np.maximum(pi,1e-300)),0),axis=1)
            denom = np.log(s['k']*support)
            record_ent = np.divide(ent+np.log(s['k']),denom,out=np.zeros_like(ent),where=denom>0)
            uncertainty = .35*record_ent+.25*(1-s['quality'])+.2*(1-w)+.2*np.clip(s['sigma']/1.5,0,1)
            out[s['unit'],3] = np.bincount(s['imp'],weights=value*uncertainty,minlength=n)[good]/total[good]
            proposed = w*mix[good]+(1-w)*shares
            norm = proposed.sum(axis=1)
            keep = norm>0
            mix[good[keep]] = proposed[keep]/norm[keep,None]
    return out, conservation


def classify(matrix, pair):
    a,b = matrix[pair[0]],matrix[pair[1]]
    diff = b-a
    eligible = (a[:,0]>=1e6)&(b[:,0]>=1e6)&np.isfinite(a[:,2])&np.isfinite(b[:,2])
    routed = (a[:,4]>=.5)&(b[:,4]>=.5)
    apparent = eligible&routed&(diff[:,1]<=-.025)
    transfer = apparent&((diff[:,2]>=.025)|(diff[:,5]>=.025))
    substantive = apparent&(diff[:,2]<=-.025)&(diff[:,5]<=-.025)&(diff[:,3]<.025)
    lower_delta = b[:,5]-np.minimum(1,a[:,5]+1-a[:,4])
    upper_delta = np.minimum(1,b[:,5]+1-b[:,4])-a[:,5]
    return dict(eligible=eligible&routed, apparent=apparent, transfer=transfer, substantive=substantive,
        transfer_lower=apparent&((diff[:,2]>=.025)|(lower_delta>=.025)),
        transfer_upper=apparent&((diff[:,2]>=.025)|(upper_delta>=.025)),
        substantive_lower=apparent&(diff[:,2]<=-.025)&(upper_delta<=-.025)&(diff[:,3]<.025),
        substantive_upper=apparent&(diff[:,2]<=-.025)&(lower_delta<=-.025)&(diff[:,3]<.025),
        origin_free_transfer_lower=apparent&(lower_delta>=.025),
        origin_free_substantive_upper=apparent&(lower_delta<=-.025), value=b[:,0])


def selections(windows):
    for fam in windows.window_family.unique():
        m = windows.window_family.eq(fam).to_numpy()
        yield str(fam),'pooled','all',m
        for year in sorted(windows.end_year.unique()):
            select = m & windows.end_year.eq(year).to_numpy()
            if select.any():
                yield str(fam),str(year),'all',select
        for metal in sorted(windows.metal.unique()):
            select = m & windows.metal.eq(metal).to_numpy()
            if select.any():
                yield str(fam),'pooled',str(metal),select


def summarize(categories, masks, draw, rho):
    rows=[]
    for family,end_year,metal,m in masks:
        a = m&categories['apparent']
        denom = categories['value'][a].sum()
        row=dict(draw=draw, temporal_rho=rho, window_family=family, end_year=end_year, metal=metal,
                 eligible_count=int((m&categories['eligible']).sum()), apparent_count=int(a.sum()), denominator_usd=float(denom))
        for key in ['transfer','substantive','transfer_lower','transfer_upper','substantive_lower',
                    'substantive_upper','origin_free_transfer_lower','origin_free_substantive_upper']:
            row[key+'_share'] = float(categories['value'][m&categories[key]].sum()/denom) if denom else np.nan
        rows.append(row)
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--draws-per-design',type=int,default=250)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    annual,corridor,seed,lanes,windows,hashes,records=load_inputs()
    blocks,pair,sizes,chokes=prepare(annual,corridor,seed,lanes,windows)
    baseline,cons=recompute(blocks,len(annual))
    errors={}
    for j,key in enumerate(FIELDS):
        frozen=annual[key].to_numpy()
        assert np.array_equal(np.isnan(frozen),np.isnan(baseline[:,j])), key+' missingness'
        delta=np.nanmax(np.abs(baseline[:,j]-frozen)/np.maximum(1,np.abs(frozen)))
        errors[key]=float(delta)
        assert delta<1e-10,(key,delta)
    cats=classify(baseline,pair)
    frozen_apparent=(windows.three_layer_evidence_eligible&windows.apparent_derisking).to_numpy()
    assert np.array_equal(cats['apparent'],frozen_apparent)
    assert np.array_equal(cats['transfer'],frozen_apparent&windows.classification.isin(['mine_origin_transfer','route_transfer','multiple_risk_transfer']).to_numpy())
    assert np.array_equal(cats['substantive'],frozen_apparent&windows.classification.eq('substantive_derisking').to_numpy())
    masks=list(selections(windows))
    rows=summarize(cats,masks,0,-1.)
    annual_rows=[]
    params=[]
    membership={rho:{k:np.zeros(len(windows),int) for k in ['eligible','apparent','transfer','substantive']} for rho in [0.,.8]}
    print(f'Baseline exact: {len(annual)} unit-years; {len(windows)} windows; {time.perf_counter()-started:.1f}s',flush=True)
    for rho in [0.,.8]:
        for draw in range(1,args.draws_per_design+1):
            inputs=draw_inputs(sizes,draw,rho)
            matrix,err=recompute(blocks,len(annual),inputs)
            assert err<1e-10
            cons=max(cons,err)
            categories=classify(matrix,pair)
            for k in membership[rho]:
                membership[rho][k]+=categories[k]
            rows.extend(summarize(categories,masks,draw,rho))
            params.append(dict(draw=draw,temporal_rho=rho,stage_weights=inputs['weights']))
            for year in range(2012,2025):
                mask=annual.year.eq(year).to_numpy()&np.isfinite(matrix[:,2])
                weights=matrix[mask,0]
                row=dict(draw=draw,temporal_rho=rho,year=year,unit_years=int(mask.sum()),value_usd=float(weights.sum()))
                row.update({FIELDS[j]:float(np.average(matrix[mask,j],weights=weights)) for j in range(1,6)})
                annual_rows.append(row)
            print(f'rho={rho:.1f} draw={draw}/{args.draws_per_design}; {time.perf_counter()-started:.1f}s',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'window_summary_all_draws.csv',index=False)
    pd.DataFrame(annual_rows).to_csv(OUT/'annual_indicator_summary_all_draws.csv',index=False)
    for rho,counts in membership.items():
        frame=windows[['importer_iso','metal','stage','window_family','baseline_year','end_year']].copy()
        for k,values in counts.items():
            frame[k+'_draw_count']=values
        frame['draws_per_design']=args.draws_per_design
        frame.to_csv(OUT/f'classification_membership_rho_{rho:.1f}.csv.gz',index=False)
    manifest=dict(source_sha256=hashes,trade_records=records,corridor_years=len(corridor),unit_years=len(annual),
        windows=len(windows),supported_minerals=int(annual.loc[annual.origin_hhi.notna(),'metal'].nunique()),
        years=list(range(2012,2025)),baseline_relative_errors=errors,maximum_conservation_error=cons,
        actual_draws=2*args.draws_per_design,random_seed_base=20260908,parameters=params,
        input_counts=sizes,chokepoints=chokes,seconds=time.perf_counter()-started,
        assumptions=dict(temporal_rho=[0.,.8],shared_supplier_variance_fraction=.5,production_log_sd=.1,
          stage_weight_uniform_halfwidth=.2,trade_sigma='sqrt(sum((HS value*sigma)^2))/corridor value; clipped [0.05,1.5]',
          trade_location='median-centred log1p reconstructed corridor value; floor 1e-12',
          covariance='supplier-stage-mineral shock shared across importing countries; corridor residual; stationary annual AR(1)',
          interpretation='uncalibrated conditional sensitivity ensemble; NOT confidence or posterior intervals'),
        coverage=dict(raw_mirror_reconstruction=False,positive_trade_support_fixed=True,trade_values=True,
          recursive_origins=True,production_seed_positive_support_fixed=True,attribution_index_recomputed=True,
          sample_screens_reapplied=True,route_weights_recomputed=True,matched_route_incidence_fixed=True,
          unmatched_routes='adversarial outer bounds per draw',commodity_voyage_data=False,
          allocation_resolved_each_draw=False,external_facility_benchmark_used_to_fit=False),
        manuscript_changed=False)
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summary=pd.DataFrame(rows)
    print(summary[(summary.draw.eq(0))&(summary.end_year.eq('pooled'))&(summary.metal.eq('all'))].to_string(index=False))
    print('Completed',manifest['actual_draws'],'conditional shared-input draws',flush=True)


if __name__=='__main__':
    main()
