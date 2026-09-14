"""Same-provider nickel product expansion through frozen mirror reconstruction."""
import io
import json
import sys
import zipfile
import run_unified as u
import pandas as pd
import numpy as np
sys.path.insert(0,str(u.ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h

OUT=u.OUT/'historical_expanded_comtrade'
EXTERNAL=u.HERE/'external/comtrade_expansion'

def load_frozen():
    folder=u.HERE/'frozen'
    modules=[]
    with zipfile.ZipFile(u.s.ARCHIVE) as z:
        for name in ['build_model_inputs.py','reconstruct_dynamic_panel.py']:
            data=z.read('reproduction/project_snapshot/modeling/'+name)
            p=folder/name
            p.parent.mkdir(exist_ok=True)
            if p.exists(): assert p.read_bytes()==data
            else: p.write_bytes(data)
            modules.append(u.s.load_module('expanded_'+name[:-3],p))
    return modules

class Sources:
    def __init__(self,paths): self.paths=paths
    def glob(self,pattern):
        assert pattern=='*.json'
        return self.paths

def reconstruct(paths,mapping,b,r,name):
    b.RAW=Sources(paths)
    b.TRADE_RAW=u.ROOT/'_raw'
    b.load_mineral_map=lambda:mapping
    obs,stats=b.read_raw_observations()
    mirror=b.reconcile_mirrors(obs)
    folder=OUT/name; folder.mkdir(parents=True,exist_ok=True)
    obs.to_csv(folder/'observations.csv.gz',index=False)
    mirror.to_csv(folder/'mirror_reconciliation.csv.gz',index=False)
    stats.to_csv(folder/'raw_file_audit.csv',index=False)
    baseline=mirror[mirror.model_eligible]
    baseline.to_csv(folder/'baseline.csv.gz',index=False)
    r.BASELINE=folder/'baseline.csv.gz'
    panel=r.reconstruct_panel(r.prepare_observed())
    panel.to_csv(folder/'reconstructed_products.csv.gz',index=False)
    return panel

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((EXTERNAL/'source_manifest.json').read_text())
    assert len(manifest)==39 and len({(m['code'],m['year']) for m in manifest})==39
    for m in manifest: assert u.s.sha(EXTERNAL/m['file'])==m['sha256']
    b,r=load_frozen()
    with zipfile.ZipFile(u.s.ARCHIVE) as z:
        old=pd.read_csv(z.open(h.BASE+'input_snapshot/dynamic_reconstruction_panel.csv.gz'),compression='gzip',dtype={'CmdCode':str})
    old=old[old.year.between(2012,2024)]
    old_ni=old[old.metal.eq('Nickel')]
    mapping={code:('Nickel',d.stage.iloc[0],d.hs_label.iloc[0]) for code,d in old_ni.groupby('CmdCode')}
    old_paths=[u.ROOT/f'_raw/{code}_{year}.json' for code in mapping for year in range(2012,2025)]
    assert all(p.exists() for p in old_paths)
    before=reconstruct(old_paths,mapping,b,r,'original_three_products_regression')
    key=['CmdCode','exporter_iso','importer_iso','year']
    a=before.set_index(key).sort_index(); ref=old_ni.set_index(key).sort_index()
    beyond=ref[ref.next_observed_year.gt(2024)]
    assert len(beyond)==163 and (~beyond.is_observed).all()
    assert ref.index.difference(a.index).equals(beyond.index),'Unexplained raw-to-panel support difference'
    ref=ref[ref.next_observed_year.le(2024)]
    assert a.index.equals(ref.index),'Within-period raw-to-panel support regression failed'
    regression={c:float((abs(a[c]-ref[c])/np.maximum(abs(ref[c]),1)).max()) for c in ['reconstructed_value_usd','data_quality_weight','observation_sigma_log']}
    assert max(regression.values())<1e-10,regression
    mapping.update({'750110':('Nickel','material','Nickel mattes'),
                    '750120':('Nickel','material','Nickel oxide sinters and other intermediate products'),
                    '283324':('Nickel','compound','Nickel sulphates')})
    paths=old_paths+[EXTERNAL/m['file'] for m in manifest]
    nickel=reconstruct(paths,mapping,b,r,'expanded_six_products')
    period_excluded=old[old.next_observed_year.gt(2024)].copy()
    assert (~period_excluded.is_observed).all()
    period_excluded.to_csv(OUT/'outside_period_interpolation_ledger.csv.gz',index=False)
    period_excluded.groupby(['metal','year'],as_index=False).reconstructed_value_usd.agg(['size','sum']).to_csv(OUT/'outside_period_interpolation_summary.csv',index=False)
    combined=pd.concat([old[old.metal.ne('Nickel')&old.next_observed_year.le(2024)],nickel],ignore_index=True)
    assert not combined.duplicated(key).any()
    combined.to_csv(OUT/'reconstructed_all_endpoints.csv.gz',index=False)
    named=combined.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&combined.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
    excluded=combined[~named].copy()
    excluded.to_csv(OUT/'non_country_endpoint_ledger.csv.gz',index=False)
    excluded.groupby(['metal','stage','year'],as_index=False).reconstructed_value_usd.sum().to_csv(OUT/'non_country_excluded_value.csv',index=False)
    combined=combined[named].copy()
    combined.to_csv(OUT/'dynamic_reconstruction_panel.csv.gz',index=False)
    # Recompute all supported minerals from HS-level records to retain the existing
    # record-entropy quality-index definition (rather than changing it silently).
    infer,lm,_=u.modules()
    import cached_routes
    sample=combined[combined.metal.eq('Nickel')&combined.year.eq(2018)].groupby(h.KEY+['exporter_iso'],as_index=False).reconstructed_value_usd.sum()
    sample_stats=lm.concentration_stats(sample,'exporter_iso','reconstructed_value_usd','direct')
    route_cache_test=cached_routes.install(lm,u.OUT/'routes.csv',sample,sample_stats)
    oldannual,_,seed,_,oldwindows,_,_=h.load_inputs()
    production=infer.build_production_mix(seed)
    ni_annual=[]; ni_origins=[]; fps=[]
    for metal in sorted(production.metal.unique()):
      for year in range(2012,2025):
        p=production[production.year.eq(year)&production.metal.eq(metal)]
        world=dict(zip(p.inferred_mine_origin_iso,p.origin_share))
        if not world: continue
        mix={country:{country:1.} for country in world}
        current=combined[combined.year.eq(year)&combined.metal.eq(metal)]
        order=['ore','material','compound','metal'] if metal=='Nickel' else sorted(current.stage.unique(),key=lambda x:infer.STAGE_ORDER[x])
        for stage in order:
            t=current[current.stage.eq(stage)]
            if t.empty: continue
            flows,mix=infer.infer_year_stage(t,mix,world,stage)
            origins=flows.groupby(h.KEY+['inferred_mine_origin_iso'],as_index=False).attributed_value_usd.sum()
            ni_origins.append(origins)
            origin_stats=lm.concentration_stats(origins,'inferred_mine_origin_iso','attributed_value_usd','origin')
            attribution_stats=lm.attribution_stats_from_corridors(lm._uncertainty_by_corridor(flows))
            ni_annual.append(lm.build_annual_risk_panel(t,origin_stats,attribution_stats,u.OUT/'routes.csv'))
            f=flows.groupby(['year','metal','stage','exporter_iso','inferred_mine_origin_iso'],as_index=False).attributed_value_usd.sum()
            f['share']=f.attributed_value_usd/f.groupby(['year','metal','stage','exporter_iso']).attributed_value_usd.transform('sum')
            fps.append(f)
        if year in [2016,2020,2024]: print(f'Expanded Comtrade attribution {metal}/{year} complete',flush=True)
    ni_annual=pd.concat(ni_annual,ignore_index=True)
    # Silicon remains in retained trade records but is outside the 23-mineral
    # source-attribution analysis, exactly because no production seed exists.
    annual=ni_annual
    analytical=combined[combined.metal.isin(annual.metal.unique())].copy()
    analytical.to_csv(OUT/'analytical_trade_panel.csv.gz',index=False)
    direct=analytical.groupby(h.KEY+['exporter_iso'],as_index=False).reconstructed_value_usd.sum()
    routed=lm.route_stats(direct,annual,u.OUT/'routes.csv')
    annual=annual.drop(columns=[c for c in routed if c not in lm.GROUP]).merge(routed,on=lm.GROUP,validate='one_to_one').sort_values(h.KEY).reset_index(drop=True)
    annual.to_csv(OUT/'annual_indicators.csv.gz',index=False)
    pd.concat(ni_origins,ignore_index=True).to_csv(OUT/'importer_origins.csv.gz',index=False)
    pd.concat(fps,ignore_index=True).to_csv(OUT/'exporter_profiles.csv.gz',index=False)
    cfg=lm.AnalysisConfig(bootstrap_reps=2000)
    windows=pd.concat([lm.build_window_panel(annual,list(zip(d.baseline_year,d.end_year)),fam,cfg)
        for fam,d in oldwindows[['window_family','baseline_year','end_year']].drop_duplicates().groupby('window_family')],ignore_index=True)
    windows.to_csv(OUT/'all_windows.csv.gz',index=False)
    lm.summarize_windows(windows,pd.DataFrame(),.025,'local_full_evidence_endpoint','none',cfg).to_csv(OUT/'historical_summary_with_bootstrap.csv',index=False)
    info=dict(raw_product_year_files=len(paths),reconstruction_regression=regression,old_nickel_rows=len(old_ni),expanded_nickel_rows=len(nickel),
        full_product_rows=len(combined),unit_years=len(annual),windows=len(windows),codes=sorted(mapping),
        source_vintage='Original three products use retained cache; additional three products retrieved 2026-09-09. No BACI values substituted.',
        unknown_endpoints='Three-letter alphabetic country-code reference scope; S19 and other non-country codes retained in a separate ledger, never mapped to a guessed country',
        non_country_excluded_rows=len(excluded),supported_minerals=int(annual.metal.nunique()),
        outside_period_interpolated_rows_removed=len(period_excluded),nickel_outside_period_rows_removed=len(beyond),
        temporal_rule='Both interpolation endpoints must lie within 2012–2024; original records using 2025 endpoints preserved as an excluded sensitivity ledger',
        route_cache_regression=route_cache_test,route_cache_code_sha256=u.s.sha(cached_routes.__file__),
        input_sha256={str(p.relative_to(u.ROOT)):u.s.sha(p) for p in paths},script_sha256=u.s.sha(__file__),
        new_source_manifest_sha256=u.s.sha(EXTERNAL/'source_manifest.json'),full_paper_updated=False)
    (OUT/'run_manifest.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in info.items() if k!='input_sha256'},indent=2),flush=True)

if __name__=='__main__': main()
