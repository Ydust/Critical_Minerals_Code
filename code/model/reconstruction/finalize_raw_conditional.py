"""Independent flow-level spot checks and complete-batch acceptance."""
from pathlib import Path
import importlib.util,sys,json,hashlib
CAP=Path(__file__).resolve().parent/'capsule'
FINALIZER_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('raw_run',CAP/'run_raw_conditional.py');p=importlib.util.module_from_spec(spec);sys.modules[spec.name]=p;spec.loader.exec_module(p)
import numpy as np
import pandas as pd
def main():
    p.verify_manifest();out=CAP/'outputs/raw_conditional'
    independent_code_hashes={}
    for rec in json.loads((CAP/'HISTORICAL_INPUT_MANIFEST.json').read_text()):
        if rec['file'].startswith('code/'):
            assert p.sha(CAP/rec['file'])==rec['sha256'];independent_code_hashes[rec['file']]=rec['sha256']
    h=p.load('conditional_engine');infer=p.load('infer_mine_origin_flows');lm=p.load('longitudinal_mrci');cache=p.load('cached_routes')
    seed=pd.read_csv(CAP/'outputs/seed/historical_production_seed_rebuilt.csv.gz');routes=CAP/'inputs/historical/routes.csv'
    producers=sorted(set(zip(seed.metal,seed.mine_origin_iso)));prod_index={k:i for i,k in enumerate(producers)}
    order=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    markers=[out/f'rho_{rho:.1f}_draw_{draw:03d}/REPRODUCTION_QA.json' for draw in range(1,21) for rho in [0.,.8]]
    reports=[json.loads(m.read_text()) for m in markers];assert all(r['accepted'] for r in reports)
    digest=p.sha(CAP/'RAW_CONDITIONAL_INPUT_MANIFEST.json');assert all(r['input_manifest_sha256']==digest for r in reports)
    spot=[]
    for marker,meta in zip(markers,reports):
        if meta['draw'] not in [2,10,20]:continue
        rho=meta['rho'];draw=meta['draw'];dest=marker.parent;rng=np.random.default_rng(202609093000+draw);shocks=h.ar_errors(rng,len(producers),rho)
        weights={k:float(np.clip(w+rng.uniform(-.2,.2),0,1)) for k,w in h.WEIGHTS.items()};assert weights==meta['weights'];infer.LOCAL_STAGE_WEIGHT.update(weights)
        predicted=pd.read_csv(dest/'annual_indicators.csv.gz').set_index(h.KEY).sort_index()
        for metal,year in [('Nickel',2018),('Copper',2024)]:
            trade=pd.read_csv(dest/metal/'reconstructed.csv.gz',dtype={'CmdCode':str});named=trade.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&trade.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False);trade=trade[named&trade.year.eq(year)]
            direct=trade.groupby(h.KEY+['exporter_iso'],as_index=False).reconstructed_value_usd.sum();cache.install(lm,routes,direct,lm.concentration_stats(direct,'exporter_iso','reconstructed_value_usd','direct'))
            q=seed[seed.metal.eq(metal)&seed.year.eq(year)].copy();q.production*=np.exp(.1*np.array([shocks[prod_index[(metal,o)],year-2012] for o in q.mine_origin_iso]));prod=infer.build_production_mix(q);world=dict(zip(prod.inferred_mine_origin_iso,prod.origin_share));mix={o:{o:1.} for o in world};parts=[]
            for stage in sorted(trade.stage.unique(),key=lambda s:order[s]):
                t=trade[trade.stage.eq(stage)];flows,mix=infer.infer_year_stage(t,mix,world,stage);origins=flows.groupby(h.KEY+['inferred_mine_origin_iso'],as_index=False).attributed_value_usd.sum();stats=lm.concentration_stats(origins,'inferred_mine_origin_iso','attributed_value_usd','origin');ats=lm.attribution_stats_from_corridors(lm._uncertainty_by_corridor(flows));parts.append(lm.build_annual_risk_panel(t,stats,ats,routes))
            actual=pd.concat(parts,ignore_index=True).set_index(h.KEY).sort_index();expected=predicted.loc[actual.index];errors={c:float((abs(actual[c]-expected[c])/np.maximum(abs(expected[c]),1)).max()) for c in h.FIELDS};assert max(errors.values())<1e-10,errors
            spot.append(dict(draw=draw,rho=rho,metal=metal,year=year,rows=len(actual),errors=errors));print(f'Flow-level check passed: {draw}/{rho}/{metal}/{year}',flush=True)
    assert len(spot)==12
    frames=[pd.read_csv(m.parent/'window_summaries.csv') for m in markers];pd.concat(frames,ignore_index=True).to_csv(out/'all_window_summaries.csv',index=False)
    report=dict(accepted=True,draws=40,rhos=[0.,.8],draws_per_design=20,annual_rows_per_draw=reports[0]['annual']['rows'],summary_rows=sum(r['summary']['rows'] for r in reports),max_annual_normalized_error=max(max(r['annual']['errors'].values()) for r in reports),max_summary_normalized_error=max(max(r['summary']['errors'].values()) for r in reports),max_conservation_error=max(r['max_conservation_error'] for r in reports),all_sample_counts_exact=all(r['counts_exact'] for r in reports),all_weights_exact=all(r['weights_exact'] for r in reports),all_changed_record_counts_exact=all(r['changed_counts_exact'] for r in reports),flow_spot_checks=spot,flow_check_scope='12 mineral-year checks at draws 2,10,20 in both designs; not exhaustive',blocked_external_reads=p.blocked,input_manifest_sha256=digest,interpretation='Uncalibrated conditional sensitivity; not independent samples, confidence intervals or observed mine-source identification',allocation_draws_rerun=False,full_pipeline_complete=False)
    report.update(finalizer_sha256=FINALIZER_SHA,independent_code_hashes=independent_code_hashes)
    (out/'INPUT_MANIFEST.json').write_bytes((CAP/'RAW_CONDITIONAL_INPUT_MANIFEST.json').read_bytes());(out/'RAW_CONDITIONAL_REPRODUCTION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ['flow_spot_checks','independent_code_hashes']},indent=2))
if __name__=='__main__':main()
