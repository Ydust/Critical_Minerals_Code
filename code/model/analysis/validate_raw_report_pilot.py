"""Independent flow-level checks of selected perturbed historical branches."""
import json
import raw_report_sensitivity_pilot as p
import cached_routes
import numpy as np
import pandas as pd

def main():
    infer,lm,_=p.u.modules()
    order=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    _,_,seed,_,_,_,_=p.h.load_inputs()
    producer_keys=sorted(set(zip(seed.metal,seed.mine_origin_iso)))
    prod_index={key:i for i,key in enumerate(producer_keys)}
    reports=[]
    path=p.u.HERE/'results/routes.csv'
    for marker in sorted(p.OUT.glob('*/complete.json')):
        meta=json.loads(marker.read_text()); rho=meta['rho']; dest=marker.parent
        rng=np.random.default_rng(202609093001)
        shocks=p.h.ar_errors(rng,len(producer_keys),rho)
        weights={k:float(np.clip(w+rng.uniform(-.2,.2),0,1)) for k,w in p.h.WEIGHTS.items()}
        assert weights==meta['weights']
        infer.LOCAL_STAGE_WEIGHT.update(weights)
        predicted=pd.read_csv(dest/'annual_indicators.csv.gz').set_index(p.h.KEY).sort_index()
        for metal,year in [('Nickel',2018),('Copper',2024)]:
            trade=pd.read_csv(dest/metal/'reconstructed.csv.gz',dtype={'CmdCode':str})
            named=trade.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)&trade.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
            trade=trade[named&trade.year.eq(year)]
            q=seed[seed.metal.eq(metal)&seed.year.eq(year)].copy()
            q.production*=np.exp(.1*np.array([shocks[prod_index[(metal,o)],year-2012] for o in q.mine_origin_iso]))
            prod=infer.build_production_mix(q)
            world=dict(zip(prod.inferred_mine_origin_iso,prod.origin_share)); mix={o:{o:1.} for o in world}
            all_stats=[]
            for stage in sorted(trade.stage.unique(),key=lambda s:order[s]):
                t=trade[trade.stage.eq(stage)]
                flows,mix=infer.infer_year_stage(t,mix,world,stage)
                origins=flows.groupby(p.h.KEY+['inferred_mine_origin_iso'],as_index=False).attributed_value_usd.sum()
                os=lm.concentration_stats(origins,'inferred_mine_origin_iso','attributed_value_usd','origin')
                ats=lm.attribution_stats_from_corridors(lm._uncertainty_by_corridor(flows))
                all_stats.append(lm.build_annual_risk_panel(t,os,ats,path))
            actual=pd.concat(all_stats,ignore_index=True).set_index(p.h.KEY).sort_index()
            expected=predicted.loc[actual.index]
            errors={c:float((abs(actual[c]-expected[c])/np.maximum(abs(expected[c]),1)).max()) for c in p.h.FIELDS}
            assert max(errors.values())<1e-10,errors
            reports.append(dict(rho=rho,metal=metal,year=year,unit_years=len(actual),errors=errors))
            print(f'Independent raw-draw origin/quality/route check: rho={rho}, {metal}/{year} passed',flush=True)
    assert len(reports)==4
    (p.OUT/'independent_flow_spot_checks.json').write_text(json.dumps(dict(accepted=True,scope='Two mineral-years per pilot; not exhaustive',checks=reports),indent=2),encoding='utf-8')

if __name__=='__main__': main()
