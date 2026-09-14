"""Factorial accounting of product, endpoint-code and interpolation-scope changes."""
import itertools
import sys
import zipfile
import json
import run_unified as u
import pandas as pd
import numpy as np
sys.path.insert(0,str(u.ROOT/'research_process/evidence_extension_20260908'))
import run_historical_joint_sensitivity as h

if __name__=='__main__':
    root=u.OUT/'historical_expanded_comtrade'
    _,_,seed,_,original_windows,_,_=h.load_inputs()
    with zipfile.ZipFile(u.s.ARCHIVE) as z:
        old=pd.read_csv(z.open(h.BASE+'input_snapshot/dynamic_reconstruction_panel.csv.gz'),compression='gzip',dtype={'CmdCode':str})
    old=old[old.year.between(2012,2024)&old.metal.isin(seed.metal.unique())]
    new=pd.read_csv(root/'expanded_six_products/reconstructed_products.csv.gz',dtype={'CmdCode':str})
    new=new[new.CmdCode.isin(['750110','750120','283324'])]
    specs=original_windows[['window_family','baseline_year','end_year']].drop_duplicates()
    h.ORDER=dict(ore=0,mineral=0,material=1,compound=2,metal=3,alloy=4,magnet=5)
    rows=[]
    for expanded,named,within in itertools.product([False,True],repeat=3):
        trade=pd.concat([old,new],ignore_index=True) if expanded else old.copy()
        if within: trade=trade[trade.next_observed_year.le(2024)]
        if named: trade=trade[trade.exporter_iso.str.fullmatch('[A-Z]{3}')&trade.importer_iso.str.fullmatch('[A-Z]{3}')]
        trade=trade.copy()
        trade['quality']=trade.data_quality_weight.fillna(.5).clip(0,1)
        sigma=trade.observation_sigma_log
        trade['sigma']=sigma.where(np.isfinite(sigma)&sigma.gt(0),.75)
        trade['variance']=(trade.reconstructed_value_usd*trade.sigma)**2
        corridor=trade.groupby(h.KEY+['exporter_iso'],as_index=False).agg(value=('reconstructed_value_usd','sum'),k=('CmdCode','size'),
            quality=('quality','mean'),sigma=('sigma','mean'),variance=('variance','sum'))
        corridor['draw_sigma']=(np.sqrt(corridor.variance)/corridor.value).clip(.05,1.5)
        annual=corridor[h.KEY].drop_duplicates().sort_values(h.KEY).reset_index(drop=True)
        windows=[]
        for spec in specs.itertuples():
            start=annual[annual.year.eq(spec.baseline_year)][h.KEY[:-1]]
            end=annual[annual.year.eq(spec.end_year)][h.KEY[:-1]]
            pairs=start.merge(end,on=h.KEY[:-1],validate='one_to_one')
            pairs['window_family']=spec.window_family
            pairs['baseline_year']=spec.baseline_year; pairs['end_year']=spec.end_year
            windows.append(pairs)
        windows=pd.concat(windows,ignore_index=True)
        blocks,pair,_,_=h.prepare(annual,corridor,seed,pd.read_csv(u.OUT/'routes.csv'),windows)
        matrix,error=h.recompute(blocks,len(annual))
        assert error<1e-10
        cats=h.classify(matrix,pair)
        masks=[x for x in h.selections(windows) if x[2]=='all']
        result=h.summarize(cats,masks,0,-1.)
        for row in result: row.update(expanded_nickel=expanded,named_country_codes=named,within_period_interpolation=within,product_rows=len(trade))
        rows.extend(result)
        if expanded and named and within:
            output=annual.copy()
            for j,col in enumerate(h.FIELDS): output[col]=matrix[:,j]
            output.to_csv(root/'independent_vectorized_reference.csv.gz',index=False)
        print(f'Correction cell expanded={expanded}, named={named}, within={within} complete',flush=True)
    pd.DataFrame(rows).to_csv(root/'correction_factorial_comparison.csv',index=False)
    (root/'correction_factorial_manifest.json').write_text(json.dumps(dict(cells=8,route_scope='Corrected global plus extended nickel routes fixed in all cells',
        interpretation='Accounting contrasts with cohort and denominator changes; not causal policy effects',
        code_sha256=u.s.sha(__file__),engine_sha256=u.s.sha(h.__file__)),indent=2),encoding='utf-8')
