"""Rebuild figure tables, preserving established analytical and display rules."""
from pathlib import Path
import ast,hashlib,json,shutil,zipfile,sys
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CAP=HERE/'capsule';WORK=HERE/'figure_rebuild';R=WORK/'results';H=R/'historical_expanded_comtrade';OUT=WORK;DATA=OUT/'source_data';KEY=['importer_iso','metal','stage'];ARCHIVE=ROOT/'manuscript_package_20260907_final/Reproduction_Code_and_Derived_Inputs.zip'
source=(ROOT/'research_process/unified_reanalysis_20260909/prepare_figure_preview.py').read_text(encoding='utf-8')
tree=ast.parse(source)
save=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='save')
main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
main.body=main.body[:next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cases' for t in n.targets))]
exec(compile(ast.Module(body=[save,main],type_ignores=[]),'frozen_figure_1_4_builder','exec'),globals())
main()
if '--complete' in sys.argv:
    frames=[]
    for rho in [0.,.8]:
        for draw in range(1,6):
            p=CAP/f'outputs/allocation_precision/rebuilt/rho_{rho:.1f}_draw_{draw:03d}'
            q=json.loads((p/'PRECISION_QA.json').read_text());assert q['accepted'],str(p)
            d=pd.read_csv(p/'annual_summary.csv');d.insert(0,'draw',draw);d.insert(1,'rho',rho);frames.append(d)
    pd.concat(frames,ignore_index=True).to_csv(DATA/'Figure5a_conditional_draws.csv',index=False)
    shutil.copy2(R/'annual_allocation_summary.csv',DATA/'Figure5a.csv')
    cases=pd.read_csv(R/'all_cases.csv.gz')
    ranks=cases[cases.scenario.eq('baseline')].groupby('metal').agg(years=('base_year','nunique'),value=('baseline_value_usd','sum'))
    metals=ranks[ranks.years.eq(13)].nlargest(4,'value').index.tolist()
    selected=cases[cases.metal.isin(metals)&cases.scenario.ne('baseline')].copy()
    selected['equal_year_weight']=selected.baseline_value_usd/selected.groupby(['base_year','metal','scenario']).baseline_value_usd.transform('sum')/13
    np.testing.assert_allclose(selected.groupby(['metal','scenario']).equal_year_weight.sum(),1)
    selected.to_csv(DATA/'Figure5b.csv',index=False)
print('Figure tables rebuilt',flush=True)
