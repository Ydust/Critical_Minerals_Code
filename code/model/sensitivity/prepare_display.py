from pathlib import Path
import shutil,json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
P=ROOT/'submission_package_strengthened_20260910';D=P/'source_data'
table=pd.read_csv(P/'supplementary_data/SD6_hidden_layer_cross_distribution.csv')
def label(r):
    x,y=r.origin_change,r.route_change
    if x=='material_decrease' and y=='material_decrease':return 'Both decrease'
    if x=='material_decrease' and y=='material_increase':return 'Origin decreases, maritime increases'
    if x=='material_increase' and y=='material_decrease':return 'Origin increases, maritime decreases'
    if x=='material_increase' and y=='material_increase':return 'Both increase'
    return 'At least one intermediate'
table['outcome']=table.apply(label,axis=1)
joint=table.groupby(['end_year','outcome'],as_index=False).agg(value_share=('value_share','sum'),windows=('windows','sum'),end_value_usd=('end_value_usd','sum'))
joint['value_share_pct']=100*joint.value_share
joint.to_csv(D/'Figure1_joint_outcomes.csv',index=False)
w=pd.read_csv(P/'supplementary_data/SD2_historical_windows.csv.gz')
w=w[w.window_family.eq('adjacent')&w.three_layer_evidence_eligible&w.apparent_derisking].copy()
assert len(w)==5427,len(w)
for letter,col,scale in [('b','delta_origin_hhi',1),('c','delta_top_chokepoint_share',100)]:
    x=np.linspace(.025,1,401);v=w.transition_value_usd
    pd.DataFrame(dict(threshold=x*scale,value_share_pct=[100*v[w[col]>=z].sum()/v.sum() for z in x])).to_csv(D/f'Figure1{letter}_exceedance.csv',index=False)
hist=pd.read_csv(HERE/'results/historical/structure_summary.csv')
hist[hist.variant.eq('no_propagation')].to_csv(D/'Figure2a_no_propagation.csv',index=False)
shutil.copy2(HERE/'results/allocation/mineral_summary.csv',D/'SupplementaryFigure1_preferences.csv')
print('Display sources complete')
