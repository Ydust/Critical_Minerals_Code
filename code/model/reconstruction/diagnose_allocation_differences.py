from pathlib import Path
import json
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;CAP=HERE/'capsule';OUT=CAP/'outputs/allocation'
K=['base_year','metal','stage','importer_iso','scenario']
a=pd.read_csv(OUT/'all_cases.csv.gz').query('scenario != "without_origin"').set_index(K).sort_index()
e=pd.read_csv(CAP/'expected/SD5_allocation_cases.csv.gz').set_index(K).sort_index()
flows=pd.read_csv(OUT/'all_flows.csv.gz').query('scenario != "without_origin"')
routes=pd.read_csv(CAP/'inputs/historical/routes.csv').rename(columns={'origin':'exporter_iso','dest':'importer_iso'})
exp=flows.set_index(K+['exporter_iso']).scenario_share
routed=flows.merge(routes[['metal','exporter_iso','importer_iso','mode','chokepoints']],on=['metal','exporter_iso','importer_iso'],how='left',validate='many_to_one')
routed=routed[routed['mode'].eq('sea')&routed.chokepoints.fillna('').ne('')].copy()
routed['chokepoint']=routed.chokepoints.str.split('|');routed=routed.explode('chokepoint')
chokes=routed.groupby(K+['chokepoint']).scenario_share.sum()
details=[]
for field,series in [('top_exporter_iso',exp),('top_chokepoint',chokes)]:
    changed=a[field].fillna('')!=e[field].fillna('')
    for key in a.index[changed]:
        new=a.loc[key,field];old=e.loc[key,field]
        ns=float(series.get((*key,new),0.));es=float(series.get((*key,old),0.))
        details.append(dict(zip(K,key),field=field,recomputed_label=new,reference_label=old,recomputed_label_share=ns,reference_label_share=es,absolute_share_gap=abs(ns-es)))
pd.DataFrame(details).to_csv(OUT/'label_difference_diagnostics.csv',index=False)
amount=(a.reallocation_value_usd-e.reallocation_value_usd).abs()
normalized=amount/np.maximum(e.reallocation_value_usd.abs(),1)
amounts=pd.DataFrame({'reference_usd':e.reallocation_value_usd,'recomputed_usd':a.reallocation_value_usd,'absolute_difference_usd':amount,'error_relative_to_reference_amount':normalized,'error_relative_to_baseline_value':amount/e.baseline_value_usd})
amounts[normalized.ge(1e-6)].to_csv(OUT/'reallocation_difference_diagnostics.csv')
flags=['joint_improvement','material_joint_reduction','risk_transfer','origin_risk_transfer','route_risk_transfer','multiple_risk_transfer']
classification={c:int((a[c]!=e[c]).sum()) for c in flags}
summary=[]
for (year,scenario),d in a.reset_index().groupby(['base_year','scenario']):
    ref=e.reset_index().query('base_year == @year and scenario == @scenario')
    row=dict(year=int(year),scenario=scenario)
    for c in ['risk_transfer','material_joint_reduction']:
        row[c+'_recomputed']=float((d.baseline_value_usd*d[c]).sum()/d.baseline_value_usd.sum())
        row[c+'_reference']=float((ref.baseline_value_usd*ref[c]).sum()/ref.baseline_value_usd.sum())
    summary.append(row)
pd.DataFrame(summary).to_csv(OUT/'scientific_summary_comparison.csv',index=False)
report=dict(strict_regression_status='failed; unchanged',classification_mismatches=classification,label_mismatch_counts=pd.DataFrame(details).field.value_counts().to_dict(),max_label_share_gap_by_field=pd.DataFrame(details).groupby('field').absolute_share_gap.max().to_dict(),reallocation_amount_fields_over_declared_tolerance=int(normalized.ge(1e-6).sum()),maximum_reallocation_absolute_difference_usd=float(amount.max()),maximum_reallocation_difference_as_baseline_share=float((amount/e.baseline_value_usd).max()),max_summary_share_difference=max(abs(r[c+'_recomputed']-r[c+'_reference']) for r in summary for c in ['risk_transfer','material_joint_reduction']),acceptance_tolerance_changed=False,release_modified=False)
(OUT/'DIFFERENCE_DIAGNOSIS.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
