from pathlib import Path
import json,hashlib,argparse
import pandas as pd
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--version',type=int,choices=[1,2],default=1);VERSION=parser.parse_args().version
HERE=Path(__file__).resolve().parent;CAP=HERE/'capsule';SRC=CAP/'outputs/allocation';OUT=SRC/f'reporting_v{VERSION}';OUT.mkdir(exist_ok=True)
K=['base_year','metal','stage','importer_iso','scenario'];EPS=1e-6
routes=pd.read_csv(CAP/'inputs/historical/routes.csv').rename(columns={'origin':'exporter_iso','dest':'importer_iso'})
def ranks(frame,label,prefix):
    d=frame.groupby(K+[label],as_index=False).scenario_share.sum()
    peak=d.groupby(K).scenario_share.transform('max')
    d=d[peak-d.scenario_share<=EPS].copy()
    result=d.groupby(K)[label].agg(lambda x:'|'.join(sorted(set(x)))).to_frame(prefix+'_candidates')
    result[prefix+'_status']=np.where(result[prefix+'_candidates'].str.contains('|',regex=False),'numerically_tied','unique')
    result[prefix+'_label']=result[prefix+'_candidates'].where(result[prefix+'_status'].eq('unique'),'')
    return result
def view(cases,flows):
    result=cases[K+['baseline_value_usd','reallocation_value_usd','reallocation_share_of_baseline','route_matrix_available','top_chokepoint_share']].set_index(K)
    result=result.join(ranks(flows,'exporter_iso','supplier'))
    d=flows.merge(routes[['metal','exporter_iso','importer_iso','mode','chokepoints']],on=['metal','exporter_iso','importer_iso'],how='left',validate='many_to_one')
    d=d[d['mode'].eq('sea')&d.chokepoints.fillna('').ne('')].copy();d['chokepoint']=d.chokepoints.str.split('|');d=d.explode('chokepoint')
    result=result.join(ranks(d,'chokepoint','chokepoint'))
    result['chokepoint_status']=result.chokepoint_status.fillna('not_available')
    for col in ['chokepoint_candidates','chokepoint_label']:result[col]=result[col].fillna('')
    if VERSION==2:
        result['chokepoint_sparse_candidates_diagnostic']=result.chokepoint_candidates
        no_evidence=~result.route_matrix_available
        unresolved=result.route_matrix_available&result.top_chokepoint_share.le(EPS)
        result.loc[no_evidence,'chokepoint_status']='not_available'
        result.loc[unresolved,'chokepoint_status']='below_numerical_resolution'
        result.loc[no_evidence|unresolved,['chokepoint_label','chokepoint_candidates']]=''
    result['reallocation_status']=np.where(result.reallocation_share_of_baseline<=EPS,'below_numerical_resolution','resolved')
    return result.sort_index()
paths=[SRC/'all_cases.csv.gz',SRC/'all_flows.csv.gz',CAP/'expected/SD5_allocation_cases.csv.gz',CAP/'expected/SD5_allocation_flows.csv.gz',CAP/'inputs/historical/routes.csv',HERE/'OUTPUT_RULES_V1.md',Path(__file__)]
if VERSION==2:paths.append(HERE/'OUTPUT_RULES_V2.md')
manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(OUT/'INPUT_CONTRACT.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
actual=pd.read_csv(paths[0]).query('scenario != "without_origin"');af=pd.read_csv(paths[1]).query('scenario != "without_origin"')
expected=pd.read_csv(paths[2]);ef=pd.read_csv(paths[3])
a=view(actual,af);e=view(expected,ef);assert a.index.equals(e.index)
a.to_csv(OUT/'recomputed_output_view.csv.gz');e.to_csv(OUT/'reference_output_view.csv.gz')
labels=['supplier_status','supplier_label','supplier_candidates','chokepoint_status','chokepoint_label','chokepoint_candidates','reallocation_status']
mismatches={c:int((a[c]!=e[c]).sum()) for c in labels}
amount_error=float(((a.reallocation_value_usd-e.reallocation_value_usd).abs()/e.baseline_value_usd).max())
identity_a=float(abs(a.reallocation_value_usd/a.baseline_value_usd-a.reallocation_share_of_baseline).max())
identity_e=float(abs(e.reallocation_value_usd/e.baseline_value_usd-e.reallocation_share_of_baseline).max())
old=json.loads((SRC/'ALLOCATION_REPRODUCTION_QA.json').read_text());physical=json.loads((SRC/'INDEPENDENT_ALLOCATION_CHECK.json').read_text())
residual_numeric={c:v for c,v in old['checks']['cases']['errors'].items() if c!='reallocation_value_usd'}
residual_labels={c:v for c,v in old['checks']['cases']['mismatches'].items() if c not in ['top_exporter_iso','top_chokepoint']}
other_pass=all(v['accepted'] for k,v in old['checks'].items() if k!='cases')
passed=not any(mismatches.values()) and amount_error<EPS and identity_a<1e-10 and identity_e<1e-10 and not residual_labels and max(residual_numeric.values())<EPS and other_pass and physical['feasibility_and_KKT_checks_passed']
report=dict(reporting_v1_equivalence_passed=bool(passed),original_strict_regression_passed=old['accepted'],raw_values_changed=False,original_tolerances_changed=False,new_reporting_contract='OUTPUT_RULES_V1.md; distinct dimensionally normalized amount and near-tie presentation rules',rows=len(a),label_mismatches=mismatches,amount_difference_as_baseline_share=amount_error,amount_share_identity_error_recomputed=identity_a,amount_share_identity_error_reference=identity_e,recomputed_supplier_ties=int(a.supplier_status.eq('numerically_tied').sum()),recomputed_chokepoint_ties=int(a.chokepoint_status.eq('numerically_tied').sum()),recomputed_unresolved_reallocations=int(a.reallocation_status.eq('below_numerical_resolution').sum()),independent_feasibility_passed=physical['feasibility_and_KKT_checks_passed'],full_pipeline_complete=False)
if VERSION==2:
    report['reporting_v2_equivalence_passed']=report.pop('reporting_v1_equivalence_passed')
    report['new_reporting_contract']='OUTPUT_RULES_V2.md; explicit below-resolution bottleneck reporting amendment to V1'
    report['recomputed_unresolved_chokepoints']=int(a.chokepoint_status.eq('below_numerical_resolution').sum())
(OUT/'REPORTING_EQUIVALENCE_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if not passed:
    different=a[labels].ne(e[labels]).any(axis=1)
    a[different].join(e[different],lsuffix='_recomputed',rsuffix='_reference').to_csv(OUT/'remaining_output_differences.csv')
assert passed,report
