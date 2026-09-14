"""Aggregate independently validated allocation draws without changing gates."""
from pathlib import Path
import hashlib,json,sys,importlib.metadata
import pandas as pd
HERE=Path(__file__).resolve().parent;CAP=HERE/'capsule';OUT=CAP/'outputs/allocation_conditional'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=CAP/'ALLOCATION_CONDITIONAL_INPUT_MANIFEST.json';digest=sha(manifest)
    for rec in json.loads(manifest.read_text()):assert sha(CAP/rec['file'])==rec['sha256']
    results=[];summaries=[];output_hashes=[]
    for draw in range(1,6):
        for rho in [0.,.8]:
            folder=OUT/f'rho_{rho:.1f}_draw_{draw:03d}';path=folder/'ALLOCATION_CONDITIONAL_QA.json';qa=json.loads(path.read_text());assert qa['input_manifest_sha256']==digest
            assert qa['draw']==draw and qa['rho']==rho and qa['independent_validation']['accepted'] and not qa['blocked_external_reads']
            results.append(qa);frame=pd.read_csv(folder/'annual_summary.csv');frame.insert(0,'rho',rho);frame.insert(0,'draw',draw);summaries.append(frame)
            for name in ['ALLOCATION_CONDITIONAL_QA.json','all_cases.csv.gz','all_flows.csv.gz','all_solver.csv.gz','annual_summary.csv','independent_validation.json']:
                p=folder/name;output_hashes.append(dict(file=p.relative_to(CAP).as_posix(),sha256=sha(p)))
    pd.concat(summaries,ignore_index=True).to_csv(OUT/'all_annual_summaries.csv',index=False)
    metrics=results[0]['independent_validation']['solver_checks']
    report=dict(accepted=True,draws=10,draws_per_design=5,rhos=[0.,.8],all_weights_exact=all(x['weights_exact'] for x in results),all_reporting_v2_equivalent=all(x['reporting_v2_equivalence_passed'] for x in results),strict_regression_passed_draws=sum(x['original_strict_regression_passed'] for x in results),strict_regression_failed_draws=[dict(draw=x['draw'],rho=x['rho']) for x in results if not x['original_strict_regression_passed']],unit_year_range=[min(x['unit_years'] for x in results),max(x['unit_years'] for x in results)],optimisation_groups=sum(x['groups'] for x in results),actual_optimisations=sum(x['independent_validation']['actual_optimisations'] for x in results),solver_maxima={k:max(x['independent_validation']['solver_checks'][k] for x in results) for k in metrics},max_independent_metric_error=max(max(x['independent_validation']['independent_metric_errors'].values()) for x in results),max_historical_capacity_input_error=max(max(x['independent_validation']['historical_capacity_input_errors'].values()) for x in results),max_relative_capacity_excess=max(x['independent_validation']['max_relative_capacity_excess'] for x in results),input_manifest_sha256=digest,output_hashes=output_hashes,python_executable=sys.executable,packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','clarabel','osqp']},full_pipeline_complete=False,interpretation='10 conditional reoptimisations; no calibrated confidence intervals or arbitrary tail probabilities; raw strict regressions remain separately reported')
    report['accepted']=all(x['accepted'] for x in results)
    report['failed_draws']=[dict(draw=x['draw'],rho=x['rho']) for x in results if not x['accepted']]
    report['all_annual_and_baseline_checks_passed']=all(x['checks']['annual']['accepted'] and x['checks']['baseline']['accepted'] for x in results)
    report['all_annual_summary_checks_passed']=all(x['checks']['annual_summary']['accepted'] for x in results)
    report['all_non_top_label_categories_exact']=all(not {k:v for k,v in x['checks']['cases']['mismatches'].items() if k not in ['top_exporter_iso','top_chokepoint']} for x in results)
    report['all_reporting_labels_exact']=all(not any(x['reporting_label_mismatches'].values()) for x in results)
    report['max_annual_normalized_error']=max(max(x['checks']['annual']['errors'].values()) for x in results)
    report['max_annual_summary_normalized_error']=max(max(x['checks']['annual_summary']['errors'].values()) for x in results)
    report['max_allocation_share_difference']=max(x['checks']['flows']['max_absolute_share_error'] for x in results)
    report['max_reallocation_difference_as_baseline_share']=max(x['amount_difference_as_baseline_share'] for x in results)
    report['finalizer_sha256']=sha(Path(__file__))
    (OUT/'ALLOCATION_CONDITIONAL_BATCH_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k!='output_hashes'},indent=2))
    assert report['accepted'],report['failed_draws']
if __name__=='__main__':main()
