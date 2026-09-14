"""Separate corrected paired-input validation from preserved legacy failures."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;CAP=HERE/'capsule';OUT=CAP/'outputs/allocation_precision';KEY=['base_year','metal','stage','importer_iso','scenario']
records=[];changes=[];annual=[]
for rho in [0.,.8]:
    for draw in range(1,6):
        name=f'rho_{rho:.1f}_draw_{draw:03d}'
        legacy=OUT/'legacy'/name;rebuilt=OUT/'rebuilt'/name
        qa=json.loads((rebuilt/'PRECISION_QA.json').read_text());prior=json.loads((legacy/'PRECISION_QA.json').read_text())
        assert qa['accepted'] and qa['reporting_v2_equivalence_passed'] and qa['checks']['flows']['accepted'],name
        assert prior['independent_validation']['accepted']
        old=pd.read_csv(CAP/'expected/allocation_conditional'/name/'all_cases.csv.gz').set_index(KEY).sort_index()
        new=pd.read_csv(rebuilt/'all_cases.csv.gz').set_index(KEY).sort_index();assert old.index.equals(new.index)
        for col in ['risk_transfer','material_joint_reduction']:assert (old[col]==new[col]).all()
        for col in ['joint_improvement']:
            bad=old[col].ne(new[col]);d=new.loc[bad,['baseline_value_usd','delta_direct_hhi','delta_origin_hhi','delta_top_chokepoint_share',col]].reset_index()
            if len(d):
                d.insert(0,'draw',draw);d.insert(1,'rho',rho);d['previous_'+col]=old.loc[bad,col].to_numpy();changes.append(d)
        summary=pd.read_csv(rebuilt/'annual_summary.csv');summary.insert(0,'draw',draw);summary.insert(1,'rho',rho);annual.append(summary)
        records.append(dict(draw=draw,rho=rho,paired_validation_passed=qa['accepted'],share_error=qa['checks']['flows']['max_absolute_share_error'],baseline_normalized_amount_error=qa['amount_difference_as_baseline_share'],groups=qa['groups'],unit_years=qa['unit_years'],legacy_to_old_report_accepted=prior['accepted'],qa_sha256=hashlib.sha256((rebuilt/'PRECISION_QA.json').read_bytes()).hexdigest()))
pd.DataFrame(records).to_csv(OUT/'paired_summary.csv',index=False)
pd.concat(annual,ignore_index=True).to_csv(OUT/'annual_summaries.csv',index=False)
(pd.concat(changes,ignore_index=True) if changes else pd.DataFrame(columns=KEY+['joint_improvement'])).to_csv(OUT/'near_zero_classification_changes.csv',index=False)
report=dict(accepted=True,meaning='Corrected high-precision original-input versus rebuilt-input validation, not acceptance of the frozen lower-precision results',draws=10,paired_full_runs=20,optimization_groups_per_branch=sum(r['groups'] for r in records),optimisations_per_branch=2*sum(r['groups'] for r in records),maximum_absolute_share_error=max(r['share_error'] for r in records),maximum_baseline_normalized_amount_error=max(r['baseline_normalized_amount_error'] for r in records),unchanged_acceptance_tolerance=1e-6,solver_stopping_tolerance=1e-13,primary_material_classifications_unchanged=True,near_zero_joint_improvement_changes=sum(len(d) for d in changes),frozen_low_precision_strict_comparison_accepted=False,original_failed_reports_preserved=True,independent_observational_validation=False)
(OUT/'PAIRED_BATCH_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
