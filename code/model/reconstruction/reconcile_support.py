"""Reconcile delivered tables and recompute bounds from the delivered solutions."""
from pathlib import Path
import importlib.util,ast,hashlib,json,shutil
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];CAP=HERE/'capsule';P=ROOT/'submission_package_verified_20260910';S=P/'supplementary_data';R=HERE/'figure_rebuild/results'
qa=json.loads((CAP/'outputs/allocation_precision/PAIRED_BATCH_QA.json').read_text());assert qa['accepted']
def mod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def copy(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
items={'SD1_historical_annual.csv.gz':'historical_rebuilt_seed/annual.csv.gz','SD1_historical_trade.csv.gz':'historical_trade_rebuilt.csv.gz','SD2_historical_windows.csv.gz':'historical_rebuilt_seed/windows.csv.gz','SD2_historical_summary.csv':'historical_rebuilt_seed/summary.csv','SD3_exporter_profiles.csv.gz':'historical_rebuilt_seed/exporter_profiles.csv.gz','SD3_importer_origins.csv.gz':'historical_rebuilt_seed/importer_origins.csv.gz','SD4_routes.csv':'routes_rebuilt/routes.csv','SD5_trade_corridors.csv.gz':'baci/trade_corridors_rebuilt.csv.gz','SD5_production_seeds.csv.gz':'seed/allocation_production_seed_rebuilt.csv.gz','SD5_annual_indicators.csv.gz':'allocation/annual_indicators.csv.gz','SD5_annual_summary.csv':'allocation/annual_summary.csv'}
for dest,src in items.items():copy(CAP/'outputs'/src,S/dest)
for dest,src in [('SD3_fixed_endpoint_cohort.csv','fixed_endpoint_cohort.csv'),('SD4_maritime_events.csv.gz','fig4_events.csv.gz'),('SD4_event_route_allocations.csv.gz','fig4_event_route_allocations.csv.gz'),('SD4_all_ranked_paths.csv','fig4_all_ranked_paths.csv')]:copy(R/'structural_integration'/src,S/dest)
for kind,new in [('cases','cases'),('flows','flows'),('solver','solver')]:
    d=pd.read_csv(CAP/f'outputs/allocation/all_{kind}.csv.gz')
    d[d.scenario.ne('without_origin')].to_csv(S/f'SD5_allocation_{new}.csv.gz',index=False)
    d[d.scenario.eq('without_origin')].to_csv(S/('SD6_origin_ablation_all_solver.csv' if kind=='solver' else f'SD6_origin_ablation_all_{new}.csv.gz'),index=False)
helper_source=(CAP/'code/allocation_conditional_helpers.py').read_text()
assert helper_source.count('len(cases)//3')==1 and helper_source.count('len(solvers)//3')==1
helper_source=helper_source.replace('len(cases)//3',"int(cases.scenario.eq('baseline').sum())").replace('len(solvers)//3',"len(solvers[['base_year','metal','stage']].drop_duplicates())")
namespace={'__file__':str(CAP/'code/allocation_conditional_helpers.py')}
exec(compile(helper_source,'reference_audit_generalized_scenario_counts','exec'),namespace)
reference_audit=namespace['audit'](CAP/'outputs/allocation',trade_input=pd.read_csv(CAP/'outputs/baci/trade_corridors_rebuilt.csv.gz'))
(S/'SD5_independent_validation.json').write_text(json.dumps(reference_audit,indent=2),encoding='utf-8')
copy(CAP/'outputs/allocation/reporting_v2/recomputed_output_view.csv.gz',S/'SD5_reporting_view.csv.gz')
summary_source=ast.parse((CAP/'code/modeling/run_capacity_constrained_derisking_counterfactual.py').read_text(encoding='utf-8-sig'))
nodes=[n for n in summary_source.body if isinstance(n,ast.FunctionDef) and n.name in ['weighted_average','summarize_cases']]
assert len(nodes)==2
summary_namespace=dict(np=np,pd=pd)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_case_summary','exec'),summary_namespace)
cases=pd.read_csv(CAP/'outputs/allocation/all_cases.csv.gz')
for grouping,name in [(['base_year','scenario','scenario_label'],'annual'),(['base_year','metal','scenario','scenario_label'],'mineral_year')]:
    summary_namespace['summarize_cases'](cases,grouping).to_csv(S/f'SD6_origin_ablation_{name}_comparison.csv',index=False)
(S/'SD6_origin_ablation_VALIDATION.json').write_text(json.dumps(reference_audit,indent=2),encoding='utf-8')
for branch in ['legacy','rebuilt']:
    for source in (CAP/f'outputs/allocation_precision/{branch}').glob('*/PRECISION_QA.json'):
        copy(source,S/'SD9_uncertainty/precision_audit'/branch/source.parent.name/source.name)
for name in ['PAIRED_BATCH_QA.json','paired_summary.csv','near_zero_classification_changes.csv']:
    copy(CAP/'outputs/allocation_precision'/name,S/'SD9_uncertainty/precision_audit'/name)
copy(CAP/'outputs/allocation_precision/annual_summaries.csv',S/'SD9_uncertainty/joint_batch_diagnostics/all_allocation_annual_summaries.csv')
bounds=mod('route_bounds',ROOT/'research_process/unified_reanalysis_20260909/allocation_route_bounds.py')
frames=[bounds.calculate(CAP/'outputs/allocation',0,-1.)]
for rho in [0.,.8]:
    for draw in range(1,6):frames.append(bounds.calculate(CAP/f'outputs/allocation_precision/rebuilt/rho_{rho:.1f}_draw_{draw:03d}',draw,rho))
b=pd.concat(frames,ignore_index=True);b=b[b.scenario.isin(['direct_partner','integrated'])]
b.to_csv(S/'SD9_uncertainty/allocation_unmatched_route_bounds/annual_bounds.csv',index=False)
p=b.pivot(index=['draw','rho','base_year'],columns='scenario',values=['transfer_lower_value_share','transfer_upper_value_share','joint_lower_value_share','joint_upper_value_share'])
c=pd.DataFrame(index=p.index)
c['transfer_comparison_margin']=p[('transfer_lower_value_share','direct_partner')]-p[('transfer_upper_value_share','integrated')]
c['joint_comparison_margin']=p[('joint_lower_value_share','integrated')]-p[('joint_upper_value_share','direct_partner')]
c['guaranteed_lower_integrated_transfer']=c.transfer_comparison_margin.gt(0)
c['guaranteed_higher_integrated_joint_reduction']=c.joint_comparison_margin.gt(0)
c.reset_index().to_csv(S/'SD9_uncertainty/allocation_unmatched_route_bounds/objective_comparison_bounds.csv',index=False)
diagnostics=mod('batch_diagnostics',ROOT/'research_process/unified_reanalysis_20260909/batch_diagnostics.py')
a=pd.read_csv(CAP/'outputs/allocation_precision/annual_summaries.csv')
ac=diagnostics.checkpoints(a[a.scenario.ne('baseline')],['rho','base_year','scenario'],['material_joint_value_share','risk_transfer_value_share'],[2,3,5],'allocation')
ac.to_csv(S/'SD9_uncertainty/joint_batch_diagnostics/allocation_cumulative_diagnostics.csv',index=False)
for source in (CAP/'outputs/routes_rebuilt').glob('*'):
    if source.suffix in ['.json','.csv'] and source.name!='sea_geometries.json':copy(source,S/'SD10_provenance/route_rebuild'/source.name)
report=dict(accepted=True,reference_physical_audit=reference_audit['accepted'],fixed_allocation_bound_rows=len(b),paired_year_comparisons=len(c),guaranteed_lower_integrated_transfer=int(c.guaranteed_lower_integrated_transfer.sum()),guaranteed_higher_integrated_joint_reduction=int(c.guaranteed_higher_integrated_joint_reduction.sum()),maximum_latest_allocation_endpoint_change_pp=float(ac[ac.draws.eq(5)].maximum_endpoint_change_pp.max()),scope='Paired numerical reproduction and fixed-allocation outer bounds; not independent observation or adversarial reoptimisation')
(P/'audit/rebuild/SUPPORT_RECONCILIATION_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
