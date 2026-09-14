"""Separate conservation, capacity and environment checks on generated output."""
from pathlib import Path
import json,hashlib,sys,importlib.metadata
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
CAP=HERE/'capsule';OUT=CAP/'outputs/allocation'
cases=pd.read_csv(OUT/'all_cases.csv.gz');flows=pd.read_csv(OUT/'all_flows.csv.gz');solver=pd.read_csv(OUT/'all_solver.csv.gz')
key=['base_year','metal','stage','importer_iso','scenario']
totals=flows.groupby(key).agg(allocated=('scenario_value_usd','sum'),shares=('scenario_share','sum'))
demand=cases.set_index(key).baseline_value_usd
assert totals.index.sort_values().equals(demand.index.sort_values())
demand_error=float(((totals.allocated-demand).abs()/demand).max());share_error=float(abs(totals.shares-1).max())
capacity_keys=['base_year','metal','stage','exporter_iso','scenario']
g=flows.groupby(capacity_keys)
assert g.outside_universe_commitment_usd.nunique().max()==1
assert g.export_envelope_value_usd.nunique().max()==1
allocated=g.scenario_value_usd.sum();outside=g.outside_universe_commitment_usd.first();envelope=g.export_envelope_value_usd.first()
excess=(allocated+outside-envelope).clip(lower=0)
capacity_error=float((excess/np.maximum(envelope,1)).max())
opt=solver[solver.scenario.ne('baseline')]
metrics={c:float(opt[c].max()) for c in ['max_abs_demand_share_error','max_scaled_inequality_violation','max_bound_violation','kkt_stationarity','kkt_complementarity','dual_sign_error']}
paths={n:str(importlib.metadata.distribution(n).locate_file('')) for n in ['numpy','pandas','scipy','clarabel','osqp']}
venv=CAP/'solver_env'
packages_isolated=all(Path(p).resolve().is_relative_to(venv.resolve()) for p in paths.values())
regression=json.loads((OUT/'ALLOCATION_REPRODUCTION_QA.json').read_text())
accepted=bool(regression['accepted'] and demand_error<=2e-6 and share_error<=2e-6 and capacity_error<=2e-7 and opt.success.all() and metrics['max_bound_violation']<=2e-8 and max(metrics[c] for c in ['kkt_stationarity','kkt_complementarity','dual_sign_error'])<2e-7 and packages_isolated)
physical=bool(demand_error<=2e-6 and share_error<=2e-6 and capacity_error<=2e-7 and opt.success.all() and metrics['max_abs_demand_share_error']<=2e-6 and metrics['max_scaled_inequality_violation']<=2e-7 and metrics['max_bound_violation']<=2e-8 and max(metrics[c] for c in ['kkt_stationarity','kkt_complementarity','dual_sign_error'])<2e-7)
report=dict(accepted=accepted,feasibility_and_KKT_checks_passed=physical,demand_relative_error=demand_error,share_sum_error=share_error,export_envelope_relative_excess=capacity_error,max_envelope_excess_usd=float(excess.max()),solver_maxima=metrics,package_locations=paths,all_numerical_packages_in_fresh_venv=packages_isolated,python_executable=sys.executable,reference_regression_passed=regression['accepted'],main_groups=int(solver[solver.scenario.eq('baseline')].shape[0]),ablation_groups=int(solver[solver.scenario.eq('without_origin')].shape[0]),full_pipeline_complete=False)
(OUT/'INDEPENDENT_ALLOCATION_CHECK.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2));assert accepted,report
