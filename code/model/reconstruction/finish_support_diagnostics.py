"""Refresh conditional mineral diagnostics from accepted rebuilt allocations."""
from pathlib import Path
import json, shutil, hashlib
import numpy as np
import pandas as pd
H=Path(__file__).resolve().parent; R=H.parents[1]; C=H/'capsule'; P=R/'submission_package_verified_20260910'; D=P/'supplementary_data/SD9_uncertainty/joint_batch_diagnostics'
qa=json.loads((C/'outputs/allocation_precision/PAIRED_BATCH_QA.json').read_text()); assert qa['accepted']
means=[]; ranks=[]; fingerprints={}
cols=['delta_direct_hhi','delta_origin_hhi','delta_top_chokepoint_share']
for rho in [0.,.8]:
 for draw in range(1,6):
  folder=C/f'outputs/allocation_precision/rebuilt/rho_{rho:.1f}_draw_{draw:03d}'
  cases=pd.read_csv(folder/'all_cases.csv.gz'); b=cases[cases.scenario.eq('baseline')]
  rank=b.groupby('metal',as_index=False).agg(years=('base_year','nunique'),value_usd=('baseline_value_usd','sum'))
  rank=rank[rank.years.eq(13)].sort_values('value_usd',ascending=False); rank['rank']=np.arange(1,len(rank)+1);rank['draw']=draw;rank['rho']=rho;ranks.append(rank)
  x=cases[cases.scenario.ne('baseline')].copy()
  for col in cols:x[col+'_weighted']=x[col]*x.baseline_value_usd
  y=x.groupby(['metal','base_year','scenario'],as_index=False)[['baseline_value_usd']+[c+'_weighted' for c in cols]].sum()
  for col in cols:y[col]=y[col+'_weighted']/y.baseline_value_usd
  m=y.groupby(['metal','scenario'],as_index=False)[cols].mean();m['draw']=draw;m['rho']=rho;means.append(m)
  fingerprints[folder.name]=hashlib.sha256((folder/'PRECISION_QA.json').read_bytes()).hexdigest()
mm=pd.concat(means,ignore_index=True);mm.to_csv(D/'mineral_equal_year_tradeoff_means.csv',index=False)
pd.concat(ranks,ignore_index=True).to_csv(D/'mineral_complete_coverage_value_ranks.csv',index=False)
b=pd.read_csv(C/'outputs/allocation/all_cases.csv.gz');b=b[b.scenario.eq('baseline')]
ref=b.groupby('metal',as_index=False).agg(years=('base_year','nunique'),value_usd=('baseline_value_usd','sum'))
ref=ref[ref.years.eq(13)].sort_values('value_usd',ascending=False);ref['rank']=np.arange(1,len(ref)+1);ref.to_csv(D/'reference_mineral_value_ranks.csv',index=False)
p=mm.pivot(index=['draw','rho','metal'],columns='scenario',values=cols);delta=pd.DataFrame(index=p.index)
for col in cols:delta[col+'_integrated_minus_direct']=p[(col,'integrated')]-p[(col,'direct_partner')]
delta['maritime_improves_relative_to_direct']=delta.delta_top_chokepoint_share_integrated_minus_direct.lt(0)
delta['supplier_reduction_smaller']=delta.delta_direct_hhi_integrated_minus_direct.gt(0)
delta['origin_reduction_smaller']=delta.delta_origin_hhi_integrated_minus_direct.gt(0)
delta=delta.reset_index();delta['reference_top_four']=delta.metal.isin(ref.head(4).metal);delta.to_csv(D/'mineral_paired_tradeoff_checks.csv',index=False)
info=json.loads((D/'SUMMARY.json').read_text());info['completion_record_sha256']=fingerprints
info['allocation_actual_optimisations']=qa['optimisations_per_branch'];info['allocation_validation']='High-precision original-versus-rebuilt paired inputs; original lower-precision failures retained in precision_audit.'
info['code_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ac=pd.read_csv(D/'allocation_cumulative_diagnostics.csv');latest=ac[ac.draws.eq(5)]
info['allocation_checkpoints']['latest_maximum_endpoint_change_pp']=float(latest.maximum_endpoint_change_pp.max())
(D/'SUMMARY.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
dest=P/'supplementary_data/SD9_uncertainty/precision_audit/original_lower_precision';dest.mkdir(parents=True,exist_ok=True)
shutil.copy2(C/'outputs/allocation_conditional/ALLOCATION_CONDITIONAL_BATCH_QA.json',dest/'ALLOCATION_CONDITIONAL_BATCH_QA.json')
print('Conditional mineral diagnostics synchronized:',len(mm),len(delta))
