"""Independent geometry, numerical, provenance and delivery checks."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

from build_routes import OUT, ROOT, OLD, SCOPE, prepare, sha
from run_route_reanalysis import CONFIGS, GEOMETRIC, COVERAGE
from audit_geometric_incidence import robust_labels
import validate_and_compare as prior_validation
import numpy as np
import pandas as pd


def clips_rectangle(x0,y0,x1,y1,xmin,ymin,xmax,ymax):
    """Liang-Barsky clipping, independent of the Shapely intersection routine."""
    dx,dy=x1-x0,y1-y0
    lower,upper=0.,1.
    for p,q in [(-dx,x0-xmin),(dx,xmax-x0),(-dy,y0-ymin),(dy,ymax-y0)]:
        if abs(p)<1e-14:
            if q<0:
                return False
        elif p<0:
            lower=max(lower,q/p)
        else:
            upper=min(upper,q/p)
        if lower>upper+1e-12:
            return False
    return True


def independent_labels(vertices,boxes):
    labels=[]
    for c in boxes:
        hit=False
        for a,b in zip(vertices[:-1],vertices[1:]):
            x0,y0=float(a[0]),float(a[1])
            x1=x0+(float(b[0])-x0+180)%360-180
            y1=float(b[1])
            lo,hi=sorted((x0,x1))
            for k in range(math.ceil((lo-c.xmax)/360),math.floor((hi-c.xmin)/360)+1):
                if clips_rectangle(x0,y0,x1,y1,c.xmin+360*k,c.ymin,c.xmax+360*k,c.ymax):
                    hit=True
                    break
            if hit:
                break
        if hit:
            labels.append(c.label)
    return labels


def geometry_checks():
    rt,template,_=prepare()
    boxes=rt.load_chokepoints(template/'inputs/chokepoint_boxes.csv')
    paths=json.loads((OUT/'all_nickel_route_geometries.json').read_text())
    audit=pd.read_csv(OUT/'geometric_incidence_audit.csv').fillna({'old_labels':'','segment_labels':'','regenerated_legacy_labels':''})
    expected=audit.set_index(['origin','dest']).segment_labels
    for key,vertices in paths.items():
        actual='|'.join(independent_labels(vertices,boxes))
        assert actual==expected.loc[tuple(key.split('|'))],key
    assert 'Panama' in independent_labels([[280.5,8.6],[280.,9.75]],boxes)
    assert 'Panama' not in independent_labels([[179.,9.],[-179.,9.]],boxes)
    assert 'Gibraltar' in independent_labels([[-7.,36.],[-4.,36.]],boxes)
    assert 'Gibraltar' not in rt.chokepoint_labels([[-7.,36.],[-4.,36.]],boxes)
    for key in ['NCL|NLD','CUB|CAN','PNG|NLD']:
        points=paths[key]
        target=independent_labels(points,boxes)
        assert target==independent_labels(list(reversed(points)),boxes)
        for shift in [-720,-360,360,720]:
            assert target==independent_labels([[x+shift,y] for x,y in points],boxes)
    old=pd.read_csv(OLD).fillna({'chokepoints':''})
    coverage=pd.read_csv(COVERAGE).fillna({'chokepoints':''})
    corrected=pd.read_csv(GEOMETRIC).fillna({'chokepoints':''})
    keys=['metal','origin','dest']
    baseline=old.set_index(keys).sort_index()
    aligned=coverage.set_index(keys).reindex(baseline.index)
    pd.testing.assert_frame_equal(baseline,aligned,check_dtype=False)
    pd.testing.assert_frame_equal(old[old.metal.ne('Nickel')].reset_index(drop=True),corrected[corrected.metal.ne('Nickel')].reset_index(drop=True))
    collapsed=audit[audit.geometry_status.eq('unresolved_collapsed_geometry')]
    known=set(zip(corrected[corrected.metal.eq('Nickel')].origin,corrected[corrected.metal.eq('Nickel')].dest))
    assert not(set(zip(collapsed.origin,collapsed.dest))&known)
    manifest=json.loads((OUT/'geometric_incidence_manifest.json').read_text())
    assert manifest['regenerated_legacy_disagreements']==0
    assert not coverage.duplicated(keys).any() and not corrected.duplicated(keys).any()
    # The detector is shared by all minerals. Record observed spillovers from the
    # audited pair set without claiming to have audited every other country pair.
    changed=audit[audit.any_incidence_change][['origin','dest','old_labels','segment_labels']]
    spill=old[old.metal.ne('Nickel')].merge(changed,on=['origin','dest'])
    assert spill.chokepoints.eq(spill.old_labels).all()
    spill.to_csv(OUT/'other_minerals_same_detector_bug_audited_pairs.csv',index=False)
    spill.groupby('metal').size().rename('affected_template_rows').reset_index().to_csv(OUT/'other_minerals_same_detector_bug_summary.csv',index=False)
    return dict(independent_geometries_checked=len(paths),longitude_shift_reverse_and_segment_tests='PASS',old_coverage_control_unchanged=True,
        non_nickel_corrected_output_unchanged=True,collapsed_routes_removed=len(collapsed),other_minerals_affected_in_audited_subset=int(spill.metal.nunique()),other_mineral_affected_rows=len(spill))


def numerical_checks():
    manifest=json.loads((OUT/'analysis_manifest.json').read_text())
    assert set(manifest['variants'])==set(CONFIGS)
    results,summaries,historical,thresholds=[],[],[],[]
    cases_by={}
    for name,config in manifest['variants'].items():
        folder=OUT/name
        source=config['origin_source_variant']
        cases=pd.read_csv(folder/'allocation/all_cases.csv.gz')
        alloc=pd.read_csv(folder/'allocation/all_allocations.csv.gz')
        fps=pd.read_csv(SCOPE/source/'exporter_origin_profiles.csv.gz')
        audits=pd.read_csv(folder/'allocation/all_solver_audits.csv')
        assert audits.success.all() and cases.solver_feasible.all()
        for col in ['kkt_stationarity','dual_sign_error','kkt_complementarity']:
            assert audits[audits.scenario.ne('baseline')][col].max()<2e-7
        assert audits.max_abs_demand_share_error.max()<2e-6
        assert audits.max_scaled_inequality_violation.max()<2e-7
        assert audits.max_bound_violation.max()<2e-8
        assert sorted(cases.base_year.unique())==config['years']
        assert not cases.duplicated(['base_year','stage','importer_iso','scenario']).any()
        index=['base_year','stage','importer_iso']
        baseline=cases[cases.scenario.eq('baseline')].set_index(index)
        for metric in ['direct_hhi','origin_hhi','top_chokepoint_share']:
            reference=cases[index].merge(baseline[[metric]],left_on=index,right_index=True,how='left')[metric].to_numpy()
            assert np.allclose(cases[metric]-reference,cases['delta_'+metric],rtol=0,atol=2e-8)
        transfer=(cases.delta_direct_hhi<=-.025)&((cases.delta_origin_hhi>=.025)|(cases.delta_top_chokepoint_share>=.025))
        joint=(cases.delta_direct_hhi<=-.025)&(cases.delta_origin_hhi<=-.025)&(cases.delta_top_chokepoint_share<=-.025)
        assert np.array_equal(transfer,cases.risk_transfer) and np.array_equal(joint,cases.material_joint_reduction)
        objectives=cases[cases.scenario.ne('baseline')].pivot(index=index,columns='scenario',values='objective_weight_direct')
        assert np.allclose(objectives.direct_partner,objectives.integrated,rtol=0,atol=0)
        for (year,scenario),sub in cases[cases.scenario.ne('baseline')].groupby(['base_year','scenario']):
            weights=sub.baseline_value_usd
            for threshold in [.01,.025,.05]:
                transfer=(sub.delta_direct_hhi<=-threshold)&((sub.delta_origin_hhi>=threshold)|(sub.delta_top_chokepoint_share>=threshold))
                joint=(sub.delta_direct_hhi<=-threshold)&(sub.delta_origin_hhi<=-threshold)&(sub.delta_top_chokepoint_share<=-threshold)
                thresholds.append(dict(variant=name,year=year,scenario=scenario,threshold=threshold,
                    risk_transfer_value_share=float(np.average(transfer,weights=weights)),material_joint_value_share=float(np.average(joint,weights=weights))))
        route=OUT/config['route_file']
        assert sha(route)==config['route_sha256']
        lanes=pd.read_csv(route).query('metal == "Nickel"')
        reports=[]
        if config['material_stage_port_alternative']:
            alternate=pd.read_csv(OUT/'material_stage_halifax_endpoint_routes.csv').query('metal == "Nickel"')
            for stage in sorted(cases.stage.unique()):
                reports.append(prior_validation.verify_allocations(cases[cases.stage.eq(stage)],alloc[alloc.stage.eq(stage)],fps,alternate if stage=='material' else lanes))
        else:
            reports.append(prior_validation.verify_allocations(cases,alloc,fps,lanes))
        results.append(dict(variant=name,optimized_group_scenarios=int(audits.scenario.ne('baseline').sum()),case_rows=len(cases),
            max_independent_metric_error=max(x['max_metric_error'] for x in reports),max_relative_demand_error=max(x['max_relative_demand_error'] for x in reports),
            max_export_envelope_excess_usd=max(x['max_export_envelope_excess_usd'] for x in reports),
            max_kkt_stationarity=float(audits.kkt_stationarity.max()),max_kkt_complementarity=float(audits.kkt_complementarity.max()),max_dual_sign_error=float(audits.dual_sign_error.max())))
        cases_by[name]=cases
        for filename in ['annual_summary.csv','annual_stage_summary.csv']:
            s=pd.read_csv(folder/'allocation'/filename)
            if 'stage' not in s:
                s['stage']='all'
            s.insert(0,'variant',name)
            summaries.append(s)
        h=pd.read_csv(folder/'historical_three_indicator_summary.csv')
        h.insert(0,'variant',name)
        historical.append(h)
        print(name,json.dumps(results[-1]),flush=True)
    # Evaluation-only control retains exactly the same supplier decisions.
    e=OUT/'expanded6_fixed_allocations/allocation'
    eval_cases=pd.read_csv(e/'all_cases.csv.gz')
    eval_alloc=pd.read_csv(e/'all_allocations.csv.gz')
    prior=pd.read_csv(SCOPE/'expanded6/allocation/all_allocations.csv.gz')
    pd.testing.assert_frame_equal(eval_alloc,prior)
    prior_validation.verify_allocations(eval_cases,eval_alloc,pd.read_csv(SCOPE/'expanded6/exporter_origin_profiles.csv.gz'),pd.read_csv(GEOMETRIC).query('metal == "Nickel"'))
    key=['base_year','stage','importer_iso','scenario']
    frozen=cases_by['expanded6_fixed_cohort'].set_index(key).sort_index()
    oldcases=pd.read_csv(SCOPE/'expanded6/allocation/all_cases.csv.gz').set_index(key).sort_index()
    assert frozen.index.equals(oldcases.index)
    assert np.allclose(frozen.baseline_value_usd,oldcases.baseline_value_usd,rtol=0,atol=1e-5)
    # A port scenario cannot repair missing national trade observations. Keep
    # record presence separate from a physical-flow zero, and expose the local
    # effect that can disappear inside an all-mineral/stage average.
    trade=pd.read_csv(SCOPE/'expanded6/trade_corridors.csv.gz')
    port_rows=[]
    for year in [2019,2022,2024]:
        flow=trade[trade.year.eq(year)&trade.stage.eq('material')&trade.exporter_iso.eq('CUB')&trade.importer_iso.eq('CAN')]
        for name in ['expanded6_extended','expanded6_halifax_endpoint']:
            c=cases_by[name]
            sub=c[c.base_year.eq(year)&c.stage.eq('material')&c.importer_iso.eq('CAN')]
            assert len(sub)==3
            for row in sub.itertuples():
                port_rows.append(dict(variant=name,year=year,scenario=row.scenario,
                    national_corridor_positive_record_present=bool(len(flow)),
                    national_corridor_recorded_value_usd=float(flow.reconstructed_value_usd.sum()) if len(flow) else None,
                    importer_stage_value_usd=row.baseline_value_usd,
                    top_chokepoint_share=row.top_chokepoint_share,route_pressure=row.route_pressure,
                    source_boundary='BACI record presence only; absence is not a verified zero physical shipment'))
    pd.DataFrame(port_rows).to_csv(OUT/'halifax_endpoint_local_and_record_presence.csv',index=False)
    for filename in ['annual_summary.csv','annual_stage_summary.csv']:
        s=pd.read_csv(e/filename)
        if 'stage' not in s:
            s['stage']='all'
        s.insert(0,'variant','expanded6_fixed_allocations')
        summaries.append(s)
    for name in ['original3','expanded6']:
        for filename in ['annual_summary.csv','annual_stage_summary.csv']:
            s=pd.read_csv(SCOPE/name/'allocation'/filename)
            if 'stage' not in s:
                s['stage']='all'
            s.insert(0,'variant',name+'_old_routes')
            summaries.append(s)
    pd.concat(summaries,ignore_index=True).to_csv(OUT/'allocation_route_comparison.csv',index=False)
    pd.concat(historical,ignore_index=True).to_csv(OUT/'historical_route_comparison.csv',index=False)
    pd.DataFrame(results).to_csv(OUT/'independent_numeric_checks.csv',index=False)
    pd.DataFrame(thresholds).to_csv(OUT/'classification_threshold_sensitivity.csv',index=False)
    return dict(optimized_group_scenarios=sum(x['optimized_group_scenarios'] for x in results),variants=len(results),
        fixed_cohort_unit_years=len(frozen)//3,old_decisions_re_evaluated_without_optimization=True,
        port_record_presence_audit_rows=len(port_rows),
        max_metric_error=max(x['max_independent_metric_error'] for x in results),max_envelope_excess_usd=max(x['max_export_envelope_excess_usd'] for x in results))


def main():
    geometry=geometry_checks()
    numeric=numerical_checks()
    sources=json.loads((OUT.parent/'external/source_manifest.json').read_text())
    for row in sources:
        assert sha(OUT.parent/'external'/row['path'])==row['sha256']
    package=ROOT/'manuscript_package_20260907_final'
    inventory=pd.read_csv(package/'PACKAGE_SHA256.csv')
    for row in inventory.itertuples():
        assert sha(package/row.file)==row.sha256
    archive_sha=sha(ROOT/'manuscript_package_20260907_final.zip')
    assert archive_sha=='0226e5422850f2b2adab605b47fc8e35d7b41d19a6899ba647e5696d96b84019'
    global_manifest=json.loads((OUT/'global_detector_audit/run_manifest.json').read_text())
    assert global_manifest['regenerated_legacy_disagreements']==0
    assert global_manifest['unit_years']==68164 and global_manifest['comparison_windows']==111480
    report=dict(status='PASS',geometry=geometry,numeric=numeric,operator_sources_verified=len(sources),unchanged_submission_files=len(inventory),
        unchanged_submission_archive_sha256=archive_sha,
        global_existing_support_detector_audit=global_manifest,
        interpretation='Static geometry and conditional allocations only. No independent validation of national mine origins or annual vessel paths.')
    (OUT/'independent_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    files=[p for p in OUT.parent.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='OUTPUT_SHA256.csv']
    pd.DataFrame([dict(path=str(p.relative_to(OUT.parent)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]).to_csv(OUT/'OUTPUT_SHA256.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
