"""Same-source nickel scope comparison; writes isolated candidate outputs only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for folder in ('scipy_scope_vendor', 'qp_vendor', 'osqp_scope_vendor'):
    sys.path.insert(0, str(ROOT / 'research_process/runtime_support' / folder))

import numpy as np
import pandas as pd
import scipy
import osqp
from scipy.sparse import csc_matrix, eye, lil_matrix, vstack

ARCHIVE = ROOT / 'manuscript_package_20260907_final/Reproduction_Code_and_Derived_Inputs.zip'
PRODUCTS = ROOT / 'research_process/evidence_extension_20260908/external/baci_nickel_intermediates/baci_nickel_six_products_2007_2024_prepared.csv.gz'
IP = 'figure_source_tables/fig5_global_coupled_2012_2024/input_build/'
LANE_MEMBER = 'figure_source_tables/longitudinal_2012_2024/input_snapshot/minerals_lanes.csv'
OUT = HERE / 'results'
KEYS = ['importer_iso', 'metal', 'stage']
GROUP = KEYS + ['year']
EDGE = ['metal', 'stage', 'exporter_iso', 'importer_iso', 'year']
STAGE = {260400: 'ore', 282540: 'compound', 750210: 'metal',
         750110: 'material', 750120: 'material', 283324: 'compound'}
ORIGINAL = [260400, 282540, 750210]
CONFIGS = {
    'original3': (ORIGINAL, ['ore', 'compound', 'metal'], .65),
    'matte4': (ORIGINAL + [750110], ['ore', 'material', 'compound', 'metal'], .65),
    'expanded6': (list(STAGE), ['ore', 'material', 'compound', 'metal'], .65),
    'expanded6_reverse': (list(STAGE), ['ore', 'material', 'metal', 'compound'], .65),
    'expanded6_w0': (list(STAGE), ['ore', 'material', 'compound', 'metal'], 0.),
    'expanded6_w1': (list(STAGE), ['ore', 'material', 'compound', 'metal'], 1.),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare():
    """Extract only fixed, small source members, preserving their relative hierarchy."""
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = HERE / 'frozen'
    members = [
        'reproduction/project_snapshot/modeling/global_coupled_counterfactual/core.py',
        'reproduction/project_snapshot/modeling/global_coupled_counterfactual/rolling_support.py',
        'reproduction/project_snapshot/modeling/run_capacity_constrained_derisking_counterfactual.py',
        'reproduction/project_snapshot/modeling/infer_mine_origin_flows.py',
        'reproduction/project_snapshot/longitudinal_2012_2024/analysis/longitudinal_mrci.py',
        LANE_MEMBER,
    ]
    hashes = {}
    with zipfile.ZipFile(ARCHIVE) as z:
        for member in members:
            data = z.read(member)
            dest = (frozen / member).resolve()
            assert dest.is_relative_to(frozen.resolve())
            assert len(data) < 10_000_000
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert dest.read_bytes() == data, f'Frozen source modified: {dest}'
            else:
                dest.write_bytes(data)
            hashes[member] = hashlib.sha256(data).hexdigest()
        seed = pd.read_csv(z.open(IP + 'mine_origin_seed_bgs_wmd_2008_2024.csv.gz'), compression='gzip')
        old_trade = pd.read_csv(z.open(IP + 'baci_hs07_trade_panel_2007_2024.csv.gz'), compression='gzip')
        old_annual = pd.read_csv(z.open(IP + 'risk_indicator_panel_baci_2008_2024.csv.gz'), compression='gzip')
    infer = load_module('scope_infer', frozen / members[3])
    lm = load_module('scope_lm', frozen / members[4])
    sys.path.insert(0, str(frozen / 'reproduction/project_snapshot/modeling/global_coupled_counterfactual'))
    import core
    core.local.FIRST_DATA_YEAR = 2007
    core.local.FIRST_IDENTIFIABLE_BASE_YEAR = 2012
    core.local.LANES = frozen / LANE_MEMBER
    (OUT / 'frozen_source_hashes.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    return infer, lm, core, seed[seed.metal.eq('Nickel')].copy(), old_trade[old_trade.metal.eq('Nickel')].copy(), old_annual[old_annual.metal.eq('Nickel')].copy()


def trade_inputs(raw, codes):
    valid = raw.hs07_code.isin(codes) & raw.trade_value_usd.gt(0)
    valid &= raw.exporter_iso.str.fullmatch('[A-Z]{3}').fillna(False)
    valid &= raw.importer_iso.str.fullmatch('[A-Z]{3}').fillna(False)
    valid &= raw.exporter_iso.ne(raw.importer_iso)
    sub = raw.loc[valid].copy()
    sub['metal'] = 'Nickel'
    sub['stage'] = sub.hs07_code.map(STAGE)
    sub = sub.rename(columns={'trade_value_usd': 'reconstructed_value_usd'})
    return sub.groupby(EDGE, as_index=False).reconstructed_value_usd.sum().sort_values(EDGE).reset_index(drop=True), int(valid.sum())


def infer_inputs(trade, seed, order, weight, infer, lm, lanes):
    """Use the frozen stage recursion, aggregating each stage before releasing memory."""
    infer.LOCAL_STAGE_WEIGHT['material'] = weight
    production = infer.build_production_mix(seed)
    origin_parts, fingerprint_rows, mass_audits = [], [], []
    for year in range(2008, 2025):
        prod = production[production.year.eq(year)]
        world = dict(zip(prod.inferred_mine_origin_iso, prod.origin_share))
        assert world and abs(sum(world.values()) - 1) < 1e-12
        mix = {origin: {origin: 1.} for origin in world}
        for stage in order:
            flows = trade[trade.year.eq(year) & trade.stage.eq(stage)]
            if flows.empty:
                continue
            inferred, mix = infer.infer_year_stage(flows, mix, world, stage)
            expected = flows.set_index(['exporter_iso', 'importer_iso']).reconstructed_value_usd
            actual = inferred.groupby(['exporter_iso', 'importer_iso']).attributed_value_usd.sum().reindex(expected.index)
            error = float(((actual - expected).abs() / np.maximum(expected, 1.)).max())
            assert error < 1e-12 and inferred.attributed_value_usd.gt(0).all()
            mass_audits.append(dict(year=year, stage=stage, corridors=len(flows), max_relative_value_error=error))
            origin_parts.append(inferred.groupby(GROUP + ['inferred_mine_origin_iso'], as_index=False).attributed_value_usd.sum())
            fp = inferred.groupby(['year', 'metal', 'stage', 'exporter_iso', 'inferred_mine_origin_iso'], as_index=False).attributed_value_usd.sum()
            fp['share'] = fp.attributed_value_usd / fp.groupby(['year', 'metal', 'stage', 'exporter_iso']).attributed_value_usd.transform('sum')
            fingerprint_rows.append(fp.drop(columns='attributed_value_usd'))
    origin = pd.concat(origin_parts, ignore_index=True)
    direct = lm.concentration_stats(trade[trade.year.ge(2008)], 'exporter_iso', 'reconstructed_value_usd', 'direct')
    annual = direct.merge(lm.concentration_stats(origin, 'inferred_mine_origin_iso', 'attributed_value_usd', 'origin'), on=GROUP, how='left', validate='one_to_one')
    routes = lm.route_stats(trade[trade.year.ge(2008)], direct, lanes)
    annual = annual.merge(routes, on=GROUP, how='left', validate='one_to_one')
    annual['origin_coverage_ratio'] = annual.origin_total_value_usd / annual.direct_total_value_usd
    assert not annual.duplicated(GROUP).any()
    assert np.isfinite(annual[['direct_hhi', 'origin_hhi', 'top_chokepoint_share']]).all().all()
    fps = pd.concat(fingerprint_rows, ignore_index=True)
    return annual, fps, pd.DataFrame(mass_audits)


def compare_original(trade, annual, old_trade, old_annual):
    a = trade.set_index(EDGE).reconstructed_value_usd.sort_index()
    b = old_trade.set_index(EDGE).reconstructed_value_usd.sort_index()
    assert a.index.equals(b.index)
    e = float(((a-b).abs() / np.maximum(b, 1.)).max())
    assert e < 1e-12
    a = annual.set_index(GROUP).sort_index()
    b = old_annual.set_index(GROUP).sort_index()
    assert a.index.equals(b.index)
    errors = {}
    for col in ('direct_total_value_usd', 'direct_hhi', 'origin_total_value_usd', 'origin_hhi', 'origin_coverage_ratio', 'route_coverage_share', 'top_chokepoint_share'):
        errors[col] = float(((a[col]-b[col]).abs() / np.maximum(b[col].abs(), 1.)).max())
    assert max(errors[c] for c in errors if c not in ('route_coverage_share', 'top_chokepoint_share')) < 1e-12
    return dict(corridor_keys=len(a), trade_corridors=len(trade), max_trade_relative_error=e, indicator_max_normalized_errors=errors)


def historical(annual):
    results, rows = [], []
    panel = annual[annual.year.ge(2012)].copy()
    for lag in (1, 4):
        start = panel.rename(columns={c: c+'_start' for c in panel if c not in KEYS})
        start['year'] = start.year_start + lag
        windows = panel.merge(start, on=KEYS + ['year'], validate='one_to_one')
        windows['lag_years'] = lag
        elig = windows.direct_total_value_usd.ge(1e6) & windows.direct_total_value_usd_start.ge(1e6)
        for col, threshold in [('route_coverage_share', .5), ('origin_coverage_ratio', .8)]:
            elig &= windows[col].ge(threshold) & windows[col+'_start'].ge(threshold)
        windows['eligible'] = elig
        for col in ('direct_hhi', 'origin_hhi', 'top_chokepoint_share'):
            windows['delta_'+col] = windows[col] - windows[col+'_start']
        decon = elig & windows.delta_direct_hhi.le(-.025)
        windows['direct_deconcentration'] = decon
        windows['three_indicator_transfer'] = decon & (windows.delta_origin_hhi.ge(.025) | windows.delta_top_chokepoint_share.ge(.025))
        windows['three_indicator_joint_reduction'] = decon & windows.delta_origin_hhi.le(-.025) & windows.delta_top_chokepoint_share.le(-.025)
        rows.append(windows)
        for stage in ['all'] + sorted(panel.stage.unique()):
            sub = windows if stage == 'all' else windows[windows.stage.eq(stage)]
            selected = sub[sub.direct_deconcentration]
            weights = selected.direct_total_value_usd
            denom = weights.sum()
            results.append(dict(lag_years=lag, stage=stage, eligible_windows=int(sub.eligible.sum()), deconcentration_windows=len(selected), deconcentration_end_value_usd=float(denom),
                transfer_value_share=float((weights*selected.three_indicator_transfer).sum()/denom) if denom else np.nan,
                three_indicator_joint_reduction_value_share=float((weights*selected.three_indicator_joint_reduction).sum()/denom) if denom else np.nan))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(results)


def fingerprints_dict(frame):
    return {int(year): {(str(metal), str(stage), str(exporter)): dict(zip(sub.inferred_mine_origin_iso, sub.share))
                       for (metal, stage, exporter), sub in annual.groupby(['metal', 'stage', 'exporter_iso'])}
            for year, annual in frame.groupby('year')}


def matched_quadratic(core, model, objective):
    if objective != 'matched_direct':
        return core.quadratic_form(model, objective)
    h = lil_matrix((model.variable_count, model.variable_count), dtype=float)
    for unit in model.units:
        factor = 2 * unit.group_weight * core.integrated_weights_for_unit(unit)[0] / max(float(unit.baseline_metrics['direct_hhi']), 1e-9)
        for idx in unit.variable_indices:
            h[idx, idx] = factor
    return h.tocsr(), np.zeros(model.variable_count)


def solve(core, model, objective):
    if model.variable_count == 0:
        return model.start.copy(), dict(success=True, message='No adjustable variables', iterations=0, kkt_stationarity=0., **core.constraint_violations(model, model.start))
    h, c = matched_quadratic(core, model, objective)
    a = vstack([model.equality.A, model.inequality.A, eye(model.variable_count)], format='csc')
    lower = np.concatenate([model.equality.lb, model.inequality.lb, model.lower])
    upper = np.concatenate([model.equality.ub, model.inequality.ub, model.upper])
    attempts = []
    # Retry conditioning/iteration settings only; objective, constraints and
    # acceptance tolerances remain unchanged. Preserve every attempted solve.
    settings = [{}, {'max_iter': 2000000, 'scaling': 50, 'adaptive_rho_interval': 50},
                {'max_iter': 2000000, 'scaling': 50, 'rho': 1., 'adaptive_rho_interval': 100}]
    for extra in settings:
        options = dict(verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=200000, polishing=True, adaptive_rho=True, check_termination=25)
        options.update(extra)
        solver = osqp.OSQP()
        solver.setup(P=csc_matrix(h), q=c, A=a, l=lower, u=upper, **options)
        solver.warm_start(x=model.start)
        result = solver.solve()
        vector = np.clip(np.asarray(result.x), model.lower, model.upper)
        audit = core.constraint_violations(model, vector)
        kkt = float(np.max(np.abs(h@result.x+c+a.T@result.y)))
        activity = a @ result.x
        dual = np.asarray(result.y)
        finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
        positive, negative = np.maximum(dual, 0.), np.maximum(-dual, 0.)
        dual_sign_error = max(float(positive[~finite_upper].max(initial=0.)), float(negative[~finite_lower].max(initial=0.)))
        complementarity = max(float(np.abs(positive[finite_upper]*(upper[finite_upper]-activity[finite_upper])).max(initial=0.)),
                              float(np.abs(negative[finite_lower]*(activity[finite_lower]-lower[finite_lower])).max(initial=0.)))
        good = result.info.status_val == 1 and audit['max_abs_demand_share_error'] <= 2e-6 and audit['max_scaled_inequality_violation'] <= 2e-7 and audit['max_bound_violation'] <= 2e-8 and max(kkt, dual_sign_error, complementarity) < 2e-7
        attempts.append(dict(options=extra, status=result.info.status, iterations=result.info.iter, kkt_stationarity=kkt, dual_sign_error=dual_sign_error, kkt_complementarity=complementarity, **audit))
        if good:
            break
    return vector, dict(success=bool(good), message=result.info.status, iterations=result.info.iter, solver='OSQP '+osqp.__version__, attempts_json=json.dumps(attempts), primal_residual=result.info.prim_res, dual_residual=result.info.dual_res, kkt_stationarity=kkt, dual_sign_error=dual_sign_error, kkt_complementarity=complementarity, objective_value=float(.5*vector@(h@vector)+c@vector), **audit)


def run_allocation(name, trade, annual, fps, years, core, resume):
    dest = OUT / name / 'allocation'
    dest.mkdir(parents=True, exist_ok=True)
    fingerprints = fingerprints_dict(fps)
    routes = core.local.load_routes()
    inventories = []
    for year in years:
        units = core.local.rolling_universe(annual, year)
        inventories.append(units)
        baseline, capacity, pools = core.local.year_trade_components(trade, year)
        for (_, stage), group in units.groupby(['metal', 'stage'], observed=True, sort=True):
            token = f'{year}_{stage}'
            marker = dest / f'{token}_complete.json'
            if resume and marker.exists():
                continue
            model = core.build_group_model(year, group, baseline, capacity, pools, fingerprints[year], routes)
            print(f'{name}/{token}: {len(model.units)} units; {model.variable_count} variables', flush=True)
            cases, allocations, audits = [], [], []
            cap = model.exporter_capacity.set_index('exporter_iso')
            for scenario, objective in [('baseline', None), ('direct_partner', 'matched_direct'), ('integrated', 'integrated')]:
                start = time.monotonic()
                if objective is None:
                    vector = model.start.copy()
                    audit = dict(success=True, message='Observed baseline', iterations=0, **core.constraint_violations(model, vector))
                else:
                    vector, audit = solve(core, model, objective)
                    if audit['success']:
                        vector = core.set_route_epigraph_to_realized(model, vector)
                audits.append(dict(base_year=year, stage=stage, scenario=scenario, units=len(model.units), variables=model.variable_count, elapsed_seconds=time.monotonic()-start, **audit))
                if not audit['success']:
                    pd.DataFrame(audits).to_csv(dest / f'{token}_failed_solver.csv', index=False)
                    raise RuntimeError(f'Unverified optimizer {name}/{token}/{scenario}: {audit}')
                for unit in model.units:
                    shares = unit.baseline_shares.copy()
                    shares[unit.active_exporter_indices] = vector[unit.variable_indices]
                    case = core.case_from_shares(unit, shares, year, scenario, True, str(audit['message']), int(audit['iterations']))
                    if scenario == 'direct_partner':
                        case['objective'] = 'matched_normalized_direct'
                        case['objective_weight_direct'] = core.integrated_weights_for_unit(unit)[0]
                        case['scenario_label'] = 'Baseline-normalized direct-only globally coupled allocation'
                    cases.append(case)
                    for j, exporter in enumerate(unit.context.exporters):
                        if shares[j] <= 0 and unit.context.baseline_value[j] <= 0:
                            continue
                        c = cap.loc[exporter]
                        allocations.append(dict(base_year=year, metal='Nickel', stage=stage, importer_iso=unit.context.importer_iso, scenario=scenario, exporter_iso=exporter,
                            baseline_value_usd=float(unit.context.baseline_value[j]), scenario_value_usd=float(shares[j]*unit.demand), scenario_share=float(shares[j]),
                            current_global_value_usd=float(c.current_global_value_usd), outside_universe_commitment_usd=float(c.outside_universe_commitment_usd), export_envelope_value_usd=float(c.envelope_value_usd)))
            pd.DataFrame(cases).to_csv(dest / f'{token}_cases.csv.gz', index=False)
            pd.DataFrame(allocations).to_csv(dest / f'{token}_allocations.csv.gz', index=False)
            pd.DataFrame(audits).to_csv(dest / f'{token}_solver.csv', index=False)
            marker.write_text(json.dumps(dict(variant=name, year=year, stage=stage, scenarios=3)), encoding='utf-8')
    pd.concat(inventories, ignore_index=True).to_csv(dest / 'cohort_inventory.csv', index=False)
    cases = pd.concat([pd.read_csv(p) for p in sorted(dest.glob('*_cases.csv.gz')) if not p.name.startswith('all_')], ignore_index=True)
    allocations = pd.concat([pd.read_csv(p) for p in sorted(dest.glob('*_allocations.csv.gz')) if not p.name.startswith('all_')], ignore_index=True)
    audits = pd.concat([pd.read_csv(p) for p in sorted(dest.glob('*_solver.csv')) if not p.name.startswith('all_') and '_failed_' not in p.name], ignore_index=True)
    assert not cases.duplicated(['base_year', 'metal', 'stage', 'importer_iso', 'scenario']).any()
    cases.to_csv(dest / 'all_cases.csv.gz', index=False)
    allocations.to_csv(dest / 'all_allocations.csv.gz', index=False)
    audits.to_csv(dest / 'all_solver_audits.csv', index=False)
    core.CF.summarize_cases(cases, ['base_year', 'scenario', 'scenario_label']).to_csv(dest / 'annual_summary.csv', index=False)
    core.CF.summarize_cases(cases, ['base_year', 'stage', 'scenario', 'scenario_label']).to_csv(dest / 'annual_stage_summary.csv', index=False)
    core.global_capacity_audit(allocations).to_csv(dest / 'capacity_audit.csv', index=False)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', default=','.join(CONFIGS))
    parser.add_argument('--skip-allocation', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    variants = args.variants.split(',')
    infer, lm, core, seed, old_trade, old_annual = prepare()
    raw = pd.read_csv(PRODUCTS)
    checks, configs = {}, {}
    for name in variants:
        codes, order, weight = CONFIGS[name]
        dest = OUT / name
        dest.mkdir(parents=True, exist_ok=True)
        trade, raw_count = trade_inputs(raw, codes)
        if args.resume and (dest / 'annual_indicators.csv.gz').exists():
            annual = pd.read_csv(dest / 'annual_indicators.csv.gz')
            fps = pd.read_csv(dest / 'exporter_origin_profiles.csv.gz')
        else:
            print(f'Inferring {name}: {raw_count} product records, {len(trade)} corridors', flush=True)
            annual, fps, mass = infer_inputs(trade, seed, order, weight, infer, lm, core.local.LANES)
            trade.to_csv(dest / 'trade_corridors.csv.gz', index=False)
            annual.to_csv(dest / 'annual_indicators.csv.gz', index=False)
            fps.to_csv(dest / 'exporter_origin_profiles.csv.gz', index=False)
            mass.to_csv(dest / 'attribution_value_conservation.csv', index=False)
        if name == 'original3':
            checks['original_basket_regression'] = compare_original(trade, annual, old_trade, old_annual)
            print(json.dumps(checks, indent=2), flush=True)
        windows, summary = historical(annual)
        windows.to_csv(dest / 'historical_three_indicator_windows.csv.gz', index=False)
        summary.to_csv(dest / 'historical_three_indicator_summary.csv', index=False)
        years = list(range(2012, 2025)) if name in ('original3', 'matte4', 'expanded6') else [2024]
        if not args.skip_allocation:
            run_allocation(name, trade, annual, fps, years, core, args.resume)
        configs[name] = dict(hs07_codes=codes, stage_order=order, material_local_weight=weight, allocation_years=[] if args.skip_allocation else years, raw_product_records=raw_count, corridor_records=len(trade), unit_years=len(annual))
    (OUT / 'run_manifest.json').write_text(json.dumps(dict(status='candidate scope reanalysis; not promoted', variants=configs, regression=checks, python=sys.version, numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, osqp=osqp.__version__, input_sha256=sha(PRODUCTS), archive_sha256=sha(ARCHIVE), script_sha256=sha(__file__), historical_classification='three-indicator only, no attribution-quality condition', matched_route_geometry='static mineral-country template', stage_weights='model assumptions, not calibrated measurements'), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
