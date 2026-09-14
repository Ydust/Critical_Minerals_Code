"""Independent numerical checks and complete scope-comparison tables."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

from run_scope_reanalysis import CONFIGS, EDGE, GROUP, IP, KEYS, OUT, ARCHIVE, PRODUCTS, STAGE, ROOT, sha
import numpy as np
import pandas as pd


def independent_profiles(trade, seed, order, weight, year):
    """Matrix implementation independent of the frozen dictionary recursion."""
    flows = trade[trade.year.eq(year)]
    prod = seed[seed.year.eq(year) & seed.production.gt(0)].groupby('mine_origin_iso').production.sum()
    origins = sorted(prod.index)
    countries = sorted(set(flows.exporter_iso) | set(flows.importer_iso) | set(origins))
    ix = {c: i for i, c in enumerate(countries)}
    world = prod.reindex(origins).to_numpy() / prod.sum()
    mix = np.zeros((len(countries), len(origins)))
    for j, origin in enumerate(origins):
        mix[ix[origin], j] = 1
    rows = []
    for stage in order:
        sub = flows[flows.stage.eq(stage)]
        if sub.empty:
            continue
        w = {'ore': .95, 'material': weight, 'compound': .55, 'metal': .45}[stage]
        has = mix.sum(axis=1) > 0
        export = np.tile(world, (len(countries), 1))
        export[has] = w*mix[has] + (1-w)*world
        export /= export.sum(axis=1)[:, None]
        matrix = np.zeros((len(countries), len(countries)))
        np.add.at(matrix, (sub.importer_iso.map(ix).to_numpy(), sub.exporter_iso.map(ix).to_numpy()), sub.reconstructed_value_usd.to_numpy())
        incoming = matrix @ export
        total = incoming.sum(axis=1)
        active = total > 0
        updated = w*mix[active] + (1-w)*incoming[active]/total[active, None]
        norm = updated.sum(axis=1)
        indices = np.flatnonzero(active)[norm > 0]
        mix[indices] = updated[norm > 0] / norm[norm > 0, None]
        for exporter in sorted(sub.exporter_iso.unique()):
            for j, origin in enumerate(origins):
                rows.append((year, stage, exporter, origin, export[ix[exporter], j]))
    return pd.DataFrame(rows, columns=['year', 'stage', 'exporter_iso', 'inferred_mine_origin_iso', 'share'])


def verify_allocations(cases, allocations, fps, lanes):
    """Reconstruct D, O and C from allocations, without invoking model metric code."""
    casekey = ['base_year', 'stage', 'importer_iso', 'scenario']
    profilekey = ['year', 'stage', 'exporter_iso']
    profiles = {k: dict(zip(g.inferred_mine_origin_iso, g.share)) for k, g in fps.groupby(profilekey)}
    routes = {}
    for key, group in lanes.groupby(['origin', 'dest']):
        labels = set()
        for value in group.chokepoints.dropna():
            labels.update(x.strip() for x in str(value).split('|') if x.strip())
        routes[key] = labels
    indexed = cases.set_index(casekey)
    max_metric_error, max_demand_relative_error = 0., 0.
    for key, group in allocations.groupby(casekey):
        year, stage, importer, scenario = key
        case = indexed.loc[key]
        direct = group.groupby('exporter_iso').scenario_value_usd.sum()
        shares = direct / direct.sum()
        max_demand_relative_error = max(max_demand_relative_error, abs(direct.sum()/case.baseline_value_usd - 1))
        origins, chokes = {}, {}
        for exporter, share in shares.items():
            for origin, fraction in profiles[(year, stage, exporter)].items():
                origins[origin] = origins.get(origin, 0.) + share*fraction
            for choke in routes.get((exporter, importer), set()):
                chokes[choke] = chokes.get(choke, 0.) + share
        d = float((shares**2).sum())
        o = sum(v*v for v in origins.values())
        c = max(chokes.values(), default=0.)
        max_metric_error = max(max_metric_error, abs(d-case.direct_hhi), abs(o-case.origin_hhi), abs(c-case.top_chokepoint_share))
    # Direct reconstruction may renormalize the tiny accepted demand residual.
    assert max_metric_error < 5e-7 and max_demand_relative_error < 2e-6
    cap = allocations.groupby(['base_year', 'stage', 'exporter_iso', 'scenario']).agg(
        chosen=('scenario_value_usd', 'sum'), outside=('outside_universe_commitment_usd', 'max'),
        envelope=('export_envelope_value_usd', 'max'))
    excess = (cap.chosen + cap.outside - cap.envelope).clip(lower=0.)
    assert excess.max() <= 1.
    assert allocations.scenario_value_usd.ge(0).all()
    return dict(max_metric_error=max_metric_error, max_relative_demand_error=max_demand_relative_error,
                max_export_envelope_excess_usd=float(excess.max()), case_rows=len(cases), allocation_rows=len(allocations))


def main():
    manifest = json.loads((OUT / 'run_manifest.json').read_text())
    assert set(manifest['variants']) == set(CONFIGS)
    assert manifest['input_sha256'] == sha(PRODUCTS)
    assert manifest['archive_sha256'] == sha(ARCHIVE)
    assert manifest['script_sha256'] == sha(OUT.parent/'run_scope_reanalysis.py')
    with zipfile.ZipFile(ARCHIVE) as z:
        seed = pd.read_csv(z.open(IP+'mine_origin_seed_bgs_wmd_2008_2024.csv.gz'), compression='gzip')
        old_annual = pd.read_csv(z.open(IP+'risk_indicator_panel_baci_2008_2024.csv.gz'), compression='gzip')
    seed = seed[seed.metal.eq('Nickel')]
    lanes = pd.read_csv(OUT.parent / 'frozen/figure_source_tables/longitudinal_2012_2024/input_snapshot/minerals_lanes.csv')
    lanes = lanes[lanes.metal.eq('Nickel')]
    base_annual = pd.read_csv(OUT/'original3/annual_indicators.csv.gz').set_index(GROUP).sort_index()
    base_trade = pd.read_csv(OUT/'original3/trade_corridors.csv.gz')
    base_cases = pd.read_csv(OUT/'original3/allocation/all_cases.csv.gz')
    base_alloc = pd.read_csv(OUT/'original3/allocation/all_allocations.csv.gz')
    validations, summaries, historical, finland, metal_changes, paired_cases = {}, [], [], [], [], []
    fpkey = ['year', 'stage', 'exporter_iso', 'inferred_mine_origin_iso']
    for name, (_, order, weight) in CONFIGS.items():
        folder = OUT/name
        trade = pd.read_csv(folder/'trade_corridors.csv.gz')
        annual = pd.read_csv(folder/'annual_indicators.csv.gz')
        fps = pd.read_csv(folder/'exporter_origin_profiles.csv.gz')
        cases = pd.read_csv(folder/'allocation/all_cases.csv.gz')
        allocations = pd.read_csv(folder/'allocation/all_allocations.csv.gz')
        solver = pd.read_csv(folder/'allocation/all_solver_audits.csv')
        expected_years = manifest['variants'][name]['allocation_years']
        assert sorted(cases.base_year.unique()) == expected_years
        assert cases.solver_feasible.all() and solver.success.all()
        assert not cases.duplicated(['base_year', 'stage', 'importer_iso', 'scenario']).any()
        assert solver[solver.scenario.ne('baseline')].kkt_stationarity.max() < 2e-7
        assert solver[solver.scenario.ne('baseline')].kkt_complementarity.max() < 2e-7
        assert solver[solver.scenario.ne('baseline')].dual_sign_error.max() < 2e-7
        assert solver.max_abs_demand_share_error.max() < 2e-6
        assert solver.max_scaled_inequality_violation.max() < 2e-7
        assert solver.max_bound_violation.max() < 2e-8
        groups = cases.groupby(['base_year', 'stage', 'importer_iso']).scenario.nunique()
        assert groups.eq(3).all()
        totals = fps.groupby(['year', 'stage', 'exporter_iso']).share.sum()
        assert float((totals-1).abs().max()) < 1e-12 and fps.share.ge(0).all()
        assert not trade.duplicated(EDGE).any()
        assert trade.exporter_iso.ne(trade.importer_iso).all()
        independent_error = 0.
        for year in (2008, 2018, 2024):
            expected = independent_profiles(trade, seed, order, weight, year).set_index(fpkey).share
            actual = fps[fps.year.eq(year)].set_index(fpkey).share
            aligned = pd.concat([actual.rename('actual'), expected.rename('expected')], axis=1).fillna(0)
            independent_error = max(independent_error, float((aligned.actual-aligned.expected).abs().max()))
        assert independent_error < 1e-12
        audited = verify_allocations(cases, allocations, fps, lanes)
        # Matched metal cohort, direct/route inputs and direct-only allocation.
        metal = annual[annual.stage.eq('metal')].set_index(GROUP).sort_index()
        reference = base_annual[base_annual.index.get_level_values('stage') == 'metal']
        assert metal.index.equals(reference.index)
        constant_error = float((metal[['direct_hhi', 'top_chokepoint_share', 'route_coverage_share']] - reference[['direct_hhi', 'top_chokepoint_share', 'route_coverage_share']]).abs().max().max())
        assert constant_error < 1e-12
        unchanged = trade[trade.stage.eq('metal')].set_index(EDGE).reconstructed_value_usd.sort_index()
        original = base_trade[base_trade.stage.eq('metal')].set_index(EDGE).reconstructed_value_usd.sort_index()
        pd.testing.assert_series_equal(unchanged, original)
        pairkey = ['base_year', 'stage', 'importer_iso', 'scenario']
        mc = cases[cases.stage.eq('metal')].set_index(pairkey).sort_index()
        bc = base_cases[base_cases.stage.eq('metal') & base_cases.base_year.isin(expected_years)].set_index(pairkey).sort_index()
        assert mc.index.equals(bc.index)
        akey = pairkey + ['exporter_iso']
        a = allocations[allocations.stage.eq('metal') & allocations.scenario.eq('direct_partner')].set_index(akey).scenario_share
        b = base_alloc[base_alloc.stage.eq('metal') & base_alloc.scenario.eq('direct_partner') & base_alloc.base_year.isin(expected_years)].set_index(akey).scenario_share
        ab = pd.concat([a.rename('a'), b.rename('b')], axis=1).fillna(0)
        direct_allocation_error = float((ab.a-ab.b).abs().max())
        assert direct_allocation_error < 1e-6
        validations[name] = dict(independent_profile_max_error=independent_error, solver_max_kkt=float(solver.kkt_stationarity.max()),
            solver_max_complementarity=float(solver.kkt_complementarity.max()), solver_max_dual_sign_error=float(solver.dual_sign_error.max()),
            optimized_group_scenarios=int(solver.scenario.ne('baseline').sum()),
            constant_metal_input_error=constant_error, metal_matched_unit_years=int(len(mc)/3), direct_only_metal_max_share_change=direct_allocation_error, **audited)
        for year in range(2012, 2025):
            m = metal[metal.index.get_level_values('year') == year]
            b = reference[reference.index.get_level_values('year') == year]
            weights = b.direct_total_value_usd
            delta = m.origin_hhi - b.origin_hhi
            metal_changes.append(dict(variant=name, year=year, unit_years=len(m), value_weighted_origin_hhi=float(np.average(m.origin_hhi, weights=weights)),
                value_weighted_delta_origin_hhi=float(np.average(delta, weights=weights)), max_abs_delta_origin_hhi=float(delta.abs().max()), units_changed_over_0025=int(delta.abs().ge(.025).sum())))
        f = fps[fps.exporter_iso.eq('FIN') & fps.stage.eq('metal') & fps.inferred_mine_origin_iso.eq('RUS')].copy()
        f.insert(0, 'variant', name)
        finland.append(f)
        for stage, fn in [('all', 'annual_summary.csv'), ('by_stage', 'annual_stage_summary.csv')]:
            s = pd.read_csv(folder/'allocation'/fn)
            if stage == 'all':
                s['stage'] = 'all'
            s.insert(0, 'variant', name)
            summaries.append(s)
        h = pd.read_csv(folder/'historical_three_indicator_summary.csv')
        h.insert(0, 'variant', name)
        historical.append(h)
        paired = mc[['origin_hhi', 'delta_origin_hhi', 'risk_transfer', 'material_joint_reduction']].join(bc[['origin_hhi', 'delta_origin_hhi', 'risk_transfer', 'material_joint_reduction']], rsuffix='_original3')
        paired.insert(0, 'variant', name)
        paired_cases.append(paired.reset_index())
        print(name, json.dumps(validations[name]), flush=True)
    for fname, frames in [('allocation_scope_comparison.csv', summaries), ('historical_scope_comparison.csv', historical), ('finland_metal_russian_origin_share.csv', finland), ('matched_metal_case_comparison.csv', paired_cases)]:
        pd.concat(frames, ignore_index=True).to_csv(OUT/fname, index=False)
    pd.DataFrame(metal_changes).to_csv(OUT/'matched_metal_origin_changes.csv', index=False)
    # Document initial-geometry differences; they are not a scope effect.
    old = old_annual[old_annual.metal.eq('Nickel')].set_index(GROUP).sort_index()
    difference = base_annual.top_chokepoint_share - old.top_chokepoint_share
    validations['archived_initial_route_geometry_difference'] = dict(unit_years_changed=int(difference.abs().gt(1e-12).sum()), max_abs_difference=float(difference.abs().max()), scope_comparison_uses_same_current_routes=True)
    validations['total_verified_optimized_group_scenarios'] = sum(v['optimized_group_scenarios'] for v in validations.values() if isinstance(v, dict) and 'optimized_group_scenarios' in v)
    validations['status'] = 'PASS'
    (OUT/'independent_validation.json').write_text(json.dumps(validations, indent=2), encoding='utf-8')
    package = ROOT/'manuscript_package_20260907_final'
    inventory = pd.read_csv(package/'PACKAGE_SHA256.csv')
    mismatches = []
    for row in inventory.itertuples(index=False):
        path = package/row.file
        if not path.exists() or path.stat().st_size != row.bytes or sha(path) != row.sha256:
            mismatches.append(row.file)
    assert not mismatches, mismatches
    archive_hash = sha(ROOT/'manuscript_package_20260907_final.zip')
    assert archive_hash == '0226e5422850f2b2adab605b47fc8e35d7b41d19a6899ba647e5696d96b84019'
    current_routes = ROOT/'research_process/numerical_checks/recalculation_tables/routes/minerals_lanes_geography_corrected.csv'
    used_routes = OUT.parent/'frozen/figure_source_tables/longitudinal_2012_2024/input_snapshot/minerals_lanes.csv'
    assert sha(current_routes) == sha(used_routes)
    (OUT/'submission_package_preservation.json').write_text(json.dumps(dict(checked_files=len(inventory), mismatches=mismatches,
        submission_zip_sha256=archive_hash, current_route_sha256=sha(used_routes), status='unchanged; not promoted'), indent=2), encoding='utf-8')
    files = [p for p in OUT.parent.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name != 'OUTPUT_SHA256.csv']
    pd.DataFrame([dict(path=str(p.relative_to(OUT.parent)), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(files)]).to_csv(OUT/'OUTPUT_SHA256.csv', index=False)


if __name__ == '__main__':
    main()
