import json,time
import numpy as np
import pandas as pd
import osqp
from scipy.sparse import lil_matrix,csc_matrix,vstack,eye
KEYS=['importer_iso','metal','stage']
GROUP=KEYS+['year']
EDGE=['metal','stage','exporter_iso','importer_iso','year']
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
