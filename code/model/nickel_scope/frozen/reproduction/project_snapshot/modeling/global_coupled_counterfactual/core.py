"""Globally coupled export-envelope allocation stress tests by metal-stage.

Unlike the locked Fig. 5 implementation, all in-scope importers within a
metal-stage group share exporter envelope constraints in a single convex
optimization.  Importers outside the rolling universe remain fixed at their
observed base-year commitments.  The objective is baseline-value weighted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.sparse import csr_matrix, lil_matrix


HERE = Path(__file__).resolve().parent
try:
    from . import rolling_support as local
except ImportError:  # Permit direct execution during isolated diagnostics.
    sys.path.insert(0, str(HERE))
    import rolling_support as local  # type: ignore[no-redef]


CF = local.CF
KEYS = local.KEYS
MAIN_HEADROOM = local.MAIN_HEADROOM
MATERIAL_CHANGE = CF.MATERIAL_CHANGE
EQUAL_WEIGHTS = CF.EQUAL_WEIGHTS
ROUTE_ZERO_BASELINE_EPSILON_C = 0.025


@dataclass
class UnitModel:
    context: object
    baseline_shares: np.ndarray
    baseline_metrics: dict[str, float | str]
    active_exporter_indices: np.ndarray
    frozen_exporter_indices: np.ndarray
    variable_indices: np.ndarray
    route_variable_index: int | None
    group_weight: float
    route_matrix_available: bool

    @property
    def demand(self) -> float:
        return float(self.context.total_value)


@dataclass
class GroupModel:
    base_year: int
    metal: str
    stage: str
    units: list[UnitModel]
    variable_count: int
    share_variable_count: int
    lower: np.ndarray
    upper: np.ndarray
    start: np.ndarray
    equality: LinearConstraint
    inequality: LinearConstraint
    capacity_rows: dict[str, int]
    route_constraint_count: int
    no_route_matrix_units: int
    exporter_capacity: pd.DataFrame


def integrated_weights_for_unit(unit: UnitModel) -> tuple[float, float, float]:
    if unit.route_matrix_available:
        return EQUAL_WEIGHTS
    # Every candidate path has zero exposure to the tracked chokepoints, so the
    # route metric is a structural constant rather than missing evidence.
    return (0.5, 0.5, 0.0)


def build_group_model(
    base_year: int,
    group_units: pd.DataFrame,
    baseline: pd.DataFrame,
    capacity: pd.DataFrame,
    pools: dict[tuple[str, str], list[str]],
    fingerprints: dict[tuple[str, str, str], dict[str, float]],
    routes: dict[tuple[str, str, str], tuple[str, ...]],
) -> GroupModel:
    metal = str(group_units["metal"].iloc[0])
    stage = str(group_units["stage"].iloc[0])
    raw_contexts = [
        local.build_context(unit, baseline, capacity, pools, fingerprints, routes)
        for _, unit in group_units.iterrows()
    ]
    raw_contexts = [context for context in raw_contexts if context.total_value > 0]
    total_group_value = sum(context.total_value for context in raw_contexts)
    if total_group_value <= 0:
        raise ValueError(f"Empty group {metal}/{stage}/{base_year}")

    units: list[UnitModel] = []
    next_index = 0
    for context in raw_contexts:
        baseline_shares = context.baseline_value / context.total_value
        route_matrix_available = context.route_matrix.shape[1] > 0
        active = np.flatnonzero(context.expandable)
        frozen = np.flatnonzero(~context.expandable)
        variable_indices = np.arange(next_index, next_index + len(active), dtype=int)
        next_index += len(active)
        units.append(
            UnitModel(
                context=context,
                baseline_shares=baseline_shares,
                baseline_metrics=CF.metric_values(context, baseline_shares),
                active_exporter_indices=active,
                frozen_exporter_indices=frozen,
                variable_indices=variable_indices,
                route_variable_index=None,
                group_weight=context.total_value / total_group_value,
                route_matrix_available=route_matrix_available,
            )
        )
    share_variable_count = next_index
    for unit in units:
        if unit.route_matrix_available:
            unit.route_variable_index = next_index
            next_index += 1
    variable_count = next_index

    lower = np.zeros(variable_count, dtype=float)
    upper = np.ones(variable_count, dtype=float)
    start = np.zeros(variable_count, dtype=float)
    active_units = [unit for unit in units if len(unit.variable_indices)]
    equality_matrix = lil_matrix((len(active_units), variable_count), dtype=float)
    equality_value = np.zeros(len(active_units), dtype=float)
    for row, unit in enumerate(active_units):
        equality_matrix[row, unit.variable_indices] = 1.0
        start[unit.variable_indices] = unit.baseline_shares[unit.active_exporter_indices]
        equality_value[row] = 1.0 - float(
            unit.baseline_shares[unit.frozen_exporter_indices].sum()
        )
    for unit in units:
        if unit.route_variable_index is not None:
            route = unit.baseline_shares @ unit.context.route_matrix
            start[unit.route_variable_index] = min(1.0, float(route.max()) + 1e-8)

    capacity_group = capacity[
        capacity["metal"].astype(str).eq(metal)
        & capacity["stage"].astype(str).eq(stage)
    ].copy()
    capacity_group["exporter_iso"] = capacity_group["exporter_iso"].astype(str)
    capacity_group = capacity_group.drop_duplicates("exporter_iso").set_index("exporter_iso")
    selected_baseline: dict[str, float] = {}
    selected_frozen: dict[str, float] = {}
    selected_active_baseline: dict[str, float] = {}
    active_pairs: dict[str, list[tuple[int, float]]] = {}
    for unit in units:
        context = unit.context
        for exporter, value in zip(context.exporters, context.baseline_value):
            selected_baseline[exporter] = selected_baseline.get(exporter, 0.0) + float(value)
        for context_idx, variable_idx in zip(
            unit.active_exporter_indices, unit.variable_indices
        ):
            exporter = context.exporters[int(context_idx)]
            active_pairs.setdefault(exporter, []).append((int(variable_idx), unit.demand))
            selected_active_baseline[exporter] = selected_active_baseline.get(exporter, 0.0) + float(
                context.baseline_value[int(context_idx)]
            )
        for context_idx in unit.frozen_exporter_indices:
            exporter = context.exporters[int(context_idx)]
            selected_frozen[exporter] = selected_frozen.get(exporter, 0.0) + float(
                context.baseline_value[int(context_idx)]
            )

    exporters = sorted(set(selected_baseline) | set(active_pairs))
    capacity_scale = max(total_group_value, 1.0)
    capacity_rows: dict[str, int] = {}
    capacity_records: list[dict[str, object]] = []
    inequality_rows: list[dict[int, float]] = []
    inequality_lower: list[float] = []
    for exporter in exporters:
        if exporter not in capacity_group.index:
            raise KeyError(f"Missing capacity row for {metal}/{stage}/{exporter}")
        peak = float(capacity_group.at[exporter, "peak_global_value_usd"])
        current = float(capacity_group.at[exporter, "current_global_value_usd"])
        baseline_selected = selected_baseline.get(exporter, 0.0)
        frozen_selected = selected_frozen.get(exporter, 0.0)
        outside_commitment = max(current - baseline_selected, 0.0)
        envelope = peak * (1.0 + MAIN_HEADROOM)
        active_capacity_true = max(envelope - outside_commitment - frozen_selected, 0.0)
        active_baseline = selected_active_baseline.get(exporter, 0.0)
        # Reserve a tiny interior margin so numerical optimizer tolerances do
        # not appear as dollar-level envelope overruns after rescaling.  The
        # margin is capped at half of the observed headroom above the feasible
        # baseline and is at most one part per million of group value.
        true_slack = max(active_capacity_true - active_baseline, 0.0)
        numerical_buffer = min(capacity_scale * 1e-6, true_slack * 0.5)
        active_capacity = max(active_capacity_true - numerical_buffer, active_baseline)
        capacity_records.append(
            {
                "base_year": base_year,
                "metal": metal,
                "stage": stage,
                "exporter_iso": exporter,
                "peak_global_value_usd": peak,
                "current_global_value_usd": current,
                "selected_baseline_value_usd": baseline_selected,
                "frozen_selected_value_usd": frozen_selected,
                "outside_universe_commitment_usd": outside_commitment,
                "envelope_value_usd": envelope,
                "active_selected_capacity_usd": active_capacity,
                "true_active_selected_capacity_usd": active_capacity_true,
                "numerical_feasibility_buffer_usd": numerical_buffer,
            }
        )
        # Scale each exporter row by its own active capacity, rather than total
        # group value.  This keeps optimization feasibility tolerances meaningful
        # for small exporters whose headroom can be only tens of dollars.
        exporter_scale = max(active_capacity_true, active_baseline, 1.0)
        coefficients = {
            variable_idx: -demand / exporter_scale
            for variable_idx, demand in active_pairs.get(exporter, [])
        }
        # -sum(active allocation) >= -active capacity.  Exporters with no
        # active pairs need no optimization constraint; their frozen baseline
        # is recorded and checked after the solve.
        if coefficients:
            capacity_rows[exporter] = len(inequality_rows)
            inequality_rows.append(coefficients)
            inequality_lower.append(-active_capacity / exporter_scale)

    route_constraint_count = 0
    for unit in units:
        if unit.route_variable_index is None:
            continue
        active_route = unit.context.route_matrix[unit.active_exporter_indices, :]
        frozen_route = unit.baseline_shares[unit.frozen_exporter_indices] @ unit.context.route_matrix[
            unit.frozen_exporter_indices, :
        ]
        for choke_idx in range(unit.context.route_matrix.shape[1]):
            coefficients: dict[int, float] = {unit.route_variable_index: 1.0}
            for variable_idx, route_value in zip(
                unit.variable_indices, active_route[:, choke_idx]
            ):
                if route_value:
                    coefficients[int(variable_idx)] = -float(route_value)
            inequality_rows.append(coefficients)
            inequality_lower.append(float(frozen_route[choke_idx]))
            route_constraint_count += 1

    inequality_matrix = lil_matrix((len(inequality_rows), variable_count), dtype=float)
    for row, coefficients in enumerate(inequality_rows):
        for column, value in coefficients.items():
            inequality_matrix[row, column] = value
    equality = LinearConstraint(
        csr_matrix(equality_matrix), equality_value, equality_value
    )
    inequality = LinearConstraint(
        csr_matrix(inequality_matrix),
        np.asarray(inequality_lower, dtype=float),
        np.full(len(inequality_lower), np.inf),
    )
    return GroupModel(
        base_year=base_year,
        metal=metal,
        stage=stage,
        units=units,
        variable_count=variable_count,
        share_variable_count=share_variable_count,
        lower=lower,
        upper=upper,
        start=start,
        equality=equality,
        inequality=inequality,
        capacity_rows=capacity_rows,
        route_constraint_count=route_constraint_count,
        no_route_matrix_units=sum(not unit.route_matrix_available for unit in units),
        exporter_capacity=pd.DataFrame(capacity_records),
    )


def objective_and_gradient(model: GroupModel, objective: str):
    if objective not in {"direct", "integrated"}:
        raise ValueError(objective)

    def evaluate(vector: np.ndarray, need_gradient: bool) -> tuple[float, np.ndarray | None]:
        total = 0.0
        gradient = np.zeros_like(vector) if need_gradient else None
        for unit in model.units:
            shares = unit.baseline_shares.copy()
            if len(unit.variable_indices):
                shares[unit.active_exporter_indices] = vector[unit.variable_indices]
            weight = unit.group_weight
            direct = float(np.square(shares).sum())
            if objective == "direct":
                total += weight * direct
                if need_gradient and len(unit.variable_indices):
                    gradient[unit.variable_indices] += (
                        2.0 * weight * shares[unit.active_exporter_indices]
                    )
                continue

            direct_scale = max(float(unit.baseline_metrics["direct_hhi"]), 1e-9)
            origin_scale = max(float(unit.baseline_metrics["origin_hhi"]), 1e-9)
            direct_weight, origin_weight, route_weight = integrated_weights_for_unit(unit)
            origin_share = shares @ unit.context.origin_matrix
            origin = float(np.square(origin_share).sum())
            total += weight * direct_weight * direct / direct_scale
            total += weight * origin_weight * origin / origin_scale
            if need_gradient and len(unit.variable_indices):
                gradient[unit.variable_indices] += (
                    2.0
                    * weight
                    * direct_weight
                    * shares[unit.active_exporter_indices]
                    / direct_scale
                )
                gradient[unit.variable_indices] += (
                    2.0
                    * weight
                    * origin_weight
                    * (unit.context.origin_matrix[unit.active_exporter_indices, :] @ origin_share)
                    / origin_scale
                )
            if unit.route_variable_index is not None:
                route_scale = max(
                    float(unit.baseline_metrics["top_chokepoint_share"]),
                    ROUTE_ZERO_BASELINE_EPSILON_C,
                )
                total += (
                    weight
                    * route_weight
                    * float(vector[unit.route_variable_index])
                    / route_scale
                )
                if need_gradient:
                    gradient[unit.route_variable_index] += (
                        weight * route_weight / route_scale
                    )
        return total, gradient

    return (
        lambda vector: evaluate(vector, False)[0],
        lambda vector: evaluate(vector, True)[1],
    )


def quadratic_form(model: GroupModel, objective: str) -> tuple[csr_matrix, np.ndarray]:
    """Return H and c for 0.5*x'H*x + c'x."""
    hessian = lil_matrix((model.variable_count, model.variable_count), dtype=float)
    linear = np.zeros(model.variable_count, dtype=float)
    for unit in model.units:
        active_vars = unit.variable_indices
        active_context = unit.active_exporter_indices
        if objective == "direct":
            for variable_idx in active_vars:
                hessian[variable_idx, variable_idx] += 2.0 * unit.group_weight
            continue
        direct_scale = max(float(unit.baseline_metrics["direct_hhi"]), 1e-9)
        origin_scale = max(float(unit.baseline_metrics["origin_hhi"]), 1e-9)
        direct_weight, origin_weight, route_weight = integrated_weights_for_unit(unit)
        for variable_idx in active_vars:
            hessian[variable_idx, variable_idx] += (
                2.0 * unit.group_weight * direct_weight / direct_scale
            )
        if len(active_vars):
            active_origin = unit.context.origin_matrix[active_context, :]
            origin_quadratic = active_origin @ active_origin.T
            origin_factor = 2.0 * unit.group_weight * origin_weight / origin_scale
            for local_row, variable_row in enumerate(active_vars):
                for local_col, variable_col in enumerate(active_vars):
                    value = float(origin_quadratic[local_row, local_col])
                    if value:
                        hessian[variable_row, variable_col] += origin_factor * value
            frozen_origin = (
                unit.baseline_shares[unit.frozen_exporter_indices]
                @ unit.context.origin_matrix[unit.frozen_exporter_indices, :]
            )
            origin_linear = (
                2.0
                * unit.group_weight
                * origin_weight
                * (active_origin @ frozen_origin)
                / origin_scale
            )
            linear[active_vars] += origin_linear
        if unit.route_variable_index is not None:
            route_scale = max(
                float(unit.baseline_metrics["top_chokepoint_share"]),
                ROUTE_ZERO_BASELINE_EPSILON_C,
            )
            linear[unit.route_variable_index] += (
                unit.group_weight * route_weight / route_scale
            )
    return csr_matrix(hessian), linear


def constraint_violations(model: GroupModel, vector: np.ndarray) -> dict[str, float]:
    equality_value = np.asarray(model.equality.A @ vector).ravel()
    equality_target = np.asarray(model.equality.lb).ravel()
    inequality_value = np.asarray(model.inequality.A @ vector).ravel()
    inequality_lower = np.asarray(model.inequality.lb).ravel()
    return {
        "max_abs_demand_share_error": float(
            np.max(np.abs(equality_value - equality_target)) if len(equality_value) else 0.0
        ),
        "max_scaled_inequality_violation": float(
            np.max(np.maximum(inequality_lower - inequality_value, 0.0))
            if len(inequality_value)
            else 0.0
        ),
        "max_bound_violation": float(
            max(
                np.max(np.maximum(model.lower - vector, 0.0)),
                np.max(np.maximum(vector - model.upper, 0.0)),
            )
        ),
    }


def solve_group(
    model: GroupModel, objective: str, start: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, object]]:
    if model.variable_count == 0:
        return model.start.copy(), {
            "success": True,
            "message": "no active variables",
            "iterations": 0,
            "objective_value": 0.0,
            **constraint_violations(model, model.start),
        }
    hessian, linear = quadratic_form(model, objective)

    def fun(vector: np.ndarray) -> float:
        return float(0.5 * vector @ (hessian @ vector) + linear @ vector)

    def jac(vector: np.ndarray) -> np.ndarray:
        return np.asarray(hessian @ vector + linear, dtype=float)

    initial = model.start.copy() if start is None else np.asarray(start, dtype=float).copy()
    result = minimize(
        fun,
        initial,
        jac=jac,
        hess=lambda vector: hessian,
        method="trust-constr",
        bounds=Bounds(model.lower, model.upper),
        constraints=[model.equality, model.inequality],
        options={
            "maxiter": 1_000,
            "gtol": 1e-8,
            "xtol": 1e-10,
            "barrier_tol": 1e-10,
            "verbose": 0,
            "sparse_jacobian": True,
            "factorization_method": "AugmentedSystem",
        },
    )
    vector = np.clip(np.asarray(result.x, dtype=float), model.lower, model.upper)
    audit = constraint_violations(model, vector)
    feasible = (
        audit["max_abs_demand_share_error"] <= 2e-6
        and audit["max_scaled_inequality_violation"] <= 2e-7
        and audit["max_bound_violation"] <= 2e-8
    )
    if (not result.success or not feasible) and model.variable_count <= 500:
        result = minimize(
            fun,
            vector,
            jac=jac,
            method="SLSQP",
            bounds=Bounds(model.lower, model.upper),
            constraints=[model.equality, model.inequality],
            options={"maxiter": 4_000, "ftol": 1e-9, "disp": False},
        )
        vector = np.clip(np.asarray(result.x, dtype=float), model.lower, model.upper)
        audit = constraint_violations(model, vector)
        feasible = (
            audit["max_abs_demand_share_error"] <= 2e-6
            and audit["max_scaled_inequality_violation"] <= 2e-7
            and audit["max_bound_violation"] <= 2e-8
        )
    return vector, {
        "success": bool(result.success and feasible),
        "optimizer_success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "objective_value": float(fun(vector)),
        **audit,
    }


def set_route_epigraph_to_realized(model: GroupModel, vector: np.ndarray) -> np.ndarray:
    adjusted = vector.copy()
    for unit in model.units:
        if unit.route_variable_index is None:
            continue
        shares = unit.baseline_shares.copy()
        if len(unit.variable_indices):
            shares[unit.active_exporter_indices] = adjusted[unit.variable_indices]
        route = shares @ unit.context.route_matrix
        adjusted[unit.route_variable_index] = float(route.max()) if len(route) else 0.0
    return adjusted


def case_from_shares(
    unit: UnitModel,
    shares: np.ndarray,
    base_year: int,
    scenario: str,
    solver_success: bool,
    solver_message: str,
    solver_iterations: int,
) -> dict[str, object]:
    metrics = CF.metric_values(unit.context, shares)
    baseline = unit.baseline_metrics
    delta_direct = float(metrics["direct_hhi"]) - float(baseline["direct_hhi"])
    delta_origin = float(metrics["origin_hhi"]) - float(baseline["origin_hhi"])
    delta_choke = float(metrics["top_chokepoint_share"]) - float(
        baseline["top_chokepoint_share"]
    )
    joint = delta_direct < -1e-6 and delta_origin < -1e-6 and delta_choke < -1e-6
    material_joint = (
        delta_direct <= -MATERIAL_CHANGE
        and delta_origin <= -MATERIAL_CHANGE
        and delta_choke <= -MATERIAL_CHANGE
    )
    risk_transfer = delta_direct <= -MATERIAL_CHANGE and (
        delta_origin >= MATERIAL_CHANGE or delta_choke >= MATERIAL_CHANGE
    )
    origin_transfer = delta_direct <= -MATERIAL_CHANGE and delta_origin >= MATERIAL_CHANGE
    route_transfer = delta_direct <= -MATERIAL_CHANGE and delta_choke >= MATERIAL_CHANGE
    reallocation = 0.5 * float(np.abs(shares - unit.baseline_shares).sum())
    objective = {
        "baseline": "observed",
        "direct_partner": "direct",
        "integrated": "integrated",
    }[scenario]
    if scenario == "baseline":
        weights = (np.nan, np.nan, np.nan)
    elif scenario == "direct_partner":
        weights = (1.0, 0.0, 0.0)
    else:
        weights = integrated_weights_for_unit(unit)
    return {
        "base_year": base_year,
        "importer_iso": unit.context.importer_iso,
        "metal": unit.context.metal,
        "stage": unit.context.stage,
        "scenario": scenario,
        "scenario_label": {
            "baseline": f"{base_year} observed baseline",
            "direct_partner": "Direct-only globally coupled allocation",
            "integrated": "Joint-objective globally coupled allocation",
        }[scenario],
        "objective": objective,
        "objective_weight_direct": weights[0],
        "objective_weight_origin": weights[1],
        "objective_weight_route": weights[2],
        "headroom_above_peak": MAIN_HEADROOM,
        "recycling_demand_share": 0.0,
        "baseline_value_usd": unit.demand,
        "external_import_requirement_usd": unit.demand,
        "external_import_requirement_ratio": 1.0,
        "candidate_exporters": len(unit.context.exporters),
        "expandable_exporters": int(unit.context.expandable.sum()),
        "reallocation_value_usd": reallocation * unit.demand,
        "reallocation_share_of_baseline": reallocation,
        "solver_feasible": solver_success,
        "solver_message": solver_message,
        "solver_iterations": solver_iterations,
        **metrics,
        "delta_direct_hhi": delta_direct,
        "delta_origin_hhi": delta_origin,
        "delta_route_pressure": float(metrics["route_pressure"])
        - float(baseline["route_pressure"]),
        "delta_top_chokepoint_share": delta_choke,
        "joint_improvement": joint,
        "material_joint_reduction": material_joint,
        "risk_transfer": risk_transfer,
        "origin_risk_transfer": origin_transfer,
        "route_risk_transfer": route_transfer,
        "multiple_risk_transfer": origin_transfer and route_transfer,
        "eligibility_start_year": base_year - local.ENDPOINT_LAG_YEARS,
        "envelope_start_year": base_year - (local.LOOKBACK_YEARS_INCLUSIVE - 1),
        "envelope_end_year": base_year,
        "trade_max_year_used": base_year,
        "origin_fingerprint_year": base_year,
        "route_fingerprint_scope": "static_representative_geometry_no_trade_weights",
        "interpretation_scope": "globally_coupled_within_metal_stage_outside_universe_fixed",
        "route_matrix_available": unit.route_matrix_available,
        "route_objective_regime": (
            "three_metric_epsilon_C_regularized_if_needed"
            if unit.route_matrix_available
            else "constant_zero_route_direct_origin_half_each"
        ),
        "route_zero_baseline_regularization_epsilon_C": (
            ROUTE_ZERO_BASELINE_EPSILON_C if unit.route_matrix_available else np.nan
        ),
    }


def run_group(
    model: GroupModel,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    cases: list[dict[str, object]] = []
    allocations: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    scenario_vectors: dict[str, np.ndarray] = {"baseline": model.start.copy()}
    baseline_audit = constraint_violations(model, model.start)
    scenario_audits: dict[str, dict[str, object]] = {
        "baseline": {
            "success": True,
            "optimizer_success": True,
            "message": "observed baseline",
            "iterations": 0,
            "objective_value": np.nan,
            **baseline_audit,
        }
    }
    direct_vector, direct_audit = solve_group(model, "direct")
    direct_vector = set_route_epigraph_to_realized(model, direct_vector)
    scenario_vectors["direct_partner"] = direct_vector
    scenario_audits["direct_partner"] = direct_audit
    integrated_vector, integrated_audit = solve_group(model, "integrated")
    integrated_vector = set_route_epigraph_to_realized(model, integrated_vector)
    scenario_vectors["integrated"] = integrated_vector
    scenario_audits["integrated"] = integrated_audit

    capacity_index = model.exporter_capacity.set_index("exporter_iso")
    for scenario, vector in scenario_vectors.items():
        audit = scenario_audits[scenario]
        solver_rows.append(
            {
                "base_year": model.base_year,
                "metal": model.metal,
                "stage": model.stage,
                "scenario": scenario,
                "units": len(model.units),
                "share_variables": model.share_variable_count,
                "total_variables": model.variable_count,
                "capacity_constraints": len(model.capacity_rows),
                "route_constraints": model.route_constraint_count,
                "no_route_matrix_units": model.no_route_matrix_units,
                **audit,
            }
        )
        for unit in model.units:
            shares = unit.baseline_shares.copy()
            if len(unit.variable_indices):
                shares[unit.active_exporter_indices] = vector[unit.variable_indices]
            cases.append(
                case_from_shares(
                    unit,
                    shares,
                    model.base_year,
                    scenario,
                    bool(audit["success"]),
                    str(audit["message"]),
                    int(audit["iterations"]),
                )
            )
            for idx, exporter in enumerate(unit.context.exporters):
                baseline_value = float(unit.context.baseline_value[idx])
                scenario_value = float(shares[idx] * unit.demand)
                if baseline_value <= 0 and scenario_value <= 0:
                    continue
                capacity_row = capacity_index.loc[exporter]
                allocations.append(
                    {
                        "base_year": model.base_year,
                        "importer_iso": unit.context.importer_iso,
                        "metal": model.metal,
                        "stage": model.stage,
                        "scenario": scenario,
                        "exporter_iso": exporter,
                        "baseline_value_usd": baseline_value,
                        "scenario_value_usd": scenario_value,
                        "scenario_share": float(shares[idx]),
                        "expandable_with_full_evidence": bool(unit.context.expandable[idx]),
                        "peak_global_value_usd": float(capacity_row["peak_global_value_usd"]),
                        "current_global_value_usd": float(
                            capacity_row["current_global_value_usd"]
                        ),
                        "outside_universe_commitment_usd": float(
                            capacity_row["outside_universe_commitment_usd"]
                        ),
                        "export_envelope_value_usd": float(
                            capacity_row["envelope_value_usd"]
                        ),
                        "route_matrix_available": unit.route_matrix_available,
                    }
                )
    return cases, allocations, solver_rows


def global_capacity_audit(allocations: pd.DataFrame) -> pd.DataFrame:
    grouped = allocations.groupby(
        ["base_year", "metal", "stage", "exporter_iso", "scenario"],
        as_index=False,
        observed=True,
    ).agg(
        selected_baseline_value_usd=("baseline_value_usd", "sum"),
        selected_scenario_value_usd=("scenario_value_usd", "sum"),
        current_global_value_usd=("current_global_value_usd", "max"),
        outside_universe_commitment_usd=("outside_universe_commitment_usd", "max"),
        envelope_value_usd=("export_envelope_value_usd", "max"),
    )
    grouped["implied_global_scenario_value_usd"] = (
        grouped["outside_universe_commitment_usd"]
        + grouped["selected_scenario_value_usd"]
    )
    grouped["envelope_excess_usd"] = (
        grouped["implied_global_scenario_value_usd"] - grouped["envelope_value_usd"]
    ).clip(lower=0.0)
    grouped["envelope_slack_usd"] = (
        grouped["envelope_value_usd"] - grouped["implied_global_scenario_value_usd"]
    )
    rows = []
    for (year, scenario), data in grouped.groupby(["base_year", "scenario"], observed=True):
        rows.append(
            {
                "base_year": int(year),
                "scenario": str(scenario),
                "exporter_metal_stage_rows": len(data),
                "violating_rows_over_one_usd": int(data["envelope_excess_usd"].gt(1.0).sum()),
                "total_envelope_excess_usd": float(data["envelope_excess_usd"].sum()),
                "max_envelope_excess_usd": float(data["envelope_excess_usd"].max()),
                "minimum_envelope_slack_usd": float(data["envelope_slack_usd"].min()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2024")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parent / "outputs" / "global_coupled_rolling_candidate",
    )
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--skip-groups", type=int, default=0)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    years = local.parse_years(args.years)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indicator = local.load_indicator()
    universes = {year: local.rolling_universe(indicator, year) for year in years}
    trade = local.load_trade()
    routes = local.load_routes()
    fingerprints = local.load_origin_fingerprints(years)
    all_cases: list[dict[str, object]] = []
    all_allocations: list[dict[str, object]] = []
    all_solver_rows: list[dict[str, object]] = []
    group_counter = 0
    for year in years:
        baseline, capacity, pools = local.year_trade_components(trade, year)
        groups = list(universes[year].groupby(["metal", "stage"], observed=True, sort=True))
        print(f"year {year}: {len(universes[year])} units in {len(groups)} metal-stage groups")
        for group_position, ((metal, stage), group_units) in enumerate(groups):
            if group_position < args.skip_groups:
                continue
            if args.max_groups is not None and group_counter >= args.max_groups:
                break
            model = build_group_model(
                year,
                group_units,
                baseline,
                capacity,
                pools,
                fingerprints[year],
                routes,
            )
            print(
                f"building {year} {metal}/{stage}: units={len(model.units)}, "
                f"share_vars={model.share_variable_count}, total_vars={model.variable_count}, "
                f"capacity={len(model.capacity_rows)}, route={model.route_constraint_count}, "
                f"no_route_units={model.no_route_matrix_units}",
                flush=True,
            )
            if args.inventory_only:
                all_solver_rows.append(
                    {
                        "base_year": year,
                        "metal": metal,
                        "stage": stage,
                        "scenario": "inventory",
                        "units": len(model.units),
                        "share_variables": model.share_variable_count,
                        "total_variables": model.variable_count,
                        "capacity_constraints": len(model.capacity_rows),
                        "route_constraints": model.route_constraint_count,
                        "no_route_matrix_units": model.no_route_matrix_units,
                        "success": True,
                    }
                )
                group_counter += 1
                continue
            cases, allocations, solver_rows = run_group(model)
            all_cases.extend(cases)
            all_allocations.extend(allocations)
            all_solver_rows.extend(solver_rows)
            group_counter += 1
            latest = solver_rows[-1]
            print(
                f"{year} {metal}/{stage}: units={len(model.units)}, vars={model.variable_count}, "
                f"integrated_success={latest['success']}, iterations={latest['iterations']}",
                flush=True,
            )
        if args.max_groups is not None and group_counter >= args.max_groups:
            break

    solver = pd.DataFrame(all_solver_rows)
    if args.inventory_only:
        solver.to_csv(output_dir / "global_coupled_group_inventory.csv", index=False)
        print(solver.to_string(index=False))
        print(output_dir)
        return
    cases = pd.DataFrame(all_cases)
    allocations = pd.DataFrame(all_allocations)
    cases = cases.sort_values(["base_year", *KEYS, "scenario"]).reset_index(drop=True)
    allocations = allocations.sort_values(
        ["base_year", *KEYS, "scenario", "scenario_value_usd"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)
    summary = CF.summarize_cases(
        cases,
        ["base_year", "scenario", "scenario_label", "eligibility_start_year", "envelope_start_year", "envelope_end_year"],
    )
    capacity_audit = global_capacity_audit(allocations)
    cases.to_csv(output_dir / "global_coupled_cases.csv.gz", index=False, compression="gzip")
    allocations.to_csv(
        output_dir / "global_coupled_supplier_allocations.csv.gz",
        index=False,
        compression="gzip",
    )
    solver.to_csv(output_dir / "global_coupled_group_solver_audit.csv", index=False)
    summary.to_csv(output_dir / "global_coupled_annual_summary.csv", index=False)
    capacity_audit.to_csv(output_dir / "global_coupled_capacity_audit.csv", index=False)
    no_route = cases[
        cases["scenario"].eq("baseline") & ~cases["route_matrix_available"].astype(bool)
    ][
        [
            "base_year",
            *KEYS,
            "baseline_value_usd",
            "route_coverage_share",
            "top_chokepoint_share",
            "interpretation_scope",
        ]
    ].copy()
    no_route["treatment"] = (
        "tracked-route exposure is structural zero; direct optimized normally; "
        "joint objective weights direct/origin at 1/2 each"
    )
    no_route.to_csv(output_dir / "global_coupled_no_route_matrix_units.csv", index=False)
    manifest = {
        "analysis": "globally coupled allocation stress test by metal-stage",
        "years": years,
        "objective_weighting": "baseline import value within each metal-stage group",
        "capacity_constraint": "shared exporter envelope with out-of-universe base-year commitments fixed",
        "candidate_support": "top-60 revealed-exporter pool with complete origin and route evidence for expandable suppliers",
        "historical_baseline_information_rule": (
            "candidate suppliers and exporter envelopes use trade years <= base year; "
            "origin fingerprint = base year; one frozen static route template generated "
            "from 2021-2023 corridor seeds is used for all baselines; not a real-time "
            "information-set backtest"
        ),
        "zero_baseline_route_regularization": {
            "rule": "when a positive-candidate route matrix exists, route denominator = max(baseline top-chokepoint share, epsilon_C)",
            "epsilon_C": ROUTE_ZERO_BASELINE_EPSILON_C,
            "constant_zero_route_treatment": "if all candidate routes have zero tracked-chokepoint exposure, optimize direct normally and weight direct/origin at 1/2 each in the joint objective",
        },
        "rows": {
            "cases": len(cases),
            "allocations": len(allocations),
            "group_solver_rows": len(solver),
        },
    }
    (output_dir / "global_coupled_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(capacity_audit.to_string(index=False))
    print(solver.groupby("scenario")["success"].agg(["count", "mean"]).to_string())
    print(output_dir)


if __name__ == "__main__":
    main()
