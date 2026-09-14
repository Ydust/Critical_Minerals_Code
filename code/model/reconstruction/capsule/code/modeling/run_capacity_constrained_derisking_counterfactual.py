"""Run capacity-constrained de-risking counterfactuals for 2024 trade flows.

The model reallocates importer-metal-stage trade across observed suppliers under
a revealed export-capacity envelope. It compares direct-partner, inferred
mine-origin, route, joint-objective and recycling-supported strategies. The exercise
is a constrained allocation stress test, not a causal or engineering-capacity
model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, minimize


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"

DYNAMIC = OUT / "dynamic_reconstruction_panel.csv.gz"
MINE = OUT / "mine_origin_stage_flows_v0.csv.gz"
RISK = OUT / "risk_transfer_indicator_panel.csv.gz"
UNIVERSE = OUT / "main_figure_analysis_universe.csv"
CANONICAL_MANIFEST = ROOT / "manuscript" / "canonical_analysis_manifest.json"
LANES = ROOT / "minerals_lanes.csv"
XU = OUT / "external_xu_2020_battery_material_demand_features.csv"

CASE_OUTPUT = OUT / "capacity_constrained_derisking_scenarios.csv.gz"
ALLOCATION_OUTPUT = OUT / "capacity_constrained_derisking_supplier_allocations.csv.gz"
SUMMARY_OUTPUT = OUT / "capacity_constrained_derisking_summary.csv"
METAL_OUTPUT = OUT / "capacity_constrained_derisking_by_metal.csv"
IMPORTER_OUTPUT = OUT / "capacity_constrained_derisking_by_importer.csv"
SENSITIVITY_OUTPUT = OUT / "capacity_constrained_derisking_capacity_sensitivity.csv"
WEIGHT_SENSITIVITY_CASE_OUTPUT = OUT / "capacity_constrained_derisking_weight_sensitivity_cases.csv.gz"
WEIGHT_SENSITIVITY_OUTPUT = OUT / "capacity_constrained_derisking_weight_sensitivity.csv"
CARD_OUTPUT = OUT / "CAPACITY_CONSTRAINED_DERISKING_CARD.md"

START_YEAR = 2019
BASE_YEAR = 2024
MIN_VALUE_USD = 10_000_000.0
MIN_ORIGIN_COVERAGE = 0.80
MIN_ROUTE_COVERAGE = 0.50
MAX_POOL_EXPORTERS = 60
MAIN_HEADROOM = 0.10
HEADROOM_SENSITIVITY = (0.05, 0.10, 0.25)
EQUAL_WEIGHTS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
WEIGHT_SENSITIVITY = {
    "equal": EQUAL_WEIGHTS,
    "direct_heavy": (0.60, 0.20, 0.20),
    "origin_heavy": (0.20, 0.60, 0.20),
    "route_heavy": (0.20, 0.20, 0.60),
}
MATERIAL_CHANGE = 0.025
TIE_BREAKER = 1e-7

KEYS = ["importer_iso", "metal", "stage"]
SCENARIO_ORDER = [
    "baseline_2024",
    "direct_partner",
    "mine_origin",
    "route",
    "integrated",
    "integrated_recycling",
]
SCENARIO_LABELS = {
    "baseline_2024": "2024 baseline",
    "direct_partner": "Direct-partner diversification",
    "mine_origin": "Mine-origin-aware allocation",
    "route": "Route-aware allocation",
    "integrated": "Joint three-metric objective allocation",
    "integrated_recycling": "Joint three-metric objective + Xu 2040 recycling",
}


@dataclass
class CaseContext:
    importer_iso: str
    metal: str
    stage: str
    exporters: list[str]
    baseline_value: np.ndarray
    peak_global_value: np.ndarray
    current_global_value: np.ndarray
    origin_matrix: np.ndarray
    route_matrix: np.ndarray
    origin_known: np.ndarray
    route_known: np.ndarray
    expandable: np.ndarray
    origin_labels: list[str]
    route_labels: list[str]

    @property
    def total_value(self) -> float:
        return float(self.baseline_value.sum())


def ensure_inputs() -> None:
    missing = [path for path in [DYNAMIC, MINE, UNIVERSE, LANES, XU] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing counterfactual inputs:\n" + "\n".join(str(path) for path in missing))


def combine_chokepoints(values: pd.Series) -> tuple[str, ...]:
    items: list[str] = []
    for value in values.dropna().astype(str):
        for item in value.split("|"):
            item = item.strip()
            if item and item not in items:
                items.append(item)
    return tuple(items)


def load_trade() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = ["metal", "stage", "exporter_iso", "importer_iso", "year", "reconstructed_value_usd"]
    trade = pd.read_csv(DYNAMIC, usecols=cols, low_memory=False)
    trade["reconstructed_value_usd"] = pd.to_numeric(
        trade["reconstructed_value_usd"], errors="coerce"
    ).fillna(0.0)
    trade = trade[
        trade["year"].between(START_YEAR, BASE_YEAR)
        & trade["reconstructed_value_usd"].gt(0)
    ].copy()
    trade = trade.groupby(cols[:-1], as_index=False, observed=True)["reconstructed_value_usd"].sum()

    baseline = trade[trade["year"].eq(BASE_YEAR)].drop(columns="year").copy()
    global_annual = (
        trade.groupby(["metal", "stage", "exporter_iso", "year"], as_index=False, observed=True)[
            "reconstructed_value_usd"
        ].sum()
    )
    peak = (
        global_annual.groupby(["metal", "stage", "exporter_iso"], as_index=False, observed=True)
        .agg(peak_global_value_usd=("reconstructed_value_usd", "max"))
    )
    current = global_annual[global_annual["year"].eq(BASE_YEAR)][
        ["metal", "stage", "exporter_iso", "reconstructed_value_usd"]
    ].rename(columns={"reconstructed_value_usd": "current_global_value_usd"})
    supplier_capacity = peak.merge(current, on=["metal", "stage", "exporter_iso"], how="left")
    supplier_capacity["current_global_value_usd"] = supplier_capacity["current_global_value_usd"].fillna(0.0)
    supplier_capacity = supplier_capacity[supplier_capacity["current_global_value_usd"].gt(0)].copy()
    return trade, baseline, supplier_capacity


def load_universe() -> pd.DataFrame:
    manifest = json.loads(CANONICAL_MANIFEST.read_text(encoding="utf-8"))
    expected_units = int(manifest["sample"]["expected_units"])
    panel = pd.read_csv(UNIVERSE)
    universe = panel[
        KEYS
        + [
            "direct_total_value_usd_2024",
            "origin_coverage_ratio_2024",
            "route_coverage_share_2024",
        ]
    ].rename(
        columns={
            "direct_total_value_usd_2024": "direct_total_value_usd",
            "origin_coverage_ratio_2024": "origin_coverage_ratio",
            "route_coverage_share_2024": "route_coverage_share",
        }
    )
    if len(universe) != expected_units:
        raise ValueError(
            f"Expected {expected_units} common main-figure units from the canonical manifest, "
            f"found {len(universe)}"
        )
    return universe.sort_values("direct_total_value_usd", ascending=False).reset_index(drop=True)


def load_origin_fingerprints() -> dict[tuple[str, str, str], dict[str, float]]:
    cols = ["metal", "stage", "year", "inferred_mine_origin_iso", "exporter_iso", "attributed_value_usd"]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(MINE, usecols=cols, chunksize=400_000, low_memory=False):
        chunk = chunk[chunk["year"].eq(BASE_YEAR)].copy()
        chunk["attributed_value_usd"] = pd.to_numeric(chunk["attributed_value_usd"], errors="coerce").fillna(0.0)
        chunk = chunk[
            chunk["attributed_value_usd"].gt(0)
            & chunk["exporter_iso"].notna()
            & chunk["inferred_mine_origin_iso"].notna()
        ]
        if chunk.empty:
            continue
        parts.append(
            chunk.groupby(
                ["metal", "stage", "exporter_iso", "inferred_mine_origin_iso"],
                as_index=False,
                observed=True,
            )["attributed_value_usd"].sum()
        )
    if not parts:
        return {}
    grouped = (
        pd.concat(parts, ignore_index=True)
        .groupby(["metal", "stage", "exporter_iso", "inferred_mine_origin_iso"], as_index=False, observed=True)[
            "attributed_value_usd"
        ].sum()
    )
    grouped["origin_share"] = grouped["attributed_value_usd"] / grouped.groupby(
        ["metal", "stage", "exporter_iso"], observed=True
    )["attributed_value_usd"].transform("sum")
    fingerprints: dict[tuple[str, str, str], dict[str, float]] = {}
    for (metal, stage, exporter), group in grouped.groupby(["metal", "stage", "exporter_iso"], observed=True):
        fingerprints[(str(metal), str(stage), str(exporter))] = dict(
            zip(group["inferred_mine_origin_iso"].astype(str), group["origin_share"].astype(float))
        )
    return fingerprints


def load_routes() -> dict[tuple[str, str, str], tuple[str, ...]]:
    lanes = pd.read_csv(LANES)
    lanes = lanes.rename(columns={"origin": "exporter_iso", "dest": "importer_iso"})
    grouped = lanes.groupby(["metal", "exporter_iso", "importer_iso"], observed=True)["chokepoints"].agg(
        combine_chokepoints
    )
    return {
        (str(metal), str(exporter), str(importer)): tuple(chokes)
        for (metal, exporter, importer), chokes in grouped.items()
    }


def load_xu_recycling() -> dict[str, float]:
    data = pd.read_csv(XU)
    data = data[data["year"].eq(2040)].copy()
    share_col = "xu_steps_ncx_recycling_potential_without_second_life_100pct_share"
    data[share_col] = pd.to_numeric(data[share_col], errors="coerce").fillna(0.0).clip(0.0, 0.95)
    return dict(zip(data["metal"].astype(str), data[share_col].astype(float)))


def top_supplier_pools(capacity: pd.DataFrame) -> dict[tuple[str, str], list[str]]:
    pools: dict[tuple[str, str], list[str]] = {}
    ordered = capacity.sort_values(
        ["metal", "stage", "peak_global_value_usd"], ascending=[True, True, False]
    )
    for (metal, stage), group in ordered.groupby(["metal", "stage"], observed=True):
        pools[(str(metal), str(stage))] = group["exporter_iso"].astype(str).head(MAX_POOL_EXPORTERS).tolist()
    return pools


def build_context(
    unit: pd.Series,
    baseline: pd.DataFrame,
    capacity: pd.DataFrame,
    pools: dict[tuple[str, str], list[str]],
    fingerprints: dict[tuple[str, str, str], dict[str, float]],
    routes: dict[tuple[str, str, str], tuple[str, ...]],
) -> CaseContext:
    importer = str(unit["importer_iso"])
    metal = str(unit["metal"])
    stage = str(unit["stage"])
    base = baseline[
        baseline["importer_iso"].astype(str).eq(importer)
        & baseline["metal"].astype(str).eq(metal)
        & baseline["stage"].astype(str).eq(stage)
    ][["exporter_iso", "reconstructed_value_usd"]].copy()
    base["exporter_iso"] = base["exporter_iso"].astype(str)
    base = base.groupby("exporter_iso", as_index=False)["reconstructed_value_usd"].sum()
    base_map = dict(zip(base["exporter_iso"], base["reconstructed_value_usd"].astype(float)))

    supplier = capacity[
        capacity["metal"].astype(str).eq(metal) & capacity["stage"].astype(str).eq(stage)
    ].copy()
    supplier["exporter_iso"] = supplier["exporter_iso"].astype(str)
    supplier = supplier.set_index("exporter_iso")
    exporters = list(dict.fromkeys(list(base_map) + pools.get((metal, stage), [])))
    exporters = [exporter for exporter in exporters if exporter in supplier.index]

    baseline_value = np.array([base_map.get(exporter, 0.0) for exporter in exporters], dtype=float)
    peak_global = np.array(
        [float(supplier.at[exporter, "peak_global_value_usd"]) for exporter in exporters], dtype=float
    )
    current_global = np.array(
        [float(supplier.at[exporter, "current_global_value_usd"]) for exporter in exporters], dtype=float
    )

    origin_dicts = [fingerprints.get((metal, stage, exporter), {}) for exporter in exporters]
    origin_known = np.array([bool(mapping) for mapping in origin_dicts], dtype=bool)
    origin_labels = sorted({origin for mapping in origin_dicts for origin in mapping}) + ["UNRESOLVED"]
    origin_index = {label: idx for idx, label in enumerate(origin_labels)}
    origin_matrix = np.zeros((len(exporters), len(origin_labels)), dtype=float)
    for row, mapping in enumerate(origin_dicts):
        if not mapping:
            origin_matrix[row, origin_index["UNRESOLVED"]] = 1.0
            continue
        total = sum(max(float(value), 0.0) for value in mapping.values())
        if total <= 0:
            origin_matrix[row, origin_index["UNRESOLVED"]] = 1.0
            origin_known[row] = False
            continue
        for origin, value in mapping.items():
            origin_matrix[row, origin_index[origin]] = max(float(value), 0.0) / total

    route_values = [routes.get((metal, exporter, importer)) for exporter in exporters]
    route_known = np.array([value is not None for value in route_values], dtype=bool)
    route_labels = sorted({choke for value in route_values if value is not None for choke in value})
    route_index = {label: idx for idx, label in enumerate(route_labels)}
    route_matrix = np.zeros((len(exporters), len(route_labels)), dtype=float)
    for row, value in enumerate(route_values):
        if value is None:
            continue
        for choke in value:
            route_matrix[row, route_index[choke]] = 1.0

    expandable = origin_known & route_known
    return CaseContext(
        importer_iso=importer,
        metal=metal,
        stage=stage,
        exporters=exporters,
        baseline_value=baseline_value,
        peak_global_value=peak_global,
        current_global_value=current_global,
        origin_matrix=origin_matrix,
        route_matrix=route_matrix,
        origin_known=origin_known,
        route_known=route_known,
        expandable=expandable,
        origin_labels=origin_labels,
        route_labels=route_labels,
    )


def bounds_for_context(
    context: CaseContext,
    headroom: float,
    target_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_value = context.total_value * target_ratio
    target_baseline = context.baseline_value * target_ratio
    other_commitments = np.maximum(context.current_global_value - context.baseline_value, 0.0)
    envelope = context.peak_global_value * (1.0 + headroom)
    available = np.maximum(envelope - other_commitments, 0.0)
    upper_value = np.maximum(target_baseline, available)
    lower_value = np.zeros_like(upper_value)
    lower_value[~context.expandable] = target_baseline[~context.expandable]
    upper_value[~context.expandable] = target_baseline[~context.expandable]
    lower = lower_value / target_value
    upper = upper_value / target_value
    start = target_baseline / target_value
    return lower, upper, start


def metric_values(context: CaseContext, shares: np.ndarray) -> dict[str, float | str]:
    origin = shares @ context.origin_matrix
    choke = shares @ context.route_matrix if context.route_matrix.shape[1] else np.zeros(0)
    top_exporter_idx = int(np.argmax(shares))
    top_origin_idx = int(np.argmax(origin))
    if len(choke):
        top_choke_idx = int(np.argmax(choke))
        top_choke = context.route_labels[top_choke_idx]
        top_choke_share = float(choke[top_choke_idx])
    else:
        top_choke = ""
        top_choke_share = 0.0
    return {
        "direct_hhi": float(np.square(shares).sum()),
        "origin_hhi": float(np.square(origin).sum()),
        "route_pressure": float(np.square(choke).sum()),
        "top_chokepoint_share": top_choke_share,
        "origin_coverage_share": float(shares[context.origin_known].sum()),
        "route_coverage_share": float(shares[context.route_known].sum()),
        "top_exporter_iso": context.exporters[top_exporter_idx],
        "top_exporter_share": float(shares[top_exporter_idx]),
        "top_origin_iso": context.origin_labels[top_origin_idx],
        "top_origin_share": float(origin[top_origin_idx]),
        "top_chokepoint": top_choke,
    }


def objective_matrix(context: CaseContext, objective: str, baseline_metrics: dict[str, float | str]) -> np.ndarray:
    count = len(context.exporters)
    direct = np.eye(count)
    origin = context.origin_matrix @ context.origin_matrix.T
    if objective == "direct":
        return direct
    if objective == "origin":
        return origin + TIE_BREAKER * direct
    raise ValueError(f"Quadratic objective not available for: {objective}")


def solve_quadratic(
    matrix: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
) -> tuple[np.ndarray, bool, str, int]:
    if lower.sum() > 1.0 + 1e-8 or upper.sum() < 1.0 - 1e-8:
        return start.copy(), False, "infeasible_bounds", 0

    def fun(x: np.ndarray) -> float:
        return float(x @ matrix @ x)

    def jac(x: np.ndarray) -> np.ndarray:
        return 2.0 * (matrix @ x)

    options = {"maxiter": 300, "ftol": 1e-11, "disp": False}
    constraints = [{"type": "eq", "fun": lambda x: float(x.sum() - 1.0), "jac": lambda x: np.ones_like(x)}]
    result = minimize(fun, start, jac=jac, method="SLSQP", bounds=list(zip(lower, upper)), constraints=constraints, options=options)
    if not result.success:
        result = minimize(
            fun,
            np.asarray(result.x, dtype=float),
            jac=jac,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": 1_000, "ftol": 1e-9, "disp": False},
        )
    solution = np.clip(np.asarray(result.x, dtype=float), lower, upper)
    total = solution.sum()
    if total > 0 and abs(total - 1.0) <= 1e-6:
        solution /= total
    feasible = bool(
        result.success
        and abs(solution.sum() - 1.0) <= 1e-6
        and np.all(solution >= lower - 1e-7)
        and np.all(solution <= upper + 1e-7)
    )
    return solution, feasible, str(result.message), int(getattr(result, "nit", 0))


def solve_route_epigraph(
    context: CaseContext,
    objective: str,
    baseline_metrics: dict[str, float | str],
    lower: np.ndarray,
    upper: np.ndarray,
    start: np.ndarray,
    integrated_weights: tuple[float, float, float] = EQUAL_WEIGHTS,
) -> tuple[np.ndarray, bool, str, int]:
    """Solve route or joint objectives using an exact max-exposure epigraph."""
    if lower.sum() > 1.0 + 1e-8 or upper.sum() < 1.0 - 1e-8:
        return start.copy(), False, "infeasible_bounds", 0
    direct_weight, origin_weight, route_weight = integrated_weights
    weight_sum = direct_weight + origin_weight + route_weight
    if objective == "integrated" and (weight_sum <= 0 or min(integrated_weights) < 0):
        raise ValueError("Joint-objective weights must be non-negative and sum to a positive value.")
    if objective == "integrated":
        direct_weight /= weight_sum
        origin_weight /= weight_sum
        route_weight /= weight_sum
    if context.route_matrix.shape[1] == 0:
        if objective == "route":
            matrix = np.eye(len(start))
        else:
            active_sum = max(direct_weight + origin_weight, 1e-12)
            matrix = (
                direct_weight
                * np.eye(len(start))
                / max(float(baseline_metrics["direct_hhi"]), 1e-9)
                + origin_weight
                * (context.origin_matrix @ context.origin_matrix.T)
                / max(float(baseline_metrics["origin_hhi"]), 1e-9)
            ) / active_sum
        return solve_quadratic(matrix, lower, upper, start)

    direct = np.eye(len(start))
    origin = context.origin_matrix @ context.origin_matrix.T
    baseline_top = float(baseline_metrics["top_chokepoint_share"])
    include_route = baseline_top > 1e-9
    if objective == "route":
        quadratic = TIE_BREAKER * direct
        top_weight = 1.0
        warm_matrix = direct
    elif objective == "integrated":
        quadratic = (
            direct_weight * direct / max(float(baseline_metrics["direct_hhi"]), 1e-9)
            + origin_weight * origin / max(float(baseline_metrics["origin_hhi"]), 1e-9)
        )
        top_weight = route_weight / baseline_top if include_route else 0.0
        warm_matrix = quadratic
    else:
        raise ValueError(f"Unknown route epigraph objective: {objective}")

    warm_start, warm_feasible, _, _ = solve_quadratic(warm_matrix, lower, upper, start)
    if not warm_feasible:
        warm_start = start
    start_top = float(np.max(warm_start @ context.route_matrix))
    start_top = min(1.0, start_top + 1e-5)
    start_augmented = np.r_[warm_start, start_top]
    bounds = list(zip(lower, upper)) + [(0.0, 1.0)]

    def fun(augmented: np.ndarray) -> float:
        shares = augmented[:-1]
        return float(shares @ quadratic @ shares + top_weight * augmented[-1])

    def jac(augmented: np.ndarray) -> np.ndarray:
        shares = augmented[:-1]
        return np.r_[2.0 * (quadratic @ shares), top_weight]

    def equality(augmented: np.ndarray) -> float:
        return float(augmented[:-1].sum() - 1.0)

    def equality_jac(augmented: np.ndarray) -> np.ndarray:
        return np.r_[np.ones(len(start)), 0.0]

    def route_constraints(augmented: np.ndarray) -> np.ndarray:
        return augmented[-1] - augmented[:-1] @ context.route_matrix

    def route_constraints_jac(augmented: np.ndarray) -> np.ndarray:
        return np.c_[-context.route_matrix.T, np.ones(context.route_matrix.shape[1])]

    constraints = [
        {"type": "eq", "fun": equality, "jac": equality_jac},
        {"type": "ineq", "fun": route_constraints, "jac": route_constraints_jac},
    ]
    result = minimize(
        fun,
        start_augmented,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-11, "disp": False},
    )
    if not result.success:
        retry = np.asarray(result.x, dtype=float)
        retry[-1] = min(1.0, max(float(retry[-1]), float(np.max(retry[:-1] @ context.route_matrix)) + 1e-6))
        result = minimize(
            fun,
            retry,
            jac=jac,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1_200, "ftol": 1e-9, "disp": False},
        )
    if not result.success:
        count = len(start)
        equality_row = np.r_[np.ones(count), 0.0][None, :]
        route_rows = np.c_[-context.route_matrix.T, np.ones(context.route_matrix.shape[1])]
        linear = LinearConstraint(
            np.vstack([equality_row, route_rows]),
            np.r_[1.0, np.zeros(context.route_matrix.shape[1])],
            np.r_[1.0, np.full(context.route_matrix.shape[1], np.inf)],
        )
        hessian = np.zeros((count + 1, count + 1), dtype=float)
        hessian[:count, :count] = 2.0 * quadratic
        retry = np.asarray(result.x, dtype=float)
        retry[-1] = min(1.0, max(float(retry[-1]), float(np.max(retry[:-1] @ context.route_matrix)) + 1e-6))
        result = minimize(
            fun,
            retry,
            jac=jac,
            hess=lambda x: hessian,
            method="trust-constr",
            bounds=Bounds(np.r_[lower, 0.0], np.r_[upper, 1.0]),
            constraints=[linear],
            options={"maxiter": 800, "gtol": 1e-8, "xtol": 1e-10, "verbose": 0},
        )
    solution = np.clip(np.asarray(result.x[:-1], dtype=float), lower, upper)
    total = solution.sum()
    if total > 0 and abs(total - 1.0) <= 1e-6:
        solution /= total
    choke = solution @ context.route_matrix
    feasible = bool(
        result.success
        and abs(solution.sum() - 1.0) <= 1e-6
        and np.all(solution >= lower - 1e-7)
        and np.all(solution <= upper + 1e-7)
        and float(result.x[-1]) >= float(choke.max()) - 1e-6
    )
    return solution, feasible, str(result.message), int(getattr(result, "nit", 0))


def evaluate_scenario(
    context: CaseContext,
    scenario: str,
    objective: str | None,
    headroom: float,
    recycling_share: float,
    baseline_metrics: dict[str, float | str],
    integrated_weights: tuple[float, float, float] = EQUAL_WEIGHTS,
) -> tuple[dict[str, object], np.ndarray]:
    target_ratio = 1.0 - recycling_share
    lower, upper, start = bounds_for_context(context, headroom, target_ratio)
    if objective is None:
        shares = start.copy()
        feasible, message, iterations = True, "observed baseline", 0
    elif objective in {"direct", "origin"}:
        matrix = objective_matrix(context, objective, baseline_metrics)
        shares, feasible, message, iterations = solve_quadratic(matrix, lower, upper, start)
    else:
        shares, feasible, message, iterations = solve_route_epigraph(
            context,
            objective,
            baseline_metrics,
            lower,
            upper,
            start,
            integrated_weights=integrated_weights,
        )
    metrics = metric_values(context, shares)
    target_value = context.total_value * target_ratio
    scenario_value = shares * target_value
    target_baseline = context.baseline_value * target_ratio
    reallocation = 0.5 * float(np.abs(scenario_value - target_baseline).sum())

    delta_direct = float(metrics["direct_hhi"]) - float(baseline_metrics["direct_hhi"])
    delta_origin = float(metrics["origin_hhi"]) - float(baseline_metrics["origin_hhi"])
    delta_choke = float(metrics["top_chokepoint_share"]) - float(baseline_metrics["top_chokepoint_share"])
    joint_improvement = delta_direct < -1e-6 and delta_origin < -1e-6 and delta_choke < -1e-6
    material_joint = (
        delta_direct <= -MATERIAL_CHANGE
        and delta_origin <= -MATERIAL_CHANGE
        and delta_choke <= -MATERIAL_CHANGE
    )
    risk_transfer = (
        delta_direct <= -MATERIAL_CHANGE
        and (delta_origin >= MATERIAL_CHANGE or delta_choke >= MATERIAL_CHANGE)
    )
    origin_risk_transfer = delta_direct <= -MATERIAL_CHANGE and delta_origin >= MATERIAL_CHANGE
    route_risk_transfer = delta_direct <= -MATERIAL_CHANGE and delta_choke >= MATERIAL_CHANGE
    multiple_risk_transfer = origin_risk_transfer and route_risk_transfer
    if objective == "direct":
        objective_weights = (1.0, 0.0, 0.0)
    elif objective == "origin":
        objective_weights = (0.0, 1.0, 0.0)
    elif objective == "route":
        objective_weights = (0.0, 0.0, 1.0)
    elif objective == "integrated":
        total_weight = max(sum(integrated_weights), 1e-12)
        objective_weights = tuple(value / total_weight for value in integrated_weights)
    else:
        objective_weights = (np.nan, np.nan, np.nan)
    result: dict[str, object] = {
        "importer_iso": context.importer_iso,
        "metal": context.metal,
        "stage": context.stage,
        "scenario": scenario,
        "scenario_label": SCENARIO_LABELS[scenario],
        "objective": objective or "observed",
        "objective_weight_direct": objective_weights[0],
        "objective_weight_origin": objective_weights[1],
        "objective_weight_route": objective_weights[2],
        "headroom_above_peak": headroom,
        "recycling_demand_share": recycling_share,
        "baseline_value_usd": context.total_value,
        "external_import_requirement_usd": target_value,
        "external_import_requirement_ratio": target_ratio,
        "candidate_exporters": len(context.exporters),
        "expandable_exporters": int(context.expandable.sum()),
        "reallocation_value_usd": reallocation,
        "reallocation_share_of_baseline": reallocation / context.total_value,
        "solver_feasible": feasible,
        "solver_message": message,
        "solver_iterations": iterations,
        **metrics,
        "delta_direct_hhi": delta_direct,
        "delta_origin_hhi": delta_origin,
        "delta_route_pressure": float(metrics["route_pressure"]) - float(baseline_metrics["route_pressure"]),
        "delta_top_chokepoint_share": delta_choke,
        "joint_improvement": joint_improvement,
        "material_joint_reduction": material_joint,
        "risk_transfer": risk_transfer,
        "origin_risk_transfer": origin_risk_transfer,
        "route_risk_transfer": route_risk_transfer,
        "multiple_risk_transfer": multiple_risk_transfer,
    }
    return result, scenario_value


def weighted_average(group: pd.DataFrame, column: str) -> float:
    weights = group["baseline_value_usd"].clip(lower=0).to_numpy(float)
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    valid = np.isfinite(weights) & np.isfinite(values) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def summarize_cases(cases: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for labels, group in cases.groupby(group_columns, observed=True, sort=False):
        if not isinstance(labels, tuple):
            labels = (labels,)
        row = dict(zip(group_columns, labels))
        total_value = float(group["baseline_value_usd"].sum())
        row.update(
            {
                "cases": len(group),
                "baseline_value_usd": total_value,
                "value_weighted_direct_hhi": weighted_average(group, "direct_hhi"),
                "value_weighted_origin_hhi": weighted_average(group, "origin_hhi"),
                "value_weighted_top_chokepoint_share": weighted_average(group, "top_chokepoint_share"),
                "value_weighted_route_pressure": weighted_average(group, "route_pressure"),
                "value_weighted_delta_direct_hhi": weighted_average(group, "delta_direct_hhi"),
                "value_weighted_delta_origin_hhi": weighted_average(group, "delta_origin_hhi"),
                "value_weighted_delta_top_chokepoint_share": weighted_average(
                    group, "delta_top_chokepoint_share"
                ),
                "value_weighted_import_requirement_ratio": weighted_average(
                    group, "external_import_requirement_ratio"
                ),
                "value_weighted_reallocation_share": weighted_average(group, "reallocation_share_of_baseline"),
                "joint_improvement_case_share": float(group["joint_improvement"].mean()),
                "material_joint_case_share": float(group["material_joint_reduction"].mean()),
                "risk_transfer_case_share": float(group["risk_transfer"].mean()),
                "origin_transfer_case_share": float(group["origin_risk_transfer"].mean()),
                "route_transfer_case_share": float(group["route_risk_transfer"].mean()),
                "multiple_transfer_case_share": float(group["multiple_risk_transfer"].mean()),
                "joint_improvement_value_share": float(
                    group.loc[group["joint_improvement"], "baseline_value_usd"].sum() / max(total_value, 1.0)
                ),
                "material_joint_value_share": float(
                    group.loc[group["material_joint_reduction"], "baseline_value_usd"].sum()
                    / max(total_value, 1.0)
                ),
                "risk_transfer_value_share": float(
                    group.loc[group["risk_transfer"], "baseline_value_usd"].sum() / max(total_value, 1.0)
                ),
                "origin_transfer_value_share": float(
                    group.loc[group["origin_risk_transfer"], "baseline_value_usd"].sum() / max(total_value, 1.0)
                ),
                "route_transfer_value_share": float(
                    group.loc[group["route_risk_transfer"], "baseline_value_usd"].sum() / max(total_value, 1.0)
                ),
                "multiple_transfer_value_share": float(
                    group.loc[group["multiple_risk_transfer"], "baseline_value_usd"].sum() / max(total_value, 1.0)
                ),
                "solver_feasible_share": float(group["solver_feasible"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_card(
    universe: pd.DataFrame,
    cases: pd.DataFrame,
    summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    weight_sensitivity: pd.DataFrame,
    total_2024_value: float,
) -> None:
    strategy = summary[summary["scenario"].ne("baseline_2024")].copy()
    best = strategy.sort_values("material_joint_value_share", ascending=False).iloc[0]
    partner = summary[summary["scenario"].eq("direct_partner")].iloc[0]
    integrated = summary[summary["scenario"].eq("integrated")].iloc[0]
    circular = summary[summary["scenario"].eq("integrated_recycling")].iloc[0]
    sensitivity_lines = []
    for row in sensitivity.sort_values("headroom_above_peak").itertuples(index=False):
        sensitivity_lines.append(
            f"- peak headroom {row.headroom_above_peak:.0%}: material joint-reduction value share "
            f"{row.material_joint_value_share:.1%}; joint-improvement value share {row.joint_improvement_value_share:.1%}"
        )
    weight_lines = []
    for row in weight_sensitivity.itertuples(index=False):
        weight_lines.append(
            f"- {row.objective_weight_profile} ({row.objective_weight_direct:.0%}/"
            f"{row.objective_weight_origin:.0%}/{row.objective_weight_route:.0%}): "
            f"material joint-reduction {row.material_joint_value_share:.1%}; "
            f"risk transfer {row.risk_transfer_value_share:.1%}"
        )
    lines = [
        "# Capacity-constrained de-risking counterfactual v0",
        "",
        "Generated by `modeling/run_capacity_constrained_derisking_counterfactual.py`.",
        "",
        "## Scope",
        f"- Analysis units: {len(universe):,} importer x metal x stage cases in {BASE_YEAR}",
        f"- Covered baseline value: US${universe['direct_total_value_usd'].sum() / 1e9:,.1f} billion",
        f"- Share of all reconstructed {BASE_YEAR} value: {universe['direct_total_value_usd'].sum() / total_2024_value:.1%}",
        f"- Main capacity envelope: historical export peak plus {MAIN_HEADROOM:.0%}",
        f"- Minimum baseline value: US${MIN_VALUE_USD / 1e6:,.0f} million",
        f"- Minimum origin / route coverage: {MIN_ORIGIN_COVERAGE:.0%} / {MIN_ROUTE_COVERAGE:.0%}",
        "",
        "## Key results",
        f"- Direct-partner strategy risk-transfer value share: {partner['risk_transfer_value_share']:.1%}.",
        f"- Direct-partner route / mine-origin / multiple-transfer value shares: "
        f"{partner['route_transfer_value_share']:.1%} / {partner['origin_transfer_value_share']:.1%} / "
        f"{partner['multiple_transfer_value_share']:.1%}.",
        f"- Joint three-metric objective material joint-reduction value share: {integrated['material_joint_value_share']:.1%}.",
        f"- Joint three-metric objective any joint-improvement value share: {integrated['joint_improvement_value_share']:.1%}.",
        f"- Joint three-metric objective + recycling weighted external-import requirement: {circular['value_weighted_import_requirement_ratio']:.1%} of baseline.",
        f"- Highest material joint-reduction value share: {best['scenario_label']} ({best['material_joint_value_share']:.1%}).",
        f"- Solver-feasible case-scenario share: {cases['solver_feasible'].mean():.1%}.",
        "",
        "## Capacity sensitivity",
        *sensitivity_lines,
        "",
        "## Joint-objective weight sensitivity",
        "Weights are reported as direct / mine-origin / route.",
        *weight_lines,
        "",
        "## Interpretation boundary",
        "The capacity envelope is inferred from 2019-2024 realised export values and is not engineering production capacity.",
        "Mine-origin fingerprints are inferred attribution shares; routes are static corridor-to-chokepoint mappings.",
        "Xu 2040 recycling shares are global battery-material stress-test inputs, not observed importer-specific recycling rates.",
        "The module does not identify causal policy effects, price equilibrium, new processing investment or dynamic vessel rerouting.",
        "",
        "## Outputs",
        f"- `{CASE_OUTPUT.relative_to(ROOT)}`",
        f"- `{ALLOCATION_OUTPUT.relative_to(ROOT)}`",
        f"- `{SUMMARY_OUTPUT.relative_to(ROOT)}`",
        f"- `{METAL_OUTPUT.relative_to(ROOT)}`",
        f"- `{IMPORTER_OUTPUT.relative_to(ROOT)}`",
        f"- `{SENSITIVITY_OUTPUT.relative_to(ROOT)}`",
        f"- `{WEIGHT_SENSITIVITY_CASE_OUTPUT.relative_to(ROOT)}`",
        f"- `{WEIGHT_SENSITIVITY_OUTPUT.relative_to(ROOT)}`",
    ]
    CARD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_inputs()
    _, baseline, capacity = load_trade()
    universe = load_universe()
    fingerprints = load_origin_fingerprints()
    routes = load_routes()
    xu_recycling = load_xu_recycling()
    pools = top_supplier_pools(capacity)

    total_2024_value = float(
        baseline.groupby(KEYS, observed=True)["reconstructed_value_usd"].sum().sum()
    )
    case_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    weight_sensitivity_rows: list[dict[str, object]] = []

    scenario_specs = [
        ("baseline_2024", None, 0.0),
        ("direct_partner", "direct", 0.0),
        ("mine_origin", "origin", 0.0),
        ("route", "route", 0.0),
        ("integrated", "integrated", 0.0),
        ("integrated_recycling", "integrated", None),
    ]

    for position, unit in universe.iterrows():
        context = build_context(unit, baseline, capacity, pools, fingerprints, routes)
        if context.total_value <= 0 or not context.exporters:
            continue
        baseline_shares = context.baseline_value / context.total_value
        baseline_metrics = metric_values(context, baseline_shares)
        recycling_share = float(xu_recycling.get(context.metal, 0.0))

        main_results: dict[str, dict[str, object]] = {}
        for scenario, objective, recycling in scenario_specs:
            scenario_recycling = recycling_share if recycling is None else float(recycling)
            result, values = evaluate_scenario(
                context,
                scenario,
                objective,
                MAIN_HEADROOM,
                scenario_recycling,
                baseline_metrics,
            )
            case_rows.append(result)
            main_results[scenario] = result
            for exporter, baseline_value, scenario_value, peak, current, expandable in zip(
                context.exporters,
                context.baseline_value,
                values,
                context.peak_global_value,
                context.current_global_value,
                context.expandable,
            ):
                if baseline_value <= 0 and scenario_value <= 0:
                    continue
                allocation_rows.append(
                    {
                        "importer_iso": context.importer_iso,
                        "metal": context.metal,
                        "stage": context.stage,
                        "scenario": scenario,
                        "exporter_iso": exporter,
                        "baseline_value_usd": baseline_value,
                        "scenario_value_usd": scenario_value,
                        "scenario_share": scenario_value / max(float(result["external_import_requirement_usd"]), 1.0),
                        "peak_global_value_usd": peak,
                        "current_global_value_usd": current,
                        "expandable_with_full_evidence": bool(expandable),
                    }
                )

        for headroom in HEADROOM_SENSITIVITY:
            if np.isclose(headroom, MAIN_HEADROOM):
                sensitivity_rows.append(dict(main_results["integrated"], headroom_above_peak=headroom))
                continue
            result, _ = evaluate_scenario(
                context,
                "integrated",
                "integrated",
                headroom,
                0.0,
                baseline_metrics,
            )
            sensitivity_rows.append(result)

        for profile, weights in WEIGHT_SENSITIVITY.items():
            if profile == "equal":
                result = dict(main_results["integrated"])
            else:
                result, _ = evaluate_scenario(
                    context,
                    "integrated",
                    "integrated",
                    MAIN_HEADROOM,
                    0.0,
                    baseline_metrics,
                    integrated_weights=weights,
                )
            result["objective_weight_profile"] = profile
            weight_sensitivity_rows.append(result)

        if (position + 1) % 50 == 0 or position + 1 == len(universe):
            print(f"processed {position + 1:,}/{len(universe):,} cases", flush=True)

    cases = pd.DataFrame(case_rows)
    cases["scenario"] = pd.Categorical(cases["scenario"], SCENARIO_ORDER, ordered=True)
    cases = cases.sort_values(KEYS + ["scenario"]).reset_index(drop=True)
    cases.to_csv(CASE_OUTPUT, index=False, compression="gzip")

    allocations = pd.DataFrame(allocation_rows)
    allocations["scenario"] = pd.Categorical(allocations["scenario"], SCENARIO_ORDER, ordered=True)
    allocations = allocations.sort_values(KEYS + ["scenario", "scenario_value_usd"], ascending=[True, True, True, True, False])
    allocations.to_csv(ALLOCATION_OUTPUT, index=False, compression="gzip")

    summary = summarize_cases(cases, ["scenario", "scenario_label", "headroom_above_peak"])
    summary["scenario"] = pd.Categorical(summary["scenario"], SCENARIO_ORDER, ordered=True)
    summary = summary.sort_values("scenario")
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    by_metal = summarize_cases(cases, ["scenario", "scenario_label", "metal"])
    by_metal["scenario"] = pd.Categorical(by_metal["scenario"], SCENARIO_ORDER, ordered=True)
    by_metal = by_metal.sort_values(["scenario", "baseline_value_usd"], ascending=[True, False])
    by_metal.to_csv(METAL_OUTPUT, index=False)

    by_importer = summarize_cases(cases, ["scenario", "scenario_label", "importer_iso"])
    by_importer["scenario"] = pd.Categorical(by_importer["scenario"], SCENARIO_ORDER, ordered=True)
    by_importer = by_importer.sort_values(["scenario", "baseline_value_usd"], ascending=[True, False])
    by_importer.to_csv(IMPORTER_OUTPUT, index=False)

    sensitivity_cases = pd.DataFrame(sensitivity_rows)
    sensitivity = summarize_cases(sensitivity_cases, ["headroom_above_peak"])
    sensitivity = sensitivity.sort_values("headroom_above_peak")
    sensitivity.to_csv(SENSITIVITY_OUTPUT, index=False)

    weight_sensitivity_cases = pd.DataFrame(weight_sensitivity_rows)
    weight_sensitivity_cases.to_csv(WEIGHT_SENSITIVITY_CASE_OUTPUT, index=False, compression="gzip")
    weight_sensitivity = summarize_cases(
        weight_sensitivity_cases,
        [
            "objective_weight_profile",
            "objective_weight_direct",
            "objective_weight_origin",
            "objective_weight_route",
        ],
    )
    profile_order = {name: position for position, name in enumerate(WEIGHT_SENSITIVITY)}
    weight_sensitivity["_order"] = weight_sensitivity["objective_weight_profile"].map(profile_order)
    weight_sensitivity = weight_sensitivity.sort_values("_order").drop(columns="_order")
    weight_sensitivity.to_csv(WEIGHT_SENSITIVITY_OUTPUT, index=False)

    write_card(universe, cases, summary, sensitivity, weight_sensitivity, total_2024_value)
    print(f"analysis cases: {len(universe):,}")
    print(f"scenario rows: {len(cases):,}")
    print(f"allocation rows: {len(allocations):,}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
