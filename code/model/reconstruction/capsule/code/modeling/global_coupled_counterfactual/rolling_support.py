"""Rolling input, cohort, metric and local-reference support.

This extension deliberately reproduces the semantics of the locked 2024 Fig. 5
module: every importer-metal-stage unit is optimized separately while all other
importers remain at their observed baseline.  Results from different units are
therefore diagnostic local stress tests and are not a jointly feasible global
allocation.  A separate aggregate-envelope audit quantifies that boundary.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[3]
ROOT = PACKAGE.parents[1]
SNAPSHOT = PACKAGE / "figure_source_tables" / "longitudinal_2012_2024" / "input_snapshot"
RELEASE = PACKAGE / "figure_source_tables" / "longitudinal_2012_2024" / "release_v3"
FORMAL_FIG5 = PACKAGE / "figure_source_tables" / "fig5_counterfactual"
CANONICAL_SCRIPT = HERE.parent / "run_capacity_constrained_derisking_counterfactual.py"

DYNAMIC = SNAPSHOT / "dynamic_reconstruction_panel.csv.gz"
ORIGIN = RELEASE / "longitudinal_mine_origin_flows_2012_2024.csv.gz"
INDICATOR = RELEASE / "longitudinal_risk_indicator_panel_2012_2024.csv.gz"
LANES = SNAPSHOT / "minerals_lanes.csv"
CANONICAL_UNIVERSE = SNAPSHOT / "canonical_main_figure_analysis_universe_704.csv"

FIRST_DATA_YEAR = 2012
LAST_DATA_YEAR = 2024
FIRST_IDENTIFIABLE_BASE_YEAR = 2017
LOOKBACK_YEARS_INCLUSIVE = 6
ENDPOINT_LAG_YEARS = 4
MIN_ENDPOINT_VALUE_USD = 1_000_000.0
MIN_BASE_VALUE_USD = 10_000_000.0
MIN_ORIGIN_COVERAGE = 0.80
MIN_ROUTE_COVERAGE = 0.50
MAIN_HEADROOM = 0.10
MAX_POOL_EXPORTERS = 60
KEYS = ["importer_iso", "metal", "stage"]
SCENARIOS = ("baseline", "direct_partner", "integrated")


def load_canonical_module():
    spec = importlib.util.spec_from_file_location("canonical_counterfactual", CANONICAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical module: {CANONICAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CF = load_canonical_module()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_years(text: str) -> list[int]:
    years: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(token))
    years = sorted(set(years))
    if not years:
        raise ValueError("At least one base year is required")
    invalid = [year for year in years if year < FIRST_IDENTIFIABLE_BASE_YEAR or year > LAST_DATA_YEAR]
    if invalid:
        raise ValueError(
            f"Base years must be {FIRST_IDENTIFIABLE_BASE_YEAR}-{LAST_DATA_YEAR}; invalid={invalid}"
        )
    return years


def combine_chokepoints(values: pd.Series) -> tuple[str, ...]:
    items: list[str] = []
    for value in values.dropna().astype(str):
        for item in value.split("|"):
            item = item.strip()
            if item and item not in items:
                items.append(item)
    return tuple(items)


def load_routes() -> dict[tuple[str, str, str], tuple[str, ...]]:
    lanes = pd.read_csv(LANES, low_memory=False).rename(
        columns={"origin": "exporter_iso", "dest": "importer_iso"}
    )
    grouped = lanes.groupby(
        ["metal", "exporter_iso", "importer_iso"], observed=True
    )["chokepoints"].agg(combine_chokepoints)
    return {
        (str(metal), str(exporter), str(importer)): tuple(chokes)
        for (metal, exporter, importer), chokes in grouped.items()
    }


def load_trade() -> pd.DataFrame:
    columns = [
        "metal",
        "stage",
        "exporter_iso",
        "importer_iso",
        "year",
        "reconstructed_value_usd",
    ]
    trade = pd.read_csv(DYNAMIC, usecols=columns, low_memory=False)
    trade["year"] = pd.to_numeric(trade["year"], errors="coerce").astype("Int64")
    trade["reconstructed_value_usd"] = pd.to_numeric(
        trade["reconstructed_value_usd"], errors="coerce"
    ).fillna(0.0)
    trade = trade[
        trade["year"].between(FIRST_DATA_YEAR, LAST_DATA_YEAR)
        & trade["reconstructed_value_usd"].gt(0)
    ].copy()
    trade["year"] = trade["year"].astype(int)
    return trade.groupby(
        ["metal", "stage", "exporter_iso", "importer_iso", "year"],
        as_index=False,
        observed=True,
    )["reconstructed_value_usd"].sum()


def load_origin_fingerprints(
    years: Iterable[int],
) -> dict[int, dict[tuple[str, str, str], dict[str, float]]]:
    year_set = set(int(year) for year in years)
    columns = [
        "metal",
        "stage",
        "year",
        "inferred_mine_origin_iso",
        "exporter_iso",
        "attributed_value_usd",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(ORIGIN, usecols=columns, chunksize=500_000, low_memory=False):
        chunk = chunk[chunk["year"].isin(year_set)].copy()
        if chunk.empty:
            continue
        chunk["attributed_value_usd"] = pd.to_numeric(
            chunk["attributed_value_usd"], errors="coerce"
        ).fillna(0.0)
        chunk = chunk[
            chunk["attributed_value_usd"].gt(0)
            & chunk["exporter_iso"].notna()
            & chunk["inferred_mine_origin_iso"].notna()
        ]
        if chunk.empty:
            continue
        parts.append(
            chunk.groupby(
                [
                    "year",
                    "metal",
                    "stage",
                    "exporter_iso",
                    "inferred_mine_origin_iso",
                ],
                as_index=False,
                observed=True,
            )["attributed_value_usd"].sum()
        )
    if not parts:
        return {year: {} for year in year_set}
    grouped = pd.concat(parts, ignore_index=True).groupby(
        ["year", "metal", "stage", "exporter_iso", "inferred_mine_origin_iso"],
        as_index=False,
        observed=True,
    )["attributed_value_usd"].sum()
    grouped["origin_share"] = grouped["attributed_value_usd"] / grouped.groupby(
        ["year", "metal", "stage", "exporter_iso"], observed=True
    )["attributed_value_usd"].transform("sum")
    result: dict[int, dict[tuple[str, str, str], dict[str, float]]] = {
        year: {} for year in year_set
    }
    for (year, metal, stage, exporter), group in grouped.groupby(
        ["year", "metal", "stage", "exporter_iso"], observed=True
    ):
        result[int(year)][(str(metal), str(stage), str(exporter))] = dict(
            zip(
                group["inferred_mine_origin_iso"].astype(str),
                group["origin_share"].astype(float),
            )
        )
    return result


def load_indicator() -> pd.DataFrame:
    frame = pd.read_csv(INDICATOR, low_memory=False)
    for column in [
        "direct_total_value_usd",
        "origin_coverage_ratio",
        "route_coverage_share",
        "direct_hhi",
        "origin_hhi",
        "top_chokepoint_share",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def rolling_universe(indicator: pd.DataFrame, base_year: int) -> pd.DataFrame:
    start_year = base_year - ENDPOINT_LAG_YEARS
    metrics = [
        "direct_total_value_usd",
        "origin_coverage_ratio",
        "route_coverage_share",
        "direct_hhi",
        "origin_hhi",
        "top_chokepoint_share",
    ]
    start = indicator[indicator["year"].eq(start_year)][KEYS + metrics].copy()
    end = indicator[indicator["year"].eq(base_year)][KEYS + metrics].copy()
    pair = start.merge(end, on=KEYS, how="inner", suffixes=("_start", "_end"))
    pair = pair[
        pair[["direct_total_value_usd_start", "direct_total_value_usd_end"]]
        .min(axis=1)
        .ge(MIN_ENDPOINT_VALUE_USD)
        & pair[["origin_coverage_ratio_start", "origin_coverage_ratio_end"]]
        .min(axis=1)
        .ge(MIN_ORIGIN_COVERAGE)
        & pair[["route_coverage_share_start", "route_coverage_share_end"]]
        .min(axis=1)
        .ge(MIN_ROUTE_COVERAGE)
        & pair["direct_total_value_usd_end"].ge(MIN_BASE_VALUE_USD)
    ].copy()
    pair["base_year"] = base_year
    pair["eligibility_start_year"] = start_year
    pair["envelope_start_year"] = base_year - (LOOKBACK_YEARS_INCLUSIVE - 1)
    pair["envelope_end_year"] = base_year
    pair["unit_key"] = pair[KEYS].astype(str).agg("|".join, axis=1)
    return pair.sort_values(["direct_total_value_usd_end", *KEYS], ascending=[False, True, True, True])


def year_trade_components(
    trade: pd.DataFrame, base_year: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, str], list[str]]]:
    envelope_start = base_year - (LOOKBACK_YEARS_INCLUSIVE - 1)
    scoped = trade[trade["year"].between(envelope_start, base_year)].copy()
    baseline = scoped[scoped["year"].eq(base_year)].drop(columns="year")
    global_annual = scoped.groupby(
        ["metal", "stage", "exporter_iso", "year"], as_index=False, observed=True
    )["reconstructed_value_usd"].sum()
    peak = global_annual.groupby(
        ["metal", "stage", "exporter_iso"], as_index=False, observed=True
    ).agg(peak_global_value_usd=("reconstructed_value_usd", "max"))
    current = global_annual[global_annual["year"].eq(base_year)][
        ["metal", "stage", "exporter_iso", "reconstructed_value_usd"]
    ].rename(columns={"reconstructed_value_usd": "current_global_value_usd"})
    capacity = peak.merge(current, on=["metal", "stage", "exporter_iso"], how="left")
    capacity["current_global_value_usd"] = capacity["current_global_value_usd"].fillna(0.0)
    capacity = capacity[capacity["current_global_value_usd"].gt(0)].copy()
    ordered = capacity.sort_values(
        ["metal", "stage", "peak_global_value_usd"], ascending=[True, True, False]
    )
    pools: dict[tuple[str, str], list[str]] = {}
    for (metal, stage), group in ordered.groupby(["metal", "stage"], observed=True):
        pools[(str(metal), str(stage))] = (
            group["exporter_iso"].astype(str).head(MAX_POOL_EXPORTERS).tolist()
        )
    return baseline, capacity, pools


def build_context(
    unit: pd.Series,
    baseline: pd.DataFrame,
    capacity: pd.DataFrame,
    pools: dict[tuple[str, str], list[str]],
    fingerprints: dict[tuple[str, str, str], dict[str, float]],
    routes: dict[tuple[str, str, str], tuple[str, ...]],
):
    importer = str(unit["importer_iso"])
    metal = str(unit["metal"])
    stage = str(unit["stage"])
    base = baseline[
        baseline["importer_iso"].astype(str).eq(importer)
        & baseline["metal"].astype(str).eq(metal)
        & baseline["stage"].astype(str).eq(stage)
    ][["exporter_iso", "reconstructed_value_usd"]].copy()
    base["exporter_iso"] = base["exporter_iso"].astype(str)
    base = base.groupby("exporter_iso", as_index=False, observed=True)[
        "reconstructed_value_usd"
    ].sum()
    base_map = dict(zip(base["exporter_iso"], base["reconstructed_value_usd"].astype(float)))

    supplier = capacity[
        capacity["metal"].astype(str).eq(metal)
        & capacity["stage"].astype(str).eq(stage)
    ].copy()
    supplier["exporter_iso"] = supplier["exporter_iso"].astype(str)
    supplier = supplier.set_index("exporter_iso")
    exporters = list(dict.fromkeys([*base_map, *pools.get((metal, stage), [])]))
    exporters = [exporter for exporter in exporters if exporter in supplier.index]
    baseline_value = np.array([base_map.get(exporter, 0.0) for exporter in exporters], dtype=float)
    peak_global = np.array(
        [float(supplier.at[exporter, "peak_global_value_usd"]) for exporter in exporters],
        dtype=float,
    )
    current_global = np.array(
        [float(supplier.at[exporter, "current_global_value_usd"]) for exporter in exporters],
        dtype=float,
    )

    origin_dicts = [fingerprints.get((metal, stage, exporter), {}) for exporter in exporters]
    origin_known = np.array([bool(mapping) for mapping in origin_dicts], dtype=bool)
    origin_labels = sorted({origin for mapping in origin_dicts for origin in mapping}) + [
        "UNRESOLVED"
    ]
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
    route_labels = sorted(
        {choke for value in route_values if value is not None for choke in value}
    )
    route_index = {label: idx for idx, label in enumerate(route_labels)}
    route_matrix = np.zeros((len(exporters), len(route_labels)), dtype=float)
    for row, value in enumerate(route_values):
        if value is None:
            continue
        for choke in value:
            route_matrix[row, route_index[choke]] = 1.0
    expandable = origin_known & route_known
    return CF.CaseContext(
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


def scenario_specifications() -> list[tuple[str, str | None]]:
    return [("baseline", None), ("direct_partner", "direct"), ("integrated", "integrated")]


def run_year(
    base_year: int,
    universe: pd.DataFrame,
    trade: pd.DataFrame,
    fingerprints: dict[tuple[str, str, str], dict[str, float]],
    routes: dict[tuple[str, str, str], tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline, capacity, pools = year_trade_components(trade, base_year)
    case_rows: list[dict[str, object]] = []
    allocation_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    for position, unit in universe.reset_index(drop=True).iterrows():
        context = build_context(unit, baseline, capacity, pools, fingerprints, routes)
        if context.total_value <= 0 or not context.exporters:
            continue
        baseline_shares = context.baseline_value / context.total_value
        baseline_metrics = CF.metric_values(context, baseline_shares)
        lower, upper, _ = CF.bounds_for_context(context, MAIN_HEADROOM, 1.0)
        other_commitments = np.maximum(context.current_global_value - context.baseline_value, 0.0)
        envelope = context.peak_global_value * (1.0 + MAIN_HEADROOM)
        available = np.maximum(envelope - other_commitments, 0.0)
        context_rows.append(
            {
                "base_year": base_year,
                "importer_iso": context.importer_iso,
                "metal": context.metal,
                "stage": context.stage,
                "baseline_value_usd": context.total_value,
                "candidate_exporters": len(context.exporters),
                "expandable_exporters": int(context.expandable.sum()),
                "baseline_value_reconciliation_usd": context.total_value
                - float(unit["direct_total_value_usd_end"]),
            }
        )
        for scenario, objective in scenario_specifications():
            canonical_scenario = "baseline_2024" if scenario == "baseline" else scenario
            result, scenario_values = CF.evaluate_scenario(
                context,
                canonical_scenario,
                objective,
                MAIN_HEADROOM,
                0.0,
                baseline_metrics,
            )
            result.update(
                {
                    "base_year": base_year,
                    "scenario": scenario,
                    "scenario_label": {
                        "baseline": f"{base_year} observed baseline",
                        "direct_partner": "Direct-only local stress test",
                        "integrated": "Joint-objective local stress test",
                    }[scenario],
                    "eligibility_start_year": base_year - ENDPOINT_LAG_YEARS,
                    "envelope_start_year": base_year - (LOOKBACK_YEARS_INCLUSIVE - 1),
                    "envelope_end_year": base_year,
                    "trade_max_year_used": base_year,
                    "origin_fingerprint_year": base_year,
                    "route_fingerprint_scope": "static_representative_geometry_no_trade_weights",
                    "interpretation_scope": "one_unit_at_a_time_other_importers_fixed",
                }
            )
            case_rows.append(result)
            for idx, exporter in enumerate(context.exporters):
                baseline_value = float(context.baseline_value[idx])
                scenario_value = float(scenario_values[idx])
                if baseline_value <= 0 and scenario_value <= 0:
                    continue
                allocation_rows.append(
                    {
                        "base_year": base_year,
                        "importer_iso": context.importer_iso,
                        "metal": context.metal,
                        "stage": context.stage,
                        "scenario": scenario,
                        "exporter_iso": exporter,
                        "baseline_value_usd": baseline_value,
                        "scenario_value_usd": scenario_value,
                        "scenario_share": scenario_value / context.total_value,
                        "lower_value_usd": float(lower[idx] * context.total_value),
                        "upper_value_usd": float(upper[idx] * context.total_value),
                        "peak_global_value_usd": float(context.peak_global_value[idx]),
                        "current_global_value_usd": float(context.current_global_value[idx]),
                        "other_importer_commitments_usd": float(other_commitments[idx]),
                        "export_envelope_value_usd": float(envelope[idx]),
                        "available_to_this_unit_usd": float(available[idx]),
                        "expandable_with_full_evidence": bool(context.expandable[idx]),
                    }
                )
        if (position + 1) % 100 == 0 or position + 1 == len(universe):
            print(
                f"year {base_year}: processed {position + 1:,}/{len(universe):,} local units",
                flush=True,
            )
    return pd.DataFrame(case_rows), pd.DataFrame(allocation_rows), pd.DataFrame(context_rows)


def summarize(cases: pd.DataFrame, cohort: str) -> pd.DataFrame:
    summary = CF.summarize_cases(
        cases,
        [
            "base_year",
            "scenario",
            "scenario_label",
            "eligibility_start_year",
            "envelope_start_year",
            "envelope_end_year",
        ],
    )
    summary.insert(0, "cohort", cohort)
    return summary


def unit_conservation_audit(
    cases: pd.DataFrame, allocations: pd.DataFrame
) -> pd.DataFrame:
    grouped = allocations.groupby(
        ["base_year", *KEYS, "scenario"], as_index=False, observed=True
    ).agg(
        allocated_value_usd=("scenario_value_usd", "sum"),
        minimum_allocation_usd=("scenario_value_usd", "min"),
        maximum_upper_excess_usd=(
            "scenario_value_usd",
            lambda values: 0.0,
        ),
    )
    # Compute bound diagnostics before group aggregation to avoid hidden cancellations.
    allocations = allocations.copy()
    allocations["upper_excess_usd"] = (
        allocations["scenario_value_usd"] - allocations["upper_value_usd"]
    ).clip(lower=0.0)
    allocations["lower_deficit_usd"] = (
        allocations["lower_value_usd"] - allocations["scenario_value_usd"]
    ).clip(lower=0.0)
    allocations["frozen_delta_usd"] = np.where(
        allocations["expandable_with_full_evidence"],
        0.0,
        np.abs(allocations["scenario_value_usd"] - allocations["baseline_value_usd"]),
    )
    bounds = allocations.groupby(
        ["base_year", *KEYS, "scenario"], as_index=False, observed=True
    ).agg(
        max_upper_excess_usd=("upper_excess_usd", "max"),
        max_lower_deficit_usd=("lower_deficit_usd", "max"),
        max_frozen_delta_usd=("frozen_delta_usd", "max"),
    )
    grouped = grouped.drop(columns="maximum_upper_excess_usd").merge(
        bounds, on=["base_year", *KEYS, "scenario"], how="left"
    )
    target = cases[
        ["base_year", *KEYS, "scenario", "external_import_requirement_usd", "solver_feasible"]
    ]
    unit = target.merge(grouped, on=["base_year", *KEYS, "scenario"], how="left")
    unit["demand_error_usd"] = (
        unit["allocated_value_usd"] - unit["external_import_requirement_usd"]
    )
    rows: list[dict[str, object]] = []
    for (year, scenario), data in unit.groupby(["base_year", "scenario"], observed=True):
        rows.append(
            {
                "base_year": int(year),
                "scenario": str(scenario),
                "units": len(data),
                "solver_feasible_share": float(data["solver_feasible"].mean()),
                "max_abs_demand_error_usd": float(data["demand_error_usd"].abs().max()),
                "max_abs_demand_error_share": float(
                    (data["demand_error_usd"].abs() / data["external_import_requirement_usd"]).max()
                ),
                "minimum_allocation_usd": float(data["minimum_allocation_usd"].min()),
                "max_upper_excess_usd": float(data["max_upper_excess_usd"].max()),
                "max_lower_deficit_usd": float(data["max_lower_deficit_usd"].max()),
                "max_frozen_delta_usd": float(data["max_frozen_delta_usd"].max()),
            }
        )
    return pd.DataFrame(rows)


def aggregate_envelope_audit(allocations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = allocations.groupby(
        ["base_year", "metal", "stage", "exporter_iso", "scenario"],
        as_index=False,
        observed=True,
    ).agg(
        selected_baseline_value_usd=("baseline_value_usd", "sum"),
        selected_scenario_value_usd=("scenario_value_usd", "sum"),
        peak_global_value_usd=("peak_global_value_usd", "max"),
        current_global_value_usd=("current_global_value_usd", "max"),
        selected_units=("importer_iso", "nunique"),
    )
    grouped["implied_global_scenario_value_usd"] = (
        grouped["current_global_value_usd"]
        - grouped["selected_baseline_value_usd"]
        + grouped["selected_scenario_value_usd"]
    )
    grouped["envelope_value_usd"] = grouped["peak_global_value_usd"] * (1.0 + MAIN_HEADROOM)
    grouped["envelope_excess_usd"] = (
        grouped["implied_global_scenario_value_usd"] - grouped["envelope_value_usd"]
    ).clip(lower=0.0)
    grouped["envelope_ratio"] = grouped["implied_global_scenario_value_usd"] / grouped[
        "envelope_value_usd"
    ].replace(0.0, np.nan)
    rows: list[dict[str, object]] = []
    for (year, scenario), data in grouped.groupby(["base_year", "scenario"], observed=True):
        rows.append(
            {
                "base_year": int(year),
                "scenario": str(scenario),
                "exporter_metal_stage_rows": len(data),
                "violating_exporter_metal_stage_rows": int(
                    data["envelope_excess_usd"].gt(1.0).sum()
                ),
                "total_envelope_excess_usd": float(data["envelope_excess_usd"].sum()),
                "maximum_envelope_ratio": float(data["envelope_ratio"].max()),
                "interpretation": "aggregate diagnostic; local unit solutions are not simultaneous",
            }
        )
    return grouped, pd.DataFrame(rows)


def metric_reconciliation_audit(
    cases: pd.DataFrame, indicator: pd.DataFrame, contexts: pd.DataFrame
) -> pd.DataFrame:
    baseline = cases[cases["scenario"].eq("baseline")].copy()
    expected = indicator[
        indicator["year"].isin(baseline["base_year"].unique())
    ][KEYS + ["year", "direct_hhi", "origin_hhi", "top_chokepoint_share"]].rename(
        columns={"year": "base_year"}
    )
    merged = baseline.merge(expected, on=["base_year", *KEYS], suffixes=("_model", "_indicator"))
    context_max = contexts.groupby("base_year", observed=True)[
        "baseline_value_reconciliation_usd"
    ].apply(lambda values: float(values.abs().max()))
    rows: list[dict[str, object]] = []
    for year, data in merged.groupby("base_year", observed=True):
        rows.append(
            {
                "base_year": int(year),
                "units": len(data),
                "max_abs_baseline_value_difference_usd": float(context_max.loc[year]),
                "max_abs_direct_hhi_difference": float(
                    (data["direct_hhi_model"] - data["direct_hhi_indicator"]).abs().max()
                ),
                "max_abs_origin_hhi_difference": float(
                    (data["origin_hhi_model"] - data["origin_hhi_indicator"]).abs().max()
                ),
                "max_abs_top_chokepoint_share_difference": float(
                    (
                        data["top_chokepoint_share_model"]
                        - data["top_chokepoint_share_indicator"]
                    ).abs().max()
                ),
            }
        )
    return pd.DataFrame(rows)


def regression_2024(cases: pd.DataFrame) -> pd.DataFrame:
    subset = cases[cases["base_year"].eq(2024)].copy()
    if subset.empty:
        return pd.DataFrame()
    formal = pd.read_csv(
        FORMAL_FIG5 / "capacity_constrained_derisking_scenarios_704x6.csv.gz",
        low_memory=False,
    )
    formal = formal[formal["scenario"].isin(["baseline_2024", "direct_partner", "integrated"])].copy()
    formal["scenario"] = formal["scenario"].replace({"baseline_2024": "baseline"})
    metrics = [
        "baseline_value_usd",
        "direct_hhi",
        "origin_hhi",
        "top_chokepoint_share",
        "delta_direct_hhi",
        "delta_origin_hhi",
        "delta_top_chokepoint_share",
        "reallocation_share_of_baseline",
    ]
    merged = subset.merge(formal[KEYS + ["scenario", *metrics]], on=KEYS + ["scenario"], suffixes=("_rolling", "_formal"))
    rows: list[dict[str, object]] = []
    for scenario, data in merged.groupby("scenario", observed=True):
        row: dict[str, object] = {
            "scenario": str(scenario),
            "matched_units": len(data),
            "rolling_units": int(subset[subset["scenario"].eq(scenario)].shape[0]),
            "formal_units": int(formal[formal["scenario"].eq(scenario)].shape[0]),
        }
        for metric in metrics:
            row[f"max_abs_difference_{metric}"] = float(
                (data[f"{metric}_rolling"] - data[f"{metric}_formal"]).abs().max()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def early_year_identifiability() -> pd.DataFrame:
    rows = []
    for base_year in range(FIRST_DATA_YEAR, FIRST_IDENTIFIABLE_BASE_YEAR):
        required_start = base_year - (LOOKBACK_YEARS_INCLUSIVE - 1)
        missing_years = list(range(required_start, FIRST_DATA_YEAR))
        rows.append(
            {
                "base_year": base_year,
                "required_envelope_start_year": required_start,
                "available_trade_start_year": FIRST_DATA_YEAR,
                "missing_prior_years": "|".join(map(str, missing_years)),
                "missing_prior_year_count": len(missing_years),
                "status": "not_identifiable_under_fixed_six_year_export_envelope",
                "minimum_alternative": "shorter variable lookback; not directly comparable",
            }
        )
    return pd.DataFrame(rows)


def write_manifest(
    output_dir: Path,
    years: list[int],
    universes: pd.DataFrame,
    cases: pd.DataFrame,
    allocations: pd.DataFrame,
) -> None:
    manifest = {
        "analysis": "rolling one-unit-at-a-time local allocation stress test",
        "base_years": years,
        "lookback_definition": "six years inclusive: y-5 through y",
        "eligibility_definition": "four-year endpoint pair y-4 and y; both endpoints >=US$1m, origin coverage >=80%, route coverage >=50%; end year >=US$10m",
        "future_information_rule": "trade and supplier pools use years <= y; mine-origin fingerprints use y; static route geometry has no trade weights",
        "interpretation_boundary": "each unit holds all other importers fixed; pooled local solutions are not a simultaneous global allocation",
        "rows": {
            "universes": len(universes),
            "cases": len(cases),
            "allocations": len(allocations),
        },
        "inputs": {
            str(path.relative_to(ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for path in [DYNAMIC, ORIGIN, INDICATOR, LANES, CANONICAL_UNIVERSE]
        },
    }
    (output_dir / "rolling_local_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017-2024")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE.parent / "outputs" / "local_one_unit_diagnostic",
    )
    args = parser.parse_args()
    years = parse_years(args.years)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    indicator = load_indicator()
    universes = {year: rolling_universe(indicator, year) for year in years}
    if 2024 in universes:
        canonical = pd.read_csv(CANONICAL_UNIVERSE, usecols=KEYS).astype(str)
        canonical_keys = set(map(tuple, canonical.itertuples(index=False, name=None)))
        rolling_keys = set(
            map(tuple, universes[2024][KEYS].astype(str).itertuples(index=False, name=None))
        )
        if canonical_keys != rolling_keys:
            raise AssertionError(
                f"2024 rolling universe differs from canonical 704: "
                f"rolling={len(rolling_keys)}, canonical={len(canonical_keys)}, "
                f"rolling_only={len(rolling_keys-canonical_keys)}, canonical_only={len(canonical_keys-rolling_keys)}"
            )

    trade = load_trade()
    if int(trade["year"].max()) != LAST_DATA_YEAR:
        raise AssertionError("Trade loader must exclude 2025 and stop at 2024")
    routes = load_routes()
    fingerprints = load_origin_fingerprints(years)

    all_cases: list[pd.DataFrame] = []
    all_allocations: list[pd.DataFrame] = []
    all_contexts: list[pd.DataFrame] = []
    for year in years:
        print(
            f"starting {year}: units={len(universes[year]):,}, "
            f"envelope={year - 5}-{year}, origin fingerprint={year}",
            flush=True,
        )
        cases, allocations, contexts = run_year(
            year,
            universes[year],
            trade,
            fingerprints[year],
            routes,
        )
        all_cases.append(cases)
        all_allocations.append(allocations)
        all_contexts.append(contexts)

    cases = pd.concat(all_cases, ignore_index=True)
    allocations = pd.concat(all_allocations, ignore_index=True)
    contexts = pd.concat(all_contexts, ignore_index=True)
    universe_table = pd.concat([universes[year] for year in years], ignore_index=True)
    cases = cases.sort_values(["base_year", *KEYS, "scenario"]).reset_index(drop=True)
    allocations = allocations.sort_values(
        ["base_year", *KEYS, "scenario", "scenario_value_usd"],
        ascending=[True, True, True, True, True, False],
    ).reset_index(drop=True)

    summaries = [summarize(cases, "rolling_four_year_endpoint_cohort")]
    balanced_years = [year for year in years if 2020 <= year <= 2024]
    if balanced_years:
        key_sets = [set(universes[year]["unit_key"]) for year in balanced_years]
        balanced_keys = set.intersection(*key_sets)
        balanced_cases = cases[
            cases["base_year"].isin(balanced_years)
            & cases[KEYS].astype(str).agg("|".join, axis=1).isin(balanced_keys)
        ].copy()
        summaries.append(summarize(balanced_cases, "balanced_2020_2024"))
    summary = pd.concat(summaries, ignore_index=True)

    unit_audit = unit_conservation_audit(cases, allocations)
    global_detail, global_summary = aggregate_envelope_audit(allocations)
    metric_audit = metric_reconciliation_audit(cases, indicator, contexts)
    regression = regression_2024(cases)
    early = early_year_identifiability()

    universe_table.to_csv(output_dir / "rolling_local_universe_2017_2024.csv", index=False)
    cases.to_csv(
        output_dir / "rolling_local_cases_2017_2024.csv.gz", index=False, compression="gzip"
    )
    allocations.to_csv(
        output_dir / "rolling_local_supplier_allocations_2017_2024.csv.gz",
        index=False,
        compression="gzip",
    )
    summary.to_csv(output_dir / "rolling_local_annual_summary_2017_2024.csv", index=False)
    unit_audit.to_csv(output_dir / "rolling_local_unit_conservation_audit.csv", index=False)
    global_detail.to_csv(
        output_dir / "rolling_local_aggregate_envelope_detail.csv.gz",
        index=False,
        compression="gzip",
    )
    global_summary.to_csv(
        output_dir / "rolling_local_aggregate_envelope_summary.csv", index=False
    )
    metric_audit.to_csv(output_dir / "rolling_local_metric_reconciliation_audit.csv", index=False)
    regression.to_csv(output_dir / "rolling_local_2024_regression_audit.csv", index=False)
    early.to_csv(output_dir / "early_year_identifiability.csv", index=False)
    write_manifest(output_dir, years, universe_table, cases, allocations)

    print(summary.to_string(index=False), flush=True)
    print(unit_audit.to_string(index=False), flush=True)
    print(global_summary.to_string(index=False), flush=True)
    print(f"wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
