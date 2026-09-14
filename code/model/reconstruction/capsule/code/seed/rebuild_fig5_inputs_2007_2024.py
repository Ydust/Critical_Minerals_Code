"""Build candidate 2007--2024 inputs for a full Fig. 5 rerun."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
ROOT = PACKAGE.parent
EXTERNAL = ROOT / "external_sources" / "baci_hs07_202601"
AUDIT = ROOT / "fig5_history_overlap_audit_2008_2012"
RELEASE = PACKAGE / "figure_source_tables" / "longitudinal_2012_2024" / "release_v3"
SNAPSHOT = PACKAGE / "figure_source_tables" / "longitudinal_2012_2024" / "input_snapshot"
OUTPUT = PACKAGE / "figure_source_tables" / "fig5_global_coupled_2012_2024" / "input_build"

BACI = EXTERNAL / "baci_hs07_202601_fig5_selected_2007_2024.csv.gz"
COUNTRIES = EXTERNAL / "country_codes_V202601.csv"
BRIDGE = EXTERNAL / "fig5_hs12_to_hs07_scope.csv"
BGS = AUDIT / "bgs_calibrated_mine_production_2008_2012.csv.gz"
WMD = RELEASE / "merged_wmd_seed_normalized.csv.gz"
LANES = SNAPSHOT / "minerals_lanes.csv"

TRADE_OUT = OUTPUT / "baci_hs07_trade_panel_2007_2024.csv.gz"
SEED_OUT = OUTPUT / "mine_origin_seed_bgs_wmd_2008_2024.csv.gz"
ORIGIN_OUT = OUTPUT / "mine_origin_flows_bgs_wmd_2008_2024.csv.gz"
INDICATOR_OUT = OUTPUT / "risk_indicator_panel_baci_2008_2024.csv.gz"

YEARS_TRADE = set(range(2007, 2025))
YEARS_ORIGIN = set(range(2008, 2025))
KEYS = ["importer_iso", "metal", "stage"]
GROUP = [*KEYS, "year"]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INFER = load_module(
    "fig5_extended_infer",
    PACKAGE / "reproduction" / "project_snapshot" / "modeling" / "infer_mine_origin_flows.py",
)
MRCI = load_module(
    "fig5_extended_mrci",
    PACKAGE / "reproduction" / "project_snapshot" / "longitudinal_2012_2024" / "analysis" / "longitudinal_mrci.py",
)
# The archived module resolves its references relative to the snapshot, whereas
# this workspace keeps the authoritative reporter crosswalk at the repository
# root.  Point the imported helper at that read-only reference explicitly.
INFER.REPORTERS = ROOT / "_raw" / "_ref_Reporters.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_inputs() -> None:
    missing = [path for path in (BACI, COUNTRIES, BRIDGE, BGS, WMD, LANES) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + ", ".join(map(str, missing)))
    gate = pd.read_csv(AUDIT / "audit_gate_summary.csv")
    if not gate["passed"].astype(bool).all():
        failed = gate.loc[~gate["passed"].astype(bool), "check"].tolist()
        raise RuntimeError(f"BGS/WMD overlap audit did not pass: {failed}")


def build_trade() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(BACI, dtype={"k": str})
    raw["k"] = raw["k"].str.zfill(6)
    raw["t"] = pd.to_numeric(raw["t"], errors="raise").astype(int)
    raw["v"] = pd.to_numeric(raw["v"], errors="coerce")
    if set(raw["t"].unique()) != YEARS_TRADE:
        raise RuntimeError("BACI selected file does not cover every year 2007--2024")
    bridge = pd.read_csv(BRIDGE, dtype={"hs07_code": str})
    bridge["hs07_code"] = bridge["hs07_code"].str.zfill(6)
    code_map = bridge[["hs07_code", "metal", "stage"]].drop_duplicates()
    if code_map["hs07_code"].duplicated().any():
        raise RuntimeError("An HS07 code maps to multiple model cells")
    countries = pd.read_csv(COUNTRIES)
    country_map = countries.set_index("country_code")["country_iso3"].astype(str).to_dict()
    raw["exporter_iso"] = raw["i"].map(country_map)
    raw["importer_iso"] = raw["j"].map(country_map)
    raw["reconstructed_value_usd"] = raw["v"] * 1000.0
    raw = raw.merge(code_map, left_on="k", right_on="hs07_code", how="left", validate="many_to_one")
    total_value = float(raw["reconstructed_value_usd"].fillna(0).sum())
    valid = (
        raw["metal"].notna()
        & raw["metal"].ne("Silicon")
        & raw["exporter_iso"].str.fullmatch(r"[A-Z]{3}", na=False)
        & raw["importer_iso"].str.fullmatch(r"[A-Z]{3}", na=False)
        & raw["exporter_iso"].ne(raw["importer_iso"])
        & raw["reconstructed_value_usd"].gt(0)
    )
    kept_value = float(raw.loc[valid, "reconstructed_value_usd"].sum())
    trade = (
        raw.loc[valid]
        .groupby(["metal", "stage", "exporter_iso", "importer_iso", "t"], as_index=False, observed=True)["reconstructed_value_usd"]
        .sum()
        .rename(columns={"t": "year"})
        .sort_values(["year", "metal", "stage", "importer_iso", "exporter_iso"])
        .reset_index(drop=True)
    )
    if trade.duplicated(["metal", "stage", "exporter_iso", "importer_iso", "year"]).any():
        raise RuntimeError("Duplicate BACI trade keys remain after aggregation")
    trade.to_csv(TRADE_OUT, index=False, compression="gzip")
    audit = trade.groupby("year", as_index=False, observed=True).agg(
        rows=("reconstructed_value_usd", "size"),
        metals=("metal", "nunique"),
        trade_value_usd=("reconstructed_value_usd", "sum"),
        exporters=("exporter_iso", "nunique"),
        importers=("importer_iso", "nunique"),
    )
    audit["source_value_retained_share"] = kept_value / total_value if total_value else np.nan
    audit["excluded_trade_scope"] = "Silicon: no mine-origin seed; not treated as zero"
    audit.to_csv(OUTPUT / "trade_panel_annual_audit.csv", index=False)
    return trade, audit


def build_seed() -> tuple[pd.DataFrame, pd.DataFrame]:
    bgs = pd.read_csv(BGS)
    bgs = bgs[bgs["year"].between(2008, 2011)].copy()
    proxy = bgs.rename(columns={"source_country_labels": "mine_origin_country_wmd"})
    proxy = INFER.map_wmd_countries(proxy)
    bgs["mine_origin_iso"] = proxy["mine_origin_iso"]
    bgs["production"] = pd.to_numeric(bgs["calibrated_production"], errors="coerce")
    bgs["metal"] = bgs["model_metal"]
    bgs["source_edition"] = "BGS2008-2012 calibrated to WMD2018 2012"
    bgs["source_dataset"] = "BGS World Mineral Production 2008-2012"
    bgs["source_url"] = "https://www2.bgs.ac.uk/mineralsuk/statistics/worldStatistics.html"
    bgs["source_confidence"] = 0.5
    total = float(bgs["production"].fillna(0).sum())
    mapped = float(bgs.loc[bgs["mine_origin_iso"].notna(), "production"].fillna(0).sum())
    coverage = mapped / total if total else 0.0
    gaps = bgs[bgs["mine_origin_iso"].isna()][["metal", "year", "source_country_labels", "production"]].copy()
    gaps.to_csv(OUTPUT / "bgs_country_crosswalk_gaps_2008_2011.csv", index=False)
    if coverage < 0.95:
        raise RuntimeError(f"BGS country mapping coverage {coverage:.4f} is below 0.95")
    common = ["metal", "year", "mine_origin_iso", "production", "commodity_label", "measurement_basis", "source_edition", "source_dataset", "source_url", "source_confidence"]
    bgs_seed = bgs[bgs["mine_origin_iso"].notna() & bgs["production"].gt(0)][common].copy()
    bgs_seed["source_component"] = "calibrated_BGS"
    wmd = pd.read_csv(WMD)
    wmd_seed = wmd[wmd["year"].between(2012, 2024)][common].copy()
    wmd_seed["source_component"] = "WMD"
    seed = pd.concat([bgs_seed, wmd_seed], ignore_index=True)
    seed = seed.groupby(
        [column for column in seed.columns if column != "production"],
        as_index=False,
        dropna=False,
        observed=True,
    )["production"].sum().sort_values(["year", "metal", "mine_origin_iso"]).reset_index(drop=True)
    coverage_table = seed.groupby(["metal", "year"], as_index=False, observed=True).agg(
        origins=("mine_origin_iso", "nunique"), production=("production", "sum")
    )
    blocks = coverage_table.groupby("metal")["year"].nunique()
    if len(blocks) != 23 or not blocks.eq(17).all():
        raise RuntimeError("Combined seed is not complete for 23 metals and 2008--2024")
    seed.to_csv(SEED_OUT, index=False, compression="gzip")
    coverage_table.to_csv(OUTPUT / "mine_origin_seed_coverage.csv", index=False)
    mapping_audit = pd.DataFrame([{
        "metric": "bgs_2008_2011_country_mapping_value_share",
        "value": coverage,
        "threshold": 0.95,
        "passed": coverage >= 0.95,
    }])
    mapping_audit.to_csv(OUTPUT / "bgs_country_mapping_gate.csv", index=False)
    return seed, mapping_audit


def build_origin_flows(trade: pd.DataFrame, seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prod = seed.rename(columns={"mine_origin_iso": "inferred_mine_origin_iso"}).copy()
    prod = INFER.normalize_mix(prod, ["metal", "year"], "production")
    parts: list[pd.DataFrame] = []
    scoped = trade[trade["year"].isin(YEARS_ORIGIN)]
    for (metal, year), sub in scoped.groupby(["metal", "year"], sort=True, observed=True):
        mix = prod[(prod["metal"].eq(metal)) & (prod["year"].eq(year))]
        if mix.empty:
            continue
        global_mix = dict(zip(mix["inferred_mine_origin_iso"], mix["origin_share"]))
        country_mix = {origin: {origin: 1.0} for origin in global_mix}
        sub = sub.copy()
        sub["data_quality_weight"] = 1.0
        sub["observation_sigma_log"] = 0.1
        for stage in sorted(sub["stage"].dropna().unique(), key=lambda value: INFER.STAGE_ORDER.get(value, 9)):
            inferred, country_mix = INFER.infer_year_stage(
                sub[sub["stage"].eq(stage)], country_mix, global_mix, stage
            )
            if not inferred.empty:
                inferred["mine_seed_component"] = "calibrated_BGS" if year <= 2011 else "WMD"
                parts.append(inferred)
        print(f"origin attribution {year} {metal}", flush=True)
    if not parts:
        raise RuntimeError("No mine-origin flows generated")
    inferred = pd.concat(parts, ignore_index=True)
    inferred.to_csv(ORIGIN_OUT, index=False, compression="gzip")
    direct = scoped.groupby(["metal", "year"], as_index=False, observed=True)["reconstructed_value_usd"].sum()
    attributed = inferred.groupby(["metal", "year"], as_index=False, observed=True)["attributed_value_usd"].sum()
    conservation = direct.merge(attributed, on=["metal", "year"], how="outer").fillna(0)
    conservation["relative_error"] = (
        conservation["attributed_value_usd"] - conservation["reconstructed_value_usd"]
    ).abs() / conservation["reconstructed_value_usd"].replace(0, np.nan)
    conservation.to_csv(OUTPUT / "mine_origin_value_conservation.csv", index=False)
    if len(conservation) != 23 * 17 or float(conservation["relative_error"].max()) > 1e-10:
        raise RuntimeError("Mine-origin value conservation gate failed")
    return inferred, conservation


def build_indicator(trade: pd.DataFrame, inferred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoped = trade[trade["year"].isin(YEARS_ORIGIN)].copy()
    direct_edges = scoped.groupby(GROUP + ["exporter_iso"], as_index=False, observed=True)["reconstructed_value_usd"].sum()
    direct = MRCI.concentration_stats(direct_edges, "exporter_iso", "reconstructed_value_usd", "direct")
    origin_edges = inferred.groupby(GROUP + ["inferred_mine_origin_iso"], as_index=False, observed=True)["attributed_value_usd"].sum()
    origin = MRCI.concentration_stats(origin_edges, "inferred_mine_origin_iso", "attributed_value_usd", "origin")
    routes = MRCI.route_stats(direct_edges, direct, LANES)
    panel = direct.merge(origin, on=GROUP, how="left").merge(routes, on=GROUP, how="left")
    panel["origin_coverage_ratio"] = panel["origin_total_value_usd"] / panel["direct_total_value_usd"].replace(0, np.nan)
    panel = panel.sort_values(GROUP).reset_index(drop=True)
    panel.to_csv(INDICATOR_OUT, index=False, compression="gzip")

    lane_keys = pd.read_csv(LANES, usecols=["metal", "origin", "dest"], low_memory=False).rename(
        columns={"origin": "exporter_iso", "dest": "importer_iso"}
    ).drop_duplicates()
    corridor = scoped.merge(lane_keys, on=["metal", "exporter_iso", "importer_iso"], how="left", indicator=True)
    corridor["matched_value_usd"] = np.where(
        corridor["_merge"].eq("both"), corridor["reconstructed_value_usd"], 0.0
    )
    route_audit = corridor.groupby("year", as_index=False, observed=True).agg(
        total_trade_value_usd=("reconstructed_value_usd", "sum"),
        route_matched_value_usd=("matched_value_usd", "sum"),
        trade_rows=("reconstructed_value_usd", "size"),
        matched_rows=("_merge", lambda values: int(values.eq("both").sum())),
    )
    route_audit["route_matched_value_share"] = route_audit["route_matched_value_usd"] / route_audit["total_trade_value_usd"]
    route_audit["passed_global_early_value_gate"] = np.where(
        route_audit["year"].between(2008, 2011),
        route_audit["route_matched_value_share"].ge(0.50),
        True,
    )
    route_audit.to_csv(OUTPUT / "early_route_mapping_audit.csv", index=False)
    if not route_audit.loc[route_audit["year"].between(2008, 2011), "passed_global_early_value_gate"].all():
        raise RuntimeError("Early route mapping value coverage is below 50%")
    return panel, route_audit


def write_manifest(trade: pd.DataFrame, seed: pd.DataFrame, inferred: pd.DataFrame, panel: pd.DataFrame, audits: list[pd.DataFrame]) -> None:
    audit_pass = all(
        bool(frame.filter(regex="^passed").fillna(True).to_numpy().all())
        for frame in audits
        if not frame.empty and any(column.startswith("passed") for column in frame.columns)
    )
    manifest = {
        "status": "passed" if audit_pass else "failed",
        "scope": "candidate Fig. 5 input rebuild; formal release files untouched",
        "trade_source": "CEPII BACI HS2007 V202601, 2007-2024",
        "mine_source": "calibrated BGS 2008-2011 plus existing WMD 2012-2024",
        "route_source": "frozen static route template retained",
        "rows": {"trade": len(trade), "seed": len(seed), "origin_flows": len(inferred), "indicator": len(panel)},
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in (TRADE_OUT, SEED_OUT, ORIGIN_OUT, INDICATOR_OUT)
        },
    }
    (OUTPUT / "input_build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not audit_pass:
        raise RuntimeError("One or more input gates failed")


def main() -> None:
    require_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trade, trade_audit = build_trade()
    seed, mapping_audit = build_seed()
    inferred, conservation = build_origin_flows(trade, seed)
    panel, route_audit = build_indicator(trade, inferred)
    write_manifest(trade, seed, inferred, panel, [mapping_audit, route_audit])
    print(f"trade rows: {len(trade):,}")
    print(f"seed rows: {len(seed):,}")
    print(f"origin rows: {len(inferred):,}")
    print(f"indicator rows: {len(panel):,}")
    print(OUTPUT)


if __name__ == "__main__":
    main()
