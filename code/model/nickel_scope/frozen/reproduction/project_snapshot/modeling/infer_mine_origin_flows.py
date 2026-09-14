from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "modeling" / "outputs"
REPORTERS = ROOT / "_raw" / "_ref_Reporters.json"
NATION_LIST = ROOT / "external_sources" / "country_reference.csv"

FLOW_FILE = OUT / "dynamic_reconstruction_panel.csv.gz"
MINE_SEED_FILE = OUT / "wmd_mine_origin_seed.csv"
LANE_FILE = ROOT / "minerals_lanes.csv"


METAL_TO_WMD = {
    # WMD reports bauxite and primary aluminium as distinct commodities.
    # Only bauxite is a mine-origin seed; aluminium belongs in the processing
    # output baseline built by build_processing_output_baseline.py.
    "Aluminium": ["Bauxite"],
    "Antimony": ["Antimony"],
    "Boron": ["Boron Minerals"],
    "Chromium": ["Chromium"],
    "Cobalt": ["Cobalt"],
    "Copper": ["Copper"],
    "Graphite": ["Graphite"],
    "Lead": ["Lead"],
    "Lithium": ["Lithium"],
    "Magnesium": ["Magnesite"],
    "Manganese": ["Manganese"],
    "Molybdenum": ["Molybdenum"],
    "Nickel": ["Nickel"],
    "Niobium/Tantalum": ["Niobium", "Tantalum"],
    "PGM": ["Palladium", "Platinum", "Rhodium"],
    "RareEarth": ["Rare Earths"],
    "Silicon": [],
    "Silver": ["Silver"],
    "Tin": ["Tin"],
    "Titanium": ["Titanium"],
    "Tungsten": ["Tungsten"],
    "Vanadium": ["Vanadium"],
    "Zinc": ["Zinc"],
    "Zirconium": ["Zircon"],
}

STAGE_ORDER = {
    "ore": 0,
    "mineral": 0,
    "material": 1,
    "compound": 1,
    "metal": 2,
    "alloy": 3,
    "magnet": 3,
}

LOCAL_STAGE_WEIGHT = {
    "ore": 0.95,
    "mineral": 0.95,
    "material": 0.65,
    "compound": 0.55,
    "metal": 0.45,
    "alloy": 0.25,
    "magnet": 0.20,
}

ALIASES = {
    "bolivia": "BOL",
    "bolivia plurinational state of": "BOL",
    "brunei": "BRN",
    "burma": "MMR",
    "congo dr": "COD",
    "congo d r": "COD",
    "congo d.r": "COD",
    "congo d r": "COD",
    "congo democratic republic": "COD",
    "congo rep": "COG",
    "cote divoire": "CIV",
    "cote d ivoire": "CIV",
    "cote d'ivoire": "CIV",
    "czechia": "CZE",
    "east timor": "TLS",
    "eswatini": "SWZ",
    "iran": "IRN",
    "iran islamic republic of": "IRN",
    "korea north": "PRK",
    "korea south": "KOR",
    "laos": "LAO",
    "laos pdr": "LAO",
    "laos p d r": "LAO",
    "moldova": "MDA",
    "new caledonia": "NCL",
    "russia": "RUS",
    "russia asia": "RUS",
    "russia europe": "RUS",
    "sao tome and principe": "STP",
    "syria": "SYR",
    "taiwan": "TWN",
    "tanzania": "TZA",
    "tanzania united republic of": "TZA",
    "turkiye": "TUR",
    "türkiye": "TUR",
    "united states": "USA",
    "united states of america": "USA",
    "venezuela": "VEN",
    "viet nam": "VNM",
    "vietnam": "VNM",
}


def norm_text(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text).strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def read_country_crosswalk() -> tuple[dict[str, str], set[str]]:
    mapping: dict[str, str] = {}
    valid_iso: set[str] = set()

    if REPORTERS.exists():
        payload = json.loads(REPORTERS.read_text(encoding="utf-8-sig"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        for row in rows:
            iso = row.get("reporterCodeIsoAlpha3")
            if not iso or row.get("isGroup"):
                continue
            valid_iso.add(iso)
            for key in ["text", "reporterDesc", "reporterNote"]:
                name = row.get(key)
                if name:
                    mapping[norm_text(name)] = iso

    if NATION_LIST.exists():
        nations = pd.read_csv(NATION_LIST)
        for _, row in nations.iterrows():
            iso = str(row.get("iso3", "")).strip()
            if len(iso) == 3:
                valid_iso.add(iso)
                mapping[norm_text(row.get("region"))] = iso

    for name, iso in ALIASES.items():
        mapping[norm_text(name)] = iso
        valid_iso.add(iso)

    return mapping, valid_iso


def map_wmd_countries(seed: pd.DataFrame) -> pd.DataFrame:
    mapping, valid_iso = read_country_crosswalk()
    seed = seed.copy()
    seed["mine_origin_iso"] = seed["mine_origin_country_wmd"].map(lambda x: mapping.get(norm_text(x)))
    seed.loc[~seed["mine_origin_iso"].isin(valid_iso), "mine_origin_iso"] = pd.NA
    return seed


def add_model_metal(seed: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for metal, labels in METAL_TO_WMD.items():
        for label in labels:
            pairs.append((metal, label))
    xwalk = pd.DataFrame(pairs, columns=["metal", "commodity_label"])
    out = seed.merge(xwalk, on="commodity_label", how="inner")
    return out


def entropy(shares: pd.Series) -> float:
    numeric = pd.to_numeric(shares, errors="coerce").dropna()
    if numeric.lt(0).any():
        raise ValueError("Origin shares must be non-negative.")
    values = numeric[numeric > 0].to_numpy(dtype=float)
    if len(values) <= 1:
        return 0.0
    values = values / values.sum()
    return float(-(values * np.log(values)).sum() / math.log(len(values)))


def normalize_mix(df: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    numeric = pd.to_numeric(df[value_col], errors="coerce")
    if numeric.dropna().lt(0).any():
        raise ValueError(f"{value_col} must be non-negative.")
    out = df.groupby(group_cols + ["inferred_mine_origin_iso"], as_index=False)[value_col].sum()
    totals = out.groupby(group_cols)[value_col].transform("sum")
    out["origin_share"] = np.where(totals > 0, out[value_col] / totals, 0.0)
    return out[out["origin_share"] > 0].drop(columns=[value_col])


def blend_mix(
    primary: dict[str, float] | None,
    fallback: dict[str, float],
    primary_weight: float,
) -> dict[str, float]:
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("primary_weight must lie in [0, 1].")
    combined: defaultdict[str, float] = defaultdict(float)
    if primary:
        for origin, share in primary.items():
            if not np.isfinite(share) or share < 0:
                raise ValueError("Primary origin shares must be finite and non-negative.")
            combined[origin] += primary_weight * share
        rest = 1.0 - primary_weight
    else:
        rest = 1.0
    for origin, share in fallback.items():
        if not np.isfinite(share) or share < 0:
            raise ValueError("Fallback origin shares must be finite and non-negative.")
        combined[origin] += rest * share
    total = sum(combined.values())
    if total <= 0:
        return {}
    return {origin: value / total for origin, value in combined.items() if value > 0}


def build_production_mix(seed: pd.DataFrame) -> pd.DataFrame:
    prod = seed.dropna(subset=["mine_origin_iso"]).copy()
    prod = prod.rename(columns={"mine_origin_iso": "inferred_mine_origin_iso"})
    prod = prod[prod["production"] > 0]
    prod_mix = normalize_mix(prod, ["metal", "year"], "production")
    return prod_mix


def infer_year_stage(
    flows_stage: pd.DataFrame,
    country_mix: dict[str, dict[str, float]],
    global_mix: dict[str, float],
    stage: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    if flows_stage.empty:
        return pd.DataFrame(), country_mix

    local_weight = LOCAL_STAGE_WEIGHT.get(stage, 0.4)
    records = []
    importer_inputs: dict[str, defaultdict[str, float]] = defaultdict(lambda: defaultdict(float))

    for row in flows_stage.itertuples(index=False):
        exporter = row.exporter_iso
        importer = row.importer_iso
        value = float(row.reconstructed_value_usd)
        if value <= 0:
            continue
        exporter_mix = blend_mix(country_mix.get(exporter), global_mix, local_weight)
        if not exporter_mix:
            continue
        quality = float(getattr(row, "data_quality_weight", 0.5))
        if not np.isfinite(quality):
            quality = 0.5
        quality = float(np.clip(quality, 0.0, 1.0))
        sigma = float(getattr(row, "observation_sigma_log", 0.75))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 0.75
        hub_penalty = max(0.0, 1.0 - local_weight)
        for origin, share in exporter_mix.items():
            attributed = value * share
            if attributed <= 0:
                continue
            importer_inputs[importer][origin] += attributed
            records.append(
                {
                    "metal": row.metal,
                    "stage": stage,
                    "year": int(row.year),
                    "inferred_mine_origin_iso": origin,
                    "exporter_iso": exporter,
                    "importer_iso": importer,
                    "attributed_value_usd": attributed,
                    "origin_share_in_flow": share,
                    "flow_quality_weight": quality,
                    "observation_sigma_log": sigma,
                    "hub_uncertainty": hub_penalty,
                    "method": "wmd_seed_recursive_stage_mix_v0",
                }
            )

    next_mix = dict(country_mix)
    for importer, mix in importer_inputs.items():
        total = sum(mix.values())
        if total <= 0:
            continue
        old = next_mix.get(importer, {})
        blended: defaultdict[str, float] = defaultdict(float)
        for origin, share in old.items():
            blended[origin] += local_weight * share
        for origin, amount in mix.items():
            blended[origin] += (1.0 - local_weight) * amount / total
        norm = sum(blended.values())
        if norm > 0:
            next_mix[importer] = {origin: value / norm for origin, value in blended.items() if value > 0}

    return pd.DataFrame(records), next_mix


def build_flows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(MINE_SEED_FILE)
    seed = add_model_metal(map_wmd_countries(seed))
    seed = seed.dropna(subset=["mine_origin_iso"])

    gaps = pd.read_csv(MINE_SEED_FILE)
    gaps = add_model_metal(map_wmd_countries(gaps))
    gaps = gaps[gaps["mine_origin_iso"].isna()][
        ["metal", "commodity_label", "mine_origin_country_wmd"]
    ].drop_duplicates()

    production_mix = build_production_mix(seed)
    flows = pd.read_csv(
        FLOW_FILE,
        usecols=[
            "metal",
            "stage",
            "exporter_iso",
            "importer_iso",
            "year",
            "reconstructed_value_usd",
            "value_p05_usd",
            "value_p95_usd",
            "data_quality_weight",
            "observation_sigma_log",
        ],
    )
    flows = flows[flows["year"].between(2020, 2024)].copy()
    flows = flows[flows["metal"].isin(seed["metal"].unique())]
    flows = flows[flows["reconstructed_value_usd"] > 0]

    inferred_parts = []
    for (metal, year), sub in flows.groupby(["metal", "year"], sort=True):
        prod = production_mix[(production_mix["metal"].eq(metal)) & (production_mix["year"].eq(year))]
        if prod.empty:
            continue
        global_mix = dict(zip(prod["inferred_mine_origin_iso"], prod["origin_share"]))
        country_mix = {origin: {origin: 1.0} for origin in global_mix}
        for stage in sorted(sub["stage"].dropna().unique(), key=lambda x: STAGE_ORDER.get(x, 9)):
            stage_flows = sub[sub["stage"].eq(stage)]
            inferred, country_mix = infer_year_stage(stage_flows, country_mix, global_mix, stage)
            if not inferred.empty:
                inferred_parts.append(inferred)

    inferred_all = pd.concat(inferred_parts, ignore_index=True) if inferred_parts else pd.DataFrame()
    return inferred_all, gaps, seed


def write_outputs(inferred: pd.DataFrame, gaps: pd.DataFrame, seed: pd.DataFrame) -> None:
    if inferred.empty:
        raise RuntimeError("No inferred mine-origin flows were generated.")

    inferred.to_csv(OUT / "mine_origin_stage_flows_v0.csv.gz", index=False)

    uncertainty = (
        inferred.groupby(["metal", "stage", "year", "exporter_iso", "importer_iso"], as_index=False)
        .agg(
            total_attributed_value_usd=("attributed_value_usd", "sum"),
            origin_count=("inferred_mine_origin_iso", "nunique"),
            attribution_entropy=("origin_share_in_flow", entropy),
            mean_flow_quality_weight=("flow_quality_weight", "mean"),
            mean_observation_sigma_log=("observation_sigma_log", "mean"),
            mean_hub_uncertainty=("hub_uncertainty", "mean"),
        )
        .sort_values("total_attributed_value_usd", ascending=False)
    )
    uncertainty["attribution_uncertainty_index"] = (
        0.35 * uncertainty["attribution_entropy"].fillna(0)
        + 0.25 * (1 - uncertainty["mean_flow_quality_weight"].fillna(0.5))
        + 0.20 * uncertainty["mean_hub_uncertainty"].fillna(0)
        + 0.20 * np.clip(uncertainty["mean_observation_sigma_log"].fillna(0.75) / 1.5, 0, 1)
    )
    uncertainty.to_csv(OUT / "mine_origin_attribution_uncertainty_v0.csv", index=False)

    metal_origin = (
        inferred.groupby(["metal", "stage", "year", "inferred_mine_origin_iso"], as_index=False)[
            "attributed_value_usd"
        ].sum()
    )
    totals = metal_origin.groupby(["metal", "stage", "year"])["attributed_value_usd"].transform("sum")
    metal_origin["origin_share"] = np.where(totals > 0, metal_origin["attributed_value_usd"] / totals, 0.0)
    metal_origin.to_csv(OUT / "mine_origin_stage_origin_shares_v0.csv", index=False)

    coverage = (
        seed.groupby(["metal", "year"], as_index=False)
        .agg(
            wmd_rows=("production", "size"),
            mapped_origin_count=("mine_origin_iso", "nunique"),
            wmd_production=("production", "sum"),
            mean_source_confidence=("source_confidence", "mean"),
        )
        .sort_values(["metal", "year"])
    )
    coverage.to_csv(OUT / "mine_origin_wmd_seed_coverage_v0.csv", index=False)
    gaps.to_csv(OUT / "mine_origin_country_crosswalk_gaps_v0.csv", index=False)

    if LANE_FILE.exists():
        lanes = pd.read_csv(LANE_FILE)
        lanes = lanes.rename(columns={"origin": "exporter_iso", "dest": "importer_iso"})
        lanes = lanes[["metal", "exporter_iso", "importer_iso", "mode", "chokepoints"]].drop_duplicates()
        routed = inferred.merge(
            lanes,
            on=["metal", "exporter_iso", "importer_iso"],
            how="inner",
        )
        routed = routed[routed["mode"].eq("sea") & routed["chokepoints"].notna()].copy()
        if not routed.empty:
            routed["chokepoint"] = routed["chokepoints"].str.split("|")
            routed = routed.explode("chokepoint")
            routed["chokepoint"] = routed["chokepoint"].astype(str).str.strip()
            routed = routed[routed["chokepoint"].ne("")]
            chokepoint = (
                routed.groupby(
                    ["metal", "stage", "year", "inferred_mine_origin_iso", "chokepoint"],
                    as_index=False,
                )
                .agg(
                    attributed_value_usd=("attributed_value_usd", "sum"),
                    mean_flow_quality_weight=("flow_quality_weight", "mean"),
                    mean_hub_uncertainty=("hub_uncertainty", "mean"),
                    corridors=("exporter_iso", "size"),
                )
                .sort_values("attributed_value_usd", ascending=False)
            )
            chokepoint.to_csv(OUT / "mine_origin_chokepoint_exposure_v0.csv", index=False)


def main() -> None:
    inferred, gaps, seed = build_flows()
    write_outputs(inferred, gaps, seed)
    print(f"inferred rows: {len(inferred):,}")
    print(f"metals: {inferred['metal'].nunique():,}")
    print(f"stages: {inferred['stage'].nunique():,}")
    print(f"years: {inferred['year'].min()}-{inferred['year'].max()}")
    print(f"crosswalk gaps: {len(gaps):,}")


if __name__ == "__main__":
    main()
