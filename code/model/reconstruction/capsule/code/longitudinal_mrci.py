"""Core algorithms for the isolated 2012--2024 longitudinal MRCI analysis.

This module deliberately does not import or write any canonical project module.
Its default formulas reproduce the frozen v2 point-estimate algorithms, while
generalising the time axis to adjacent and four-year endpoint windows.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


KEYS = ["importer_iso", "metal", "stage"]
GROUP = [*KEYS, "year"]
MINE_FLOW_KEY = [
    "CmdCode",
    "metal",
    "stage",
    "year",
    "inferred_mine_origin_iso",
    "exporter_iso",
    "importer_iso",
]
DYNAMIC_FLOW_KEY = [
    "CmdCode",
    "metal",
    "stage",
    "exporter_iso",
    "importer_iso",
    "year",
]

DEFAULT_START_YEAR = 2012
DEFAULT_END_YEAR = 2024
DEFAULT_THRESHOLD = 0.025
DEFAULT_THRESHOLDS = (0.0, 0.01, 0.025, 0.05)
WMD_EDITION_BOUNDARY_YEARS = (2015, 2020)
WMD_EDITION_EXCLUSION_RULE = "exclude_wmd_edition_boundary_crossings"
MIN_ENDPOINT_VALUE_USD = 1_000_000.0
MIN_ORIGIN_COVERAGE = 0.80
MIN_ROUTE_COVERAGE = 0.50

TRANSFER_CLASSES = {
    "mine_origin_transfer",
    "route_transfer",
    "multiple_risk_transfer",
}

CATEGORY_ORDER = [
    "substantive_derisking",
    "mine_origin_transfer",
    "route_transfer",
    "multiple_risk_transfer",
    "evidence_weakened",
    "partial_derisking",
    "insufficient_route_evidence",
    "no_apparent_derisking",
    "outside_analysis_universe",
]

METRICS = [
    "direct_total_value_usd",
    "direct_source_count",
    "direct_hhi",
    "direct_top_share",
    "direct_top_source_iso",
    "origin_total_value_usd",
    "origin_source_count",
    "origin_hhi",
    "origin_top_share",
    "origin_top_source_iso",
    "origin_coverage_ratio",
    "attribution_uncertainty_index",
    "mean_flow_quality_weight",
    "mean_observation_sigma_log",
    "mean_hub_uncertainty",
    "route_coverage_share",
    "sea_value_share",
    "top_chokepoint",
    "top_chokepoint_value_usd",
    "top_chokepoint_share",
]

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

# This is the frozen v2 commodity-to-model mapping. WMD Aluminium is not a
# mine-origin seed; Aluminium uses Bauxite. Silicon has no canonical WMD seed.
METAL_TO_WMD = {
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

ALIASES = {
    "bolivia": "BOL",
    "bolivia plurinational state of": "BOL",
    "brunei": "BRN",
    "burma": "MMR",
    "congo dr": "COD",
    "congo d r": "COD",
    "congo d.r": "COD",
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
    "macedonia": "MKD",
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
    "turkey": "TUR",
    "united states": "USA",
    "united states of america": "USA",
    "venezuela": "VEN",
    "viet nam": "VNM",
    "vietnam": "VNM",
}


@dataclass(frozen=True)
class AnalysisConfig:
    start_year: int = DEFAULT_START_YEAR
    end_year: int = DEFAULT_END_YEAR
    minimum_endpoint_value_usd: float = MIN_ENDPOINT_VALUE_USD
    minimum_origin_coverage: float = MIN_ORIGIN_COVERAGE
    minimum_route_coverage: float = MIN_ROUTE_COVERAGE
    default_threshold: float = DEFAULT_THRESHOLD
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS
    bootstrap_reps: int = 2_000
    random_seed: int = 20260812

    def validate(self) -> None:
        if self.end_year <= self.start_year:
            raise ValueError("end_year must be greater than start_year")
        if self.minimum_endpoint_value_usd < 0:
            raise ValueError("minimum_endpoint_value_usd must be non-negative")
        for value in (
            self.minimum_origin_coverage,
            self.minimum_route_coverage,
        ):
            if not 0 <= value <= 1:
                raise ValueError("coverage thresholds must lie in [0, 1]")
        if self.default_threshold < 0 or any(value < 0 for value in self.thresholds):
            raise ValueError("MRCI thresholds must be non-negative")
        if self.bootstrap_reps < 0:
            raise ValueError("bootstrap_reps must be non-negative")


def norm_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base_seed: int, value: object) -> int:
    token = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    digest = hashlib.sha256(f"{base_seed}|{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)


def normalize_source_edition_label(value: object) -> str:
    """Return a compact WMD edition label while preserving unknown sources."""

    if pd.isna(value) or not str(value).strip():
        return "unspecified"
    text = str(value).strip()
    match = re.search(
        r"(?:world\s+mining\s+data|wmd)\s*[-:]?\s*(20\d{2})",
        text,
        flags=re.IGNORECASE,
    )
    return f"WMD{match.group(1)}" if match else text


def read_country_crosswalk(
    reporters_reference: Path | None,
    country_reference: Path | None = None,
) -> tuple[dict[str, str], set[str]]:
    mapping: dict[str, str] = {}
    valid_iso: set[str] = set()

    if reporters_reference is not None and reporters_reference.exists():
        payload = json.loads(reporters_reference.read_text(encoding="utf-8-sig"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        for row in rows:
            iso = str(row.get("reporterCodeIsoAlpha3") or "").strip().upper()
            if len(iso) != 3 or row.get("isGroup"):
                continue
            valid_iso.add(iso)
            for key in ("text", "reporterDesc", "reporterNote"):
                name = row.get(key)
                if name:
                    mapping[norm_text(name)] = iso

    if country_reference is not None and country_reference.exists():
        nations = pd.read_csv(country_reference, low_memory=False)
        iso_col = next(
            (name for name in ("iso3", "iso", "country_iso") if name in nations),
            None,
        )
        name_cols = [
            name
            for name in ("region", "country", "country_name", "name")
            if name in nations
        ]
        if iso_col:
            for row in nations.itertuples(index=False):
                iso = str(getattr(row, iso_col, "")).strip().upper()
                if len(iso) != 3:
                    continue
                valid_iso.add(iso)
                for name_col in name_cols:
                    name = getattr(row, name_col, None)
                    if pd.notna(name):
                        mapping[norm_text(name)] = iso

    for name, iso in ALIASES.items():
        mapping[norm_text(name)] = iso
        valid_iso.add(iso)
    return mapping, valid_iso


def _model_metal_from_commodity(seed: pd.DataFrame) -> pd.Series:
    lookup = {
        norm_text(label): metal
        for metal, labels in METAL_TO_WMD.items()
        for label in labels
    }
    commodity_col = next(
        (name for name in ("commodity_label", "commodity_wmd") if name in seed),
        None,
    )
    if commodity_col is None:
        return pd.Series(pd.NA, index=seed.index, dtype="object")
    return seed[commodity_col].map(lambda value: lookup.get(norm_text(value)))


def normalize_wmd_seed(
    seed: pd.DataFrame,
    reporters_reference: Path | None,
    country_reference: Path | None,
    start_year: int,
    end_year: int,
    expected_metal_count: int | None = 23,
    minimum_mapped_production_share: float = 0.95,
    allow_incomplete_seed: bool = False,
    allow_edition_overlap: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize a merged WMD seed and return mapped rows, coverage and gaps.

    Accepted source schemas include the canonical WMD table and the historical
    parser table. A pre-normalized table may instead supply ``metal`` and
    ``mine_origin_iso``.
    """

    required = {"year", "production"}
    missing = required - set(seed.columns)
    if missing:
        raise ValueError(f"WMD seed is missing columns: {sorted(missing)}")

    work = seed.copy()
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work["production"] = pd.to_numeric(work["production"], errors="coerce")
    work = work[
        work["year"].between(start_year, end_year)
        & work["production"].gt(0)
    ].copy()
    work["year"] = work["year"].astype(int)
    if work.empty:
        raise ValueError("WMD seed has no positive rows in the requested years")

    derived_metal = _model_metal_from_commodity(work)
    if "model_metal" in work:
        model_metal = work["model_metal"].where(work["model_metal"].notna(), derived_metal)
    elif "metal" in work:
        model_metal = work["metal"].where(work["metal"].notna(), derived_metal)
    else:
        model_metal = derived_metal
    work["metal"] = model_metal.astype("string").str.strip()
    work.loc[work["metal"].isin(["", "<NA>"]), "metal"] = pd.NA
    work = work[work["metal"].notna()].copy()

    mapping, valid_iso = read_country_crosswalk(reporters_reference, country_reference)
    iso_col = next(
        (
            name
            for name in ("mine_origin_iso", "inferred_mine_origin_iso")
            if name in work
        ),
        None,
    )
    name_col = next(
        (
            name
            for name in ("mine_origin_country_wmd", "country_wmd", "country")
            if name in work
        ),
        None,
    )
    if iso_col is not None:
        supplied_iso = work[iso_col].astype("string").str.strip().str.upper()
    else:
        supplied_iso = pd.Series(pd.NA, index=work.index, dtype="string")
    if name_col is not None:
        mapped_iso = work[name_col].map(lambda value: mapping.get(norm_text(value)))
        raw_iso = work[name_col].astype("string").str.strip().str.upper()
        mapped_iso = mapped_iso.where(mapped_iso.notna(), raw_iso.where(raw_iso.str.fullmatch(r"[A-Z]{3}", na=False)))
    else:
        mapped_iso = pd.Series(pd.NA, index=work.index, dtype="string")
    work["mine_origin_iso"] = supplied_iso.where(
        supplied_iso.str.fullmatch(r"[A-Z]{3}", na=False),
        mapped_iso,
    )
    if valid_iso:
        work.loc[~work["mine_origin_iso"].isin(valid_iso), "mine_origin_iso"] = pd.NA

    if "source_confidence" not in work:
        work["source_confidence"] = 0.50
    work["source_confidence"] = pd.to_numeric(
        work["source_confidence"], errors="coerce"
    ).fillna(0.50)
    if "source_edition" not in work:
        work["source_edition"] = pd.NA
    edition_fallback = work.get(
        "source_dataset",
        pd.Series("unspecified", index=work.index),
    ).fillna("unspecified")
    supplied_edition = work["source_edition"].astype("string").str.strip()
    work["source_edition"] = supplied_edition.where(
        supplied_edition.notna() & supplied_edition.ne(""),
        edition_fallback,
    ).map(normalize_source_edition_label)
    if "commodity_label" not in work:
        work["commodity_label"] = work.get(
            "commodity_wmd",
            pd.Series("", index=work.index),
        )

    collision_keys = ["metal", "year", "mine_origin_iso", "commodity_label"]
    edition_counts = (
        work.dropna(subset=["mine_origin_iso"])
        .groupby(collision_keys, observed=True)["source_edition"]
        .nunique()
    )
    collisions = edition_counts[edition_counts.gt(1)]
    if len(collisions) and not allow_edition_overlap:
        example = list(collisions.index[:5])
        raise ValueError(
            "Merged seed contains overlapping source editions for the same "
            f"metal-origin-year-commodity; examples={example}"
        )

    total = (
        work.groupby(["metal", "year"], as_index=False, observed=True)["production"]
        .sum()
        .rename(columns={"production": "total_seed_production"})
    )
    mapped = (
        work.dropna(subset=["mine_origin_iso"])
        .groupby(["metal", "year"], as_index=False, observed=True)["production"]
        .sum()
        .rename(columns={"production": "mapped_seed_production"})
    )
    coverage = total.merge(mapped, on=["metal", "year"], how="left")
    coverage["mapped_seed_production"] = coverage["mapped_seed_production"].fillna(0.0)
    coverage["mapped_production_share"] = (
        coverage["mapped_seed_production"]
        / coverage["total_seed_production"].replace(0, np.nan)
    )
    row_stats = work.groupby(["metal", "year"], as_index=False, observed=True).agg(
        seed_rows=("production", "size"),
        mapped_origin_count=("mine_origin_iso", "nunique"),
        mean_source_confidence=("source_confidence", "mean"),
        source_editions=("source_edition", lambda values: "|".join(sorted(set(map(str, values))))),
    )
    coverage = coverage.merge(row_stats, on=["metal", "year"], how="left")
    coverage = coverage.sort_values(["year", "metal"]).reset_index(drop=True)

    provenance_cols = [
        "commodity_wmd",
        "commodity_label",
        "model_metal",
        "measurement_basis",
        name_col,
        "unit",
        "unit_raw",
        "source_rem",
        "source_flag",
        "source_confidence",
        "source_edition",
        "source_dataset",
        "source_url",
        "source_pdf_page",
        "value_status",
        "input_seed_file",
    ]
    gaps_cols = [
        name
        for name in ["metal", "year", "production", *provenance_cols]
        if name is not None and name in work
    ]
    gaps_cols = list(dict.fromkeys(gaps_cols))
    gaps = work[work["mine_origin_iso"].isna()][gaps_cols].copy()

    requested_years = set(range(start_year, end_year + 1))
    observed_years = set(map(int, coverage["year"].unique()))
    missing_years = sorted(requested_years - observed_years)
    metal_count_by_year = coverage.groupby("year")["metal"].nunique()
    bad_metal_years: dict[int, int] = {}
    if expected_metal_count is not None:
        bad_metal_years = {
            int(year): int(count)
            for year, count in metal_count_by_year.items()
            if int(count) != expected_metal_count
        }
    low_mapping = coverage[
        coverage["mapped_production_share"].lt(minimum_mapped_production_share)
    ]
    if not allow_incomplete_seed and (missing_years or bad_metal_years or len(low_mapping)):
        low_examples = low_mapping[
            ["metal", "year", "mapped_production_share"]
        ].head(10).to_dict("records")
        raise ValueError(
            "Merged WMD seed failed completeness checks: "
            f"missing_years={missing_years}, metal_counts={bad_metal_years}, "
            f"low_mapping_examples={low_examples}"
        )

    mapped_cols = [
        name
        for name in [
            "metal",
            "year",
            "mine_origin_iso",
            "production",
            *provenance_cols,
        ]
        if name is not None and name in work
    ]
    mapped_cols = list(dict.fromkeys(mapped_cols))
    mapped_rows = work.dropna(subset=["mine_origin_iso"])[mapped_cols].copy()
    return mapped_rows, coverage, gaps


def entropy(shares: pd.Series) -> float:
    numeric = pd.to_numeric(shares, errors="coerce").dropna()
    if numeric.lt(0).any():
        raise ValueError("Origin shares must be non-negative")
    values = numeric[numeric.gt(0)].to_numpy(float)
    if len(values) <= 1:
        return 0.0
    values = values / values.sum()
    return float(-(values * np.log(values)).sum() / math.log(len(values)))


def normalize_mix(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
) -> pd.DataFrame:
    numeric = pd.to_numeric(frame[value_col], errors="coerce")
    if numeric.dropna().lt(0).any():
        raise ValueError(f"{value_col} must be non-negative")
    grouped = (
        frame.groupby([*group_cols, "inferred_mine_origin_iso"], as_index=False, observed=True)[value_col]
        .sum()
    )
    totals = grouped.groupby(list(group_cols), observed=True)[value_col].transform("sum")
    grouped["origin_share"] = np.where(totals.gt(0), grouped[value_col] / totals, 0.0)
    return grouped[grouped["origin_share"].gt(0)].drop(columns=value_col)


def blend_mix(
    primary: dict[str, float] | None,
    fallback: dict[str, float],
    primary_weight: float,
) -> dict[str, float]:
    if not 0 <= primary_weight <= 1:
        raise ValueError("primary_weight must lie in [0, 1]")
    combined: defaultdict[str, float] = defaultdict(float)
    if primary:
        for origin, share in primary.items():
            if not np.isfinite(share) or share < 0:
                raise ValueError("Primary shares must be finite and non-negative")
            combined[origin] += primary_weight * share
        fallback_weight = 1 - primary_weight
    else:
        fallback_weight = 1.0
    for origin, share in fallback.items():
        if not np.isfinite(share) or share < 0:
            raise ValueError("Fallback shares must be finite and non-negative")
        combined[origin] += fallback_weight * share
    total = sum(combined.values())
    if total <= 0:
        return {}
    return {origin: value / total for origin, value in combined.items() if value > 0}


def infer_year_stage(
    flows_stage: pd.DataFrame,
    country_mix: dict[str, dict[str, float]],
    global_mix: dict[str, float],
    stage: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Frozen recursive stage-mixing v0 algorithm."""

    if flows_stage.empty:
        return pd.DataFrame(), country_mix
    local_weight = LOCAL_STAGE_WEIGHT.get(stage, 0.4)
    records: list[dict[str, object]] = []
    importer_inputs: dict[str, defaultdict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
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
                    "CmdCode": row.CmdCode,
                    "metal": row.metal,
                    "stage": stage,
                    "hs_label": row.hs_label,
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
        previous = next_mix.get(importer, {})
        blended: defaultdict[str, float] = defaultdict(float)
        for origin, share in previous.items():
            blended[origin] += local_weight * share
        for origin, amount in mix.items():
            blended[origin] += (1 - local_weight) * amount / total
        normalizer = sum(blended.values())
        if normalizer > 0:
            next_mix[importer] = {
                origin: value / normalizer
                for origin, value in blended.items()
                if value > 0
            }
    return pd.DataFrame(records), next_mix


def top_label(
    frame: pd.DataFrame,
    group: Sequence[str],
    value: str,
    label: str,
    output: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[*group, output])
    idx = frame.groupby(list(group), sort=False, observed=True)[value].idxmax()
    return frame.loc[idx, [*group, label]].rename(columns={label: output})


def concentration_stats(
    edges: pd.DataFrame,
    source_col: str,
    value_col: str,
    prefix: str,
) -> pd.DataFrame:
    edge_group = [*GROUP, source_col]
    grouped = (
        edges.groupby(edge_group, as_index=False, observed=True)[value_col]
        .sum()
    )
    grouped = grouped[grouped[value_col].gt(0)].copy()
    if grouped.empty:
        return pd.DataFrame(
            columns=[
                *GROUP,
                f"{prefix}_total_value_usd",
                f"{prefix}_source_count",
                f"{prefix}_hhi",
                f"{prefix}_top_share",
                f"{prefix}_top_source_iso",
            ]
        )
    totals = grouped.groupby(GROUP, observed=True)[value_col].transform("sum")
    grouped["share"] = grouped[value_col] / totals
    grouped["share_sq"] = grouped["share"].pow(2)
    stats = (
        grouped.groupby(GROUP, as_index=False, observed=True)
        .agg(
            total_value=(value_col, "sum"),
            source_count=(source_col, "nunique"),
            hhi=("share_sq", "sum"),
            top_share=("share", "max"),
        )
        .rename(
            columns={
                "total_value": f"{prefix}_total_value_usd",
                "source_count": f"{prefix}_source_count",
                "hhi": f"{prefix}_hhi",
                "top_share": f"{prefix}_top_share",
            }
        )
    )
    top = top_label(
        grouped,
        GROUP,
        "share",
        source_col,
        f"{prefix}_top_source_iso",
    )
    return stats.merge(top, on=GROUP, how="left")


def _uncertainty_by_corridor(inferred: pd.DataFrame) -> pd.DataFrame:
    uncertainty = (
        inferred.groupby(
            ["metal", "stage", "year", "exporter_iso", "importer_iso"],
            as_index=False,
            observed=True,
        )
        .agg(
            total_attributed_value_usd=("attributed_value_usd", "sum"),
            origin_count=("inferred_mine_origin_iso", "nunique"),
            attribution_entropy=("origin_share_in_flow", entropy),
            mean_flow_quality_weight=("flow_quality_weight", "mean"),
            mean_observation_sigma_log=("observation_sigma_log", "mean"),
            mean_hub_uncertainty=("hub_uncertainty", "mean"),
        )
    )
    uncertainty["attribution_uncertainty_index"] = (
        0.35 * uncertainty["attribution_entropy"].fillna(0)
        + 0.25 * (1 - uncertainty["mean_flow_quality_weight"].fillna(0.5))
        + 0.20 * uncertainty["mean_hub_uncertainty"].fillna(0)
        + 0.20
        * np.clip(
            uncertainty["mean_observation_sigma_log"].fillna(0.75) / 1.5,
            0,
            1,
        )
    )
    return uncertainty.sort_values("total_attributed_value_usd", ascending=False)


def attribution_stats_from_corridors(uncertainty: pd.DataFrame) -> pd.DataFrame:
    data = uncertainty.copy()
    weight = data["total_attributed_value_usd"].clip(lower=0).fillna(0.0)
    metrics = [
        "attribution_uncertainty_index",
        "mean_flow_quality_weight",
        "mean_observation_sigma_log",
        "mean_hub_uncertainty",
    ]
    for metric in metrics:
        data[f"weighted_{metric}"] = (
            pd.to_numeric(data[metric], errors="coerce").fillna(0.0) * weight
        )
    grouped = data.groupby(GROUP, as_index=False, observed=True).agg(
        attribution_weight_usd=("total_attributed_value_usd", "sum"),
        weighted_attribution_uncertainty_index=(
            "weighted_attribution_uncertainty_index",
            "sum",
        ),
        weighted_mean_flow_quality_weight=("weighted_mean_flow_quality_weight", "sum"),
        weighted_mean_observation_sigma_log=(
            "weighted_mean_observation_sigma_log",
            "sum",
        ),
        weighted_mean_hub_uncertainty=("weighted_mean_hub_uncertainty", "sum"),
    )
    denominator = grouped["attribution_weight_usd"].replace(0, np.nan)
    for metric in metrics:
        grouped[metric] = grouped.pop(f"weighted_{metric}") / denominator
    return grouped


def load_trade_panel(path: Path, start_year: int, end_year: int) -> pd.DataFrame:
    columns = [
        "CmdCode",
        "metal",
        "stage",
        "hs_label",
        "exporter_iso",
        "importer_iso",
        "year",
        "reconstructed_value_usd",
        "data_quality_weight",
        "observation_sigma_log",
    ]
    data = pd.read_csv(
        path,
        usecols=columns,
        dtype={"CmdCode": "string", "hs_label": "string"},
        low_memory=False,
    )
    data["year"] = pd.to_numeric(data["year"], errors="coerce")
    data["reconstructed_value_usd"] = pd.to_numeric(
        data["reconstructed_value_usd"], errors="coerce"
    ).fillna(0.0)
    data = data[
        data["year"].between(start_year, end_year)
        & data["reconstructed_value_usd"].gt(0)
    ].copy()
    data["year"] = data["year"].astype(int)
    if data["CmdCode"].isna().any() or data["CmdCode"].str.strip().eq("").any():
        raise ValueError("Dynamic trade panel has blank CmdCode provenance")
    if data["hs_label"].isna().any() or data["hs_label"].str.strip().eq("").any():
        raise ValueError("Dynamic trade panel has blank hs_label provenance")
    duplicated = data.duplicated(DYNAMIC_FLOW_KEY, keep=False)
    if duplicated.any():
        examples = data.loc[duplicated, DYNAMIC_FLOW_KEY].head(10).to_dict("records")
        raise ValueError(
            "Dynamic trade panel is not unique on its HS corridor-year key; "
            f"examples={examples}"
        )
    missing_years = sorted(
        set(range(start_year, end_year + 1)) - set(map(int, data["year"].unique()))
    )
    if missing_years:
        raise ValueError(f"Dynamic trade panel is missing years: {missing_years}")
    return data


def infer_mine_origin_panel(
    trade: pd.DataFrame,
    mapped_seed: pd.DataFrame,
    mine_flow_output: Path | None,
    uncertainty_output: Path,
    conservation_output: Path,
    flow_key_audit_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Infer origins year-by-year and stream potentially large outputs to disk."""

    seed = mapped_seed.rename(columns={"mine_origin_iso": "inferred_mine_origin_iso"})
    production_mix = normalize_mix(seed, ["metal", "year"], "production")
    eligible_metals = set(seed["metal"].dropna().astype(str).unique())
    flows = trade[trade["metal"].isin(eligible_metals)].copy()

    origin_stats_parts: list[pd.DataFrame] = []
    attribution_parts: list[pd.DataFrame] = []
    conservation_parts: list[pd.DataFrame] = []
    flow_key_audit_rows: list[dict[str, object]] = []
    mine_handle = None
    uncertainty_handle = None
    wrote_mine_header = False
    wrote_uncertainty_header = False
    mine_flow_rows = 0
    try:
        if mine_flow_output is not None:
            mine_handle = gzip.open(mine_flow_output, "wt", encoding="utf-8", newline="")
        uncertainty_handle = uncertainty_output.open("w", encoding="utf-8", newline="")

        for (metal, year), sub in flows.groupby(["metal", "year"], sort=True):
            prod = production_mix[
                production_mix["metal"].eq(metal)
                & production_mix["year"].eq(int(year))
            ]
            if prod.empty:
                continue
            global_mix = dict(
                zip(prod["inferred_mine_origin_iso"], prod["origin_share"])
            )
            country_mix = {origin: {origin: 1.0} for origin in global_mix}
            inferred_parts: list[pd.DataFrame] = []
            stages = sorted(
                sub["stage"].dropna().unique(),
                key=lambda value: STAGE_ORDER.get(value, 9),
            )
            for stage in stages:
                inferred, country_mix = infer_year_stage(
                    sub[sub["stage"].eq(stage)],
                    country_mix,
                    global_mix,
                    str(stage),
                )
                if not inferred.empty:
                    inferred_parts.append(inferred)
            if not inferred_parts:
                continue
            inferred = pd.concat(inferred_parts, ignore_index=True)
            unique_keys = int(len(inferred.drop_duplicates(MINE_FLOW_KEY)))
            duplicate_rows = int(len(inferred) - unique_keys)
            if duplicate_rows:
                duplicated = inferred.duplicated(MINE_FLOW_KEY, keep=False)
                examples = inferred.loc[duplicated, MINE_FLOW_KEY].head(10).to_dict("records")
                raise RuntimeError(
                    "Mine-origin flow provenance key is not unique; the dynamic "
                    f"input must retain one row per HS corridor-year. examples={examples}"
                )
            mine_flow_rows += len(inferred)
            flow_key_audit_rows.append(
                {
                    "metal": metal,
                    "year": int(year),
                    "rows": int(len(inferred)),
                    "unique_keys": unique_keys,
                    "duplicate_rows": duplicate_rows,
                    "key_columns": "|".join(MINE_FLOW_KEY),
                    "validation_method": "exact pandas drop_duplicates within metal-year block",
                }
            )

            if mine_handle is not None:
                inferred.to_csv(
                    mine_handle,
                    index=False,
                    header=not wrote_mine_header,
                )
                wrote_mine_header = True

            uncertainty = _uncertainty_by_corridor(inferred)
            uncertainty.to_csv(
                uncertainty_handle,
                index=False,
                header=not wrote_uncertainty_header,
            )
            wrote_uncertainty_header = True

            origin_edges = (
                inferred.groupby(
                    GROUP + ["inferred_mine_origin_iso"],
                    as_index=False,
                    observed=True,
                )["attributed_value_usd"]
                .sum()
            )
            origin_stats_parts.append(
                concentration_stats(
                    origin_edges,
                    "inferred_mine_origin_iso",
                    "attributed_value_usd",
                    "origin",
                )
            )
            attribution_parts.append(attribution_stats_from_corridors(uncertainty))

            direct_totals = (
                sub.groupby(GROUP, as_index=False, observed=True)[
                    "reconstructed_value_usd"
                ]
                .sum()
                .rename(columns={"reconstructed_value_usd": "direct_value_usd"})
            )
            attributed_totals = (
                inferred.groupby(GROUP, as_index=False, observed=True)[
                    "attributed_value_usd"
                ]
                .sum()
            )
            audit = direct_totals.merge(attributed_totals, on=GROUP, how="outer")
            audit[["direct_value_usd", "attributed_value_usd"]] = audit[
                ["direct_value_usd", "attributed_value_usd"]
            ].fillna(0.0)
            audit["absolute_error_usd"] = (
                audit["attributed_value_usd"] - audit["direct_value_usd"]
            ).abs()
            audit["relative_error"] = (
                audit["absolute_error_usd"]
                / audit["direct_value_usd"].replace(0, np.nan)
            )
            conservation_parts.append(audit)
    finally:
        if mine_handle is not None:
            mine_handle.close()
        if uncertainty_handle is not None:
            uncertainty_handle.close()

    if not origin_stats_parts:
        raise RuntimeError("No mine-origin flows were inferred")
    origin_stats = pd.concat(origin_stats_parts, ignore_index=True)
    attribution_stats = pd.concat(attribution_parts, ignore_index=True)
    conservation = pd.concat(conservation_parts, ignore_index=True)
    conservation.to_csv(conservation_output, index=False)
    flow_key_audit = pd.DataFrame(flow_key_audit_rows).sort_values(
        ["year", "metal"]
    )
    flow_key_audit.to_csv(flow_key_audit_output, index=False)
    max_error = float(conservation["relative_error"].fillna(0).max())
    if max_error > 1e-10:
        raise RuntimeError(f"Mine-origin value conservation failed: max relative error={max_error}")
    mine_flow_key_audit: dict[str, object] = {
        "key_columns": MINE_FLOW_KEY,
        "rows": int(mine_flow_rows),
        "unique_keys": int(flow_key_audit["unique_keys"].sum()),
        "duplicate_rows": int(flow_key_audit["duplicate_rows"].sum()),
        "metal_year_blocks": int(len(flow_key_audit)),
        "audit_filename": flow_key_audit_output.name,
        "validation_method": (
            "exact pandas duplicated check within every disjoint metal-year block"
        ),
    }
    return origin_stats, attribution_stats, conservation, mine_flow_key_audit


def combine_text(values: pd.Series) -> str:
    items: list[str] = []
    for value in values.dropna().astype(str):
        for item in value.split("|"):
            item = item.strip()
            if item and item not in items:
                items.append(item)
    return "|".join(items)


def route_stats(direct_edges: pd.DataFrame, direct_stats: pd.DataFrame, lanes_path: Path) -> pd.DataFrame:
    lanes = pd.read_csv(lanes_path, low_memory=False)
    lanes = lanes.rename(columns={"origin": "exporter_iso", "dest": "importer_iso"})
    required = {"metal", "exporter_iso", "importer_iso", "mode", "chokepoints"}
    missing = required - set(lanes.columns)
    if missing:
        raise ValueError(f"Lane table is missing columns: {sorted(missing)}")
    lane_keys = ["metal", "exporter_iso", "importer_iso"]
    lanes = lanes.groupby(lane_keys, as_index=False, observed=True).agg(
        mode=(
            "mode",
            lambda values: "sea"
            if values.astype(str).eq("sea").any()
            else str(values.dropna().iloc[0])
            if values.notna().any()
            else "",
        ),
        chokepoints=("chokepoints", combine_text),
    )
    routed = direct_edges.merge(lanes, on=lane_keys, how="left", indicator=True)
    routed["route_matched_value_usd"] = np.where(
        routed["_merge"].eq("both"), routed["reconstructed_value_usd"], 0.0
    )
    routed["sea_value_usd"] = np.where(
        routed["mode"].eq("sea"), routed["reconstructed_value_usd"], 0.0
    )
    coverage = routed.groupby(GROUP, as_index=False, observed=True).agg(
        route_matched_value_usd=("route_matched_value_usd", "sum"),
        sea_value_usd=("sea_value_usd", "sum"),
    )
    coverage = coverage.merge(
        direct_stats[GROUP + ["direct_total_value_usd"]],
        on=GROUP,
        how="left",
    )
    denominator = coverage["direct_total_value_usd"].replace(0, np.nan)
    coverage["route_coverage_share"] = coverage["route_matched_value_usd"] / denominator
    coverage["sea_value_share"] = coverage["sea_value_usd"] / denominator

    choke = routed[
        routed["mode"].eq("sea") & routed["chokepoints"].fillna("").ne("")
    ].copy()
    if choke.empty:
        coverage["chokepoint_count"] = 0
        coverage["top_chokepoint"] = ""
        coverage["top_chokepoint_value_usd"] = 0.0
        coverage["top_chokepoint_share"] = 0.0
        return coverage.drop(columns="direct_total_value_usd")
    choke["chokepoint"] = choke["chokepoints"].str.split("|")
    choke = choke.explode("chokepoint")
    choke["chokepoint"] = choke["chokepoint"].astype(str).str.strip()
    choke = choke[choke["chokepoint"].ne("")]
    choke_group = choke.groupby(
        GROUP + ["chokepoint"], as_index=False, observed=True
    )["reconstructed_value_usd"].sum()
    choke_stats = choke_group.groupby(GROUP, as_index=False, observed=True).agg(
        chokepoint_count=("chokepoint", "nunique"),
        top_chokepoint_value_usd=("reconstructed_value_usd", "max"),
    )
    top = top_label(
        choke_group,
        GROUP,
        "reconstructed_value_usd",
        "chokepoint",
        "top_chokepoint",
    )
    coverage = coverage.merge(choke_stats.merge(top, on=GROUP), on=GROUP, how="left")
    coverage["top_chokepoint_value_usd"] = coverage[
        "top_chokepoint_value_usd"
    ].fillna(0.0)
    coverage["top_chokepoint_share"] = (
        coverage["top_chokepoint_value_usd"] / denominator
    )
    coverage["chokepoint_count"] = coverage["chokepoint_count"].fillna(0).astype(int)
    coverage["top_chokepoint"] = coverage["top_chokepoint"].fillna("")
    return coverage.drop(columns="direct_total_value_usd")


def build_annual_risk_panel(
    trade: pd.DataFrame,
    origin_stats: pd.DataFrame,
    attribution_stats: pd.DataFrame,
    lanes_path: Path,
) -> pd.DataFrame:
    direct_edges = (
        trade.groupby(GROUP + ["exporter_iso"], as_index=False, observed=True)[
            "reconstructed_value_usd"
        ]
        .sum()
        .sort_values(GROUP + ["exporter_iso"])
    )
    direct = concentration_stats(
        direct_edges,
        "exporter_iso",
        "reconstructed_value_usd",
        "direct",
    )
    routes = route_stats(direct_edges, direct, lanes_path)
    panel = (
        direct.merge(origin_stats, on=GROUP, how="left")
        .merge(attribution_stats, on=GROUP, how="left")
        .merge(routes, on=GROUP, how="left")
    )
    panel["origin_coverage_ratio"] = (
        panel["origin_total_value_usd"]
        / panel["direct_total_value_usd"].replace(0, np.nan)
    )
    for name in (
        "route_coverage_share",
        "sea_value_share",
        "top_chokepoint_share",
        "top_chokepoint_value_usd",
    ):
        panel[name] = panel[name].fillna(0.0)
    panel["chokepoint_count"] = panel["chokepoint_count"].fillna(0).astype(int)
    panel["top_chokepoint"] = panel["top_chokepoint"].fillna("")
    return panel.sort_values(GROUP).reset_index(drop=True)


def classify_mrci(data: pd.DataFrame, threshold: float) -> pd.Series:
    if threshold < 0:
        raise ValueError("MRCI threshold must be non-negative")
    if threshold == 0:
        direct_down = data["delta_direct_hhi"].lt(0)
        origin_up = data["delta_origin_hhi"].gt(0)
        origin_down = data["delta_origin_hhi"].lt(0)
        route_up = data["delta_top_chokepoint_share"].gt(0)
        route_down = data["delta_top_chokepoint_share"].lt(0)
        uncertainty_up = data["delta_attribution_uncertainty_index"].gt(0)
        uncertainty_not_up = data["delta_attribution_uncertainty_index"].le(0)
    else:
        direct_down = data["delta_direct_hhi"].le(-threshold)
        origin_up = data["delta_origin_hhi"].ge(threshold)
        origin_down = data["delta_origin_hhi"].le(-threshold)
        route_up = data["delta_top_chokepoint_share"].ge(threshold)
        route_down = data["delta_top_chokepoint_share"].le(-threshold)
        uncertainty_up = data["delta_attribution_uncertainty_index"].ge(threshold)
        uncertainty_not_up = data["delta_attribution_uncertainty_index"].lt(threshold)
    eligible = data["analysis_eligible"].fillna(False).astype(bool)
    route_evidence = data["route_evidence_eligible"].fillna(False).astype(bool)
    category = pd.Series("partial_derisking", index=data.index, dtype="object")
    category.loc[~eligible] = "outside_analysis_universe"
    category.loc[eligible & ~direct_down] = "no_apparent_derisking"
    apparent = eligible & direct_down
    category.loc[
        apparent & ~route_evidence & ~origin_up & ~uncertainty_up
    ] = "insufficient_route_evidence"
    category.loc[
        apparent & route_evidence & origin_down & route_down & uncertainty_not_up
    ] = "substantive_derisking"
    category.loc[
        apparent & origin_up & ~(route_up & route_evidence)
    ] = "mine_origin_transfer"
    category.loc[
        apparent & route_up & route_evidence & ~origin_up
    ] = "route_transfer"
    category.loc[
        apparent & origin_up & route_up & route_evidence
    ] = "multiple_risk_transfer"
    category.loc[
        apparent & ~origin_up & ~(route_up & route_evidence) & uncertainty_up
    ] = "evidence_weakened"
    return category


def apparent_mask(data: pd.DataFrame, threshold: float) -> pd.Series:
    eligible = data["analysis_eligible"].fillna(False).astype(bool)
    if threshold == 0:
        return eligible & data["delta_direct_hhi"].lt(0)
    return eligible & data["delta_direct_hhi"].le(-threshold)


def add_mrci_columns(data: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = data.copy()
    output["classification"] = classify_mrci(output, threshold)
    output["classification_threshold"] = threshold
    output["mrci_classification"] = output["classification"]
    output["mrci_threshold"] = threshold
    output["apparent_derisking"] = apparent_mask(output, threshold)
    output["direct_risk_reduction"] = -output["delta_direct_hhi"]
    output["origin_risk_reduction"] = -output["delta_origin_hhi"]
    output["route_risk_reduction"] = np.where(
        output["route_evidence_eligible"],
        -output["delta_top_chokepoint_share"],
        np.nan,
    )
    output["multilayer_derisking_margin"] = output[
        ["direct_risk_reduction", "origin_risk_reduction", "route_risk_reduction"]
    ].min(axis=1, skipna=False)
    output.loc[
        ~(output["analysis_eligible"] & output["route_evidence_eligible"]),
        "multilayer_derisking_margin",
    ] = np.nan
    output["apparent_derisking_indicator"] = output["apparent_derisking"]
    output["risk_transfer_indicator"] = output["classification"].isin(TRANSFER_CLASSES)
    output["substantive_derisking_indicator"] = output["classification"].eq(
        "substantive_derisking"
    )
    output["evidence_weakened_indicator"] = output["classification"].eq(
        "evidence_weakened"
    )
    route_worsening = np.where(
        output["route_evidence_eligible"],
        output["delta_top_chokepoint_share"],
        np.nan,
    )
    hidden_worsening = np.fmax(
        output["delta_origin_hhi"].to_numpy(float), route_worsening
    )
    output["risk_transfer_magnitude"] = np.where(
        output["risk_transfer_indicator"],
        np.maximum(hidden_worsening, 0.0),
        0.0,
    )
    output.loc[~output["analysis_eligible"], "risk_transfer_magnitude"] = np.nan
    return output


def adjacent_windows(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(year - 1, year) for year in range(start_year + 1, end_year + 1)]


def rolling_four_year_windows(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(year, year + 4) for year in range(start_year, end_year - 3)]


def nonoverlap_four_year_windows(start_year: int, end_year: int) -> list[tuple[int, int]]:
    windows: list[tuple[int, int]] = []
    year = start_year
    while year + 4 <= end_year:
        windows.append((year, year + 4))
        year += 4
    return windows


def build_window_panel(
    annual: pd.DataFrame,
    windows: Sequence[tuple[int, int]],
    window_family: str,
    config: AnalysisConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for baseline_year, end_year in windows:
        baseline = annual[annual["year"].eq(baseline_year)][KEYS + METRICS].copy()
        endpoint = annual[annual["year"].eq(end_year)][KEYS + METRICS].copy()
        baseline = baseline.rename(
            columns={metric: f"{metric}_previous" for metric in METRICS}
        )
        panel = endpoint.merge(
            baseline,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )
        panel["year"] = end_year
        panel["previous_year"] = baseline_year
        panel["baseline_year"] = baseline_year
        panel["end_year"] = end_year
        panel["window_family"] = window_family
        panel["window_type"] = window_family
        panel["window_length_years"] = end_year - baseline_year
        panel["transition"] = f"{baseline_year}\u2192{end_year}"
        panel["window_label"] = panel["transition"]
        panel["pandemic_transition"] = end_year in {2020, 2021}
        panel["pandemic_overlap"] = baseline_year <= 2021 and end_year >= 2020
        panel["wmd_edition_boundary_crossing"] = any(
            baseline_year < boundary_year <= end_year
            for boundary_year in WMD_EDITION_BOUNDARY_YEARS
        )
        panel["delta_direct_hhi"] = (
            panel["direct_hhi"] - panel["direct_hhi_previous"]
        )
        panel["delta_origin_hhi"] = (
            panel["origin_hhi"] - panel["origin_hhi_previous"]
        )
        panel["delta_top_chokepoint_share"] = (
            panel["top_chokepoint_share"]
            - panel["top_chokepoint_share_previous"]
        )
        panel["delta_attribution_uncertainty_index"] = (
            panel["attribution_uncertainty_index"]
            - panel["attribution_uncertainty_index_previous"]
        )
        panel["analysis_eligible"] = (
            panel[
                ["direct_total_value_usd", "direct_total_value_usd_previous"]
            ]
            .min(axis=1)
            .ge(config.minimum_endpoint_value_usd)
            & panel[["origin_coverage_ratio", "origin_coverage_ratio_previous"]]
            .min(axis=1)
            .ge(config.minimum_origin_coverage)
        )
        panel["route_evidence_eligible"] = (
            panel[["route_coverage_share", "route_coverage_share_previous"]]
            .min(axis=1)
            .ge(config.minimum_route_coverage)
        )
        panel["three_layer_evidence_eligible"] = (
            panel["analysis_eligible"] & panel["route_evidence_eligible"]
        )
        panel = add_mrci_columns(panel, config.default_threshold)
        panel["transition_value_usd"] = panel["direct_total_value_usd"]
        panel["window_value_usd"] = panel["direct_total_value_usd"]
        frames.append(panel)
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    return output.sort_values(
        ["end_year", "transition_value_usd", *KEYS],
        ascending=[True, False, True, True, True],
    ).reset_index(drop=True)


def build_balanced_sample(
    annual: pd.DataFrame,
    config: AnalysisConfig,
    canonical_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    years = list(range(config.start_year, config.end_year + 1))
    work = annual[annual["year"].isin(years)].copy()
    work["single_year_analysis_eligible"] = (
        work["direct_total_value_usd"].ge(config.minimum_endpoint_value_usd)
        & work["origin_coverage_ratio"].ge(config.minimum_origin_coverage)
    )
    work["single_year_route_evidence_eligible"] = work[
        "route_coverage_share"
    ].ge(config.minimum_route_coverage)
    work["single_year_full_evidence_eligible"] = (
        work["single_year_analysis_eligible"]
        & work["single_year_route_evidence_eligible"]
    )
    output = work.groupby(KEYS, as_index=False, observed=True).agg(
        years_present=("year", "nunique"),
        years_analysis_eligible=("single_year_analysis_eligible", "sum"),
        years_route_evidence_eligible=(
            "single_year_route_evidence_eligible",
            "sum",
        ),
        years_full_evidence_eligible=("single_year_full_evidence_eligible", "sum"),
        minimum_annual_trade_value_usd=("direct_total_value_usd", "min"),
        minimum_annual_origin_coverage=("origin_coverage_ratio", "min"),
        minimum_annual_route_coverage=("route_coverage_share", "min"),
    )
    expected_years = len(years)
    output["balanced_analysis_eligible"] = (
        output["years_present"].eq(expected_years)
        & output["years_analysis_eligible"].eq(expected_years)
    )
    output["balanced_full_evidence_eligible"] = (
        output["years_present"].eq(expected_years)
        & output["years_full_evidence_eligible"].eq(expected_years)
    )
    output["canonical_704_member"] = False
    if canonical_universe is not None:
        canonical_keys = canonical_universe[KEYS].drop_duplicates().assign(
            canonical_704_member=True
        )
        output = output.drop(columns="canonical_704_member").merge(
            canonical_keys,
            on=KEYS,
            how="left",
        )
        output["canonical_704_member"] = output["canonical_704_member"].fillna(False)
    return output.sort_values(KEYS).reset_index(drop=True)


def annual_evidence_coverage(annual: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    work = annual.copy()
    work["trade_value_eligible"] = work["direct_total_value_usd"].ge(
        config.minimum_endpoint_value_usd
    )
    work["origin_evidence_eligible"] = work["origin_coverage_ratio"].ge(
        config.minimum_origin_coverage
    )
    work["route_evidence_eligible"] = work["route_coverage_share"].ge(
        config.minimum_route_coverage
    )
    work["full_evidence_eligible"] = (
        work["trade_value_eligible"]
        & work["origin_evidence_eligible"]
        & work["route_evidence_eligible"]
    )
    rows: list[dict[str, object]] = []
    for year, frame in work.groupby("year", sort=True):
        total_value = float(frame["direct_total_value_usd"].sum())
        origin_value = float(frame["origin_total_value_usd"].fillna(0.0).sum())
        route_value = float(frame["route_matched_value_usd"].fillna(0.0).sum())
        full = frame["full_evidence_eligible"]
        rows.append(
            {
                "year": int(year),
                "direct_units": len(frame),
                "trade_value_eligible_units": int(frame["trade_value_eligible"].sum()),
                "origin_evidence_eligible_units": int(frame["origin_evidence_eligible"].sum()),
                "route_evidence_eligible_units": int(frame["route_evidence_eligible"].sum()),
                "full_evidence_eligible_units": int(full.sum()),
                "direct_value_usd": total_value,
                "mine_origin_attributed_value_usd": origin_value,
                "route_matched_value_usd": route_value,
                "mine_origin_coverage_share": origin_value / max(total_value, 1.0),
                "route_exposure_coverage_share": route_value / max(total_value, 1.0),
                "full_evidence_value_usd": float(
                    frame.loc[full, "direct_total_value_usd"].sum()
                ),
                "full_evidence_value_share": float(
                    frame.loc[full, "direct_total_value_usd"].sum()
                    / max(total_value, 1.0)
                ),
                "importers": int(frame["importer_iso"].nunique()),
                "metals": int(frame["metal"].nunique()),
                "stages": int(frame["stage"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _filter_sample(
    panel: pd.DataFrame,
    balanced: pd.DataFrame,
    sample_mode: str,
) -> pd.DataFrame:
    if sample_mode == "local_endpoint":
        return panel.copy()
    if sample_mode == "local_full_evidence_endpoint":
        return panel[panel["three_layer_evidence_eligible"].astype(bool)].copy()
    modes = {
        "balanced_analysis_2012_2024": "balanced_analysis_eligible",
        "balanced_full_evidence_2012_2024": "balanced_full_evidence_eligible",
        "canonical_704_bridge": "canonical_704_member",
    }
    if sample_mode not in modes:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")
    flag = modes[sample_mode]
    keys = balanced.loc[balanced[flag].astype(bool), KEYS]
    return panel.merge(keys, on=KEYS, how="inner", validate="many_to_one")


def _bootstrap_ratios(
    data: pd.DataFrame,
    reps: int,
    seed: int,
) -> dict[str, float]:
    if reps <= 0 or data.empty:
        return {
            "risk_transfer_value_share_ci_low": np.nan,
            "risk_transfer_value_share_ci_high": np.nan,
            "substantive_value_share_ci_low": np.nan,
            "substantive_value_share_ci_high": np.nan,
            "risk_transfer_minus_substantive_value_share_ci_low": np.nan,
            "risk_transfer_minus_substantive_value_share_ci_high": np.nan,
            "bootstrap_valid_reps": 0,
        }
    work = data.copy()
    apparent = work["apparent_at_threshold"].astype(bool)
    work["apparent_value"] = np.where(apparent, work["transition_value_usd"], 0.0)
    work["transfer_value"] = np.where(
        apparent & work["classification_at_threshold"].isin(TRANSFER_CLASSES),
        work["transition_value_usd"],
        0.0,
    )
    work["substantive_value"] = np.where(
        apparent
        & work["classification_at_threshold"].eq("substantive_derisking"),
        work["transition_value_usd"],
        0.0,
    )
    clusters = work.groupby("importer_iso", observed=True)[
        ["apparent_value", "transfer_value", "substantive_value"]
    ].sum()
    values = clusters.to_numpy(float)
    if len(values) == 0:
        return {
            "risk_transfer_value_share_ci_low": np.nan,
            "risk_transfer_value_share_ci_high": np.nan,
            "substantive_value_share_ci_low": np.nan,
            "substantive_value_share_ci_high": np.nan,
            "risk_transfer_minus_substantive_value_share_ci_low": np.nan,
            "risk_transfer_minus_substantive_value_share_ci_high": np.nan,
            "bootstrap_valid_reps": 0,
        }
    rng = np.random.default_rng(seed)
    transfer_draws: list[np.ndarray] = []
    substantive_draws: list[np.ndarray] = []
    remaining = reps
    while remaining:
        batch = min(remaining, 250)
        indices = rng.integers(0, len(values), size=(batch, len(values)))
        totals = values[indices].sum(axis=1)
        valid = totals[:, 0] > 0
        transfer_draws.append(
            np.where(valid, totals[:, 1] / totals[:, 0], np.nan)
        )
        substantive_draws.append(
            np.where(valid, totals[:, 2] / totals[:, 0], np.nan)
        )
        remaining -= batch
    transfer = np.concatenate(transfer_draws)
    substantive = np.concatenate(substantive_draws)
    paired_difference = transfer - substantive
    valid_reps = int(np.isfinite(paired_difference).sum())
    return {
        "risk_transfer_value_share_ci_low": float(np.nanquantile(transfer, 0.025))
        if valid_reps
        else np.nan,
        "risk_transfer_value_share_ci_high": float(np.nanquantile(transfer, 0.975))
        if valid_reps
        else np.nan,
        "substantive_value_share_ci_low": float(np.nanquantile(substantive, 0.025))
        if valid_reps
        else np.nan,
        "substantive_value_share_ci_high": float(np.nanquantile(substantive, 0.975))
        if valid_reps
        else np.nan,
        "risk_transfer_minus_substantive_value_share_ci_low": float(
            np.nanquantile(paired_difference, 0.025)
        )
        if valid_reps
        else np.nan,
        "risk_transfer_minus_substantive_value_share_ci_high": float(
            np.nanquantile(paired_difference, 0.975)
        )
        if valid_reps
        else np.nan,
        "bootstrap_valid_reps": valid_reps,
    }


def _summarize_one(
    data: pd.DataFrame,
    threshold: float,
    sample_mode: str,
    pandemic_rule: str,
    aggregation_scope: str,
    window_family: str,
    window_label: str,
    baseline_year: int,
    end_year: int,
    windows_included: int,
    config: AnalysisConfig,
) -> dict[str, object]:
    classified = data.copy()
    classified["classification_at_threshold"] = classify_mrci(classified, threshold)
    classified["apparent_at_threshold"] = apparent_mask(classified, threshold)
    eligible = classified[classified["analysis_eligible"]]
    apparent = classified[classified["apparent_at_threshold"]]
    transfer = apparent[
        apparent["classification_at_threshold"].isin(TRANSFER_CLASSES)
    ]
    mine_origin_reconcentration = apparent[
        apparent["classification_at_threshold"].isin(
            {"mine_origin_transfer", "multiple_risk_transfer"}
        )
    ]
    maritime_bottleneck_reconcentration = apparent[
        apparent["classification_at_threshold"].isin(
            {"route_transfer", "multiple_risk_transfer"}
        )
    ]
    substantive = apparent[
        apparent["classification_at_threshold"].eq("substantive_derisking")
    ]
    apparent_value = float(apparent["transition_value_usd"].sum())
    transfer_value = float(transfer["transition_value_usd"].sum())
    mine_origin_reconcentration_value = float(
        mine_origin_reconcentration["transition_value_usd"].sum()
    )
    maritime_bottleneck_reconcentration_value = float(
        maritime_bottleneck_reconcentration["transition_value_usd"].sum()
    )
    substantive_value = float(substantive["transition_value_usd"].sum())
    pandemic_exclusion = pandemic_rule in {
        "exclude_transition_ends_2020_2021",
        "exclude_overlap_2020_2021",
    }
    wmd_edition_exclusion = pandemic_rule == WMD_EDITION_EXCLUSION_RULE
    analysis_id = (
        f"{window_family}|{window_label}|tau={threshold:.6f}|"
        f"sample={sample_mode}|exclusion={pandemic_rule}|scope={aggregation_scope}"
    )
    result: dict[str, object] = {
        "analysis_id": analysis_id,
        "aggregation_scope": aggregation_scope,
        "window_family": window_family,
        "window_label": window_label,
        "baseline_year": int(baseline_year),
        "start_year": int(baseline_year),
        "end_year": int(end_year),
        "window_length_years": int(end_year - baseline_year),
        "windows_included": int(windows_included),
        "threshold": float(threshold),
        "sample_mode": sample_mode,
        "exclusion_rule": pandemic_rule,
        "exclude_pandemic": pandemic_exclusion,
        "pandemic_exclusion_rule": pandemic_rule if pandemic_exclusion else "none",
        "exclude_wmd_edition_boundary_crossings": wmd_edition_exclusion,
        "wmd_edition_boundary_exclusion_rule": (
            pandemic_rule if wmd_edition_exclusion else "none"
        ),
        "eligible_units": len(eligible),
        "eligible_importers": int(eligible["importer_iso"].nunique()),
        "eligible_metals": int(eligible["metal"].nunique()),
        "eligible_end_value_usd": float(eligible["transition_value_usd"].sum()),
        "apparent_units": len(apparent),
        "apparent_end_value_usd": apparent_value,
        "risk_transfer_units": len(transfer),
        "risk_transfer_end_value_usd": transfer_value,
        "risk_transfer_unit_share_of_apparent": len(transfer) / max(len(apparent), 1),
        "risk_transfer_value_share_of_apparent": transfer_value
        / max(apparent_value, 1.0),
        "mine_origin_reconcentration_units": len(mine_origin_reconcentration),
        "mine_origin_reconcentration_end_value_usd": mine_origin_reconcentration_value,
        "mine_origin_reconcentration_unit_share_of_apparent": len(
            mine_origin_reconcentration
        )
        / max(len(apparent), 1),
        "mine_origin_reconcentration_value_share_of_apparent": (
            mine_origin_reconcentration_value / max(apparent_value, 1.0)
        ),
        "maritime_bottleneck_reconcentration_units": len(
            maritime_bottleneck_reconcentration
        ),
        "maritime_bottleneck_reconcentration_end_value_usd": (
            maritime_bottleneck_reconcentration_value
        ),
        "maritime_bottleneck_reconcentration_unit_share_of_apparent": len(
            maritime_bottleneck_reconcentration
        )
        / max(len(apparent), 1),
        "maritime_bottleneck_reconcentration_value_share_of_apparent": (
            maritime_bottleneck_reconcentration_value / max(apparent_value, 1.0)
        ),
        "substantive_units": len(substantive),
        "substantive_end_value_usd": substantive_value,
        "substantive_unit_share_of_apparent": len(substantive)
        / max(len(apparent), 1),
        "substantive_value_share_of_apparent": substantive_value
        / max(apparent_value, 1.0),
        "risk_transfer_minus_substantive_value_share_of_apparent": (
            (transfer_value - substantive_value) / max(apparent_value, 1.0)
        ),
        "route_evidence_share_of_eligible": float(
            eligible["route_evidence_eligible"].mean()
        )
        if len(eligible)
        else np.nan,
        "trade_year_exposure": aggregation_scope == "pooled_family",
    }
    result.update(
        _bootstrap_ratios(
            classified,
            config.bootstrap_reps,
            stable_seed(config.random_seed, analysis_id),
        )
    )
    return result


def summarize_windows(
    panel: pd.DataFrame,
    balanced: pd.DataFrame,
    threshold: float,
    sample_mode: str,
    pandemic_rule: str,
    config: AnalysisConfig,
) -> pd.DataFrame:
    data = _filter_sample(panel, balanced, sample_mode)
    if pandemic_rule == "exclude_overlap_2020_2021":
        data = data[~data["pandemic_overlap"].astype(bool)].copy()
    elif pandemic_rule == "exclude_transition_ends_2020_2021":
        data = data[~data["pandemic_transition"].astype(bool)].copy()
    elif pandemic_rule == WMD_EDITION_EXCLUSION_RULE:
        data = data[~data["wmd_edition_boundary_crossing"].astype(bool)].copy()
    elif pandemic_rule != "none":
        raise ValueError(f"Unknown pandemic_rule: {pandemic_rule}")

    rows: list[dict[str, object]] = []
    for (family, label, baseline_year, end_year), frame in data.groupby(
        ["window_family", "window_label", "baseline_year", "end_year"],
        sort=True,
        observed=True,
    ):
        rows.append(
            _summarize_one(
                frame,
                threshold,
                sample_mode,
                pandemic_rule,
                "window",
                str(family),
                str(label),
                int(baseline_year),
                int(end_year),
                1,
                config,
            )
        )
    for family, frame in data.groupby("window_family", sort=True, observed=True):
        window_count = int(frame["window_label"].nunique())
        if window_count == 0:
            continue
        rows.append(
            _summarize_one(
                frame,
                threshold,
                sample_mode,
                pandemic_rule,
                "pooled_family",
                str(family),
                f"all_{family}",
                int(frame["baseline_year"].min()),
                int(frame["end_year"].max()),
                window_count,
                config,
            )
        )
    return pd.DataFrame(rows)


def build_default_and_robustness_summaries(
    all_windows: pd.DataFrame,
    balanced: pd.DataFrame,
    config: AnalysisConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    default = summarize_windows(
        all_windows,
        balanced,
        config.default_threshold,
        "local_full_evidence_endpoint",
        "none",
        config,
    )
    scenarios: list[pd.DataFrame] = []
    for threshold in config.thresholds:
        scenarios.append(
            summarize_windows(
                all_windows,
                balanced,
                threshold,
                "local_full_evidence_endpoint",
                "none",
                config,
            ).assign(robustness_dimension="threshold")
        )
    scenarios.append(
        summarize_windows(
            all_windows,
            balanced,
            config.default_threshold,
            "local_endpoint",
            "none",
            config,
        ).assign(robustness_dimension="local_endpoint_semantic_bridge")
    )
    for sample_mode in (
        "balanced_analysis_2012_2024",
        "balanced_full_evidence_2012_2024",
    ):
        scenarios.append(
            summarize_windows(
                all_windows,
                balanced,
                config.default_threshold,
                sample_mode,
                "none",
                config,
            ).assign(robustness_dimension="balanced_sample")
        )
    if balanced["canonical_704_member"].any():
        scenarios.append(
            summarize_windows(
                all_windows,
                balanced,
                config.default_threshold,
                "canonical_704_bridge",
                "none",
                config,
            ).assign(robustness_dimension="canonical_bridge")
        )
    for pandemic_rule in (
        "exclude_transition_ends_2020_2021",
        "exclude_overlap_2020_2021",
    ):
        scenarios.append(
            summarize_windows(
                all_windows,
                balanced,
                config.default_threshold,
                "local_full_evidence_endpoint",
                pandemic_rule,
                config,
            ).assign(robustness_dimension="pandemic_exclusion")
        )
    scenarios.append(
        summarize_windows(
            all_windows,
            balanced,
            config.default_threshold,
            "local_full_evidence_endpoint",
            WMD_EDITION_EXCLUSION_RULE,
            config,
        ).assign(robustness_dimension="wmd_edition_boundary_exclusion")
    )
    robustness = pd.concat(scenarios, ignore_index=True) if scenarios else pd.DataFrame()
    return default, robustness


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    normalized = series.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    result = normalized.map(mapping)
    if result.isna().any():
        examples = sorted(set(normalized[result.isna()].dropna().head(5)))
        raise ValueError(f"Cannot coerce boolean values: {examples}")
    return result.astype(bool)


def compare_frames(
    generated: pd.DataFrame,
    canonical: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str],
    rtol: float = 1e-10,
    atol: float = 1e-8,
) -> dict[str, object]:
    left = generated.sort_values(list(keys)).reset_index(drop=True)
    right = canonical.sort_values(list(keys)).reset_index(drop=True)
    report: dict[str, object] = {
        "generated_rows": len(left),
        "canonical_rows": len(right),
        "key_columns": list(keys),
        "column_results": {},
    }
    if len(left) != len(right):
        report["passed"] = False
        report["reason"] = "row_count_mismatch"
        return report
    key_match = left[list(keys)].astype("string").fillna("").equals(
        right[list(keys)].astype("string").fillna("")
    )
    report["keys_match"] = key_match
    passed = key_match
    for column in columns:
        if column not in left or column not in right:
            report["column_results"][column] = {"passed": False, "missing": True}
            passed = False
            continue
        a = left[column]
        b = right[column]
        if pd.api.types.is_bool_dtype(a) or pd.api.types.is_bool_dtype(b):
            match = _as_bool(a).equals(_as_bool(b))
            result = {"passed": bool(match), "type": "boolean"}
        elif pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av = pd.to_numeric(a, errors="coerce").to_numpy(float)
            bv = pd.to_numeric(b, errors="coerce").to_numpy(float)
            finite = np.isfinite(av) & np.isfinite(bv)
            same_missing = np.array_equal(np.isnan(av), np.isnan(bv))
            match = same_missing and np.allclose(
                av[finite], bv[finite], rtol=rtol, atol=atol
            )
            differences = np.abs(av[finite] - bv[finite])
            relative = differences / np.maximum(np.abs(bv[finite]), atol)
            result = {
                "passed": bool(match),
                "type": "numeric",
                "max_absolute_difference": float(differences.max())
                if len(differences)
                else 0.0,
                "max_relative_difference": float(relative.max())
                if len(relative)
                else 0.0,
            }
        else:
            match = a.astype("string").fillna("").equals(
                b.astype("string").fillna("")
            )
            result = {"passed": bool(match), "type": "string"}
        report["column_results"][column] = result
        passed = passed and bool(result["passed"])
    report["passed"] = bool(passed)
    return report


RISK_REGRESSION_COLUMNS = [
    "direct_total_value_usd",
    "direct_source_count",
    "direct_hhi",
    "direct_top_share",
    "direct_top_source_iso",
    "origin_total_value_usd",
    "origin_source_count",
    "origin_hhi",
    "origin_top_share",
    "origin_top_source_iso",
    "attribution_weight_usd",
    "attribution_uncertainty_index",
    "mean_flow_quality_weight",
    "mean_observation_sigma_log",
    "mean_hub_uncertainty",
    "route_matched_value_usd",
    "sea_value_usd",
    "route_coverage_share",
    "sea_value_share",
    "chokepoint_count",
    "top_chokepoint_value_usd",
    "top_chokepoint",
    "top_chokepoint_share",
    "origin_coverage_ratio",
]

TRANSITION_REGRESSION_COLUMNS = [
    *METRICS,
    *[f"{metric}_previous" for metric in METRICS],
    "previous_year",
    "transition",
    "delta_direct_hhi",
    "delta_origin_hhi",
    "delta_top_chokepoint_share",
    "delta_attribution_uncertainty_index",
    "analysis_eligible",
    "route_evidence_eligible",
    "apparent_derisking",
    "classification",
    "classification_threshold",
    "mrci_classification",
    "mrci_threshold",
    "direct_risk_reduction",
    "origin_risk_reduction",
    "route_risk_reduction",
    "multilayer_derisking_margin",
    "apparent_derisking_indicator",
    "risk_transfer_indicator",
    "substantive_derisking_indicator",
    "evidence_weakened_indicator",
    "risk_transfer_magnitude",
    "transition_value_usd",
]


def canonical_regression_report(
    annual: pd.DataFrame,
    adjacent: pd.DataFrame,
    canonical_risk_path: Path,
    canonical_transition_path: Path,
) -> dict[str, object]:
    canonical_risk = pd.read_csv(canonical_risk_path, low_memory=False)
    generated_risk = annual[annual["year"].between(2020, 2024)].copy()
    risk_report = compare_frames(
        generated_risk,
        canonical_risk,
        [*GROUP],
        RISK_REGRESSION_COLUMNS,
    )
    canonical_transitions = pd.read_csv(canonical_transition_path, low_memory=False)
    generated_transitions = adjacent[adjacent["end_year"].between(2021, 2024)].copy()
    transition_report = compare_frames(
        generated_transitions,
        canonical_transitions,
        [*KEYS, "year"],
        TRANSITION_REGRESSION_COLUMNS,
    )
    return {
        "check": "2020-2024 canonical v2 semantic regression",
        "canonical_risk_path": str(canonical_risk_path),
        "canonical_risk_sha256": sha256(canonical_risk_path),
        "canonical_transition_path": str(canonical_transition_path),
        "canonical_transition_sha256": sha256(canonical_transition_path),
        "risk_panel": risk_report,
        "annual_transitions": transition_report,
        "passed": bool(risk_report["passed"] and transition_report["passed"]),
    }
