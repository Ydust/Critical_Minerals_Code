"""Country-share comparisons across overlapping WMD vintages."""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path

from wmd_parser import MODEL_METAL_BY_COMMODITY, measurement_basis, normalized_token


COUNTRY_COMPARE_ALIASES = {
    "burma": "myanmar",
    "czech republic": "czechia",
    "macedonia": "north macedonia",
    "swaziland": "eswatini",
    "turkiye": "turkey",
    "viet nam": "vietnam",
}


SHARE_DETAIL_FIELDS = [
    "bridge",
    "edition_a",
    "edition_b",
    "commodity_label",
    "model_metal",
    "year",
    "country_comparison_key",
    "country_a",
    "country_b",
    "production_a",
    "production_b",
    "total_production_a",
    "total_production_b",
    "share_a",
    "share_b",
    "share_difference_b_minus_a",
    "absolute_share_difference",
    "unit_a",
    "unit_b",
    "unit_match",
    "presence_status",
]


SHARE_SUMMARY_FIELDS = [
    "bridge",
    "edition_a",
    "edition_b",
    "commodity_label",
    "model_metal",
    "year",
    "total_production_a",
    "total_production_b",
    "reported_countries_a",
    "reported_countries_b",
    "shared_reported_countries",
    "only_a_countries",
    "only_b_countries",
    "total_variation_distance",
    "share_overlap_coefficient",
    "hhi_a",
    "hhi_b",
    "hhi_difference_b_minus_a",
    "top_country_a",
    "top_country_b",
    "top_share_a",
    "top_share_b",
    "maximum_absolute_country_share_shift",
    "all_units_match",
]


def country_compare_key(value: str) -> str:
    key = normalized_token(value)
    return COUNTRY_COMPARE_ALIASES.get(key, key)


def read_wmd2026_seed(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row["commodity_label"].strip()
            if label not in MODEL_METAL_BY_COMMODITY:
                continue
            production = float(row["production"])
            records.append(
                {
                    "commodity_wmd": row["commodity_wmd"],
                    "commodity_label": label,
                    "model_metal": MODEL_METAL_BY_COMMODITY[label],
                    "measurement_basis": measurement_basis(row["commodity_wmd"], label),
                    "mine_origin_country_wmd": row["mine_origin_country_wmd"].strip(),
                    "year": int(row["year"]),
                    "production": production,
                    "unit": row["unit"].strip(),
                    "unit_raw": row["unit"].strip(),
                    "source_rem": row["source_flag"].strip(),
                    "source_flag": row["source_flag"].strip(),
                    "source_confidence": float(row["source_confidence"]),
                    "source_edition": "WMD2026",
                    "source_dataset": row["source_dataset"].strip(),
                    "source_url": row["source_url"].strip(),
                    "source_pdf_page": "",
                    "value_status": "reported_zero" if production == 0 else "reported_positive",
                }
            )
    return records


def _aggregate(records: list[dict], years: set[int]):
    values = defaultdict(float)
    names = {}
    units = defaultdict(set)
    labels = {}
    for row in records:
        if row["year"] not in years or not row.get("model_metal"):
            continue
        group = (row["commodity_label"], row["year"])
        country = country_compare_key(row["mine_origin_country_wmd"])
        key = (*group, country)
        values[key] += float(row["production"])
        names[key] = row["mine_origin_country_wmd"]
        units[group].add(row["unit"])
        labels[group] = row["model_metal"]
    return values, names, units, labels


def compare_share_vintages(
    records_a: list[dict],
    records_b: list[dict],
    edition_a: str,
    edition_b: str,
    years: set[int],
) -> tuple[list[dict], list[dict]]:
    values_a, names_a, units_a, labels_a = _aggregate(records_a, years)
    values_b, names_b, units_b, labels_b = _aggregate(records_b, years)
    groups = sorted({key[:2] for key in values_a} | {key[:2] for key in values_b})
    detail = []
    summary = []
    bridge = f"{edition_a}_vs_{edition_b}"

    for commodity, year in groups:
        countries = sorted(
            {key[2] for key in values_a if key[:2] == (commodity, year)}
            | {key[2] for key in values_b if key[:2] == (commodity, year)}
        )
        total_a = sum(values_a.get((commodity, year, country), 0.0) for country in countries)
        total_b = sum(values_b.get((commodity, year, country), 0.0) for country in countries)
        if total_a <= 0 or total_b <= 0:
            raise ValueError(f"Cannot calculate positive-denominator shares for {(commodity, year)}")
        group_rows = []
        for country in countries:
            key = (commodity, year, country)
            in_a = key in values_a
            in_b = key in values_b
            production_a = values_a.get(key, 0.0)
            production_b = values_b.get(key, 0.0)
            share_a = production_a / total_a
            share_b = production_b / total_b
            units_match = units_a.get((commodity, year), set()) == units_b.get((commodity, year), set())
            group_rows.append(
                {
                    "bridge": bridge,
                    "edition_a": edition_a,
                    "edition_b": edition_b,
                    "commodity_label": commodity,
                    "model_metal": labels_a.get((commodity, year), labels_b.get((commodity, year), "")),
                    "year": year,
                    "country_comparison_key": country,
                    "country_a": names_a.get(key, ""),
                    "country_b": names_b.get(key, ""),
                    "production_a": production_a if in_a else None,
                    "production_b": production_b if in_b else None,
                    "total_production_a": total_a,
                    "total_production_b": total_b,
                    "share_a": share_a,
                    "share_b": share_b,
                    "share_difference_b_minus_a": share_b - share_a,
                    "absolute_share_difference": abs(share_b - share_a),
                    "unit_a": " | ".join(sorted(units_a.get((commodity, year), set()))),
                    "unit_b": " | ".join(sorted(units_b.get((commodity, year), set()))),
                    "unit_match": units_match,
                    "presence_status": "both_reported" if in_a and in_b else (
                        f"only_{edition_a}_reported" if in_a else f"only_{edition_b}_reported"
                    ),
                }
            )
        detail.extend(group_rows)
        top_a = max(group_rows, key=lambda row: row["share_a"])
        top_b = max(group_rows, key=lambda row: row["share_b"])
        tv = 0.5 * sum(row["absolute_share_difference"] for row in group_rows)
        summary.append(
            {
                "bridge": bridge,
                "edition_a": edition_a,
                "edition_b": edition_b,
                "commodity_label": commodity,
                "model_metal": group_rows[0]["model_metal"],
                "year": year,
                "total_production_a": total_a,
                "total_production_b": total_b,
                "reported_countries_a": sum(row["production_a"] is not None for row in group_rows),
                "reported_countries_b": sum(row["production_b"] is not None for row in group_rows),
                "shared_reported_countries": sum(row["production_a"] is not None and row["production_b"] is not None for row in group_rows),
                "only_a_countries": sum(row["production_a"] is not None and row["production_b"] is None for row in group_rows),
                "only_b_countries": sum(row["production_a"] is None and row["production_b"] is not None for row in group_rows),
                "total_variation_distance": tv,
                "share_overlap_coefficient": 1.0 - tv,
                "hhi_a": sum(row["share_a"] ** 2 for row in group_rows),
                "hhi_b": sum(row["share_b"] ** 2 for row in group_rows),
                "hhi_difference_b_minus_a": sum(row["share_b"] ** 2 - row["share_a"] ** 2 for row in group_rows),
                "top_country_a": top_a["country_comparison_key"],
                "top_country_b": top_b["country_comparison_key"],
                "top_share_a": top_a["share_a"],
                "top_share_b": top_b["share_b"],
                "maximum_absolute_country_share_shift": max(row["absolute_share_difference"] for row in group_rows),
                "all_units_match": all(row["unit_match"] for row in group_rows),
            }
        )
    return detail, summary

