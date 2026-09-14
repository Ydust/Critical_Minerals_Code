"""Build reproducible WMD production files, vintage overlaps, and audits."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from bridge_shares import (
    SHARE_DETAIL_FIELDS,
    SHARE_SUMMARY_FIELDS,
    compare_share_vintages,
    country_compare_key,
    read_wmd2026_seed,
)
from wmd_parser import EDITIONS, MODEL_METAL_BY_COMMODITY, key_duplicates, normalized_token, parse_coordinates


LONG_FIELDS = [
    "commodity_wmd",
    "commodity_label",
    "model_metal",
    "measurement_basis",
    "mine_origin_country_wmd",
    "year",
    "production",
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
]


SOURCE_URLS = {
    "WMD2018": "https://www.world-mining-data.info/wmd/downloads/PDF/WMD2018.pdf",
    "WMD2021": "https://www.world-mining-data.info/wmd/downloads/PDF/WMD2021.pdf",
    "WMD2025": "https://www.world-mining-data.info/wmd/downloads/PDF/WMD%202025.pdf",
}


def sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Iterable[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def with_dataset(records: list[dict]) -> list[dict]:
    output = []
    for row in records:
        copy = dict(row)
        copy["source_dataset"] = f"World Mining Data {copy['source_edition'][-4:]} chapter 6.4"
        copy["source_url"] = SOURCE_URLS[copy["source_edition"]]
        output.append(copy)
    return output


def overlap_comparison(old_records: list[dict], new_records: list[dict]) -> tuple[list[dict], list[dict]]:
    def index(records: list[dict]):
        result = {}
        collisions = []
        for row in records:
            if row["year"] not in {2015, 2016}:
                continue
            key = (row["commodity_label"], country_compare_key(row["mine_origin_country_wmd"]), row["year"])
            if key in result:
                collisions.append({"key": key, "rows": [result[key], row]})
            result[key] = row
        return result, collisions

    old, old_collisions = index(old_records)
    new, new_collisions = index(new_records)
    rows = []
    for key in sorted(set(old) | set(new)):
        left = old.get(key)
        right = new.get(key)
        left_value = left["production"] if left else None
        right_value = right["production"] if right else None
        if left is None:
            status = "only_wmd2021_reported"
        elif right is None:
            status = "only_wmd2018_reported"
        elif left_value == right_value:
            status = "both_equal"
        else:
            status = "both_revised"
        difference = None if left is None or right is None else right_value - left_value
        relative = None
        if difference is not None and left_value != 0:
            relative = difference / abs(left_value)
        rows.append(
            {
                "commodity_label": key[0],
                "model_metal": (left or right).get("model_metal", ""),
                "country_comparison_key": key[1],
                "year": key[2],
                "country_wmd2018": left["mine_origin_country_wmd"] if left else "",
                "country_wmd2021": right["mine_origin_country_wmd"] if right else "",
                "production_wmd2018": left_value,
                "production_wmd2021": right_value,
                "absolute_difference_wmd2021_minus_wmd2018": difference,
                "relative_difference_vs_wmd2018": relative,
                "unit_wmd2018": left["unit"] if left else "",
                "unit_wmd2021": right["unit"] if right else "",
                "unit_match": bool(left and right and left["unit"] == right["unit"]),
                "commodity_wmd2018": left["commodity_wmd"] if left else "",
                "commodity_wmd2021": right["commodity_wmd"] if right else "",
                "measurement_basis_wmd2018": left["measurement_basis"] if left else "",
                "measurement_basis_wmd2021": right["measurement_basis"] if right else "",
                "measurement_basis_match": bool(
                    left and right and left["measurement_basis"] == right["measurement_basis"]
                ),
                "comparison_status": status,
            }
        )
    return rows, old_collisions + new_collisions


def flatten_table_audit(audits: list[dict]) -> list[dict]:
    rows = []
    for audit in audits:
        for year in audit["year_totals"]:
            row = {key: value for key, value in audit.items() if key != "year_totals"}
            row.update(year)
            rows.append(row)
    return rows


def summarize_overlap_by_commodity_year(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        key = (row["commodity_label"], row["model_metal"], row["year"])
        group = groups.setdefault(
            key,
            {
                "both_equal_rows": 0,
                "both_revised_rows": 0,
                "only_wmd2018_rows": 0,
                "only_wmd2021_rows": 0,
                "total_wmd2018_reported": 0,
                "total_wmd2021_reported": 0,
            },
        )
        status_field = {
            "both_equal": "both_equal_rows",
            "both_revised": "both_revised_rows",
            "only_wmd2018_reported": "only_wmd2018_rows",
            "only_wmd2021_reported": "only_wmd2021_rows",
        }[row["comparison_status"]]
        group[status_field] += 1
        if row["production_wmd2018"] is not None:
            group["total_wmd2018_reported"] += row["production_wmd2018"]
        if row["production_wmd2021"] is not None:
            group["total_wmd2021_reported"] += row["production_wmd2021"]
    output = []
    for key, group in sorted(groups.items()):
        shared = group["both_equal_rows"] + group["both_revised_rows"]
        difference = group["total_wmd2021_reported"] - group["total_wmd2018_reported"]
        denominator = group["total_wmd2018_reported"]
        output.append(
            {
                "commodity_label": key[0],
                "model_metal": key[1],
                "year": key[2],
                **group,
                "shared_rows": shared,
                "revised_share_among_shared": group["both_revised_rows"] / shared if shared else None,
                "total_difference_wmd2021_minus_wmd2018": difference,
                "relative_total_difference_vs_wmd2018": difference / abs(denominator) if denominator else None,
            }
        )
    return output


def summarize_edition(result, records: list[dict], quarantined_records: list[dict] | None = None) -> dict:
    target = [row for row in records if row["model_metal"]]
    quarantined_records = quarantined_records or []
    failed = [row for row in result.table_audit if not row["all_year_totals_reconciled"]]
    target_audits = [row for row in result.table_audit if row["model_metal"]]
    failed_target = [row for row in target_audits if not row["all_year_totals_reconciled"]]
    return {
        "coordinate_pdf_page_count": result.page_count,
        "configured_chapter_pages": [EDITIONS[result.edition].first_page, EDITIONS[result.edition].last_page],
        "tables_parsed_sections_6_4_1_to_6_4_4": len(result.table_audit),
        "target_mine_origin_tables": sum(bool(row["model_metal"]) for row in result.table_audit),
        "target_tables_with_all_five_printed_totals_reconciled": len(target_audits) - len(failed_target),
        "target_tables_failing_at_least_one_printed_total": [row["commodity_wmd"] for row in failed_target],
        "country_year_records_all_nonfuel_minerals": len(records),
        "country_year_records_quarantined": len(quarantined_records),
        "country_year_records_target_minerals": len(target),
        "unique_countries_all_nonfuel_minerals": len({row["mine_origin_country_wmd"] for row in records}),
        "explicit_reported_zeros": sum(row["production"] == 0 for row in records),
        "blank_cells_not_imputed": sum(row["blank_cells"] for row in result.table_audit),
        "tables_with_all_five_printed_totals_reconciled": len(result.table_audit) - len(failed),
        "tables_failing_at_least_one_printed_total": [row["commodity_wmd"] for row in failed],
        "parser_warnings": result.warnings,
    }


def summarize_share_bridge(summary_rows: list[dict]) -> dict:
    distances = [row["total_variation_distance"] for row in summary_rows]
    ranked = sorted(summary_rows, key=lambda row: row["total_variation_distance"], reverse=True)
    return {
        "commodity_year_vectors": len(summary_rows),
        "mean_total_variation_distance": mean(distances),
        "median_total_variation_distance": median(distances),
        "maximum_total_variation_distance": max(distances),
        "all_units_match": all(row["all_units_match"] for row in summary_rows),
        "largest_distribution_revisions": [
            {
                "commodity_label": row["commodity_label"],
                "model_metal": row["model_metal"],
                "year": row["year"],
                "total_variation_distance": row["total_variation_distance"],
                "top_country_a": row["top_country_a"],
                "top_country_b": row["top_country_b"],
            }
            for row in ranked[:10]
        ],
    }


def build(args) -> dict:
    old_result = parse_coordinates(args.wmd2018_coordinates, EDITIONS["WMD2018"])
    new_result = parse_coordinates(args.wmd2021_coordinates, EDITIONS["WMD2021"])
    latest_result = parse_coordinates(args.wmd2025_coordinates, EDITIONS["WMD2025"])
    old_records = with_dataset(old_result.records)
    new_records = with_dataset(new_result.records)
    latest_records_unchecked = with_dataset(latest_result.records)
    latest_failed_labels = {
        row["commodity_label"] for row in latest_result.table_audit
        if not row["all_year_totals_reconciled"]
    }
    latest_quarantined = [
        row for row in latest_records_unchecked if row["commodity_label"] in latest_failed_labels
    ]
    latest_records = [
        row for row in latest_records_unchecked if row["commodity_label"] not in latest_failed_labels
    ]
    wmd2026_records = read_wmd2026_seed(args.wmd2026_seed)

    duplicates = key_duplicates(old_records + new_records + latest_records + wmd2026_records)
    if duplicates:
        raise ValueError(f"Duplicate edition/commodity/country/year keys: {duplicates[:10]}")

    combined = [row for row in old_records if row["year"] <= 2014] + new_records
    combined.sort(key=lambda row: (
        row["commodity_label"], normalized_token(row["mine_origin_country_wmd"]), row["year"]
    ))
    target = [row for row in combined if row["commodity_label"] in MODEL_METAL_BY_COMMODITY]
    latest_target = [row for row in latest_records if row["commodity_label"] in MODEL_METAL_BY_COMMODITY]

    overlap_all, overlap_collisions = overlap_comparison(old_records, new_records)
    if overlap_collisions:
        raise ValueError(f"Country-normalized overlap collisions: {overlap_collisions[:3]}")
    overlap_target = [row for row in overlap_all if row["model_metal"]]

    output = args.output_dir
    write_csv(output / "wmd2018_parsed_2012_2016.csv", old_records, LONG_FIELDS)
    write_csv(output / "wmd2021_parsed_2015_2019.csv", new_records, LONG_FIELDS)
    write_csv(output / "wmd2025_parsed_2019_2023.csv", latest_records, LONG_FIELDS)
    write_csv(output / "wmd2025_quarantined_source_rows.csv", latest_quarantined, LONG_FIELDS)
    write_csv(output / "wmd_2019_2023_mine_origin_seed_wmd2025.csv", latest_target, LONG_FIELDS)
    write_csv(output / "wmd_2012_2019_country_mineral_production.csv", combined, LONG_FIELDS)
    write_csv(output / "wmd_2012_2019_mine_origin_seed.csv", target, LONG_FIELDS)

    overlap_fields = [
        "commodity_label", "model_metal", "country_comparison_key", "year",
        "country_wmd2018", "country_wmd2021", "production_wmd2018", "production_wmd2021",
        "absolute_difference_wmd2021_minus_wmd2018", "relative_difference_vs_wmd2018",
        "unit_wmd2018", "unit_wmd2021", "unit_match", "commodity_wmd2018", "commodity_wmd2021",
        "measurement_basis_wmd2018", "measurement_basis_wmd2021", "measurement_basis_match",
        "comparison_status",
    ]
    write_csv(output / "wmd_2015_2016_overlap_comparison_all.csv", overlap_all, overlap_fields)
    write_csv(output / "wmd_2015_2016_overlap_comparison_target.csv", overlap_target, overlap_fields)
    overlap_summary = summarize_overlap_by_commodity_year(overlap_target)
    overlap_summary_fields = [
        "commodity_label", "model_metal", "year", "both_equal_rows", "both_revised_rows",
        "only_wmd2018_rows", "only_wmd2021_rows", "shared_rows", "revised_share_among_shared",
        "total_wmd2018_reported", "total_wmd2021_reported",
        "total_difference_wmd2021_minus_wmd2018", "relative_total_difference_vs_wmd2018",
    ]
    write_csv(output / "wmd_2015_2016_overlap_summary_target.csv", overlap_summary, overlap_summary_fields)

    bridge_2019_detail, bridge_2019_summary = compare_share_vintages(
        new_records, latest_records, "WMD2021", "WMD2025", {2019}
    )
    bridge_2020_2023_detail, bridge_2020_2023_summary = compare_share_vintages(
        latest_records, wmd2026_records, "WMD2025", "WMD2026", {2020, 2021, 2022, 2023}
    )
    write_csv(
        output / "wmd2021_vs_wmd2025_2019_country_share_comparison.csv",
        bridge_2019_detail,
        SHARE_DETAIL_FIELDS,
    )
    write_csv(
        output / "wmd2021_vs_wmd2025_2019_share_summary.csv",
        bridge_2019_summary,
        SHARE_SUMMARY_FIELDS,
    )
    write_csv(
        output / "wmd2025_vs_wmd2026_2020_2023_country_share_comparison.csv",
        bridge_2020_2023_detail,
        SHARE_DETAIL_FIELDS,
    )
    write_csv(
        output / "wmd2025_vs_wmd2026_2020_2023_share_summary.csv",
        bridge_2020_2023_summary,
        SHARE_SUMMARY_FIELDS,
    )

    flat_audit = flatten_table_audit(
        old_result.table_audit + new_result.table_audit + latest_result.table_audit
    )
    audit_fields = [
        "edition", "table_id", "section", "commodity_wmd", "commodity_label", "model_metal",
        "measurement_basis", "header_pdf_page", "unit", "country_rows", "parsed_records",
        "blank_cells", "unparsed_numeric_cells", "all_year_totals_reconciled", "year",
        "printed_total", "parsed_country_sum", "difference", "reconciled",
    ]
    write_csv(output / "wmd_parse_table_year_audit.csv", flat_audit, audit_fields)

    status_counts = {}
    for row in overlap_target:
        status_counts[row["comparison_status"]] = status_counts.get(row["comparison_status"], 0) + 1
    both = [row for row in overlap_target if row["comparison_status"] in {"both_equal", "both_revised"}]
    basis_changes = sorted({
        (row["commodity_label"], row["measurement_basis_wmd2018"], row["measurement_basis_wmd2021"])
        for row in both if not row["measurement_basis_match"]
    })
    heading_changes = sorted({
        (row["commodity_label"], row["commodity_wmd2018"], row["commodity_wmd2021"])
        for row in both if row["commodity_wmd2018"] != row["commodity_wmd2021"]
    })
    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parser": "coordinate-aware WMD chapter 6.4 parser v1",
        "scope": "All commodities in official chapter sections 6.4.1-6.4.4; target seed is the canonical critical-mineral subset.",
        "year_merge_policy": "WMD2018 supplies 2012-2014; the later WMD2021 vintage supplies 2015-2019.",
        "missing_value_policy": "A blank year cell produces no row and is never converted to zero. Printed zero remains production=0 with value_status=reported_zero.",
        "unit_policy": "Values are not converted. The reported unit is retained; metric-tonne typography is normalized to 'metr. t'.",
        "input_hashes_sha256": {
            "wmd2018_coordinates": sha256(args.wmd2018_coordinates),
            "wmd2021_coordinates": sha256(args.wmd2021_coordinates),
            "wmd2025_coordinates": sha256(args.wmd2025_coordinates),
            "wmd2018_pdf": sha256(args.wmd2018_pdf),
            "wmd2021_pdf": sha256(args.wmd2021_pdf),
            "wmd2025_pdf": sha256(args.wmd2025_pdf),
            "wmd2026_seed": sha256(args.wmd2026_seed),
        },
        "official_source_urls": SOURCE_URLS,
        "editions": {
            "WMD2018": summarize_edition(old_result, old_records),
            "WMD2021": summarize_edition(new_result, new_records),
            "WMD2025": summarize_edition(latest_result, latest_records, latest_quarantined),
        },
        "merged_outputs": {
            "all_nonfuel_country_year_records": len(combined),
            "target_mine_origin_country_year_records": len(target),
            "target_model_metals": sorted({row["model_metal"] for row in target}),
            "target_commodity_labels": sorted({row["commodity_label"] for row in target}),
            "coverage_years": sorted({row["year"] for row in target}),
        },
        "wmd2025_target_output": {
            "country_year_records": len(latest_target),
            "coverage_years": sorted({row["year"] for row in latest_target}),
            "target_commodity_labels": sorted({row["commodity_label"] for row in latest_target}),
            "source_url": SOURCE_URLS["WMD2025"],
            "quarantined_non_target_commodity_labels": sorted(latest_failed_labels),
            "quarantine_reason": "The official PDF's Phosphate Rock table interleaves duplicate/next-vintage rows and cannot reconcile country rows to its printed totals. It is outside the target critical-mineral mapping and is excluded without imputation.",
        },
        "overlap_2015_2016_target": {
            "row_status_counts": status_counts,
            "both_vintages_rows": len(both),
            "revised_rows": sum(row["comparison_status"] == "both_revised" for row in both),
            "equal_rows": sum(row["comparison_status"] == "both_equal" for row in both),
            "raw_heading_changes": [
                {"commodity_label": x[0], "WMD2018": x[1], "WMD2021": x[2]} for x in heading_changes
            ],
            "measurement_basis_title_changes": [
                {"commodity_label": x[0], "WMD2018": x[1], "WMD2021": x[2]} for x in basis_changes
            ],
        },
        "share_bridges": {
            "WMD2021_vs_WMD2025_2019": summarize_share_bridge(bridge_2019_summary),
            "WMD2025_vs_WMD2026_2020_2023": summarize_share_bridge(bridge_2020_2023_summary),
            "interpretation": "TV distance is half the L1 distance between two country-share vectors: 0 means identical shares and 1 means disjoint origin distributions.",
        },
        "comparability_cautions": [
            "WMD editions revise historical observations: use the overlap comparison rather than assuming 2015-2016 equality.",
            "Raw commodity titles changed for Boron, Rare Earths, and Zircon; canonical labels preserve longitudinal keys.",
            "Vanadium's printed basis title changes from V2O5 content (WMD2018) to V content (WMD2021), although overlapping numeric magnitudes do not show a stoichiometric break. No conversion is imposed; the basis fields retain this warning.",
            "Source remarks are retained verbatim. Only r/p/e receive the same confidence defaults used by the current project seed; all other remark letters receive 0.50 rather than an invented confidence ranking.",
            "Bridge shares use each vintage's reported-country total as the denominator. A country absent from one vintage receives comparison share zero only for vector alignment; this is not an imputation into the production seed.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "wmd_parse_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    old_summary = audit["editions"]["WMD2018"]
    new_summary = audit["editions"]["WMD2021"]
    latest_summary = audit["editions"]["WMD2025"]
    overlap = audit["overlap_2015_2016_target"]
    bridge_2019 = audit["share_bridges"]["WMD2021_vs_WMD2025_2019"]
    bridge_later = audit["share_bridges"]["WMD2025_vs_WMD2026_2020_2023"]
    markdown = f"""# WMD 2018/2021/2025 coordinate parsing and bridge audit

## Result

- Parsed all chapter 6.4.1-6.4.4 tables: {old_summary['tables_parsed_sections_6_4_1_to_6_4_4']} in WMD2018, {new_summary['tables_parsed_sections_6_4_1_to_6_4_4']} in WMD2021, and {latest_summary['tables_parsed_sections_6_4_1_to_6_4_4']} in WMD2025.
- Reconciled every printed commodity-year total in WMD2018 and WMD2021. In WMD2025, {latest_summary['tables_with_all_five_printed_totals_reconciled']} of {latest_summary['tables_parsed_sections_6_4_1_to_6_4_4']} tables reconcile; all {latest_summary['target_tables_with_all_five_printed_totals_reconciled']} target tables reconcile.
- Quarantined the non-target WMD2025 Phosphate Rock table ({latest_summary['country_year_records_quarantined']} extracted cells) because the official PDF interleaves duplicate/next-vintage country rows; no value was guessed.
- Preserved {old_summary['blank_cells_not_imputed'] + new_summary['blank_cells_not_imputed'] + latest_summary['blank_cells_not_imputed']} blank cells as missing and {old_summary['explicit_reported_zeros'] + new_summary['explicit_reported_zeros'] + latest_summary['explicit_reported_zeros']} printed zeros as zero.
- The merged critical-mineral seed contains {audit['merged_outputs']['target_mine_origin_country_year_records']} country-mineral-year observations over 2012-2019.
- The standalone WMD2025 target seed contains {audit['wmd2025_target_output']['country_year_records']} observations over 2019-2023; it does not alter the main merge policy.

## Vintage overlap, 2015-2016 target minerals

- Shared country-mineral-year cells: {overlap['both_vintages_rows']}.
- Numerically revised in WMD2021: {overlap['revised_rows']} ({overlap['revised_rows'] / overlap['both_vintages_rows']:.1%}).
- Numerically unchanged: {overlap['equal_rows']}.
- Reported only in WMD2018: {overlap['row_status_counts'].get('only_wmd2018_reported', 0)}; only in WMD2021: {overlap['row_status_counts'].get('only_wmd2021_reported', 0)}.

## Country-share bridge audits

- WMD2021 versus WMD2025 at 2019: {bridge_2019['commodity_year_vectors']} commodity vectors; mean TV distance {bridge_2019['mean_total_variation_distance']:.4f}, median {bridge_2019['median_total_variation_distance']:.4f}, maximum {bridge_2019['maximum_total_variation_distance']:.4f}.
- WMD2025 versus WMD2026 at 2020-2023: {bridge_later['commodity_year_vectors']} commodity-year vectors; mean TV distance {bridge_later['mean_total_variation_distance']:.4f}, median {bridge_later['median_total_variation_distance']:.4f}, maximum {bridge_later['maximum_total_variation_distance']:.4f}.
- TV distance is half the L1 distance between country-share vectors: 0 means identical shares and 1 means disjoint distributions. Country absence is assigned share zero only inside the comparison vector, never in a production seed.

## Merge and missing-value rules

WMD2018 supplies 2012-2014 and WMD2021 supplies 2015-2019. A blank PDF cell emits no value; it is never shifted to an adjacent year or imputed as zero. Printed zero is retained explicitly. Values remain in their reported units.

## Comparability cautions

- Historical values are revised between editions, so the overlap comparison must accompany any longitudinal use.
- Boron, Rare Earths, and Zircon use different raw titles across editions; canonical labels retain a stable key while preserving each raw heading.
- Vanadium changes its printed basis title from V2O5 content to V content. The overlap magnitudes do not show a stoichiometric jump, but the parser performs no speculative conversion and retains both basis labels.
- Palladium, Platinum, Rhodium, and Silver are reported in kg; most other target commodities are in metric tonnes. Cross-mineral level aggregation is invalid without an explicit scientific conversion and weighting scheme.

## Official source URLs

- WMD2018: {SOURCE_URLS['WMD2018']}
- WMD2021: {SOURCE_URLS['WMD2021']}
- WMD2025: {SOURCE_URLS['WMD2025']}
"""
    (output / "wmd_parse_audit.md").write_text(markdown, encoding="utf-8")
    return audit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wmd2018-coordinates", type=Path, required=True)
    parser.add_argument("--wmd2021-coordinates", type=Path, required=True)
    parser.add_argument("--wmd2025-coordinates", type=Path, required=True)
    parser.add_argument("--wmd2018-pdf", type=Path)
    parser.add_argument("--wmd2021-pdf", type=Path)
    parser.add_argument("--wmd2025-pdf", type=Path)
    parser.add_argument("--wmd2026-seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps(result["merged_outputs"], ensure_ascii=False, indent=2))
