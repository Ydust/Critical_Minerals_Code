"""
Build auditable model inputs for critical-mineral trade reconstruction.

The script reads the cached UN Comtrade JSON files, preserves reporting-quality
flags that were not kept in all_minerals_long.csv, harmonizes import/export
records into exporter -> importer edges, and reconciles mirror reports into a
baseline edge panel with uncertainty proxies.
"""
from __future__ import annotations

import ast
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = ROOT / "_raw"
TRADE_RAW = RAW
OUT = HERE / "outputs"


def read_json(path: Path):
    # Comtrade reference downloads may include a UTF-8 BOM; utf-8-sig accepts
    # both BOM-prefixed and ordinary UTF-8 JSON without changing content.
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _reference_rows(payload: object) -> list[dict]:
    """Normalize cached Comtrade reference payloads to a row list."""
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("Comtrade reference payload must contain a list of row objects.")
    return payload


def _raw_cache_key(path: Path) -> tuple[str, int] | None:
    """Parse only canonical `{HS6}_{year}.json` trade-cache filenames."""
    if path.suffix.lower() != ".json":
        return None
    match = re.fullmatch(r"(\d{6})_(\d{4})", path.stem)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def load_mineral_map() -> dict[str, tuple[str, str, str]]:
    """Read the authoritative HS -> metal/stage mapping from download_minerals.py."""
    src = (ROOT / "download_minerals.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    minerals = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MINERALS":
                    minerals = ast.literal_eval(node.value)
                    break
        if minerals is not None:
            break
    if minerals is None:
        raise RuntimeError("Could not find MINERALS in download_minerals.py")
    return {
        str(hs): (metal, stage, label)
        for metal, entries in minerals.items()
        for hs, stage, label in entries
    }


def load_area_reference() -> tuple[dict[int, str], dict[int, str], dict[int, bool]]:
    """Load UN Comtrade reporter and partner reference tables from the local cache."""
    iso: dict[int, str] = {0: "W00"}
    name: dict[int, str] = {0: "World"}
    is_group: dict[int, bool] = {0: True}

    refs = [
        (TRADE_RAW / "_ref_Reporters.json", "reporterCodeIsoAlpha3", "reporterDesc"),
        (TRADE_RAW / "_ref_partnerAreas.json", "PartnerCodeIsoAlpha3", "PartnerDesc"),
    ]
    for path, iso_key, desc_key in refs:
        for row in _reference_rows(read_json(path)):
            code = row.get("id")
            if code is None:
                continue
            code = int(code)
            val = str(row.get(iso_key) or "").strip()
            if val:
                iso[code] = val
            name[code] = str(row.get(desc_key) or row.get("text") or "").strip()
            is_group[code] = bool(is_group.get(code, False)) or bool(
                row.get("isGroup", False)
            )
    return iso, name, is_group


def num(row: dict, key: str) -> float:
    val = row.get(key)
    if val in (None, ""):
        return float("nan")
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _is_kg_unit(value: object) -> bool:
    """Return whether a possibly malformed Comtrade unit code denotes kilograms."""
    try:
        return int(value) == 8
    except (TypeError, ValueError, OverflowError):
        return False


def preferred_qty_kg(row: dict) -> float:
    """Use net weight first; fall back to alternate quantity or quantity if unit is kg."""
    net = num(row, "netWgt")
    if net > 0:
        return net
    alt = num(row, "altQty")
    if alt > 0 and _is_kg_unit(row.get("altQtyUnitCode")):
        return alt
    qty = num(row, "qty")
    if qty > 0 and _is_kg_unit(row.get("qtyUnitCode")):
        return qty
    return float("nan")


def clean_iso(value: str | None) -> str:
    return str(value or "").strip()


def is_special_iso(value: str | None) -> bool:
    iso = clean_iso(value)
    if not iso or iso == "W00":
        return True
    if iso.startswith("_"):
        return True
    if len(iso) != 3:
        return True
    return False


def endpoint_is_special(
    exporter_iso: str | None,
    importer_iso: str | None,
    reporter_code: int,
    partner_code: int,
    is_group: dict[int, bool],
) -> bool:
    """Flag world, special-code, or geographic-group trade endpoints."""
    return (
        is_special_iso(exporter_iso)
        or is_special_iso(importer_iso)
        or bool(is_group.get(reporter_code, False))
        or bool(is_group.get(partner_code, False))
    )


def read_raw_observations() -> tuple[pd.DataFrame, pd.DataFrame]:
    code_meta = load_mineral_map()
    iso, name, group = load_area_reference()
    observations: list[dict] = []
    file_stats: list[dict] = []

    for path in sorted(RAW.glob("*.json")):
        cache_key = _raw_cache_key(path)
        if cache_key is None:
            continue
        code_from_file, year_from_file = cache_key
        rows = read_json(path)
        stats = {
            "source_file": path.name,
            "CmdCode": code_from_file,
            "RefYear": year_from_file,
            "raw_rows": len(rows),
            "world_partner_rows": 0,
            "bilateral_rows": 0,
            "import_rows": 0,
            "export_rows": 0,
            "reported_rows": 0,
            "aggregate_rows": 0,
            "qty_estimated_rows": 0,
            "net_wgt_estimated_rows": 0,
        }
        for row in rows:
            flow = row.get("flowCode")
            partner_code = int(row.get("partnerCode") or 0)
            reporter_code = int(row.get("reporterCode") or 0)
            stats["world_partner_rows"] += int(partner_code == 0)
            stats["bilateral_rows"] += int(partner_code != 0)
            stats["import_rows"] += int(flow == "M")
            stats["export_rows"] += int(flow == "X")
            stats["reported_rows"] += int(bool(row.get("isReported", False)))
            stats["aggregate_rows"] += int(bool(row.get("isAggregate", False)))
            stats["qty_estimated_rows"] += int(bool(row.get("isQtyEstimated", False)))
            stats["net_wgt_estimated_rows"] += int(bool(row.get("isNetWgtEstimated", False)))

            if flow not in {"M", "X"} or partner_code == 0:
                continue
            code = str(row.get("cmdCode") or code_from_file)
            metal, stage, hs_label = code_meta.get(code, ("UNKNOWN", "UNKNOWN", ""))
            reporter_iso = clean_iso(iso.get(reporter_code))
            partner_iso = clean_iso(iso.get(partner_code))
            if flow == "X":
                exporter_iso, importer_iso, reporter_role = reporter_iso, partner_iso, "exporter"
            else:
                exporter_iso, importer_iso, reporter_role = partner_iso, reporter_iso, "importer"

            observations.append(
                {
                    "RefYear": int(row.get("refYear") or year_from_file),
                    "CmdCode": code,
                    "metal": metal,
                    "stage": stage,
                    "hs_label": hs_label,
                    "FlowCode": flow,
                    "reporter_role": reporter_role,
                    "ReporterCode": reporter_code,
                    "ReporterISO": reporter_iso,
                    "ReporterDesc": name.get(reporter_code, ""),
                    "PartnerCode": partner_code,
                    "PartnerISO": partner_iso,
                    "PartnerDesc": name.get(partner_code, ""),
                    "exporter_iso": exporter_iso,
                    "importer_iso": importer_iso,
                    "value_usd": num(row, "primaryValue"),
                    "fob_value_usd": num(row, "fobvalue"),
                    "cif_value_usd": num(row, "cifvalue"),
                    "qty_kg": preferred_qty_kg(row),
                    "qty_unit_code": row.get("qtyUnitCode"),
                    "source_is_reported": bool(row.get("isReported", False)),
                    "source_is_aggregate": bool(row.get("isAggregate", False)),
                    "qty_is_estimated": bool(row.get("isQtyEstimated", False)),
                    "net_wgt_is_estimated": bool(row.get("isNetWgtEstimated", False)),
                    "legacy_estimation_flag": row.get("legacyEstimationFlag"),
                    "classification_code": row.get("classificationCode"),
                    "is_original_classification": bool(row.get("isOriginalClassification", False)),
                    "reporter_is_group": bool(group.get(reporter_code, False)),
                    "partner_is_group": bool(group.get(partner_code, False)),
                    "endpoint_is_special": endpoint_is_special(
                        exporter_iso,
                        importer_iso,
                        reporter_code,
                        partner_code,
                        group,
                    ),
                    "source_file": path.name,
                }
            )
        file_stats.append(stats)

    obs = pd.DataFrame(observations)
    stats_df = pd.DataFrame(file_stats)
    if obs.empty:
        raise RuntimeError("No bilateral import/export observations found.")
    return obs, stats_df


def role_aggregate(obs: pd.DataFrame) -> pd.DataFrame:
    idx = ["RefYear", "CmdCode", "metal", "stage", "hs_label", "exporter_iso", "importer_iso", "reporter_role"]
    return (
        obs.groupby(idx, dropna=False)
        .agg(
            value_usd=("value_usd", "sum"),
            qty_kg=("qty_kg", "sum"),
            n_rows=("value_usd", "size"),
            n_source_reported=("source_is_reported", "sum"),
            n_source_aggregate=("source_is_aggregate", "sum"),
            n_qty_estimated=("qty_is_estimated", "sum"),
            n_net_wgt_estimated=("net_wgt_is_estimated", "sum"),
            endpoint_is_special=("endpoint_is_special", "max"),
        )
        .reset_index()
    )


def reconcile_mirrors(obs: pd.DataFrame) -> pd.DataFrame:
    idx = ["RefYear", "CmdCode", "metal", "stage", "hs_label", "exporter_iso", "importer_iso"]
    agg = role_aggregate(obs)
    exp = agg[agg["reporter_role"] == "exporter"].drop(columns=["reporter_role"])
    imp = agg[agg["reporter_role"] == "importer"].drop(columns=["reporter_role"])
    mirror = exp.merge(imp, on=idx, how="outer", suffixes=("_exporter_reported", "_importer_reported"))

    for role in ["exporter_reported", "importer_reported"]:
        for col in ["value_usd", "qty_kg", "n_rows", "n_source_reported", "n_source_aggregate",
                    "n_qty_estimated", "n_net_wgt_estimated"]:
            full = f"{col}_{role}"
            if full not in mirror:
                mirror[full] = np.nan
        full = f"endpoint_is_special_{role}"
        if full not in mirror:
            mirror[full] = False

    ev = mirror["value_usd_exporter_reported"].fillna(0.0).clip(lower=0.0)
    iv = mirror["value_usd_importer_reported"].fillna(0.0).clip(lower=0.0)
    eq = mirror["qty_kg_exporter_reported"].fillna(0.0).clip(lower=0.0)
    iq = mirror["qty_kg_importer_reported"].fillna(0.0).clip(lower=0.0)
    both_value = (ev > 0) & (iv > 0)
    both_qty = (eq > 0) & (iq > 0)

    mirror["has_exporter_report"] = ev > 0
    mirror["has_importer_report"] = iv > 0
    mirror["mirror_status"] = np.select(
        [both_value, ev > 0, iv > 0],
        ["both", "exporter_only", "importer_only"],
        default="no_positive_value",
    )
    mirror["preferred_value_usd"] = np.where(ev > 0, ev, iv)
    mirror["reconciled_value_usd"] = np.where(both_value, np.sqrt(ev * iv), mirror["preferred_value_usd"])
    mirror["value_low_usd"] = np.where(both_value, np.minimum(ev, iv), mirror["preferred_value_usd"])
    mirror["value_high_usd"] = np.where(both_value, np.maximum(ev, iv), mirror["preferred_value_usd"])
    mirror["preferred_qty_kg"] = np.where(eq > 0, eq, iq)
    mirror["reconciled_qty_kg"] = np.where(both_qty, np.sqrt(eq * iq), mirror["preferred_qty_kg"])
    mirror["mirror_abs_log_gap_value"] = np.where(both_value, np.abs(np.log1p(ev) - np.log1p(iv)), np.nan)
    mirror["mirror_rel_gap_value"] = np.where(
        both_value,
        np.abs(ev - iv) / np.maximum((ev + iv) / 2.0, 1.0),
        np.nan,
    )
    mirror["mirror_abs_log_gap_qty"] = np.where(both_qty, np.abs(np.log1p(eq) - np.log1p(iq)), np.nan)
    mirror["endpoint_is_special"] = (
        mirror["endpoint_is_special_exporter_reported"].fillna(False)
        | mirror["endpoint_is_special_importer_reported"].fillna(False)
        | mirror["exporter_iso"].map(is_special_iso)
        | mirror["importer_iso"].map(is_special_iso)
    )
    mirror["is_self_loop"] = mirror["exporter_iso"] == mirror["importer_iso"]
    any_qty_est = (
        mirror["n_qty_estimated_exporter_reported"].fillna(0)
        + mirror["n_qty_estimated_importer_reported"].fillna(0)
        + mirror["n_net_wgt_estimated_exporter_reported"].fillna(0)
        + mirror["n_net_wgt_estimated_importer_reported"].fillna(0)
    ) > 0
    base_weight = np.where(both_value, 1.0 / (1.0 + mirror["mirror_abs_log_gap_value"].fillna(0.0)), 0.45)
    base_weight = np.where(mirror["mirror_status"] == "no_positive_value", 0.0, base_weight)
    mirror["data_quality_weight"] = np.where(any_qty_est, base_weight * 0.85, base_weight)
    mirror["quality_tier"] = np.select(
        [
            (mirror["mirror_status"] == "both") & (mirror["mirror_rel_gap_value"].fillna(9) <= 0.25),
            (mirror["mirror_status"] == "both") & (mirror["mirror_rel_gap_value"].fillna(9) <= 0.75),
            mirror["mirror_status"].isin(["exporter_only", "importer_only"]),
        ],
        ["A_mirror_close", "B_mirror_wide", "C_single_side"],
        default="D_low_information",
    )
    mirror["model_eligible"] = (
        (mirror["reconciled_value_usd"] > 0)
        & (~mirror["endpoint_is_special"])
        & (~mirror["is_self_loop"])
    )
    return mirror.sort_values(idx).reset_index(drop=True)


def write_outputs(obs: pd.DataFrame, stats: pd.DataFrame, mirror: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT / "observations_raw_flags.csv.gz", index=False, compression="gzip")
    obs.head(10000).to_csv(OUT / "observations_raw_flags_sample.csv", index=False)
    mirror.to_csv(OUT / "mirror_reconciled_flows.csv.gz", index=False, compression="gzip")

    model_cols = [
        "RefYear", "CmdCode", "metal", "stage", "hs_label", "exporter_iso", "importer_iso",
        "reconciled_value_usd", "preferred_value_usd", "value_low_usd", "value_high_usd",
        "reconciled_qty_kg", "preferred_qty_kg", "mirror_status", "mirror_rel_gap_value",
        "mirror_abs_log_gap_value", "data_quality_weight", "quality_tier",
    ]
    model = mirror.loc[mirror["model_eligible"], model_cols].copy()
    model.to_csv(OUT / "model_flows_baseline.csv.gz", index=False, compression="gzip")

    stats.to_csv(OUT / "source_audit_by_file.csv", index=False)
    code_year = (
        stats.groupby(["CmdCode", "RefYear"], as_index=False)
        .sum(numeric_only=True)
        .sort_values(["CmdCode", "RefYear"])
    )
    code_year.to_csv(OUT / "source_audit_by_code_year.csv", index=False)

    mirror_for_quality = mirror.copy()
    mirror_for_quality["model_eligible_value_usd"] = np.where(
        mirror_for_quality["model_eligible"],
        mirror_for_quality["reconciled_value_usd"],
        0.0,
    )
    quality = (
        mirror_for_quality.groupby(["metal", "stage", "RefYear"], as_index=False)
        .agg(
            edges=("reconciled_value_usd", "size"),
            model_eligible_edges=("model_eligible", "sum"),
            both_reports=("mirror_status", lambda s: int((s == "both").sum())),
            exporter_only=("mirror_status", lambda s: int((s == "exporter_only").sum())),
            importer_only=("mirror_status", lambda s: int((s == "importer_only").sum())),
            total_mirror_value_usd=("reconciled_value_usd", "sum"),
            model_eligible_value_usd=("model_eligible_value_usd", "sum"),
            median_mirror_rel_gap_value=("mirror_rel_gap_value", "median"),
            median_quality_weight=("data_quality_weight", "median"),
        )
        .sort_values(["metal", "stage", "RefYear"])
    )
    quality.to_csv(OUT / "mirror_quality_by_metal_stage_year.csv", index=False)

    countries = pd.DataFrame(
        {"iso": sorted(set(model["exporter_iso"]).union(set(model["importer_iso"])))}
    )
    countries.to_csv(OUT / "model_country_universe.csv", index=False)

    write_audit_markdown(obs, stats, mirror, model, quality)


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_float(value: float | int, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"


def write_audit_markdown(
    obs: pd.DataFrame,
    stats: pd.DataFrame,
    mirror: pd.DataFrame,
    model: pd.DataFrame,
    quality: pd.DataFrame,
) -> None:
    status_counts = mirror["mirror_status"].value_counts(dropna=False)
    tier_counts = mirror["quality_tier"].value_counts(dropna=False)
    years = f"{int(obs['RefYear'].min())}-{int(obs['RefYear'].max())}"
    lines = [
        "# Critical-mineral modeling data audit",
        "",
        "Generated by `modeling/build_model_inputs.py` from cached UN Comtrade JSON.",
        "",
        "## Scope",
        f"- Raw cache files read: {fmt_int(stats['source_file'].nunique())}",
        f"- Raw JSON rows: {fmt_int(stats['raw_rows'].sum())}",
        f"- Bilateral import/export observations retained: {fmt_int(len(obs))}",
        f"- Mirror-reconciled exporter->importer edges: {fmt_int(len(mirror))}",
        f"- Model-eligible positive country-country edges: {fmt_int(len(model))}",
        f"- HS codes: {fmt_int(obs['CmdCode'].nunique())}",
        f"- Metals: {fmt_int(obs['metal'].nunique())}",
        f"- Years: {years}",
        "",
        "## Mirror status",
    ]
    for key, value in status_counts.items():
        lines.append(f"- {key}: {fmt_int(value)}")
    lines.extend(["", "## Quality tiers"])
    for key, value in tier_counts.items():
        lines.append(f"- {key}: {fmt_int(value)}")
    lines.extend(
        [
            "",
            "## Baseline reconciliation rule",
            "- Direction is normalized to exporter -> importer.",
            "- Export reports use `FlowCode == X`; import mirror reports use `FlowCode == M`.",
            "- If both sides report a positive value, `reconciled_value_usd` is the geometric mean.",
            "- If only one side reports a positive value, that side is used and marked `C_single_side`.",
            "- `data_quality_weight` is lower for large mirror gaps and for quantity/net-weight estimated rows.",
            "- Special endpoints, World aggregates, and self-loops are retained in the raw/mirror table but excluded from `model_flows_baseline.csv.gz`.",
            "",
            "## Main output files",
            "- `observations_raw_flags.csv.gz`: normalized raw observations with Comtrade flags.",
            "- `mirror_reconciled_flows.csv.gz`: exporter/importer mirror reconciliation.",
            "- `model_flows_baseline.csv.gz`: positive country-country edges for first-stage models.",
            "- `source_audit_by_code_year.csv`: raw file coverage and reporting flags.",
            "- `mirror_quality_by_metal_stage_year.csv`: mirror coverage and uncertainty summary.",
            "",
            "## Highest-value metal-stage-years in the model-eligible baseline panel",
            "",
            "| metal | stage | year | value_usd | eligible_edges | median_gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    top = quality.sort_values("model_eligible_value_usd", ascending=False).head(20)
    for _, row in top.iterrows():
        lines.append(
            f"| {row['metal']} | {row['stage']} | {int(row['RefYear'])} | "
            f"{fmt_float(row['model_eligible_value_usd'], 0)} | "
            f"{fmt_int(row['model_eligible_edges'])} | "
            f"{fmt_float(row['median_mirror_rel_gap_value'], 3)} |"
        )
    lines.append("")
    (OUT / "DATA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    obs, stats = read_raw_observations()
    mirror = reconcile_mirrors(obs)
    write_outputs(obs, stats, mirror)
    print(f"observations: {len(obs):,}")
    print(f"mirror edges: {len(mirror):,}")
    print(f"model edges: {int(mirror['model_eligible'].sum()):,}")
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main()
