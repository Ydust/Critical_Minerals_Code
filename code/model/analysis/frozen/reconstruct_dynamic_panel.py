"""
Create a conservative annual OD reconstruction panel for critical minerals.

This is a first modeling layer inspired by the migration-flow paper's core
principle: keep origin-destination-year flows, combine multiple source reports,
and carry uncertainty into the reconstructed annual panel.

The model deliberately avoids inventing new corridors. It only fills short
internal gaps for corridors that are observed before and after the missing year.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
BASELINE = OUT / "model_flows_baseline.csv.gz"
MAX_INTERNAL_MISSING_YEARS = 2
Z90 = 1.6448536269514722

KEY = ["CmdCode", "metal", "stage", "hs_label", "exporter_iso", "importer_iso"]
SORT = ["CmdCode", "exporter_iso", "importer_iso", "year"]

TIER_SIGMA = {
    "A_mirror_close": 0.15,
    "B_mirror_wide": 0.45,
    "C_single_side": 0.75,
    "D_low_information": 1.10,
}


def safe_log1p(value: float) -> float:
    if value is None or pd.isna(value) or value <= 0:
        return 0.0
    return math.log1p(float(value))


def safe_expm1(value: float) -> float:
    return max(math.expm1(float(value)), 0.0)


def prepare_observed() -> pd.DataFrame:
    if not BASELINE.exists():
        raise FileNotFoundError(f"Run build_model_inputs.py first: {BASELINE}")
    df = pd.read_csv(BASELINE, dtype={"CmdCode": str})
    df["year"] = df["RefYear"].astype(int)
    df = df[df["reconciled_value_usd"].fillna(0) > 0].copy()

    center = np.log1p(df["reconciled_value_usd"].clip(lower=0))
    low = np.log1p(df["value_low_usd"].fillna(df["reconciled_value_usd"]).clip(lower=0))
    high = np.log1p(df["value_high_usd"].fillna(df["reconciled_value_usd"]).clip(lower=0))
    interval_sigma = np.maximum(center - low, high - center) / Z90
    tier_sigma = df["quality_tier"].map(TIER_SIGMA).fillna(1.10)
    df["observation_sigma_log"] = np.maximum(interval_sigma, tier_sigma)
    df["reconstruction_status"] = "observed"
    df["is_observed"] = True
    df["prev_observed_year"] = df["year"]
    df["next_observed_year"] = df["year"]
    df["interpolation_gap_years"] = 0
    df["interpolation_alpha"] = 0.0
    df["reconstructed_value_usd"] = df["reconciled_value_usd"]
    df["value_p05_usd"] = df["value_low_usd"]
    df["value_p95_usd"] = df["value_high_usd"]
    df["reconstructed_qty_kg"] = df["reconciled_qty_kg"]
    df["source_support"] = df["mirror_status"]
    return df


def interpolate_row(prev: pd.Series, nxt: pd.Series, year: int, gap: int) -> dict:
    previous_year = int(prev["year"])
    next_year = int(nxt["year"])
    if next_year <= previous_year:
        raise ValueError("Interpolation endpoints must be strictly ordered by year.")
    if not previous_year < year < next_year:
        raise ValueError("Interpolation year must lie strictly between its endpoints.")
    expected_gap = next_year - previous_year - 1
    if int(gap) != expected_gap:
        raise ValueError(
            f"Interpolation gap {gap} does not match endpoint gap {expected_gap}."
        )
    alpha = (year - previous_year) / (next_year - previous_year)
    prev_log = safe_log1p(prev["reconciled_value_usd"])
    next_log = safe_log1p(nxt["reconciled_value_usd"])
    log_value = (1.0 - alpha) * prev_log + alpha * next_log

    prev_sigma = float(prev.get("observation_sigma_log", 1.0))
    next_sigma = float(nxt.get("observation_sigma_log", 1.0))
    sigma = (1.0 - alpha) * prev_sigma + alpha * next_sigma
    sigma += 0.20 * gap

    prev_qty = prev.get("reconciled_qty_kg")
    next_qty = nxt.get("reconciled_qty_kg")
    if pd.notna(prev_qty) and pd.notna(next_qty) and prev_qty > 0 and next_qty > 0:
        qty = safe_expm1((1.0 - alpha) * safe_log1p(prev_qty) + alpha * safe_log1p(next_qty))
    else:
        qty = np.nan

    rec = {col: prev[col] for col in KEY}
    rec.update(
        {
            "RefYear": year,
            "year": year,
            "reconciled_value_usd": np.nan,
            "preferred_value_usd": np.nan,
            "value_low_usd": np.nan,
            "value_high_usd": np.nan,
            "reconciled_qty_kg": np.nan,
            "preferred_qty_kg": np.nan,
            "mirror_status": "interpolated_short_gap",
            "mirror_rel_gap_value": np.nan,
            "mirror_abs_log_gap_value": np.nan,
            "data_quality_weight": min(
                float(prev.get("data_quality_weight", 0.0)),
                float(nxt.get("data_quality_weight", 0.0)),
            )
            * 0.40
            / (1.0 + 0.25 * gap),
            "quality_tier": "I_interpolated",
            "observation_sigma_log": sigma,
            "reconstruction_status": "interpolated_short_gap",
            "is_observed": False,
            "prev_observed_year": int(prev["year"]),
            "next_observed_year": int(nxt["year"]),
            "interpolation_gap_years": gap,
            "interpolation_alpha": alpha,
            "reconstructed_value_usd": safe_expm1(log_value),
            "value_p05_usd": safe_expm1(log_value - Z90 * sigma),
            "value_p95_usd": safe_expm1(log_value + Z90 * sigma),
            "reconstructed_qty_kg": qty,
            "source_support": "temporal_neighbors",
        }
    )
    return rec


def reconstruct_panel(observed: pd.DataFrame) -> pd.DataFrame:
    observed_cols = [
        *KEY,
        "RefYear",
        "year",
        "reconciled_value_usd",
        "preferred_value_usd",
        "value_low_usd",
        "value_high_usd",
        "reconciled_qty_kg",
        "preferred_qty_kg",
        "mirror_status",
        "mirror_rel_gap_value",
        "mirror_abs_log_gap_value",
        "data_quality_weight",
        "quality_tier",
        "observation_sigma_log",
        "reconstruction_status",
        "is_observed",
        "prev_observed_year",
        "next_observed_year",
        "interpolation_gap_years",
        "interpolation_alpha",
        "reconstructed_value_usd",
        "value_p05_usd",
        "value_p95_usd",
        "reconstructed_qty_kg",
        "source_support",
    ]

    ordered = observed.sort_values([*KEY, "year"]).drop_duplicates([*KEY, "year"], keep="first")
    observed_out = ordered[observed_cols].copy()
    grouped = ordered.groupby(KEY, sort=False, dropna=False)
    ordered = ordered.copy()
    ordered["_next_year"] = grouped["year"].shift(-1)
    for col in [
        "reconciled_value_usd",
        "reconciled_qty_kg",
        "data_quality_weight",
        "observation_sigma_log",
    ]:
        ordered[f"_next_{col}"] = grouped[col].shift(-1)
    ordered["_gap"] = ordered["_next_year"] - ordered["year"] - 1

    candidates = ordered[
        (ordered["_gap"] >= 1) & (ordered["_gap"] <= MAX_INTERNAL_MISSING_YEARS)
    ].copy()
    imputed_frames: list[pd.DataFrame] = []
    for offset in range(1, MAX_INTERNAL_MISSING_YEARS + 1):
        sub = candidates[candidates["_gap"] >= offset].copy()
        if sub.empty:
            continue
        alpha = offset / (sub["_next_year"] - sub["year"])
        gap = sub["_gap"].astype(float)
        prev_log = np.log1p(sub["reconciled_value_usd"].clip(lower=0))
        next_log = np.log1p(sub["_next_reconciled_value_usd"].clip(lower=0))
        log_value = (1.0 - alpha) * prev_log + alpha * next_log
        sigma = (
            (1.0 - alpha) * sub["observation_sigma_log"].fillna(1.0)
            + alpha * sub["_next_observation_sigma_log"].fillna(1.0)
            + 0.20 * gap
        )

        prev_qty = sub["reconciled_qty_kg"]
        next_qty = sub["_next_reconciled_qty_kg"]
        qty_mask = prev_qty.notna() & next_qty.notna() & (prev_qty > 0) & (next_qty > 0)
        qty = np.where(
            qty_mask,
            np.expm1(
                (1.0 - alpha) * np.log1p(prev_qty.where(qty_mask, 0))
                + alpha * np.log1p(next_qty.where(qty_mask, 0))
            ),
            np.nan,
        )

        imp = sub[KEY].copy()
        imp["RefYear"] = (sub["year"] + offset).astype(int)
        imp["year"] = imp["RefYear"]
        imp["reconciled_value_usd"] = np.nan
        imp["preferred_value_usd"] = np.nan
        imp["value_low_usd"] = np.nan
        imp["value_high_usd"] = np.nan
        imp["reconciled_qty_kg"] = np.nan
        imp["preferred_qty_kg"] = np.nan
        imp["mirror_status"] = "interpolated_short_gap"
        imp["mirror_rel_gap_value"] = np.nan
        imp["mirror_abs_log_gap_value"] = np.nan
        imp["data_quality_weight"] = (
            np.minimum(
                sub["data_quality_weight"].fillna(0),
                sub["_next_data_quality_weight"].fillna(0),
            )
            * 0.40
            / (1.0 + 0.25 * gap)
        )
        imp["quality_tier"] = "I_interpolated"
        imp["observation_sigma_log"] = sigma
        imp["reconstruction_status"] = "interpolated_short_gap"
        imp["is_observed"] = False
        imp["prev_observed_year"] = sub["year"].astype(int)
        imp["next_observed_year"] = sub["_next_year"].astype(int)
        imp["interpolation_gap_years"] = gap.astype(int)
        imp["interpolation_alpha"] = alpha
        imp["reconstructed_value_usd"] = np.maximum(np.expm1(log_value), 0.0)
        imp["value_p05_usd"] = np.maximum(np.expm1(log_value - Z90 * sigma), 0.0)
        imp["value_p95_usd"] = np.maximum(np.expm1(log_value + Z90 * sigma), 0.0)
        imp["reconstructed_qty_kg"] = qty
        imp["source_support"] = "temporal_neighbors"
        imputed_frames.append(imp[observed_cols])

    frames = [observed_out, *imputed_frames]
    panel = pd.concat(frames, ignore_index=True)
    panel["year"] = panel["year"].astype(int)
    panel["RefYear"] = panel["year"]
    panel["is_observed"] = panel["is_observed"].astype(bool)
    return panel.sort_values([*KEY, "year"]).reset_index(drop=True)


def add_sequence_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["log_reconstructed_value"] = np.log1p(panel["reconstructed_value_usd"].clip(lower=0))
    panel["hs_exporter_total_usd"] = panel.groupby(["CmdCode", "year", "exporter_iso"])[
        "reconstructed_value_usd"
    ].transform("sum")
    panel["hs_importer_total_usd"] = panel.groupby(["CmdCode", "year", "importer_iso"])[
        "reconstructed_value_usd"
    ].transform("sum")
    panel["hs_world_total_usd"] = panel.groupby(["CmdCode", "year"])[
        "reconstructed_value_usd"
    ].transform("sum")
    panel["share_of_exporter_hs"] = np.where(
        panel["hs_exporter_total_usd"] > 0,
        panel["reconstructed_value_usd"] / panel["hs_exporter_total_usd"],
        np.nan,
    )
    panel["share_of_importer_hs"] = np.where(
        panel["hs_importer_total_usd"] > 0,
        panel["reconstructed_value_usd"] / panel["hs_importer_total_usd"],
        np.nan,
    )
    panel["share_of_world_hs"] = np.where(
        panel["hs_world_total_usd"] > 0,
        panel["reconstructed_value_usd"] / panel["hs_world_total_usd"],
        np.nan,
    )

    corridor = ["CmdCode", "exporter_iso", "importer_iso"]
    panel = panel.sort_values([*corridor, "year"]).reset_index(drop=True)
    prev_year = panel.groupby(corridor, dropna=False)["year"].shift(1)
    prev2_year = panel.groupby(corridor, dropna=False)["year"].shift(2)
    lag1 = panel.groupby(corridor, dropna=False)["log_reconstructed_value"].shift(1)
    lag2 = panel.groupby(corridor, dropna=False)["log_reconstructed_value"].shift(2)
    panel["lag1_log_value"] = np.where(panel["year"] - prev_year == 1, lag1, np.nan)
    panel["lag2_log_value"] = np.where(
        panel["year"] - prev2_year == 2,
        lag2,
        np.nan,
    )
    panel["delta_log_value"] = panel["log_reconstructed_value"] - panel["lag1_log_value"]
    return panel.sort_values(SORT).reset_index(drop=True)


def write_summary(panel: pd.DataFrame) -> pd.DataFrame:
    summary = (
        panel.groupby(["metal", "stage", "year"], as_index=False)
        .agg(
            rows=("reconstructed_value_usd", "size"),
            observed_edges=("is_observed", "sum"),
            total_reconstructed_value_usd=("reconstructed_value_usd", "sum"),
            p05_value_usd=("value_p05_usd", "sum"),
            p95_value_usd=("value_p95_usd", "sum"),
            median_quality_weight=("data_quality_weight", "median"),
            exporters=("exporter_iso", "nunique"),
            importers=("importer_iso", "nunique"),
        )
        .sort_values(["metal", "stage", "year"])
    )
    summary["interpolated_edges"] = summary["rows"] - summary["observed_edges"]
    observed_value = (
        panel.loc[panel["is_observed"]]
        .groupby(["metal", "stage", "year"])["reconstructed_value_usd"]
        .sum()
        .rename("observed_value_usd")
    )
    interpolated_value = (
        panel.loc[~panel["is_observed"]]
        .groupby(["metal", "stage", "year"])["reconstructed_value_usd"]
        .sum()
        .rename("interpolated_value_usd")
    )
    summary = summary.merge(observed_value, on=["metal", "stage", "year"], how="left")
    summary = summary.merge(interpolated_value, on=["metal", "stage", "year"], how="left")
    summary["observed_value_usd"] = summary["observed_value_usd"].fillna(0.0)
    summary["interpolated_value_usd"] = summary["interpolated_value_usd"].fillna(0.0)
    summary["interpolated_value_share"] = np.where(
        summary["total_reconstructed_value_usd"] > 0,
        summary["interpolated_value_usd"] / summary["total_reconstructed_value_usd"],
        0.0,
    )
    return summary


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_float(value: float | int, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):,.{digits}f}"


def write_model_card(panel: pd.DataFrame, summary: pd.DataFrame) -> None:
    observed = int(panel["is_observed"].sum())
    interpolated = int((~panel["is_observed"]).sum())
    lines = [
        "# Dynamic reconstruction model card",
        "",
        "## Purpose",
        "Build a traceable annual exporter-importer panel for critical-mineral trade flows.",
        "The model is designed as a conservative first-stage reconstruction, not a forecast.",
        "",
        "## Source data",
        "- Primary observations: cached UN Comtrade annual HS bilateral import/export JSON.",
        "- Source manifest: `modeling/source_manifest.csv`.",
        "- Preprocessing audit: `modeling/outputs/DATA_AUDIT.md`.",
        "",
        "## Method",
        "- Harmonize raw import/export reports to exporter -> importer direction.",
        "- Reconcile mirror reports with geometric mean when both sides report positive values.",
        "- Preserve Comtrade reporting flags and mirror gaps as uncertainty inputs.",
        f"- Fill only short internal temporal gaps of at most {MAX_INTERNAL_MISSING_YEARS} missing year(s).",
        "- Interpolate values in log space between the nearest observed years for the same HS-exporter-importer corridor.",
        "- Produce 5%-95% uncertainty bands from mirror-disagreement tiers plus an interpolation penalty.",
        "- Do not create corridors outside their observed time window.",
        "",
        "## Panel size",
        f"- Observed edges: {fmt_int(observed)}",
        f"- Interpolated short-gap edges: {fmt_int(interpolated)}",
        f"- Total panel rows: {fmt_int(len(panel))}",
        f"- Years: {int(panel['year'].min())}-{int(panel['year'].max())}",
        f"- HS codes: {fmt_int(panel['CmdCode'].nunique())}",
        f"- Metals: {fmt_int(panel['metal'].nunique())}",
        "",
        "## Output files",
        "- `dynamic_reconstruction_panel.csv.gz`: observed plus conservative short-gap reconstructed flows.",
        "- `dynamic_reconstruction_summary_by_metal_stage_year.csv`: metal-stage-year totals and uncertainty.",
        "- `sequence_model_features.csv.gz`: lag, share, and exporter/importer-total features for later RNN/state-space models.",
        "",
        "## Caveats",
        "- Missing years outside a corridor's observed window remain missing.",
        "- Single-sided mirror observations are retained but carry wider uncertainty.",
        "- The 2025 panel should be treated cautiously if source-country reporting was incomplete when the cache was downloaded.",
    ]
    top = summary.sort_values("total_reconstructed_value_usd", ascending=False).head(15)
    lines.extend(
        [
            "",
            "## Highest-value reconstructed metal-stage-years",
            "",
            "| metal | stage | year | value_usd | interpolated_share | rows |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in top.iterrows():
        lines.append(
            f"| {row['metal']} | {row['stage']} | {int(row['year'])} | "
            f"{fmt_float(row['total_reconstructed_value_usd'], 0)} | "
            f"{fmt_float(row['interpolated_value_share'], 3)} | {fmt_int(row['rows'])} |"
        )
    lines.append("")
    (OUT / "DYNAMIC_MODEL_CARD.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    observed = prepare_observed()
    panel = reconstruct_panel(observed)
    panel = add_sequence_features(panel)
    summary = write_summary(panel)

    panel.to_csv(OUT / "dynamic_reconstruction_panel.csv.gz", index=False, compression="gzip")
    summary.to_csv(OUT / "dynamic_reconstruction_summary_by_metal_stage_year.csv", index=False)

    feature_cols = [
        "year",
        "CmdCode",
        "metal",
        "stage",
        "exporter_iso",
        "importer_iso",
        "log_reconstructed_value",
        "lag1_log_value",
        "lag2_log_value",
        "delta_log_value",
        "share_of_exporter_hs",
        "share_of_importer_hs",
        "share_of_world_hs",
        "hs_exporter_total_usd",
        "hs_importer_total_usd",
        "hs_world_total_usd",
        "data_quality_weight",
        "observation_sigma_log",
        "is_observed",
        "reconstruction_status",
    ]
    panel[feature_cols].to_csv(OUT / "sequence_model_features.csv.gz", index=False, compression="gzip")
    write_model_card(panel, summary)

    print(f"observed edges: {int(panel['is_observed'].sum()):,}")
    print(f"interpolated edges: {int((~panel['is_observed']).sum()):,}")
    print(f"total panel rows: {len(panel):,}")
    print(f"outputs: {OUT}")


if __name__ == "__main__":
    main()
