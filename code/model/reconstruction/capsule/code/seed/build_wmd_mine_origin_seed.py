from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external_sources"
OUT = ROOT / "modeling" / "outputs"

WMD_64 = EXTERNAL / "wmd_2026_ch6_4_country_by_mineral.xlsx"
WMD_65 = EXTERNAL / "wmd_2026_ch6_5_country_share_2024.xlsx"
WMD_63C = EXTERNAL / "wmd_2026_ch6_3c_political_stability.xlsx"

WMD_64_URL = (
    "https://www.bmf.gv.at/dam/jcr:ed3d811d-6edd-4bf9-b564-40cacaedac94/"
    "6.4.%20Production_of_Mineral_Raw_Materials_of_individual_Countries_by_Minerals.xlsx"
)
WMD_65_URL = (
    "https://www.bmf.gv.at/dam/jcr:829ef51d-a568-434f-b9df-4e53dfe1c620/"
    "6.5.%20Share_of_World_Mineral_Production_2024_by_Countries.xlsx"
)
WMD_63C_URL = (
    "https://www.bmf.gv.at/dam/jcr:14bc053e-88c6-4d3e-bdac-5d6f8df27580/"
    "6.3c.%20Political_stability.xlsx"
)


FLAG_CONFIDENCE = {
    "r": 0.90,  # reported
    "p": 0.75,  # provisional
    "e": 0.55,  # estimated
}


def clean_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_sheet_with_header(path: Path, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    header_idx = None
    for idx, row in raw.iterrows():
        values = {clean_name(v).lower() for v in row.tolist()}
        if "country" in values or "political stability" in values or "rank 2024" in values:
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not find header row in {path.name}:{sheet_name}")
    header = [clean_name(v) for v in raw.iloc[header_idx].tolist()]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = header
    return df.dropna(how="all")


def normalize_year_columns(df: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:
    year_cols: list[str] = []
    for col in df.columns:
        if isinstance(col, (int, float)) and not pd.isna(col):
            year_cols.append(col)
            continue
        text = clean_name(col)
        if re.fullmatch(r"20\d{2}(\.0)?", text):
            year_cols.append(col)
    long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="year", value_name="production")
    long["year"] = long["year"].astype(float).astype(int)
    long["production"] = pd.to_numeric(long["production"], errors="coerce")
    return long.dropna(subset=["production"])


def build_mine_origin_seed() -> pd.DataFrame:
    rows = []
    xl = pd.ExcelFile(WMD_64)
    for sheet in xl.sheet_names:
        df = read_sheet_with_header(WMD_64, sheet)
        df = df.rename(columns={c: clean_name(c).lower().replace(" ", "_") for c in df.columns})
        if not {"country", "unit", "data_source"}.issubset(df.columns):
            continue
        df["mine_origin_country_wmd"] = df["country"].map(clean_name)
        df = df[~df["mine_origin_country_wmd"].str.lower().eq("total")].copy()
        df["unit"] = df["unit"].map(clean_name)
        df["source_flag"] = df["data_source"].map(lambda x: clean_name(x).lower())
        long = normalize_year_columns(
            df,
            ["mine_origin_country_wmd", "unit", "source_flag"],
        )
        long["commodity_wmd"] = sheet
        long["commodity_label"] = re.sub(r"\s*\([^)]*\)", "", sheet).strip()
        long["source_confidence"] = long["source_flag"].map(FLAG_CONFIDENCE).fillna(0.50)
        long["source_dataset"] = "World Mining Data 2026 chapter 6.4"
        long["source_url"] = WMD_64_URL
        rows.append(long)
    out = pd.concat(rows, ignore_index=True)
    out = out[
        [
            "commodity_wmd",
            "commodity_label",
            "mine_origin_country_wmd",
            "year",
            "production",
            "unit",
            "source_flag",
            "source_confidence",
            "source_dataset",
            "source_url",
        ]
    ]
    return out.sort_values(["commodity_label", "mine_origin_country_wmd", "year"])


def build_country_share_2024() -> pd.DataFrame:
    rows = []
    xl = pd.ExcelFile(WMD_65)
    for sheet in xl.sheet_names:
        df = read_sheet_with_header(WMD_65, sheet)
        df = df.rename(columns={c: clean_name(c).lower().replace(" ", "_").replace(".", "") for c in df.columns})
        needed = {"rank_2024", "country", "unit", "production_2024", "share_in_%", "share_hhi"}
        if not needed.issubset(df.columns):
            continue
        keep = df[list(needed)].copy()
        keep["commodity_wmd"] = sheet
        keep["commodity_label"] = re.sub(r"\s*\([^)]*\)", "", sheet).strip()
        keep["rank_2024"] = pd.to_numeric(keep["rank_2024"], errors="coerce")
        keep["production_2024"] = pd.to_numeric(keep["production_2024"], errors="coerce")
        keep["share_in_percent"] = pd.to_numeric(keep["share_in_%"], errors="coerce")
        keep["share_hhi_component"] = pd.to_numeric(keep["share_hhi"], errors="coerce")
        keep["country_wmd"] = keep["country"].map(clean_name)
        keep["unit"] = keep["unit"].map(clean_name)
        keep["source_dataset"] = "World Mining Data 2026 chapter 6.5"
        keep["source_url"] = WMD_65_URL
        rows.append(keep)
    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["rank_2024", "country_wmd"])
    out = out[
        [
            "commodity_wmd",
            "commodity_label",
            "rank_2024",
            "country_wmd",
            "production_2024",
            "unit",
            "share_in_percent",
            "share_hhi_component",
            "source_dataset",
            "source_url",
        ]
    ]
    return out.sort_values(["commodity_label", "rank_2024"])


def build_political_stability() -> pd.DataFrame:
    rows = []
    xl = pd.ExcelFile(WMD_63C)
    for sheet in xl.sheet_names:
        df = read_sheet_with_header(WMD_63C, sheet)
        df = df.rename(columns={c: clean_name(c).lower().replace(" ", "_") for c in df.columns})
        if not {"political_stability", "unit"}.issubset(df.columns):
            continue
        df["political_stability"] = df["political_stability"].map(clean_name)
        df["unit"] = df["unit"].map(clean_name)
        long = normalize_year_columns(df, ["political_stability", "unit"])
        long["commodity_wmd"] = sheet
        long["commodity_label"] = re.sub(r"\s*\([^)]*\)", "", sheet).strip()
        long["source_dataset"] = "World Mining Data 2026 chapter 6.3c"
        long["source_url"] = WMD_63C_URL
        rows.append(long)
    out = pd.concat(rows, ignore_index=True)
    out = out[
        [
            "commodity_wmd",
            "commodity_label",
            "political_stability",
            "year",
            "production",
            "unit",
            "source_dataset",
            "source_url",
        ]
    ]
    return out.sort_values(["commodity_label", "year", "political_stability"])


def write_summary(seed: pd.DataFrame, share: pd.DataFrame, stability: pd.DataFrame) -> None:
    coverage = (
        seed.groupby("commodity_label")
        .agg(
            countries=("mine_origin_country_wmd", "nunique"),
            years=("year", "nunique"),
            production_2024=("production", lambda s: s[seed.loc[s.index, "year"].eq(2024)].sum()),
            reported_share=("source_flag", lambda s: (s == "r").mean()),
            estimated_share=("source_flag", lambda s: (s == "e").mean()),
        )
        .reset_index()
        .sort_values("production_2024", ascending=False)
    )
    coverage.to_csv(OUT / "wmd_mine_origin_coverage_by_commodity.csv", index=False)

    top = (
        share.sort_values(["commodity_label", "rank_2024"])
        .groupby("commodity_label")
        .head(5)
        .copy()
    )
    top.to_csv(OUT / "wmd_top5_mine_origin_countries_2024.csv", index=False)

    stability_totals = stability[stability["political_stability"].str.lower().eq("total")]
    stability_totals.to_csv(OUT / "wmd_political_stability_totals.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in [WMD_64, WMD_65, WMD_63C]:
        if not path.exists():
            raise FileNotFoundError(f"Missing source workbook: {path}")

    seed = build_mine_origin_seed()
    share = build_country_share_2024()
    stability = build_political_stability()

    seed.to_csv(OUT / "wmd_mine_origin_seed.csv", index=False)
    share.to_csv(OUT / "wmd_country_share_2024.csv", index=False)
    stability.to_csv(OUT / "wmd_political_stability_by_commodity_year.csv", index=False)
    write_summary(seed, share, stability)

    print(f"mine_origin_seed rows: {len(seed):,}")
    print(f"commodities: {seed['commodity_label'].nunique():,}")
    print(f"countries: {seed['mine_origin_country_wmd'].nunique():,}")
    print(f"years: {seed['year'].min()}-{seed['year'].max()}")
    print(f"written: {OUT / 'wmd_mine_origin_seed.csv'}")


if __name__ == "__main__":
    main()
