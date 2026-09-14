"""Audit BGS 2008-2012 mine-production data against WMD 2016.

This is a quality gate for the proposed Fig. 5 historical extension.  It does
not rebuild Fig. 5.  BGS covers 2008-2012; WMD 2016 country-by-mineral tables
cover 2010-2014, so the cross-source overlap is 2010-2012.  The primary audit
target is each mineral-year country-share distribution because those shares,
not cross-mineral production tonnages, seed the mine-origin layer.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

try:
    import pdfplumber
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "pdfplumber is required. Run this script in the project Conda pvsim "
        "environment with the workspace-local PDF audit dependencies on PYTHONPATH."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "critical_minerals_manuscript_package_zh"
AUDIT_SOURCE_DIR = ROOT / "external_sources" / "fig5_history_audit_2008_2012"
OUTPUT_DIR = ROOT / "fig5_history_overlap_audit_2008_2012"
OUTPUT_PREFIX = ""

BGS_PDF = AUDIT_SOURCE_DIR / "BGS_WMP_2008_2012.pdf"
WMD_PDF = AUDIT_SOURCE_DIR / "WMD2016_archived_official.pdf"
WMD_COORDINATES = OUTPUT_DIR / f"{OUTPUT_PREFIX}WMD2016_coordinates.jsonl.gz"
WMD2018_SEED = (
    PACKAGE
    / "figure_source_tables"
    / "longitudinal_2012_2024"
    / "input_snapshot"
    / "wmd_2012_2019_mine_origin_seed.csv"
)


def output_path(name: str) -> Path:
    return OUTPUT_DIR / f"{OUTPUT_PREFIX}{name}"

BGS_SOURCE_URL = "https://nora.nerc.ac.uk/id/eprint/507092/1/WMP2008-2012.pdf"
WMD_ORIGINAL_URL = (
    "http://www.bmwfw.gv.at/EnergieUndBergbau/WeltBergbauDaten/Documents/"
    "WMD2016%28tw.barrierefrei%29.pdf"
)
WMD_ARCHIVE_URL = (
    "https://web.archive.org/web/20160610001916id_/" + WMD_ORIGINAL_URL
)

WMD_PARSER_DIR = (
    PACKAGE
    / "reproduction"
    / "project_snapshot"
    / "longitudinal_2012_2024"
    / "wmd_parser"
)
sys.path.insert(0, str(WMD_PARSER_DIR))
from wmd_parser import canonical_commodity, measurement_basis  # noqa: E402


YEARS_BGS = tuple(range(2008, 2013))
YEARS_OVERLAP = (2010, 2011, 2012)
YEARS_CALIBRATION_VALIDATION = (2010, 2011)
TARGET_METALS = (
    "Aluminium",
    "Antimony",
    "Boron",
    "Chromium",
    "Cobalt",
    "Copper",
    "Graphite",
    "Lead",
    "Lithium",
    "Magnesium",
    "Manganese",
    "Molybdenum",
    "Nickel",
    "Niobium/Tantalum",
    "PGM",
    "RareEarth",
    "Silver",
    "Tin",
    "Titanium",
    "Tungsten",
    "Vanadium",
    "Zinc",
    "Zirconium",
)

# Thresholds are declared before the data are compared.  They are display- and
# model-purpose thresholds, not statistical significance cut-offs.
GATE_THRESHOLDS = {
    "required_target_metals": 23,
    "required_bgs_years_per_metal": 5,
    "required_overlap_years_per_metal": 3,
    "median_tvd_max": 0.10,
    "p90_tvd_max": 0.20,
    "per_metal_median_tvd_max": 0.25,
    "top_producer_match_rate_min": 0.90,
    "p90_absolute_hhi_difference_max": 0.05,
    "bgs_listed_share_of_printed_total_min": 0.90,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_token(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower().replace("&", " and "))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


COUNTRY_ALIASES = {
    "bosnia and herzegovina": "bosnia herzegovina",
    "bosnia herzegovina": "bosnia herzegovina",
    "burma": "myanmar",
    "congo d r": "congo democratic republic",
    "congo democratic republic": "congo democratic republic",
    "czech republic": "czechia",
    "ireland republic of": "ireland",
    "korea dem p r of": "korea north",
    "korea north": "korea north",
    "korea rep of": "korea south",
    "korea south": "korea south",
    "lao p d r": "laos",
    "macedonia": "north macedonia",
    "new caledonia": "new caledonia",
    "russia asia": "russia",
    "russia europe": "russia",
    "swaziland": "eswatini",
    "united states": "united states",
    "usa": "united states",
    "viet nam": "vietnam",
}


def country_key(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)\s*", " ", value)
    key = normalized_token(value)
    return COUNTRY_ALIASES.get(key, key)


BGS_TABLES: tuple[tuple[re.Pattern[str], str, str, str], ...] = (
    (re.compile(r"^Production of bauxite\b", re.I), "Aluminium", "Bauxite", "crude ore"),
    (re.compile(r"^Mine production of antimony\b", re.I), "Antimony", "Antimony", "metal content"),
    (re.compile(r"^Production of borates\b", re.I), "Boron", "Boron Minerals", "gross borate products"),
    (re.compile(r"^Production of chromium ores and concentrates\b", re.I), "Chromium", "Chromium", "gross ores and concentrates"),
    (re.compile(r"^Mine production of cobalt\b", re.I), "Cobalt", "Cobalt", "metal content or recovered cobalt as reported"),
    (re.compile(r"^Mine production of copper\b", re.I), "Copper", "Copper", "metal content"),
    (re.compile(r"^Production of graphite\b", re.I), "Graphite", "Graphite", "natural graphite as reported"),
    (re.compile(r"^Mine production of lead\b", re.I), "Lead", "Lead", "metal content"),
    (re.compile(r"^Production of lithium minerals\b", re.I), "Lithium", "Lithium", "mixed lithium minerals/compounds as reported"),
    (re.compile(r"^Production of magnesite\b", re.I), "Magnesium", "Magnesite", "gross magnesite"),
    (re.compile(r"^Production of manganese ore\b", re.I), "Manganese", "Manganese", "gross manganese ore"),
    (re.compile(r"^Mine production of molybdenum\b", re.I), "Molybdenum", "Molybdenum", "metal content"),
    (re.compile(r"^Mine production of nickel\b", re.I), "Nickel", "Nickel", "metal content"),
    (re.compile(r"^Production of tantalum and niobium minerals\b", re.I), "Niobium/Tantalum", "Niobium/Tantalum", "gross concentrates"),
    (re.compile(r"^Mine production of platinum group metals\b", re.I), "PGM", "PGM", "PGM metal content as reported"),
    (re.compile(r"^Production of rare earth oxides\b", re.I), "RareEarth", "Rare Earths", "REO content"),
    (re.compile(r"^Mine production of silver\b", re.I), "Silver", "Silver", "metal content"),
    (re.compile(r"^Mine production of tin\b", re.I), "Tin", "Tin", "metal content"),
    (re.compile(r"^Production of titanium minerals\b", re.I), "Titanium", "Titanium", "gross mineral concentrates"),
    (re.compile(r"^Mine production of tungsten\b", re.I), "Tungsten", "Tungsten", "metal content"),
    (re.compile(r"^Mine production of vanadium\b", re.I), "Vanadium", "Vanadium", "V content"),
    (re.compile(r"^Mine production of zinc\b", re.I), "Zinc", "Zinc", "metal content"),
    (re.compile(r"^Production of zirconium minerals\b", re.I), "Zirconium", "Zircon", "gross zirconium minerals"),
)

SUBTYPE_PREFIXES = tuple(
    normalized_token(value)
    for value in (
        "Lepidolite",
        "Spodumene",
        "Platinum",
        "Palladium",
        "Other platinum metals",
        "Columbite-tantalite",
        "Tantalite",
        "Columbite",
        "Pyrochlore",
        "Struverite",
        "Ilmenite",
        "Rutile",
        "Leucoxene",
        "Metallurgical",
        "Chemical",
    )
)


def classify_bgs_heading(text: str):
    for pattern, metal, commodity, basis in BGS_TABLES:
        if pattern.search(text):
            return metal, commodity, basis
    return None


def visual_rows(page) -> list[list[dict]]:
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    groups: list[list[dict]] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        target = None
        for group in groups[-4:]:
            if abs(group[0]["top"] - word["top"]) <= 0.65:
                target = group
                break
        if target is None:
            groups.append([word])
        else:
            target.append(word)
    for group in groups:
        group.sort(key=lambda item: item["x0"])
    return groups


def row_text(row: list[dict]) -> str:
    return " ".join(str(item["text"]).strip() for item in row if str(item["text"]).strip())


def parse_bgs_cell(words: list[str]) -> tuple[float | None, str]:
    raw = " ".join(words).strip()
    if not raw or any(mark in raw for mark in ("—", "...")):
        return None, "missing"
    digits = re.sub(r"[^0-9+-]", "", raw)
    if not digits or not re.fullmatch(r"[+-]?\d+", digits):
        return None, "unparsed"
    return float(int(digits)), "estimated" if "*" in raw else "reported"


def parse_bgs() -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    totals: list[dict] = []
    current_heading = None
    active = None
    pending_country: str | None = None

    with pdfplumber.open(BGS_PDF) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            for row in visual_rows(page):
                text = row_text(row)
                heading = classify_bgs_heading(text)
                if re.match(
                    r"^(?:Mine production|Production of|Smelter(?:/refinery)? production|Refinery production)\b",
                    text,
                    re.I,
                ):
                    current_heading = heading
                    active = None
                    pending_country = None
                    continue

                tokens = {str(item["text"]).strip() for item in row}
                if "Country" in tokens and all(str(year) in tokens for year in YEARS_BGS):
                    if current_heading is None:
                        active = None
                        continue
                    year_right = {
                        int(str(item["text"]).strip()): float(item["x1"])
                        for item in row
                        if str(item["text"]).strip() in {str(year) for year in YEARS_BGS}
                    }
                    active = {
                        "metal": current_heading[0],
                        "commodity": current_heading[1],
                        "basis": current_heading[2],
                        "year_right": year_right,
                    }
                    pending_country = None
                    continue

                if active is None:
                    continue
                if text.startswith("Note(s)") or text.startswith("Abbreviations"):
                    active = None
                    pending_country = None
                    continue

                rights = active["year_right"]
                ordered = sorted(rights)
                cells: dict[int, list[str]] = {year: [] for year in ordered}
                residual: list[str] = []
                for item in row:
                    token = str(item["text"]).strip()
                    if not token:
                        continue
                    x1 = float(item["x1"])
                    if x1 < 170.0:
                        residual.append(token)
                        continue
                    # BGS prints thousands-separated values as several PDF words.
                    # Assign every word to the first annual column whose right edge
                    # contains it.  Centre-point boundaries incorrectly attach the
                    # leading group of the next year to the preceding year.
                    year = next(
                        (candidate for candidate in ordered if x1 <= rights[candidate] + 3.5),
                        None,
                    )
                    if year is not None:
                        cells[year].append(token)

                parsed = {year: parse_bgs_cell(cells[year]) for year in ordered}
                numeric_count = sum(value is not None for value, _ in parsed.values())
                residual_text = " ".join(residual).strip()
                residual_norm = normalized_token(residual_text)

                if residual_norm.startswith("world total"):
                    for year, (value, status) in parsed.items():
                        if value is not None:
                            totals.append(
                                {
                                    "model_metal": active["metal"],
                                    "commodity_label": active["commodity"],
                                    "year": year,
                                    "printed_world_total": value,
                                    "printed_total_basis": residual_text,
                                    "source_pdf_page": page_number,
                                }
                            )
                    active = None
                    pending_country = None
                    continue

                if numeric_count == 0:
                    if residual_text and not residual_norm.startswith(("production", "mine production")):
                        pending_country = residual_text
                    continue

                is_subtype = any(residual_norm.startswith(prefix) for prefix in SUBTYPE_PREFIXES)
                if pending_country and is_subtype:
                    country = pending_country
                    product_type = residual_text
                else:
                    country = residual_text
                    product_type = ""
                    pending_country = None
                if not country:
                    continue

                if normalized_token(country).startswith("world total"):
                    for year, (value, status) in parsed.items():
                        if value is not None:
                            totals.append(
                                {
                                    "model_metal": active["metal"],
                                    "commodity_label": active["commodity"],
                                    "year": year,
                                    "printed_world_total": value,
                                    "printed_total_basis": f"{country} | {product_type}",
                                    "source_pdf_page": page_number,
                                }
                            )
                    active = None
                    pending_country = None
                    continue

                for year, (value, status) in parsed.items():
                    if value is None:
                        continue
                    records.append(
                        {
                            "model_metal": active["metal"],
                            "commodity_label": active["commodity"],
                            "measurement_basis": active["basis"],
                            "mine_origin_country_source": re.sub(r"\s*\([^)]*\)\s*", " ", country).strip(),
                            "country_comparison_key": country_key(country),
                            "product_type": re.sub(r"\s*\([^)]*\)\s*", " ", product_type).strip(),
                            "year": year,
                            "production": value,
                            "value_status": status,
                            "source_edition": "BGS WMP 2008-2012",
                            "source_url": BGS_SOURCE_URL,
                            "source_pdf_page": page_number,
                        }
                    )

    raw = pd.DataFrame(records)
    if raw.empty:
        raise RuntimeError("BGS parser produced no records")
    grouped = (
        raw.groupby(
            [
                "model_metal",
                "commodity_label",
                "measurement_basis",
                "country_comparison_key",
                "year",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            production=("production", "sum"),
            source_country_labels=("mine_origin_country_source", lambda x: " | ".join(sorted(set(x)))),
            product_types=("product_type", lambda x: " | ".join(sorted(v for v in set(x) if v))),
            any_estimated=("value_status", lambda x: bool((x == "estimated").any())),
            source_pdf_pages=("source_pdf_page", lambda x: " | ".join(map(str, sorted(set(x))))),
        )
        .sort_values(["model_metal", "year", "country_comparison_key"])
        .reset_index(drop=True)
    )
    totals_df = pd.DataFrame(totals)
    return grouped, totals_df


def write_wmd_coordinates() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if WMD_COORDINATES.exists() and WMD_COORDINATES.stat().st_size > 0:
        return
    with pdfplumber.open(WMD_PDF) as pdf, gzip.open(WMD_COORDINATES, "wt", encoding="utf-8") as handle:
        for page_number in range(94, 138):
            page = pdf.pages[page_number - 1]
            words = page.extract_words(use_text_flow=False, keep_blank_chars=True)
            handle.write(
                json.dumps(
                    {
                        "record_type": "page",
                        "page": page_number,
                        "width": page.width,
                        "height": page.height,
                        "item_count": len(words),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for index, word in enumerate(words):
                handle.write(
                    json.dumps(
                        {
                            "record_type": "text",
                            "page": page_number,
                            "item_index": index,
                            "x": float(word["x0"]),
                            "y": float(page.height - word["bottom"]),
                            "width": float(word["x1"] - word["x0"]),
                            "height": float(word["bottom"] - word["top"]),
                            "font_name": "pdfplumber_word",
                            "text": str(word["text"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


WMD_MODEL_METAL = {
    "Bauxite": "Aluminium",
    "Antimony": "Antimony",
    "Boron Minerals": "Boron",
    "Chromium": "Chromium",
    "Cobalt": "Cobalt",
    "Copper": "Copper",
    "Graphite": "Graphite",
    "Lead": "Lead",
    "Lithium": "Lithium",
    "Magnesite": "Magnesium",
    "Manganese": "Manganese",
    "Molybdenum": "Molybdenum",
    "Nickel": "Nickel",
    "Niobium": "Niobium/Tantalum",
    "Tantalum": "Niobium/Tantalum",
    "Palladium": "PGM",
    "Platinum": "PGM",
    "Rhodium": "PGM",
    "Rare Earths": "RareEarth",
    "Silver": "Silver",
    "Tin": "Tin",
    "Titanium": "Titanium",
    "Tungsten": "Tungsten",
    "Vanadium": "Vanadium",
    "Zinc": "Zinc",
    "Zircon": "Zirconium",
}


def parse_wmd() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    write_wmd_coordinates()
    years = (2010, 2011, 2012, 2013, 2014)
    records: list[dict] = []
    totals: list[dict] = []
    warnings: list[dict] = []
    recent: list[str] = []
    active = None

    def recent_heading() -> str:
        skip = (
            r"^DATA$",
            r"^Total\b",
            r"^6\.4(?:\.\d)?\b",
            r"^Production of Mineral",
            r"^of individual Countries",
            r"^by Minerals$",
            r"^Abbreviations",
            r"^World Mining Data",
            r"^metr\. t",
            r"^kg(?:\s+kg)*$",
            r"^ct(?:\s+ct)*$",
            r"^\d+$",
        )
        for candidate in reversed(recent[-12:]):
            if not re.search(r"[A-Za-z]", candidate):
                continue
            if any(re.search(pattern, candidate, re.I) for pattern in skip):
                continue
            return candidate
        raise RuntimeError("Could not identify WMD commodity heading")

    with pdfplumber.open(WMD_PDF) as pdf:
        for page_number in range(94, 138):
            page = pdf.pages[page_number - 1]
            words = page.extract_words(use_text_flow=False, keep_blank_chars=True)
            rows: list[list[dict]] = []
            for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
                target = None
                for group in rows[-4:]:
                    if abs(group[0]["top"] - word["top"]) <= 0.65:
                        target = group
                        break
                if target is None:
                    rows.append([word])
                else:
                    target.append(word)
            for row in rows:
                row.sort(key=lambda item: item["x0"])
                text = row_text(row)
                clean_tokens = [str(item["text"]).strip() for item in row]
                token_set = set(clean_tokens)
                countryish = "Country" in text or "C o u n t r" in text
                if countryish and all(str(year) in token_set for year in years) and "Rem" in token_set:
                    raw_heading = recent_heading()
                    commodity = canonical_commodity(raw_heading)
                    year_right = {
                        int(str(item["text"]).strip()): float(item["x1"])
                        for item in row
                        if str(item["text"]).strip() in {str(year) for year in years}
                    }
                    rem_right = max(
                        float(item["x1"])
                        for item in row
                        if str(item["text"]).strip() == "Rem"
                    )
                    active = {
                        "raw_heading": raw_heading,
                        "commodity": commodity,
                        "basis": measurement_basis(raw_heading, commodity),
                        "year_right": year_right,
                        "rem_right": rem_right,
                        "units": {},
                    }
                    recent.append(text)
                    continue

                if active is not None:
                    rights = active["year_right"]
                    ordered = sorted(rights)
                    boundaries = [
                        (rights[left] + rights[right]) / 2
                        for left, right in zip(ordered[:-1], ordered[1:])
                    ]
                    last_to_rem = (rights[ordered[-1]] + active["rem_right"]) / 2
                    cells: dict[int | str, list[str]] = {year: [] for year in ordered}
                    cells["rem"] = []
                    residual: list[str] = []
                    for item in row:
                        token = str(item["text"]).strip()
                        if not token:
                            continue
                        xmid = (float(item["x0"]) + float(item["x1"])) / 2
                        if xmid < 150.0:
                            residual.append(token)
                        elif xmid > last_to_rem:
                            cells["rem"].append(token)
                        else:
                            column = 0
                            while column < len(boundaries) and xmid > boundaries[column]:
                                column += 1
                            cells[ordered[column]].append(token)
                    parsed = {
                        year: parse_bgs_cell(cells[year])
                        for year in ordered
                    }
                    residual_text = " ".join(residual).strip()
                    residual_norm = normalized_token(residual_text)
                    numeric_count = sum(value is not None for value, _ in parsed.values())

                    if numeric_count and not active["units"] and residual_norm == "":
                        for year in ordered:
                            raw_unit = " ".join(cells[year]).strip()
                            if raw_unit:
                                active["units"][year] = raw_unit
                        recent.append(text)
                        continue

                    if residual_norm == "total":
                        for year, (value, status) in parsed.items():
                            if value is not None:
                                totals.append(
                                    {
                                        "commodity_label": active["commodity"],
                                        "year": year,
                                        "printed_world_total": value,
                                        "source_pdf_page": page_number,
                                    }
                                )
                        active = None
                        recent.append(text)
                        continue

                    if numeric_count and residual_text and active["commodity"] in WMD_MODEL_METAL:
                        source_rem = " ".join(cells["rem"]).strip()
                        for year, (value, status) in parsed.items():
                            if value is None:
                                continue
                            records.append(
                                {
                                    "commodity_label": active["commodity"],
                                    "model_metal": WMD_MODEL_METAL[active["commodity"]],
                                    "measurement_basis": active["basis"],
                                    "mine_origin_country_wmd": residual_text,
                                    "country_comparison_key": country_key(residual_text),
                                    "year": year,
                                    "production": value,
                                    "unit": active["units"].get(year, ""),
                                    "source_rem": source_rem,
                                    "source_pdf_page": page_number,
                                }
                            )
                recent.append(text)
                if len(recent) > 30:
                    recent = recent[-30:]

    frame = pd.DataFrame(records)
    frame = frame[frame["year"].isin(YEARS_OVERLAP)].copy()
    grouped = (
        frame.groupby(
            ["model_metal", "country_comparison_key", "year"],
            as_index=False,
            observed=True,
        )
        .agg(
            production=("production", "sum"),
            source_country_labels=("mine_origin_country_wmd", lambda x: " | ".join(sorted(set(x)))),
            commodity_labels=("commodity_label", lambda x: " | ".join(sorted(set(x)))),
            measurement_bases=("measurement_basis", lambda x: " | ".join(sorted(set(x)))),
            units=("unit", lambda x: " | ".join(sorted(set(x)))),
            source_flags=("source_rem", lambda x: " | ".join(sorted(set(x)))),
            source_pdf_pages=("source_pdf_page", lambda x: " | ".join(map(str, sorted(set(x))))),
        )
        .sort_values(["model_metal", "year", "country_comparison_key"])
        .reset_index(drop=True)
    )
    raw = pd.DataFrame(records)
    raw_sums = (
        raw.groupby(["commodity_label", "year"], as_index=False, observed=True)["production"]
        .sum()
        .rename(columns={"production": "parsed_country_sum"})
    )
    target_commodities = set(WMD_MODEL_METAL)
    totals_frame = pd.DataFrame(totals)
    totals_frame = totals_frame[totals_frame["commodity_label"].isin(target_commodities)].copy()
    audits = raw_sums.merge(
        totals_frame, on=["commodity_label", "year"], how="outer", validate="one_to_one"
    )
    audits["difference"] = audits["parsed_country_sum"] - audits["printed_world_total"]
    audits["relative_difference"] = audits["difference"] / audits["printed_world_total"]
    audits["all_year_totals_reconciled"] = audits["difference"].fillna(np.inf).eq(0)
    missing_totals = audits[audits["printed_world_total"].isna()]
    for row in missing_totals.itertuples(index=False):
        warnings.append(
            {
                "commodity_label": row.commodity_label,
                "year": row.year,
                "warning": "missing printed total in direct WMD parser",
            }
        )
    return grouped, audits, warnings


BASIS_CLASS = {
    "Aluminium": "directly comparable",
    "Antimony": "directly comparable",
    "Boron": "share comparison; gross products",
    "Chromium": "share comparison; gross ore and Cr2O3 bases differ",
    "Cobalt": "share comparison; recovery conventions differ for some countries",
    "Copper": "directly comparable",
    "Graphite": "share comparison; reported product conventions vary",
    "Lead": "directly comparable",
    "Lithium": "share comparison; mixed products and Li2O bases differ",
    "Magnesium": "directly comparable",
    "Manganese": "share comparison; gross ore and WMD reported bases differ",
    "Molybdenum": "directly comparable",
    "Nickel": "directly comparable",
    "Niobium/Tantalum": "share comparison; gross concentrates and oxide-content bases differ",
    "PGM": "share comparison; BGS all PGM and WMD Pt/Pd/Rh scope differ",
    "RareEarth": "directly comparable",
    "Silver": "directly comparable",
    "Tin": "directly comparable",
    "Titanium": "share comparison; gross concentrates and TiO2 bases differ",
    "Tungsten": "directly comparable",
    "Vanadium": "shares directly comparable after a constant V/V2O5 conversion",
    "Zinc": "directly comparable",
    "Zirconium": "share comparison; zirconium-mineral scope differs slightly",
}


def rank_correlation(left: pd.Series, right: pd.Series) -> float:
    if len(left) < 2:
        return math.nan
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank))


def build_comparison(bgs: pd.DataFrame, wmd: pd.DataFrame):
    bgs_overlap = bgs[bgs["year"].isin(YEARS_OVERLAP)].copy()
    left = bgs_overlap.rename(columns={"production": "production_bgs"})
    right = wmd.rename(columns={"production": "production_wmd"})
    detail = left.merge(
        right,
        on=["model_metal", "country_comparison_key", "year"],
        how="outer",
        suffixes=("_bgs", "_wmd"),
        validate="one_to_one",
    )
    detail["production_bgs"] = detail["production_bgs"].fillna(0.0)
    detail["production_wmd"] = detail["production_wmd"].fillna(0.0)
    totals_bgs = detail.groupby(["model_metal", "year"], observed=True)["production_bgs"].transform("sum")
    totals_wmd = detail.groupby(["model_metal", "year"], observed=True)["production_wmd"].transform("sum")
    detail["share_bgs"] = detail["production_bgs"] / totals_bgs
    detail["share_wmd"] = detail["production_wmd"] / totals_wmd
    detail["share_difference_wmd_minus_bgs"] = detail["share_wmd"] - detail["share_bgs"]
    detail["absolute_share_difference"] = detail["share_difference_wmd_minus_bgs"].abs()
    detail["presence_status"] = np.select(
        [
            detail["production_bgs"].gt(0) & detail["production_wmd"].gt(0),
            detail["production_bgs"].gt(0),
            detail["production_wmd"].gt(0),
        ],
        ["both_positive", "bgs_only_positive", "wmd_only_positive"],
        default="both_zero",
    )
    detail["basis_comparability"] = detail["model_metal"].map(BASIS_CLASS)

    summaries = []
    for (metal, year), group in detail.groupby(["model_metal", "year"], observed=True):
        bgs_share = group["share_bgs"]
        wmd_share = group["share_wmd"]
        top_bgs = group.loc[bgs_share.idxmax(), "country_comparison_key"]
        top_wmd = group.loc[wmd_share.idxmax(), "country_comparison_key"]
        top3_bgs = set(group.nlargest(3, "share_bgs")["country_comparison_key"])
        top3_wmd = set(group.nlargest(3, "share_wmd")["country_comparison_key"])
        summaries.append(
            {
                "model_metal": metal,
                "year": int(year),
                "basis_comparability": BASIS_CLASS[metal],
                "total_production_bgs": group["production_bgs"].sum(),
                "total_production_wmd": group["production_wmd"].sum(),
                "wmd_to_bgs_total_ratio": (
                    group["production_wmd"].sum() / group["production_bgs"].sum()
                    if group["production_bgs"].sum() > 0
                    else math.nan
                ),
                "countries_union": int(len(group)),
                "countries_both_positive": int((group["presence_status"] == "both_positive").sum()),
                "countries_bgs_only_positive": int((group["presence_status"] == "bgs_only_positive").sum()),
                "countries_wmd_only_positive": int((group["presence_status"] == "wmd_only_positive").sum()),
                "total_variation_distance": 0.5 * group["absolute_share_difference"].sum(),
                "share_overlap_coefficient": 1.0 - 0.5 * group["absolute_share_difference"].sum(),
                "hhi_bgs": float((bgs_share**2).sum()),
                "hhi_wmd": float((wmd_share**2).sum()),
                "absolute_hhi_difference": abs(float((wmd_share**2).sum() - (bgs_share**2).sum())),
                "spearman_country_share": rank_correlation(bgs_share, wmd_share),
                "top_country_bgs": top_bgs,
                "top_country_wmd": top_wmd,
                "top_country_match": bool(top_bgs == top_wmd),
                "top3_jaccard": len(top3_bgs & top3_wmd) / len(top3_bgs | top3_wmd),
                "maximum_absolute_country_share_shift": group["absolute_share_difference"].max(),
            }
        )
    summary = pd.DataFrame(summaries).sort_values(["model_metal", "year"]).reset_index(drop=True)
    mineral_summary = (
        summary.groupby("model_metal", as_index=False, observed=True)
        .agg(
            overlap_years=("year", "nunique"),
            median_tvd=("total_variation_distance", "median"),
            maximum_tvd=("total_variation_distance", "max"),
            mean_share_overlap=("share_overlap_coefficient", "mean"),
            median_absolute_hhi_difference=("absolute_hhi_difference", "median"),
            maximum_absolute_hhi_difference=("absolute_hhi_difference", "max"),
            top_producer_match_rate=("top_country_match", "mean"),
            median_spearman=("spearman_country_share", "median"),
        )
        .sort_values("model_metal")
        .reset_index(drop=True)
    )
    mineral_summary["basis_comparability"] = mineral_summary["model_metal"].map(BASIS_CLASS)
    return detail, summary, mineral_summary


def load_wmd2018_reference() -> pd.DataFrame:
    frame = pd.read_csv(WMD2018_SEED)
    frame = frame[
        frame["model_metal"].isin(TARGET_METALS)
        & frame["year"].eq(2012)
        & frame["source_edition"].eq("WMD2018")
        & frame["production"].gt(0)
    ].copy()
    frame["country_comparison_key"] = frame["mine_origin_country_wmd"].map(country_key)
    return (
        frame.groupby(["model_metal", "country_comparison_key", "year"], as_index=False)
        .agg(
            production=("production", "sum"),
            source_country_labels=(
                "mine_origin_country_wmd",
                lambda values: " | ".join(sorted(set(values))),
            ),
            source_pdf_pages=(
                "source_pdf_page",
                lambda values: " | ".join(map(str, sorted(set(values)))),
            ),
        )
        .sort_values(["model_metal", "country_comparison_key"])
        .reset_index(drop=True)
    )


def build_wmd2018_anchored_calibration(
    bgs: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bgs_anchor = bgs[bgs["year"].eq(2012)][
        ["model_metal", "country_comparison_key", "production"]
    ].rename(columns={"production": "bgs_2012_production"})
    ref_anchor = reference[["model_metal", "country_comparison_key", "production"]].rename(
        columns={"production": "wmd2018_2012_production"}
    )
    bridge = bgs_anchor.merge(
        ref_anchor,
        on=["model_metal", "country_comparison_key"],
        how="outer",
        validate="one_to_one",
    )
    bridge["anchor_status"] = np.select(
        [
            bridge["bgs_2012_production"].notna()
            & bridge["wmd2018_2012_production"].notna(),
            bridge["bgs_2012_production"].notna(),
            bridge["wmd2018_2012_production"].notna(),
        ],
        ["country_anchor", "bgs_only_2012", "wmd2018_only_2012"],
        default="neither",
    )
    bridge["country_calibration_factor"] = (
        bridge["wmd2018_2012_production"] / bridge["bgs_2012_production"]
    )

    bgs_total = bgs_anchor.groupby("model_metal")["bgs_2012_production"].sum()
    ref_total = ref_anchor.groupby("model_metal")["wmd2018_2012_production"].sum()
    fallback = (ref_total / bgs_total).rename("mineral_total_fallback_factor").reset_index()
    bridge = bridge.merge(fallback, on="model_metal", how="left", validate="many_to_one")
    bridge["factor_used_for_history"] = bridge["country_calibration_factor"].fillna(
        bridge["mineral_total_fallback_factor"]
    )

    country_factors = bridge.loc[
        bridge["anchor_status"].eq("country_anchor"),
        ["model_metal", "country_comparison_key", "country_calibration_factor"],
    ]
    calibrated = bgs.merge(
        country_factors,
        on=["model_metal", "country_comparison_key"],
        how="left",
        validate="many_to_one",
    ).merge(fallback, on="model_metal", how="left", validate="many_to_one")
    calibrated["calibration_method"] = np.where(
        calibrated["country_calibration_factor"].notna(),
        "country-specific WMD2018/BGS 2012 anchor",
        "mineral-total WMD2018/BGS 2012 fallback",
    )
    calibrated["calibration_factor"] = calibrated["country_calibration_factor"].fillna(
        calibrated["mineral_total_fallback_factor"]
    )
    calibrated["calibrated_production"] = (
        calibrated["production"] * calibrated["calibration_factor"]
    )

    coverage = (
        calibrated[calibrated["year"].isin(range(2008, 2012))]
        .assign(
            anchored_value=lambda frame: np.where(
                frame["country_calibration_factor"].notna(), frame["production"], 0.0
            ),
            anchored_row=lambda frame: frame["country_calibration_factor"].notna(),
            calibrated_value=lambda frame: np.where(
                frame["calibration_factor"].notna()
                & np.isfinite(frame["calibration_factor"])
                & frame["calibration_factor"].gt(0),
                frame["production"],
                0.0,
            ),
            calibrated_row=lambda frame: frame["calibration_factor"].notna()
            & np.isfinite(frame["calibration_factor"])
            & frame["calibration_factor"].gt(0),
        )
        .groupby(["model_metal", "year"], as_index=False, observed=True)
        .agg(
            raw_bgs_production=("production", "sum"),
            country_anchored_raw_production=("anchored_value", "sum"),
            source_rows=("country_comparison_key", "size"),
            country_anchored_rows=("anchored_row", "sum"),
            calibration_assigned_raw_production=("calibrated_value", "sum"),
            calibration_assigned_rows=("calibrated_row", "sum"),
        )
    )
    coverage["country_anchor_value_coverage"] = (
        coverage["country_anchored_raw_production"] / coverage["raw_bgs_production"]
    )
    coverage["country_anchor_row_coverage"] = (
        coverage["country_anchored_rows"] / coverage["source_rows"]
    )
    coverage["calibration_assigned_value_coverage"] = (
        coverage["calibration_assigned_raw_production"]
        / coverage["raw_bgs_production"]
    )
    coverage["calibration_assigned_row_coverage"] = (
        coverage["calibration_assigned_rows"] / coverage["source_rows"]
    )
    return calibrated, bridge, coverage


def build_calibrated_validation(
    calibrated: pd.DataFrame, wmd2016: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    left = calibrated[calibrated["year"].isin(YEARS_CALIBRATION_VALIDATION)].copy()
    left = left.drop(columns=["production"]).rename(
        columns={"calibrated_production": "production"}
    )
    right = wmd2016[wmd2016["year"].isin(YEARS_CALIBRATION_VALIDATION)].copy()
    return build_comparison(left, right)


def calibration_gate_results(
    reference: pd.DataFrame,
    coverage: pd.DataFrame,
    validation: pd.DataFrame,
    mineral_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    validation_year_counts = validation.groupby("model_metal", observed=True)["year"].nunique()
    checks = [
        (
            "wmd2018_anchor_target_metal_coverage",
            int(reference["model_metal"].nunique()),
            GATE_THRESHOLDS["required_target_metals"],
            int(reference["model_metal"].nunique())
            == GATE_THRESHOLDS["required_target_metals"],
            "equal",
        ),
        (
            "minimum_valid_calibration_factor_value_coverage_2008_2011",
            float(coverage["calibration_assigned_value_coverage"].min()),
            1.0,
            bool(
                np.isclose(
                    coverage["calibration_assigned_value_coverage"].min(),
                    1.0,
                    atol=1e-12,
                )
            ),
            "minimum",
        ),
        (
            "minimum_validation_years_per_metal",
            int(validation_year_counts.min()),
            len(YEARS_CALIBRATION_VALIDATION),
            int(validation_year_counts.min()) >= len(YEARS_CALIBRATION_VALIDATION),
            "minimum",
        ),
        (
            "validation_median_total_variation_distance",
            float(validation["total_variation_distance"].median()),
            GATE_THRESHOLDS["median_tvd_max"],
            bool(
                validation["total_variation_distance"].median()
                <= GATE_THRESHOLDS["median_tvd_max"]
            ),
            "maximum",
        ),
        (
            "validation_p90_total_variation_distance",
            float(validation["total_variation_distance"].quantile(0.90)),
            GATE_THRESHOLDS["p90_tvd_max"],
            bool(
                validation["total_variation_distance"].quantile(0.90)
                <= GATE_THRESHOLDS["p90_tvd_max"]
            ),
            "maximum",
        ),
        (
            "validation_maximum_mineral_median_tvd",
            float(mineral_validation["median_tvd"].max()),
            GATE_THRESHOLDS["per_metal_median_tvd_max"],
            bool(
                mineral_validation["median_tvd"].max()
                <= GATE_THRESHOLDS["per_metal_median_tvd_max"]
            ),
            "maximum",
        ),
        (
            "validation_top_producer_match_rate",
            float(validation["top_country_match"].mean()),
            GATE_THRESHOLDS["top_producer_match_rate_min"],
            bool(
                validation["top_country_match"].mean()
                >= GATE_THRESHOLDS["top_producer_match_rate_min"]
            ),
            "minimum",
        ),
        (
            "validation_p90_absolute_hhi_difference",
            float(validation["absolute_hhi_difference"].quantile(0.90)),
            GATE_THRESHOLDS["p90_absolute_hhi_difference_max"],
            bool(
                validation["absolute_hhi_difference"].quantile(0.90)
                <= GATE_THRESHOLDS["p90_absolute_hhi_difference_max"]
            ),
            "maximum",
        ),
    ]
    gate = pd.DataFrame(checks, columns=["check", "observed", "threshold", "passed", "criterion"])
    return gate, bool(gate["passed"].all())


def build_bgs_total_audit(bgs: pd.DataFrame, printed: pd.DataFrame) -> pd.DataFrame:
    parsed = (
        bgs.groupby(["model_metal", "year"], as_index=False, observed=True)["production"]
        .sum()
        .rename(columns={"production": "parsed_listed_country_total"})
    )
    printed = printed.sort_values(["model_metal", "year", "source_pdf_page"]).drop_duplicates(
        ["model_metal", "year"], keep="first"
    )
    audit = parsed.merge(printed, on=["model_metal", "year"], how="left", validate="one_to_one")
    audit["listed_share_of_printed_total"] = (
        audit["parsed_listed_country_total"] / audit["printed_world_total"]
    )
    audit["printed_total_same_basis"] = ~audit["model_metal"].isin(
        ["Lithium", "Titanium"]
    )
    return audit.sort_values(["model_metal", "year"]).reset_index(drop=True)


def gate_results(
    bgs: pd.DataFrame,
    wmd: pd.DataFrame,
    bgs_total_audit: pd.DataFrame,
    wmd_table_audit: pd.DataFrame,
    summary: pd.DataFrame,
    mineral_summary: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    bgs_year_counts = bgs.groupby("model_metal", observed=True)["year"].nunique()
    wmd_year_counts = wmd.groupby("model_metal", observed=True)["year"].nunique()
    printed_available = (
        bgs_total_audit["printed_world_total"].notna()
        & bgs_total_audit["printed_total_same_basis"]
    )
    listed_share = bgs_total_audit.loc[printed_available, "listed_share_of_printed_total"]
    checks = [
        (
            "target_metal_coverage_bgs",
            int(bgs["model_metal"].nunique()),
            GATE_THRESHOLDS["required_target_metals"],
            int(bgs["model_metal"].nunique()) == GATE_THRESHOLDS["required_target_metals"],
            "equal",
        ),
        (
            "target_metal_coverage_wmd",
            int(wmd["model_metal"].nunique()),
            GATE_THRESHOLDS["required_target_metals"],
            int(wmd["model_metal"].nunique()) == GATE_THRESHOLDS["required_target_metals"],
            "equal",
        ),
        (
            "minimum_bgs_years_per_metal",
            int(bgs_year_counts.min()),
            GATE_THRESHOLDS["required_bgs_years_per_metal"],
            int(bgs_year_counts.min()) >= GATE_THRESHOLDS["required_bgs_years_per_metal"],
            "minimum",
        ),
        (
            "minimum_overlap_years_per_metal",
            int(wmd_year_counts.min()),
            GATE_THRESHOLDS["required_overlap_years_per_metal"],
            int(wmd_year_counts.min()) >= GATE_THRESHOLDS["required_overlap_years_per_metal"],
            "minimum",
        ),
        (
            "wmd_all_printed_totals_reconciled",
            int(wmd_table_audit["all_year_totals_reconciled"].fillna(False).sum()),
            int(len(wmd_table_audit)),
            bool(wmd_table_audit["all_year_totals_reconciled"].fillna(False).all()),
            "all_true",
        ),
        (
            "minimum_bgs_listed_share_of_printed_total",
            float(listed_share.min()) if len(listed_share) else math.nan,
            GATE_THRESHOLDS["bgs_listed_share_of_printed_total_min"],
            bool(len(listed_share) and listed_share.min() >= GATE_THRESHOLDS["bgs_listed_share_of_printed_total_min"]),
            "minimum",
        ),
        (
            "median_total_variation_distance",
            float(summary["total_variation_distance"].median()),
            GATE_THRESHOLDS["median_tvd_max"],
            bool(summary["total_variation_distance"].median() <= GATE_THRESHOLDS["median_tvd_max"]),
            "maximum",
        ),
        (
            "p90_total_variation_distance",
            float(summary["total_variation_distance"].quantile(0.90)),
            GATE_THRESHOLDS["p90_tvd_max"],
            bool(summary["total_variation_distance"].quantile(0.90) <= GATE_THRESHOLDS["p90_tvd_max"]),
            "maximum",
        ),
        (
            "maximum_mineral_median_tvd",
            float(mineral_summary["median_tvd"].max()),
            GATE_THRESHOLDS["per_metal_median_tvd_max"],
            bool(mineral_summary["median_tvd"].max() <= GATE_THRESHOLDS["per_metal_median_tvd_max"]),
            "maximum",
        ),
        (
            "top_producer_match_rate",
            float(summary["top_country_match"].mean()),
            GATE_THRESHOLDS["top_producer_match_rate_min"],
            bool(summary["top_country_match"].mean() >= GATE_THRESHOLDS["top_producer_match_rate_min"]),
            "minimum",
        ),
        (
            "p90_absolute_hhi_difference",
            float(summary["absolute_hhi_difference"].quantile(0.90)),
            GATE_THRESHOLDS["p90_absolute_hhi_difference_max"],
            bool(summary["absolute_hhi_difference"].quantile(0.90) <= GATE_THRESHOLDS["p90_absolute_hhi_difference_max"]),
            "maximum",
        ),
    ]
    gate = pd.DataFrame(checks, columns=["check", "observed", "threshold", "passed", "criterion"])
    return gate, bool(gate["passed"].all())


def write_outputs() -> bool:
    for path in (BGS_PDF, WMD_PDF, WMD2018_SEED):
        if not path.exists():
            raise FileNotFoundError(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bgs, bgs_printed = parse_bgs()
    wmd, wmd_audit, wmd_warnings = parse_wmd()
    detail, annual, mineral = build_comparison(bgs, wmd)
    bgs_total_audit = build_bgs_total_audit(bgs, bgs_printed)
    raw_gate, _ = gate_results(bgs, wmd, bgs_total_audit, wmd_audit, annual, mineral)
    reference = load_wmd2018_reference()
    calibrated, bridge, calibration_coverage = build_wmd2018_anchored_calibration(
        bgs, reference
    )
    validation_detail, validation, validation_mineral = build_calibrated_validation(
        calibrated, wmd
    )
    calibration_gate, calibration_passed = calibration_gate_results(
        reference, calibration_coverage, validation, validation_mineral
    )
    structural_gate = raw_gate.iloc[:6].copy()
    structural_gate["check"] = "source_structure_" + structural_gate["check"]
    gate = pd.concat([structural_gate, calibration_gate], ignore_index=True)
    passed = bool(gate["passed"].all() and calibration_passed)

    bgs.to_csv(output_path("bgs_mine_production_2008_2012_normalized.csv.gz"), index=False)
    wmd.to_csv(output_path("wmd2016_mine_production_2010_2012_normalized.csv.gz"), index=False)
    detail.to_csv(output_path("bgs_wmd_country_share_detail_2010_2012.csv.gz"), index=False)
    annual.to_csv(output_path("bgs_wmd_mineral_year_audit_2010_2012.csv"), index=False)
    mineral.to_csv(output_path("bgs_wmd_mineral_summary_audit_2010_2012.csv"), index=False)
    bgs_total_audit.to_csv(output_path("bgs_printed_total_reconciliation_2008_2012.csv"), index=False)
    wmd_audit.to_csv(output_path("wmd2016_table_reconciliation.csv"), index=False)
    pd.DataFrame(wmd_warnings).to_csv(output_path("wmd2016_parser_warnings.csv"), index=False)
    raw_gate.to_csv(output_path("raw_uncalibrated_audit_gate_summary.csv"), index=False)
    reference.to_csv(output_path("wmd2018_reference_seed_2012_normalized.csv.gz"), index=False)
    bridge.to_csv(output_path("bgs_to_wmd2018_country_calibration_bridge_2012.csv"), index=False)
    calibrated.to_csv(output_path("bgs_calibrated_mine_production_2008_2012.csv.gz"), index=False)
    calibration_coverage.to_csv(output_path("calibration_anchor_coverage_2008_2011.csv"), index=False)
    validation_detail.to_csv(output_path("calibrated_holdout_country_share_detail_2010_2011.csv.gz"), index=False)
    validation.to_csv(output_path("calibrated_holdout_mineral_year_audit_2010_2011.csv"), index=False)
    validation_mineral.to_csv(output_path("calibrated_holdout_mineral_summary_2010_2011.csv"), index=False)
    gate.to_csv(output_path("audit_gate_summary.csv"), index=False)

    manifest = {
        "audit_scope": "BGS 2008-2012 coverage; BGS-WMD country-share overlap 2010-2012",
        "gate_passed": passed,
        "thresholds_predeclared": GATE_THRESHOLDS,
        "sources": [
            {
                "name": BGS_PDF.name,
                "sha256": sha256(BGS_PDF),
                "source_url": BGS_SOURCE_URL,
                "pages": 126,
            },
            {
                "name": WMD_PDF.name,
                "sha256": sha256(WMD_PDF),
                "original_official_url": WMD_ORIGINAL_URL,
                "archive_url": WMD_ARCHIVE_URL,
                "archive_timestamp_utc": "2016-06-10T00:19:16Z",
                "pages": 250,
            },
            {
                "name": WMD2018_SEED.name,
                "sha256": sha256(WMD2018_SEED),
                "source_edition": "WMD2018",
                "role": "2012 country-specific calibration anchor",
            },
        ],
        "output_row_counts": {
            "bgs_normalized": int(len(bgs)),
            "wmd_normalized": int(len(wmd)),
            "country_share_detail": int(len(detail)),
            "mineral_year_audit": int(len(annual)),
            "mineral_summary": int(len(mineral)),
            "calibration_bridge": int(len(bridge)),
            "calibration_holdout_mineral_year_audit": int(len(validation)),
            "gate_checks": int(len(gate)),
        },
        "calibration_design": {
            "anchor": "WMD2018 country production in 2012 divided by BGS 2012 country production",
            "fallback": "WMD2018/BGS 2012 mineral-total ratio for BGS countries without a country anchor",
            "validation": "apply the frozen 2012 factors to BGS 2010-2011 and compare with WMD2016",
            "factor_clipping": "none",
        },
    }
    output_path("source_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    failed = gate.loc[~gate["passed"], "check"].tolist()
    raw_failed = raw_gate.loc[~raw_gate["passed"], "check"].tolist()
    report = [
        "# BGS-WMD 2008-2012 mine-origin consistency audit",
        "",
        f"Overall gate: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "BGS supplies five-year coverage for 2008-2012. WMD 2016 country-by-mineral tables cover 2010-2014, so the defensible cross-source overlap is 2010-2012; 2008-2009 are coverage-only years.",
        "",
        "The raw comparison is diagnostic because several BGS tables mix gross products or compounds whereas WMD reports content measures, and later WMD editions revise some country series.",
        "",
        f"Raw cross-source median TVD: {annual['total_variation_distance'].median():.4f}; p90 TVD: {annual['total_variation_distance'].quantile(0.90):.4f}; top-producer match: {annual['top_country_match'].mean():.1%}; p90 |HHI difference|: {annual['absolute_hhi_difference'].quantile(0.90):.4f}.",
        "",
        "Calibration uses the existing WMD2018 2012 seed as a country-specific anchor. The 2012 factors are frozen, applied backward to BGS, and tested out of sample against WMD2016 for 2010-2011. No factor is clipped.",
        "",
        f"Calibrated holdout median TVD: {validation['total_variation_distance'].median():.4f}; p90 TVD: {validation['total_variation_distance'].quantile(0.90):.4f}; top-producer match: {validation['top_country_match'].mean():.1%}; p90 |HHI difference|: {validation['absolute_hhi_difference'].quantile(0.90):.4f}.",
        "",
        "Failed checks: " + (", ".join(failed) if failed else "none"),
        "",
        "Raw diagnostic checks not used as the calibrated release gate: "
        + (", ".join(raw_failed) if raw_failed else "none"),
        "",
        "The release gate tests source structure, anchor coverage and out-of-sample country-share concordance after the pre-specified overlap calibration. Raw production totals remain diagnostic where measurement bases differ. No Fig. 5 optimization is authorized unless this calibrated gate passes.",
    ]
    output_path("AUDIT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"BGS normalized rows: {len(bgs):,}")
    print(f"WMD normalized rows: {len(wmd):,}")
    print(f"Mineral-year comparisons: {len(annual):,}")
    print(f"Gate: {'PASS' if passed else 'FAIL'}")
    if failed:
        print("Failed checks: " + ", ".join(failed))
    return passed


if __name__ == "__main__":
    # A failed scientific gate is reported in the outputs rather than treated
    # as a software crash; downstream Fig. 5 scripts must read the gate flag.
    write_outputs()
