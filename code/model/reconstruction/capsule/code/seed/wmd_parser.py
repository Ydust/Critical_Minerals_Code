"""Coordinate-aware parser for World Mining Data chapter 6.4 PDF tables.

The PDF tables right-align each observation to a fixed year-column edge.  That
geometry is essential: plain extracted text collapses internal blank cells and
can silently shift an observation to the wrong year.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import gzip
import json
from pathlib import Path
import re
from typing import Iterable, Iterator
import unicodedata


@dataclass(frozen=True)
class EditionConfig:
    edition: str
    years: tuple[int, ...]
    first_page: int
    last_page: int


EDITIONS = {
    "WMD2018": EditionConfig("WMD2018", (2012, 2013, 2014, 2015, 2016), 103, 146),
    "WMD2021": EditionConfig("WMD2021", (2015, 2016, 2017, 2018, 2019), 112, 155),
    "WMD2025": EditionConfig("WMD2025", (2019, 2020, 2021, 2022, 2023), 112, 155),
}


MODEL_METAL_BY_COMMODITY = {
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


SOURCE_CONFIDENCE = {"r": 0.90, "p": 0.75, "e": 0.55}


@dataclass
class TextItem:
    page: int
    item_index: int
    x: float
    y: float
    width: float
    height: float
    text: str

    @property
    def right(self) -> float:
        return self.x + self.width


@dataclass
class VisualRow:
    page: int
    y: float
    items: list[TextItem]

    @property
    def text(self) -> str:
        return clean_spaces(" ".join(item.text for item in self.items if item.text.strip()))


@dataclass
class ActiveTable:
    table_id: int
    section: str
    commodity_wmd: str
    commodity_label: str
    measurement_basis: str
    header_page: int
    years: tuple[int, ...]
    year_right: dict[int, float]
    rem_right: float
    units: dict[int, str] = field(default_factory=dict)
    unit_raw: dict[int, str] = field(default_factory=dict)
    country_rows: int = 0
    blank_cells: int = 0
    total_values: dict[int, int] = field(default_factory=dict)
    parsed_sums: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    parsed_records: int = 0
    unparsed_numeric_cells: int = 0


@dataclass
class ParseResult:
    edition: str
    records: list[dict]
    table_audit: list[dict]
    warnings: list[dict]
    page_count: int


def clean_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalized_token(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.lower().replace("&", " and "))
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return clean_spaces(value)


def canonical_commodity(heading: str) -> str:
    normalized = normalized_token(heading)
    special = {
        "rare earths concentrates reo content": "Rare Earths",
        "rare earth minerals reo content": "Rare Earths",
        "boron": "Boron Minerals",
        "boron minerals": "Boron Minerals",
        "zirconium": "Zircon",
        "zircon": "Zircon",
        "diamonds gem": "Diamonds (Gem)",
        "diamonds ind": "Diamonds (Industrial)",
    }
    if normalized in special:
        return special[normalized]
    base = re.sub(r"\s*\([^)]*\)\s*$", "", heading).strip()
    normalized_base = normalized_token(base)
    for key, value in special.items():
        if normalized_base == key:
            return value
    words = []
    for word in normalized_base.split():
        if word in {"and", "of"}:
            words.append(word)
        else:
            words.append(word.capitalize())
    return " ".join(words)


def measurement_basis(heading: str, commodity_label: str) -> str:
    text = normalized_token(heading)
    manual = {
        "Iron": "Fe content",
        "Chromium": "Cr2O3 content",
        "Niobium": "Nb2O5 content",
        "Tantalum": "Ta2O5 content",
        "Titanium": "TiO2 content",
        "Tungsten": "W content",
        "Lithium": "Li2O content",
        "Rare Earths": "REO content",
        "Bauxite": "crude ore",
        "Phosphates": "P2O5 content",
        "Potash": "K2O content",
    }
    if commodity_label == "Vanadium":
        return "V2O5 content" if "v2o5" in text or "v2o5" in heading.lower().replace(" ", "") else "V content"
    return manual.get(commodity_label, "as reported commodity")


def normalize_unit(value: str) -> str:
    raw = clean_spaces(value)
    token = normalized_token(raw)
    if token.replace(" ", "") in {"metrt", "metrictonnes", "metrict"}:
        return "metr. t"
    if token == "kg":
        return "kg"
    if token == "ct":
        return "ct"
    return raw


def parse_integer(value: str) -> int | None:
    compact = re.sub(r"[\s\u00a0]", "", value)
    if compact in {"", "-", "–", "—", "..", "..."}:
        return None
    if not re.fullmatch(r"[+-]?\d+", compact):
        return None
    return int(compact)


def split_grouped_integers(value: str, expected: int) -> list[str] | None:
    """Split a PDF.js item that accidentally fused adjacent table cells.

    WMD2018's Iron total fuses the 2013 and 2014 strings into one PDF text
    object.  Thousands grouping makes the intended partition recoverable.
    """
    tokens = value.split()

    def valid_group(parts: list[str]) -> bool:
        if not parts or not re.fullmatch(r"[+-]?\d{1,3}", parts[0]):
            return False
        return all(re.fullmatch(r"\d{3}", token) is not None for token in parts[1:])

    solutions: list[list[str]] = []

    def visit(start: int, remaining: int, built: list[str]) -> None:
        if remaining == 0:
            if start == len(tokens):
                solutions.append(list(built))
            return
        for end in range(start + 1, len(tokens) + 1):
            parts = tokens[start:end]
            if valid_group(parts):
                visit(end, remaining - 1, built + [" ".join(parts)])

    visit(0, expected, [])
    return solutions[0] if len(solutions) == 1 else None


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def iter_visual_rows(path: Path, config: EditionConfig) -> tuple[Iterator[VisualRow], int]:
    pages: dict[int, list[TextItem]] = defaultdict(list)
    page_count = 0
    with open_text(path) as handle:
        for line in handle:
            payload = json.loads(line)
            page = int(payload["page"])
            page_count = max(page_count, page)
            if payload["record_type"] != "text" or not config.first_page <= page <= config.last_page:
                continue
            if not str(payload["text"]).strip():
                continue
            pages[page].append(
                TextItem(
                    page=page,
                    item_index=int(payload["item_index"]),
                    x=float(payload["x"]),
                    y=float(payload["y"]),
                    width=float(payload["width"]),
                    height=float(payload["height"]),
                    text=str(payload["text"]),
                )
            )

    def generate() -> Iterator[VisualRow]:
        for page in range(config.first_page, config.last_page + 1):
            groups: list[list[TextItem]] = []
            for item in sorted(pages.get(page, []), key=lambda x: (-x.y, x.x, x.item_index)):
                target = None
                for group in groups[-3:]:
                    if abs(group[0].y - item.y) <= 0.55:
                        target = group
                        break
                if target is None:
                    groups.append([item])
                else:
                    target.append(item)
            groups.sort(key=lambda group: -sum(x.y for x in group) / len(group))
            for group in groups:
                group.sort(key=lambda x: (x.x, x.item_index))
                yield VisualRow(page, sum(x.y for x in group) / len(group), group)

    return generate(), page_count


def is_table_header(row: VisualRow, years: tuple[int, ...]) -> bool:
    texts = {clean_spaces(item.text) for item in row.items}
    countryish = "Country" in row.text or "C o u n t r y" in row.text
    return countryish and all(str(year) in texts for year in years) and "Rem" in texts


def extract_header_geometry(row: VisualRow, years: tuple[int, ...]) -> tuple[dict[int, float], float]:
    year_right = {}
    rem_right = None
    for item in row.items:
        text = clean_spaces(item.text)
        if text in {str(year) for year in years}:
            year_right[int(text)] = item.right
        elif text == "Rem":
            rem_right = item.right
    if set(year_right) != set(years) or rem_right is None:
        raise ValueError(f"Incomplete header geometry on PDF page {row.page}: {row.text}")
    return year_right, rem_right


def heading_from_recent(recent: list[VisualRow]) -> str:
    skip_patterns = (
        r"^DATA$",
        r"^Total\b",
        r"^6\.4(?:\.\d)?\b",
        r"^PRODUCTION OF\b",
        r"^OF INDIVIDUAL\b",
        r"^by Minerals$",
        r"^Abbreviations",
        r"^World Mining Data",
        r"^\d+$",
    )
    for row in reversed(recent[-8:]):
        text = row.text
        if not re.search(r"[A-Za-z]", text):
            continue
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in skip_patterns):
            continue
        return text
    raise ValueError("Could not identify commodity heading immediately before table header")


def nearest_column(item: TextItem, column_rights: dict[int | str, float], tolerance: float = 2.25):
    distances = sorted((abs(item.right - edge), key) for key, edge in column_rights.items())
    if distances and distances[0][0] <= tolerance:
        return distances[0][1]
    return None


def parse_row_cells(row: VisualRow, table: ActiveTable):
    columns: dict[int | str, float] = dict(table.year_right)
    columns["rem"] = table.rem_right
    assigned: dict[int | str, list[str]] = defaultdict(list)
    residual: list[str] = []
    ordered_years = sorted(table.years, key=lambda year: table.year_right[year])
    for item in row.items:
        column = nearest_column(item, columns)
        if column is None:
            residual.append(item.text)
        else:
            if isinstance(column, int):
                end_index = ordered_years.index(column)
                spanned = [
                    year for year in ordered_years[: end_index + 1]
                    if table.year_right[year] >= item.x - 0.5
                ]
                if len(spanned) > 1:
                    split = split_grouped_integers(item.text, len(spanned))
                    if split is not None:
                        for year, text in zip(spanned, split):
                            assigned[year].append(text)
                        continue
            assigned[column].append(item.text)
    values = {key: clean_spaces(" ".join(parts)) for key, parts in assigned.items()}
    residual_text = clean_spaces(" ".join(residual))
    return values, residual_text


def source_fields(source_rem: str) -> tuple[str, float]:
    match = re.search(r"([A-Za-z])\s*$", source_rem)
    flag = match.group(1).lower() if match else ""
    return flag, SOURCE_CONFIDENCE.get(flag, 0.50)


def table_audit_row(table: ActiveTable, edition: str) -> dict:
    year_rows = []
    all_reconciled = True
    for year in table.years:
        printed = table.total_values.get(year)
        parsed = table.parsed_sums.get(year, 0)
        delta = None if printed is None else parsed - printed
        reconciled = printed is not None and delta == 0
        all_reconciled = all_reconciled and reconciled
        year_rows.append(
            {
                "year": year,
                "printed_total": printed,
                "parsed_country_sum": parsed,
                "difference": delta,
                "reconciled": reconciled,
            }
        )
    units = sorted({value for value in table.units.values() if value})
    return {
        "edition": edition,
        "table_id": table.table_id,
        "section": table.section,
        "commodity_wmd": table.commodity_wmd,
        "commodity_label": table.commodity_label,
        "model_metal": MODEL_METAL_BY_COMMODITY.get(table.commodity_label, ""),
        "measurement_basis": table.measurement_basis,
        "header_pdf_page": table.header_page,
        "unit": " | ".join(units),
        "country_rows": table.country_rows,
        "parsed_records": table.parsed_records,
        "blank_cells": table.blank_cells,
        "unparsed_numeric_cells": table.unparsed_numeric_cells,
        "all_year_totals_reconciled": all_reconciled,
        "year_totals": year_rows,
    }


def parse_coordinates(path: Path, config: EditionConfig) -> ParseResult:
    visual_rows, page_count = iter_visual_rows(path, config)
    recent: list[VisualRow] = []
    records: list[dict] = []
    audits: list[dict] = []
    warnings: list[dict] = []
    section = ""
    active: ActiveTable | None = None
    table_counter = 0

    for row in visual_rows:
        text = row.text
        section_match = re.search(r"\b6\.4\.([1-5])\b", text)
        if section_match:
            section = f"6.4.{section_match.group(1)}"
            if section == "6.4.5":
                active = None

        if is_table_header(row, config.years):
            raw_heading = heading_from_recent(recent)
            table_counter += 1
            label = canonical_commodity(raw_heading)
            year_right, rem_right = extract_header_geometry(row, config.years)
            active = ActiveTable(
                table_id=table_counter,
                section=section,
                commodity_wmd=raw_heading,
                commodity_label=label,
                measurement_basis=measurement_basis(raw_heading, label),
                header_page=row.page,
                years=config.years,
                year_right=year_right,
                rem_right=rem_right,
            ) if section in {"6.4.1", "6.4.2", "6.4.3", "6.4.4"} else None
            recent.append(row)
            continue

        if active is not None:
            values, residual_text = parse_row_cells(row, active)
            year_text = {year: values.get(year, "") for year in config.years}
            parsed_year = {year: parse_integer(value) for year, value in year_text.items()}
            has_unit = any(value and parse_integer(value) is None for value in year_text.values())
            if has_unit and not active.units:
                for year in config.years:
                    raw = year_text[year]
                    if raw:
                        active.unit_raw[year] = raw
                        active.units[year] = normalize_unit(raw)
                recent.append(row)
                continue

            if normalized_token(residual_text) == "total":
                for year, number in parsed_year.items():
                    if number is not None:
                        active.total_values[year] = number
                audits.append(table_audit_row(active, config.edition))
                active = None
                recent.append(row)
                continue

            numeric_count = sum(number is not None for number in parsed_year.values())
            source_rem = values.get("rem", "")
            if numeric_count and residual_text and normalized_token(residual_text) not in {
                "abbreviations see explanation", "data"
            }:
                active.country_rows += 1
                active.blank_cells += len(config.years) - numeric_count
                active.unparsed_numeric_cells += sum(
                    bool(year_text[year]) and parsed_year[year] is None for year in config.years
                )
                flag, confidence = source_fields(source_rem)
                for year, number in parsed_year.items():
                    if number is None:
                        continue
                    unit_raw = active.unit_raw.get(year, "")
                    unit = active.units.get(year, normalize_unit(unit_raw))
                    records.append(
                        {
                            "commodity_wmd": active.commodity_wmd,
                            "commodity_label": active.commodity_label,
                            "model_metal": MODEL_METAL_BY_COMMODITY.get(active.commodity_label, ""),
                            "measurement_basis": active.measurement_basis,
                            "mine_origin_country_wmd": residual_text,
                            "year": year,
                            "production": number,
                            "unit": unit,
                            "unit_raw": unit_raw,
                            "source_rem": source_rem,
                            "source_flag": flag,
                            "source_confidence": confidence,
                            "source_edition": config.edition,
                            "source_pdf_page": row.page,
                            "value_status": "reported_zero" if number == 0 else "reported_positive",
                        }
                    )
                    active.parsed_sums[year] += number
                    active.parsed_records += 1
            elif source_rem and residual_text and not numeric_count:
                warnings.append(
                    {
                        "edition": config.edition,
                        "pdf_page": row.page,
                        "commodity_wmd": active.commodity_wmd,
                        "row_text": text,
                        "warning": "country-like row has source remark but no parseable production cell",
                    }
                )

        recent.append(row)
        if len(recent) > 20:
            recent = recent[-20:]

    if active is not None:
        warnings.append(
            {
                "edition": config.edition,
                "pdf_page": config.last_page,
                "commodity_wmd": active.commodity_wmd,
                "row_text": "",
                "warning": "table remained open at end of configured page range",
            }
        )
        audits.append(table_audit_row(active, config.edition))

    return ParseResult(config.edition, records, audits, warnings, page_count)


def key_duplicates(records: Iterable[dict]) -> list[tuple]:
    counts: dict[tuple, int] = defaultdict(int)
    for row in records:
        key = (
            row["source_edition"],
            row["commodity_label"],
            normalized_token(row["mine_origin_country_wmd"]),
            row["year"],
        )
        counts[key] += 1
    return [key for key, count in counts.items() if count > 1]
