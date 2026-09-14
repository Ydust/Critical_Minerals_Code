"""Generate the checksum-locked critical-mineral route template.

The generator reproduces the historical method exactly: a representative-port
override, country-centre fallback, a strict same-continent 1,500 km overland
rule, searoute 1.6.0 paths, and inclusive vertex-in-rectangle chokepoint tags.
It writes only the six-column template consumed by downstream analyses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
import platform
from typing import NamedTuple

import networkx as nx
import pandas as pd
import searoute as sr
from searoute.classes.passages import Passage


HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
EXPECTED_SEAROUTE_VERSION = "1.6.0"
EXPECTED_PYTHON_VERSION = "3.12.13"
EXPECTED_PANDAS_VERSION = "3.0.3"
EXPECTED_NETWORKX_VERSION = "3.5"
EXPECTED_SEAROUTE_MODULE_SHA256 = (
    "4ab006ba097e2d01807ccbe6e215756ccab5338920bb47042db9599b467dcaf0"
)
EXPECTED_MARNET_SHA256 = (
    "3622fff283e71b940ba5d58280b07c5cb2e95959ffb48f647b39b500b6c50ea3"
)
EXPECTED_INPUT_SHA256 = {
    "corridor seed": "85cd1b2b04ac5e3d7e911a85baa098e900a671db23714a5eb76fe71b12dfe8fa",
    "country coordinates": "799b1b9cb17830893b44c0c2b336267c161226f06be9efb9b051b408ac5095f9",
    "representative ports": "d7e8409f18afa7cf64ae9d88f2081b6bdbb7caae4e08c0df2b96b32686422509",
    "chokepoint boxes": "d24df68e7e7361ae6f3829a768f0d1fe91b2b5f5d8078a67ddf9d1b7e979c787",
}
EXPECTED_SEED_ROWS = 29_714
EXPECTED_OUTPUT_ROWS = 26_682
EXPECTED_SKIPPED_ROWS = 3_032
EARTH_RADIUS_KM = 6_371.0
OVERLAND_THRESHOLD_KM = 1_500.0


class Chokepoint(NamedTuple):
    order: int
    label: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the frozen 26,682-row route template."
    )
    parser.add_argument(
        "--corridor-seed",
        type=Path,
        default=INPUTS / "route_corridor_seed_2021_2023.csv",
    )
    parser.add_argument(
        "--country-coordinates",
        type=Path,
        default=INPUTS / "nation_lat_lon.csv",
    )
    parser.add_argument(
        "--representative-ports",
        type=Path,
        default=INPUTS / "representative_ports.csv",
    )
    parser.add_argument(
        "--chokepoints",
        type=Path,
        default=INPUTS / "chokepoint_boxes.csv",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--fallback-log",
        type=Path,
        help="Optional CSV receiving missing-coordinate and searoute errors.",
    )
    return parser.parse_args()


def verify_route_engine() -> None:
    observed_runtime = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "networkx": nx.__version__,
        "searoute": version("searoute"),
    }
    expected_runtime = {
        "python": EXPECTED_PYTHON_VERSION,
        "pandas": EXPECTED_PANDAS_VERSION,
        "networkx": EXPECTED_NETWORKX_VERSION,
        "searoute": EXPECTED_SEAROUTE_VERSION,
    }
    if observed_runtime != expected_runtime:
        raise RuntimeError(
            f"Route-engine version mismatch: expected {expected_runtime}; "
            f"observed {observed_runtime}."
        )
    package_root = Path(sr.__file__).resolve().parent
    route_module = package_root / "searoute.py"
    marnet = package_root / "data" / "marnet_dict.py"
    for path in [route_module, marnet]:
        if not path.is_file():
            raise RuntimeError(f"Missing searoute asset: {path}")
    observed_module_hash = sha256(route_module)
    if observed_module_hash != EXPECTED_SEAROUTE_MODULE_SHA256:
        raise RuntimeError(
            "searoute module checksum mismatch: "
            f"expected {EXPECTED_SEAROUTE_MODULE_SHA256}, "
            f"observed {observed_module_hash}."
        )
    observed_hash = sha256(marnet)
    if observed_hash != EXPECTED_MARNET_SHA256:
        raise RuntimeError(
            "searoute marnet checksum mismatch: "
            f"expected {EXPECTED_MARNET_SHA256}, observed {observed_hash}."
        )


def verify_input(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    expected = EXPECTED_INPUT_SHA256[label]
    if observed != expected:
        raise ValueError(
            f"{label} checksum mismatch: expected {expected}, observed {observed}."
        )


def load_coordinates(
    path: Path,
) -> tuple[dict[str, tuple[float, float]], dict[str, str]]:
    data = pd.read_csv(path)
    required = {"iso3", "continent", "lat", "lon"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Coordinate file is missing columns: {missing}")
    data = data.dropna(subset=["iso3", "lat", "lon", "continent"])
    if data["iso3"].duplicated().any():
        raise ValueError("Coordinate file contains duplicate ISO3 codes.")
    coordinates = {
        str(row.iso3): (float(row.lon), float(row.lat))
        for row in data.itertuples(index=False)
    }
    continents = {
        str(row.iso3): str(row.continent) for row in data.itertuples(index=False)
    }
    return coordinates, continents


def load_ports(path: Path) -> dict[str, tuple[float, float]]:
    data = pd.read_csv(path)
    required = {"iso3", "lon", "lat"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Port file is missing columns: {missing}")
    if data["iso3"].duplicated().any():
        raise ValueError("Representative-port file contains duplicate ISO3 codes.")
    return {
        str(row.iso3): (float(row.lon), float(row.lat))
        for row in data.itertuples(index=False)
    }


def load_chokepoints(path: Path) -> list[Chokepoint]:
    data = pd.read_csv(path)
    required = {"order", "label", "xmin", "ymin", "xmax", "ymax"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Chokepoint file is missing columns: {missing}")
    data = data.sort_values("order", kind="stable")
    if data["order"].duplicated().any() or data["label"].duplicated().any():
        raise ValueError("Chokepoint order and labels must be unique.")
    return [
        Chokepoint(
            int(row.order),
            str(row.label),
            float(row.xmin),
            float(row.ymin),
            float(row.xmax),
            float(row.ymax),
        )
        for row in data.itertuples(index=False)
    ]


def haversine_km(origin: tuple[float, float], destination: tuple[float, float]) -> float:
    lon1, lat1, lon2, lat2 = map(
        radians,
        [origin[0], origin[1], destination[0], destination[1]],
    )
    value = sin((lat2 - lat1) / 2) ** 2 + (
        cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * asin(sqrt(value))


def chokepoint_labels(
    coordinates: list[list[float]], chokepoints: list[Chokepoint]
) -> list[str]:
    labels: list[str] = []
    for item in chokepoints:
        if any(
            item.xmin <= float(point[0]) <= item.xmax
            and item.ymin <= float(point[1]) <= item.ymax
            for point in coordinates
        ):
            labels.append(item.label)
    return labels


def write_fallback_log(path: Path | None, rows: list[dict[str, str]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["origin", "dest", "reason", "detail"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    verify_route_engine()
    for path, label in [
        (args.corridor_seed, "corridor seed"),
        (args.country_coordinates, "country coordinates"),
        (args.representative_ports, "representative ports"),
        (args.chokepoints, "chokepoint boxes"),
    ]:
        verify_input(path, label)

    seed = pd.read_csv(args.corridor_seed, low_memory=False)
    expected_columns = ["metal", "origin", "dest", "value_usd_raw"]
    if seed.columns.tolist() != expected_columns:
        raise ValueError(
            f"Corridor seed columns must be {expected_columns}; found {seed.columns.tolist()}."
        )
    if len(seed) != EXPECTED_SEED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SEED_ROWS:,} seed rows, found {len(seed):,}."
        )
    if seed.duplicated(["metal", "origin", "dest"]).any():
        raise ValueError("Corridor seed contains duplicate keys.")

    country_coordinates, continents = load_coordinates(args.country_coordinates)
    ports = load_ports(args.representative_ports)
    chokepoints = load_chokepoints(args.chokepoints)

    route_cache: dict[tuple[str, str], dict[str, object] | None] = {}
    failures: list[dict[str, str]] = []

    def coordinate(iso3: str) -> tuple[float, float] | None:
        return ports.get(iso3) or country_coordinates.get(iso3)

    def route(origin_iso: str, dest_iso: str) -> dict[str, object] | None:
        key = (origin_iso, dest_iso)
        if key in route_cache:
            return route_cache[key]
        origin = coordinate(origin_iso)
        destination = coordinate(dest_iso)
        if origin is None or destination is None:
            missing = origin_iso if origin is None else dest_iso
            failures.append(
                {
                    "origin": origin_iso,
                    "dest": dest_iso,
                    "reason": "missing_coordinate",
                    "detail": missing,
                }
            )
            result = None
        elif (
            continents.get(origin_iso) == continents.get(dest_iso)
            and haversine_km(origin, destination) < OVERLAND_THRESHOLD_KM
        ):
            result = {"mode": "overland", "chokepoints": []}
        else:
            try:
                geometry = sr.searoute(
                    origin,
                    destination,
                    units="km",
                    speed_knot=24,
                    append_orig_dest=False,
                    restrictions=[Passage.northwest],
                    include_ports=False,
                    port_params={},
                    return_passages=False,
                    algorithm=None,
                    backend="networkx",
                ).geometry["coordinates"]
                result = {
                    "mode": "sea",
                    "chokepoints": chokepoint_labels(geometry, chokepoints),
                }
            except Exception as exc:  # preserve the historical fallback, but audit it
                failures.append(
                    {
                        "origin": origin_iso,
                        "dest": dest_iso,
                        "reason": "searoute_exception",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
                result = {"mode": "sea_approx", "chokepoints": []}
        route_cache[key] = result
        return result

    output_rows: list[dict[str, object]] = []
    skipped_rows = 0
    for row in seed.itertuples(index=False):
        result = route(str(row.origin), str(row.dest))
        if result is None:
            skipped_rows += 1
            continue
        output_rows.append(
            {
                "metal": row.metal,
                "origin": row.origin,
                "dest": row.dest,
                "value_usd": round(float(row.value_usd_raw)),
                "mode": result["mode"],
                "chokepoints": "|".join(result["chokepoints"]),
            }
        )

    write_fallback_log(args.fallback_log, failures)
    route_exceptions = [
        item for item in failures if item["reason"] == "searoute_exception"
    ]
    if route_exceptions:
        raise RuntimeError(
            f"Route generation produced {len(route_exceptions):,} searoute failures; "
            "see --fallback-log and do not publish this output."
        )
    if skipped_rows != EXPECTED_SKIPPED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_SKIPPED_ROWS:,} seed rows to be excluded for "
            f"missing coordinates, observed {skipped_rows:,}."
        )
    if len(output_rows) != EXPECTED_OUTPUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_OUTPUT_ROWS:,} output rows, "
            f"found {len(output_rows):,}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(output_rows).to_csv(
        args.output,
        index=False,
        encoding="utf-8",
        # The frozen Windows artifact has one CRLF terminator per record.
        # Pinning this makes the validator meaningful on every host platform.
        lineterminator="\r\n",
    )
    print(
        f"Wrote {len(output_rows):,} rows to {args.output} "
        f"(sha256={sha256(args.output)})."
    )
    print(
        f"Excluded {skipped_rows:,} seed rows across "
        f"{sum(item['reason'] == 'missing_coordinate' for item in failures):,} "
        "unique origin-destination pairs with missing coordinates."
    )


if __name__ == "__main__":
    main()
