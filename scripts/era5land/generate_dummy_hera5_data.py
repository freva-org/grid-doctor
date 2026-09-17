#!/usr/bin/env python3
"""Create a compact, valid ERA5-Land- and ERA5-like GRIB fixture tree for CLI testing.

The generated files mirror the source layouts consumed by ``heal-era5`` and
contain real GRIB1 messages written with ecCodes. Their deliberately coarse
12x9 regular-latitude/longitude grid maps to a very low HEALPix level while
remaining large enough to exercise the workflow's spatial-data validation.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
from eccodes import codes_grib_new_from_samples, codes_release, codes_set, codes_set_array, codes_write
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn


@dataclass(frozen=True)
class Variable:
    """GRIB metadata and deterministic payload parameters for one fixture variable."""

    name: str
    stream: str
    data_type: str
    parameter: int
    level_type: str
    levels: tuple[int, ...]
    base_value: float


VARIABLES = {
    "tas": Variable("tas", "sf", "fc", 167, "heightAboveGround", (2,), 273.15),
    "pr": Variable("pr", "sf", "fc", 228, "surface", (0,), 0.001),
    "zg": Variable("zg", "pl", "an", 129, "isobaricInhPa", (1000, 850, 500), 50_000.0),
}
E1_YEAR_RANGE = range(2001, 2007)
ERA5_DATASET_CODES = ("E5", "ET")
ERA5LAND_DATASET_CODE = "EL"
PROGRESS_DATASET_CODES = ("E1", "E5", "ET", "EL")
FREQUENCY_CODES = {"1hr": "1H", "day": "1D", "mon": "1M"}
GRID_NI = 12
GRID_NJ = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Fixture source root. heal-era5 receives this same path through --root.",
    )
    parser.add_argument(
        "--start",
        "--start-date",
        "--start-year",
        dest="start",
        default="2020",
        help="Inclusive start: YYYY, YYYY-MM, YYYYMMDD, or YYYY-MM-DD (default: 2020).",
    )
    parser.add_argument(
        "--end",
        "--end-date",
        "--end-year",
        dest="end",
        default=str(datetime.now(UTC).year),
        help="Inclusive end: YYYY, YYYY-MM, YYYYMMDD, or YYYY-MM-DD (default: current year).",
    )
    parser.add_argument(
        "--variables",
        default="tas,pr,zg",
        help="Comma-separated fixture variables; supported values: tas, pr, zg.",
    )
    parser.add_argument(
        "--frequencies",
        default="1hr,day,mon",
        help="Comma-separated fixture frequencies; supported values: 1hr, day, mon.",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Delete --root before generating; without this flag, add to the existing fixture tree.",
    )
    return parser.parse_args()


def parse_names(value: str, *, choices: Iterable[str], option: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    invalid = sorted(set(names).difference(choices))
    if not names or invalid:
        valid = ", ".join(choices)
        raise ValueError(f"{option} must contain one or more of: {valid}; invalid values: {', '.join(invalid)}")
    return names


def parse_date_bound(value: str, *, end: bool) -> date:
    """Parse a calendar bound, expanding partial dates to their full period."""

    try:
        if len(value) == 4 and value.isdecimal():
            year = int(value)
            return date(year, 12, 31) if end else date(year, 1, 1)
        if len(value) == 7 and value[4] == "-":
            year, month = (int(part) for part in value.split("-"))
            if end:
                next_month = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
                return next_month - timedelta(days=1)
            return date(year, month, 1)
        if len(value) == 8 and value.isdecimal():
            return datetime.strptime(value, "%Y%m%d").date()
        if len(value) == 10 and value[4] == value[7] == "-":
            return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date: {value}") from exc
    raise ValueError(f"Date must use YYYY, YYYY-MM, YYYYMMDD, or YYYY-MM-DD: {value}")


def dates_for_frequency(start_date: date, end_date: date, frequency: str) -> Iterable[datetime]:
    start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
    stop = datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC) + timedelta(days=1)
    if frequency == "1hr":
        current = start
        while current < stop:
            yield current
            current += timedelta(hours=1)
    elif frequency == "day":
        current = start
        while current < stop:
            yield current
            current += timedelta(days=1)
    elif frequency == "mon":
        current = datetime(start.year, start.month, 1, tzinfo=UTC)
        while current < stop:
            yield current
            current = datetime(current.year + int(current.month == 12), current.month % 12 + 1, 1, tzinfo=UTC)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")


def file_dates(start_date: date, end_date: date, frequency: str) -> Iterable[tuple[str, tuple[datetime, ...]]]:
    """Group selected timestamps into the daily, monthly, or annual source files."""

    label_formats = {"1hr": "%Y-%m-%d", "day": "%Y-%m", "mon": "%Y"}
    try:
        label_format = label_formats[frequency]
    except KeyError as exc:
        raise ValueError(f"Unsupported frequency: {frequency}") from exc

    groups: dict[str, list[datetime]] = {}
    for timestamp in dates_for_frequency(start_date, end_date, frequency):
        groups.setdefault(timestamp.strftime(label_format), []).append(timestamp)
    yield from ((label, tuple(timestamps)) for label, timestamps in groups.items())


def write_message(handle: object, variable: Variable, timestamp: datetime, level: int, sequence: int) -> None:
    """Append one minimal regular-grid GRIB2 message."""

    # The workflow inventory reads ERA-style GRIB1 P1/P2 fields, which the
    # GRIB1 sample provides while retaining a standard regular-latlon payload.
    gid = codes_grib_new_from_samples("regular_ll_sfc_grib1")
    try:
        codes_set(gid, "paramId", variable.parameter)
        codes_set(gid, "typeOfLevel", variable.level_type)
        codes_set(gid, "level", level)
        codes_set(gid, "gridType", "regular_ll")
        codes_set(gid, "Ni", GRID_NI)
        codes_set(gid, "Nj", GRID_NJ)
        codes_set(gid, "latitudeOfFirstGridPointInDegrees", 90.0)
        codes_set(gid, "longitudeOfFirstGridPointInDegrees", 0.0)
        codes_set(gid, "latitudeOfLastGridPointInDegrees", -90.0)
        codes_set(gid, "longitudeOfLastGridPointInDegrees", 330.0)
        codes_set(gid, "iDirectionIncrementInDegrees", 30.0)
        codes_set(gid, "jDirectionIncrementInDegrees", 22.5)
        codes_set(gid, "dataDate", int(timestamp.strftime("%Y%m%d")))
        codes_set(gid, "dataTime", timestamp.hour * 100)
        codes_set(gid, "stepType", "instant")
        codes_set(gid, "step", 0)
        values = variable.base_value + sequence + np.arange(GRID_NI * GRID_NJ, dtype=np.float64) / 1000
        codes_set_array(gid, "values", values)
        codes_write(gid, handle)
    finally:
        codes_release(gid)


def release_mtime(timestamps: tuple[datetime, ...], *, now: datetime) -> datetime:
    """Return a realistic EL release timestamp for one complete source file.

    EL data becomes permanent on the first day of the coverage-end month plus
    three months. Fixtures appear eight days into that month, matching the
    observed release cadence while ensuring the permanent eligibility test is
    satisfied. Data whose simulated release is still in the future retains the
    current timestamp and remains temporary.
    """

    coverage_end = timestamps[-1]
    month_index = coverage_end.month - 1 + 3
    year = coverage_end.year + month_index // 12
    month = month_index % 12 + 1
    simulated_release = datetime(year, month, 8, tzinfo=UTC)
    return min(simulated_release, now)


def write_file(
    path: Path,
    variable: Variable,
    timestamps: tuple[datetime, ...],
    *,
    modified_time: datetime | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for sequence, timestamp in enumerate(timestamps):
            for level in variable.levels:
                write_message(handle, variable, timestamp, level, sequence)
    if modified_time is not None:
        os.utime(path, (modified_time.timestamp(), modified_time.timestamp()))


def dataset_codes_for(variable: Variable, year: int) -> tuple[str, ...]:
    """Return fixture source collections available for a variable and year."""

    era5_codes = ERA5_DATASET_CODES
    if year in E1_YEAR_RANGE:
        era5_codes = ("E1", *era5_codes)
    if variable.stream == "sf":
        return (ERA5LAND_DATASET_CODE, *era5_codes)
    return era5_codes


def data_type_for(variable: Variable, dataset_code: str) -> str:
    """Return the source data type used by a variable in one collection."""

    # ERA5-Land supplies 2 m temperature as a forecast field, whereas the
    # ERA5 E1/E5/ET collections supply it as an analysis field.
    if variable.name == "tas" and dataset_code != ERA5LAND_DATASET_CODE:
        return "an"
    return variable.data_type


def prepare_root(root: Path, *, from_scratch: bool) -> None:
    """Create the fixture root, optionally clearing its existing contents."""

    if not from_scratch:
        root.mkdir(parents=True, exist_ok=True)
        return
    if root == Path(root.anchor):
        raise ValueError("--from-scratch refuses to delete the filesystem root")
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"--root must be a directory: {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)


def main() -> int:
    args = parse_args()
    start_date = parse_date_bound(args.start, end=False)
    end_date = parse_date_bound(args.end, end=True)
    if start_date > end_date:
        raise ValueError("--start cannot be after --end")
    variables = tuple(VARIABLES[name] for name in parse_names(args.variables, choices=VARIABLES, option="--variables"))
    frequencies = parse_names(args.frequencies, choices=FREQUENCY_CODES, option="--frequencies")
    frequency_file_dates = {
        frequency: tuple(file_dates(start_date, end_date, frequency)) for frequency in frequencies
    }
    root = args.root.resolve()
    prepare_root(root, from_scratch=args.from_scratch)
    now = datetime.now(UTC)
    written = 0
    progress_totals = {
        dataset_code: sum(
            sum(dataset_code in dataset_codes_for(variable, timestamps[0].year) for variable in variables)
            for frequency in frequencies
            for _, timestamps in frequency_file_dates[frequency]
        )
        for dataset_code in PROGRESS_DATASET_CODES
    }
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        tasks = {
            dataset_code: progress.add_task(f"Writing {dataset_code} GRIB files", total=total)
            for dataset_code, total in progress_totals.items()
            if total
        }
        for frequency in frequencies:
            code = FREQUENCY_CODES[frequency]
            for variable in variables:
                # ERA5-Land supplies surface forecast fields only. The fixture
                # supplies every requested variable from each applicable source
                # tier. E1 covers only 2001--2006; E5 and provisional ET cover
                # every selected year.
                for label, timestamps in frequency_file_dates[frequency]:
                    for dataset_code in dataset_codes_for(variable, timestamps[0].year):
                        data_type = data_type_for(variable, dataset_code)
                        directory = root / dataset_code / variable.stream / data_type / code / str(variable.parameter)
                        filename = f"{dataset_code}{variable.stream}12_{code}_{label}_{variable.parameter}.grb"
                        modified_time = release_mtime(timestamps, now=now)
                        write_file(directory / filename, variable, timestamps, modified_time=modified_time)
                        written += 1
                        progress.advance(tasks[dataset_code])

    print(f"Wrote {written} GRIB files under {root}")
    print("Use --root with this directory and --highest-level-only for a compact HEALPix fixture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
