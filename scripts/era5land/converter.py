#! /usr/bin/env python3
"""Unified entry point for the ERA5/ERA5-Land conversion workflow."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from helpers.file_fetcher import (
    load_json,
    load_variable_requests,
    parse_interval,
    resolve_records,
    selected_variables,
    split_csv_list,
    unresolved_records,
)
from helpers.formatter import normalise_frequencies
from helpers.mapper import map_grib_to_healpix, update_healpix_attrs_only

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAR_TABLE = SCRIPT_DIR / "assets" / "default_variables.csv"
DEFAULT_SOURCE_MAPPER = SCRIPT_DIR / "assets" / "source_mapper.json"
DEFAULT_CMOR_TABLES = SCRIPT_DIR / "tables" / "era5-cmor-tables" / "Tables"
FREQUENCIES = ("1hr", "day", "mon", "fx")
UNRESOLVED_REASON = (
    "not found in CMOR table, unsupported stream/frequency, "
    "or has no DKRZ_ID/grib_paramID"
)


def parse_arg_list(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse a comma-separated CLI option."""

    if value is None:
        return None
    return split_csv_list(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    source_mapper = load_json(DEFAULT_SOURCE_MAPPER)

    parser = argparse.ArgumentParser(
        description="ERA5/ERA5-Land source discovery and conversion tools."
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch = subparsers.add_parser(
        "fetch-files",
        help="Resolve source GRIB files from the CMOR tables.",
    )
    fetch.add_argument("--dataset", choices=("era5land", "era5"), default="era5land")
    fetch.add_argument("--var", dest="variables", help="Comma-separated variables.")
    fetch.add_argument(
        "--freq",
        default="all",
        help="Comma-separated frequencies: 1hr,day,mon,fx. Default: all.",
    )
    fetch.add_argument(
        "--interval",
        help="Date interval as yyyymmdd1,yyyymmdd2. Empty end means today.",
    )
    fetch.add_argument(
        "--root",
        help="Override /pool/data/ERA5 for tests or alternate mounts.",
    )
    fetch.add_argument(
        "--json",
        action="store_true",
        help="Print records, missing matches, and unresolved requests as JSON.",
    )
    fetch.add_argument(
        "--show-patterns",
        action="store_true",
        help="Print resolved glob patterns instead of matching files.",
    )
    fetch.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any resolved source has no matching files.",
    )

    convert = subparsers.add_parser(
        "convert-healpix",
        help="Resolve GRIB files and convert them to HEALPix Zarr pyramids.",
    )
    convert.add_argument("--dataset", choices=("era5land", "era5"), default="era5land")
    convert.add_argument("--var", dest="variables", help="Comma-separated variables.")
    convert.add_argument(
        "--freq",
        default="all",
        help="Comma-separated frequencies: 1hr,day,mon,fx. Default: all.",
    )
    convert.add_argument(
        "--interval",
        help="Date interval as yyyymmdd1,yyyymmdd2. Empty end means today.",
    )
    convert.add_argument(
        "--root",
        help="Override /pool/data/ERA5 for tests or alternate mounts.",
    )
    convert.add_argument(
        "--time-chunk",
        type=int,
        default=48,
        help="Optional time chunk size before remapping.",
    )
    convert.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        help="Zarr format version for the output pyramid.",
    )
    convert.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable cached GRIB inventories and cached multi-file opens.",
    )
    convert.set_defaults(use_cache=True)
    convert.add_argument(
        "--weights-dir",
        default=str(source_mapper["weights_path"]),
        help="Directory where HEALPix weight files are stored and reused.",
    )
    convert.add_argument(
        "--clean",
        action="store_true",
        help="Overwrite existing Zarr outputs instead of updating them incrementally.",
    )
    convert.add_argument(
        "--attrs-only",
        action="store_true",
        help="Refresh variable attrs on existing Zarr outputs without remapping data.",
    )
    return parser


def parse_frequencies(value: str) -> Tuple[str, ...]:
    """Parse and validate a frequency CLI option."""

    frequencies = (
        normalise_frequencies(FREQUENCIES)
        if value == "all"
        else normalise_frequencies(split_csv_list(value))
    )
    unknown_freqs = sorted(set(frequencies) - set(FREQUENCIES))
    if unknown_freqs:
        raise ValueError(f"Unsupported frequencies: {', '.join(unknown_freqs)}")
    return tuple(frequencies)


def selected_requests(
    *,
    dataset: str,
    variables: Optional[Tuple[str, ...]],
):
    """Resolve the requested variables for one dataset selection."""

    source_mapper = load_json(DEFAULT_SOURCE_MAPPER)
    dataset_codes = tuple(
        str(code) for code in source_mapper["datasets"][dataset]["priority"]
    )
    requests = selected_variables(
        load_variable_requests(DEFAULT_VAR_TABLE),
        allowed_codes=dataset_codes,
        variables=variables,
    )
    return source_mapper, requests


def run_fetch_files(args: argparse.Namespace) -> int:
    """Resolve source files and print either JSON records or paths."""

    variables = parse_arg_list(args.variables)
    frequencies = parse_frequencies(args.freq)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)
    records = resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
        mapper_path=DEFAULT_SOURCE_MAPPER,
        dataset=args.dataset,
        variables=variables,
        frequencies=frequencies,
        interval=parse_interval(args.interval),
        root=args.root,
        glob_files=not args.show_patterns,
    )

    missing = [record for record in records if not record.files]
    unresolved = unresolved_records(requests, frequencies, records, UNRESOLVED_REASON)
    if args.strict and (missing or unresolved):
        for record in missing:
            print(
                f"missing: {record.variable} {record.frequency} {record.pattern}",
                file=sys.stderr,
            )
        for record in unresolved:
            print(
                f"unresolved: {record.variable} {record.frequency}: {record.reason}",
                file=sys.stderr,
            )
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "records": [record._asdict() for record in records],
                    "missing": [record._asdict() for record in missing],
                    "unresolved": [record._asdict() for record in unresolved],
                },
                indent=2,
            )
        )
        return 0

    if args.show_patterns:
        for record in records:
            print(record.pattern)
        return 0

    for record in records:
        for file in record.files:
            print(file)
    return 0


def run_convert_healpix(args: argparse.Namespace) -> int:
    """Resolve source files, remap them with grid_doctor, and write Zarr output."""

    variables = parse_arg_list(args.variables)
    frequencies = parse_frequencies(args.freq)
    interval = parse_interval(args.interval)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)

    records = resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
        mapper_path=DEFAULT_SOURCE_MAPPER,
        dataset=args.dataset,
        variables=variables,
        frequencies=frequencies,
        interval=interval,
        root=args.root,
        glob_files=True,
    )

    if args.attrs_only:
        update_healpix_attrs_only(
            records,
            frequencies=frequencies,
        )
        return 0

    map_grib_to_healpix(
        records,
        frequencies=frequencies,
        interval=interval,
        time_chunk=args.time_chunk,
        zarr_format=args.zarr_format,
        use_cache=args.use_cache,
        weights_dir=args.weights_dir,
        clean=args.clean,
    )

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ERA5-Land converter."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "fetch-files":
        return run_fetch_files(args)
    if args.command == "convert-healpix":
        return run_convert_healpix(args)
    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
