#! /usr/bin/env python3
"""Unified entry point for the ERA5/ERA5-Land conversion workflow."""

import argparse
import json
import logging
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
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
RESET_COLOR = "\033[0m"
LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[37m",
    logging.WARNING: "\033[93m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}
STAGE_COLORS = {
    "convert_start": "\033[1;36m",
    "frequency_start": "\033[1;94m",
    "grib_read_parallel": "\033[94m",
    "grib_merge_done": "\033[36m",
    "weight_calculation": "\033[93m",
    "remap_start": "\033[1;95m",
    "remap_materialize_done": "\033[95m",
    "coarsen_source_open": "\033[36m",
    "zarr_write_start": "\033[32m",
    "frequency_done": "\033[1;32m",
    "frequency_skip_empty": "\033[90m",
    "attrs_only": "\033[32m",
}


class StageColorFormatter(logging.Formatter):
    """Format log records with ANSI colors when writing to an interactive terminal."""

    def __init__(self, fmt: str, *, use_color: bool) -> None:
        """Store the base format and whether ANSI colors should be emitted."""

        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        """Return one formatted log line, optionally colorized by stage or level."""

        message = super().format(record)
        if not self.use_color:
            return message

        stage_name = self._stage_name(record)
        color = STAGE_COLORS.get(stage_name, LEVEL_COLORS.get(record.levelno))
        if color is None:
            return message
        return f"{color}{message}{RESET_COLOR}"

    @staticmethod
    def _stage_name(record: logging.LogRecord) -> Optional[str]:
        """Extract the structured stage token from the rendered log message."""

        message = record.getMessage()
        for token in message.split():
            if token.startswith("stage="):
                return token.split("=", 1)[1]
        return None


def configure_logging() -> None:
    """Configure terminal logging with ANSI colors for interactive stderr."""

    handler = logging.StreamHandler()
    handler.setFormatter(
        StageColorFormatter(
            LOG_FORMAT,
            use_color=sys.stderr.isatty(),
        )
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)


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
        "-ao","--attrs-only",
        action="store_true",
        help="Refresh variable attrs on existing Zarr outputs without remapping data.",
    )
    mode_group = convert.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-hlo","--highest-level-only",
        action="store_true",
        help="Only remap and write the finest HEALPix zoom level for each frequency.",
    )
    mode_group.add_argument(
        "-co","--coarsen-only",
        action="store_true",
        help="Skip GRIB remapping and derive lower zoom levels from an existing highest-level Zarr store.",
    )
    convert.add_argument(
        "-ps","--pyramid-strategy",
        choices=("lazy", "stepwise"),
        default="lazy",
        help=(
            "Build the HEALPix pyramid lazily with grid_doctor's default path, "
            "or materialize the highest zoom first and coarsen stepwise in memory."
        ),
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

    logger = logging.getLogger(__name__)
    variables = parse_arg_list(args.variables)
    frequencies = parse_frequencies(args.freq)
    interval = parse_interval(args.interval)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)

    if args.highest_level_only and args.pyramid_strategy != "stepwise":
        logger.info(
            "Forcing pyramid strategy to 'stepwise' because --highest-level-only was requested."
        )
        args.pyramid_strategy = "stepwise"
    if args.coarsen_only and args.pyramid_strategy != "lazy":
        logger.info(
            "Ignoring --pyramid-strategy=%s because --coarsen-only does not remap.",
            args.pyramid_strategy,
        )
        args.pyramid_strategy = "lazy"

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
        zarr_format=args.zarr_format,
        use_cache=args.use_cache,
        weights_dir=args.weights_dir,
        clean=args.clean,
        pyramid_strategy=args.pyramid_strategy,
        highest_level_only=args.highest_level_only,
        coarsen_only=args.coarsen_only,
    )

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ERA5-Land converter."""

    configure_logging()
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
