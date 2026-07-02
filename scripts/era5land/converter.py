#! /usr/bin/env python3
"""Unified entry point for the ERA5/ERA5-Land conversion workflow."""

import argparse
import json
import logging
import shutil
import sys
from datetime import date, timedelta
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
from helpers.special import split_special_variables
from helpers.formatter import dataset_output_root, normalise_frequencies
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
        "--batches",
        type=int,
        metavar="MONTHS",
        help=(
            "Split the requested interval into sequential batches of N months "
            "and process each batch in a loop."
        ),
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
        "--no-inventory-cache",
        action="store_false",
        dest="use_inventory_cache",
        help="Disable cached GRIB inventories.",
    )
    convert.set_defaults(use_inventory_cache=True)
    convert.add_argument(
        "--cache-input-datasets",
        action="store_true",
        dest="use_input_cache",
        help="Enable cached multi-file input dataset pickles.",
    )
    convert.set_defaults(use_input_cache=False)
    convert.add_argument(
        "--record-threads",
        action="store_true",
        dest="use_record_threads",
        help="Open source records in parallel within each frequency merge.",
    )
    convert.set_defaults(use_record_threads=False)
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
        "--from-scratch",
        action="store_true",
        help="Delete the whole dataset output root before writing any new stores.",
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
        default="stepwise",
        help=(
            "Build the HEALPix pyramid lazily with grid_doctor's default path, "
            "or materialize the highest zoom first and coarsen stepwise in memory."
        ),
    )
    return parser


def add_months(current: date, months: int) -> date:
    """Return ``current`` shifted forward by ``months`` calendar months."""

    year = current.year + (current.month - 1 + months) // 12
    month = (current.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def batched_intervals(
    interval: Tuple[Optional[date], Optional[date]],
    *,
    batch_months: Optional[int],
) -> Tuple[Tuple[Optional[date], Optional[date]], ...]:
    """Split one inclusive interval into inclusive month-sized batches.

    Args:
        interval: Inclusive ``(start, end)`` bounds used by the converter.
        batch_months: Number of calendar months per batch, or ``None`` to keep
            the original interval unchanged.

    Returns:
        A tuple containing one or more inclusive intervals.

    Raises:
        ValueError: If batching is requested without a fully bounded interval or
            with a non-positive month count.
    """

    if batch_months is None:
        return (interval,)
    if batch_months <= 0:
        raise ValueError("--batches must be a positive integer number of months.")

    start, end = interval
    if start is None or end is None:
        raise ValueError("--batches requires a bounded --interval with a start and end date.")

    intervals: list[Tuple[date, date]] = []
    current_start = start
    while current_start <= end:
        next_start = add_months(date(current_start.year, current_start.month, 1), batch_months)
        current_end = min(end, next_start - timedelta(days=1))
        intervals.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return tuple(intervals)


def format_interval(interval: Tuple[Optional[date], Optional[date]]) -> str:
    """Render one interval tuple for logs."""

    start, end = interval
    start_text = start.isoformat() if start is not None else ""
    end_text = end.isoformat() if end is not None else ""
    return f"{start_text},{end_text}"


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


def extend_frequencies_for_special_variables(
    frequencies: Tuple[str, ...],
    requested_variables: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Add the `fx` publication pass when special variables are requested."""

    _, special_variables = split_special_variables(requested_variables)
    if not special_variables or "fx" in frequencies:
        return frequencies
    return tuple((*frequencies, "fx"))


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
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, _ = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )
    records = resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
        mapper_path=DEFAULT_SOURCE_MAPPER,
        dataset=args.dataset,
        variables=source_variables,
        frequencies=frequencies,
        interval=parse_interval(args.interval),
        root=args.root,
        glob_files=not args.show_patterns,
    )

    missing = [record for record in records if not record.files]
    unresolved = unresolved_records(
        [request for request in requests if request.name in source_variables],
        frequencies,
        records,
        UNRESOLVED_REASON,
    )
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
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, _ = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )

    if args.from_scratch and args.attrs_only:
        raise ValueError("--from-scratch cannot be combined with --attrs-only.")

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

    if args.from_scratch:
        root_path = dataset_output_root(args.dataset)
        if root_path.exists():
            logger.warning("Deleting dataset output root %s", root_path)
            shutil.rmtree(root_path)
        else:
            logger.info("Dataset output root %s does not exist; nothing to delete.", root_path)

    def run_single_interval(
        current_interval: Tuple[Optional[date], Optional[date]],
        *,
        clean: bool,
    ) -> None:
        """Process one interval with the existing record-resolution pipeline."""

        records = resolve_records(
            var_table=DEFAULT_VAR_TABLE,
            cmor_tables_dir=DEFAULT_CMOR_TABLES,
            mapper_path=DEFAULT_SOURCE_MAPPER,
            dataset=args.dataset,
            variables=source_variables,
            frequencies=frequencies,
            interval=current_interval,
            root=args.root,
            glob_files=True,
        )

        if args.attrs_only:
            update_healpix_attrs_only(
                records,
                dataset=args.dataset,
                frequencies=effective_frequencies,
                requested_variables=requested_variable_names,
            )
            return

        map_grib_to_healpix(
            records,
            dataset=args.dataset,
            frequencies=effective_frequencies,
            requested_variables=requested_variable_names,
            interval=current_interval,
            zarr_format=args.zarr_format,
            use_inventory_cache=args.use_inventory_cache,
            use_input_cache=args.use_input_cache,
            use_record_threads=args.use_record_threads,
            weights_dir=args.weights_dir,
            clean=clean,
            pyramid_strategy=args.pyramid_strategy,
            highest_level_only=args.highest_level_only,
            coarsen_only=args.coarsen_only,
        )

    intervals = batched_intervals(interval, batch_months=args.batches)
    if len(intervals) > 1:
        logger.info(
            "Processing %s interval batches of %s month(s) each.",
            len(intervals),
            args.batches,
        )

    for index, current_interval in enumerate(intervals, start=1):
        if len(intervals) > 1:
            logger.info(
                "Starting batch %s/%s for interval %s",
                index,
                len(intervals),
                format_interval(current_interval),
            )
        run_single_interval(current_interval, clean=(args.clean and index == 1))

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
