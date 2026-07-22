#! /usr/bin/env python3
"""Unified entry point for the ERA5/ERA5-Land conversion workflow."""

import argparse
import hashlib
import json
import logging
import os
from rich_argparse import RichHelpFormatter
import shutil
import signal
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

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

VERSION_SERIES = "2026.07"
VERSION_MAJOR = 0
VERSION_MINOR = 4
BETA_REVISION = 1
__version__ = f"{VERSION_SERIES}.{VERSION_MAJOR}.{VERSION_MINOR}b{BETA_REVISION}"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAR_TABLE = SCRIPT_DIR / "assets" / "default_variables.csv"
DEFAULT_SOURCE_MAPPER = SCRIPT_DIR / "assets" / "source_mapper.json"
DEFAULT_CMOR_TABLES = SCRIPT_DIR / "tables" / "era5-cmor-tables" / "Tables"
FREQUENCIES = ("1hr", "day", "mon", "fx")
BATCH_EXECUTION_MODES = ("subprocess", "inprocess")
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
_ACTIVE_BATCH_STATE_PATH: Optional[Path] = None


class RichDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    RichHelpFormatter,
):
    """Rich help with argparse default values."""


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


def parse_coarsen_levels(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse optional HEALPix levels for `--coarsen-only`.

    Accepts comma-separated integers like ``8,0`` and descending ranges like
    ``8-0``. Multiple comma-separated ranges may be combined.
    """

    if value in (None, "all"):
        return None

    levels: list[int] = []
    for token in split_csv_list(value):
        if "-" in token:
            start_text, end_text = token.split("-", maxsplit=1)
            try:
                start_level = int(start_text)
                end_level = int(end_text)
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported coarsen level range {token!r}; use values like 8-0."
                ) from exc
            if start_level < 0 or end_level < 0:
                raise ValueError("Coarsen levels must be non-negative integers.")
            if start_level < end_level:
                raise ValueError(
                    f"Unsupported ascending coarsen range {token!r}; use descending ranges like 8-0."
                )
            levels.extend(range(start_level, end_level - 1, -1))
            continue

        try:
            level = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported coarsen level {token!r}; use non-negative integers like 8,0 or ranges like 8-0."
            ) from exc
        if level < 0:
            raise ValueError("Coarsen levels must be non-negative integers.")
        levels.append(level)

    if not levels:
        raise ValueError("--coarsen-only requires at least one level when a value is provided.")
    return tuple(sorted(dict.fromkeys(levels), reverse=True))


def parse_truncate_after(value: Optional[str]) -> Optional[str]:
    """Parse an optional truncation cutoff date for existing Zarr stores.

    The accepted formats mirror ``--interval`` date tokens: ``YYYY``, ``YYYYMM``,
    ``YYYYMMDD`` and their hyphenated equivalents. The returned ISO date string is
    used as an inclusive upper bound when selecting timestamps to keep.
    """

    if value in (None, ""):
        return None
    start, end = parse_interval(f"{value},{value}")
    if start is None or end is None:
        raise ValueError("--truncate-after requires a bounded date value.")
    return start.isoformat()


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    source_mapper = load_json(DEFAULT_SOURCE_MAPPER)

    parser = argparse.ArgumentParser(
        description=f"ERA5/ERA5-Land source discovery and conversion tools (v{__version__})",
        formatter_class=RichDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v","--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show the converter version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch = subparsers.add_parser(
        "fetch-files",
        help="Resolve source GRIB files from the CMOR tables.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    fetch.add_argument(
        "--dataset",
        choices=("era5land", "era5"),
        default="era5land",
        help="Dataset to process.",
    )
    fetch.add_argument(
        "--var",
        dest="variables",
        default=None,
        help="Comma-separated variables.",
    )
    fetch.add_argument(
        "--freq",
        default="all",
        help="Comma-separated frequencies: 1hr,day,mon,fx.",
    )
    fetch.add_argument(
        "--interval",
        default=None,
        help=(
            "Date interval (START,END) where each date may be YYYY, YYYYMM, YYYYMMDD "
            "(hyphens optional). Empty END means today."
        ),
    )
    fetch.add_argument(
        "--root",
        default=None,
        help="Override /pool/data/ERA5 for tests or alternate mounts.",
    )
    fetch.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print records, missing matches, and unresolved requests as JSON.",
    )
    fetch.add_argument(
        "--show-patterns",
        action="store_true",
        default=False,
        help="Print resolved glob patterns instead of matching files.",
    )
    fetch.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero if any resolved source has no matching files.",
    )

    convert = subparsers.add_parser(
        "convert-healpix",
        help="Resolve GRIB files and convert them to HEALPix Zarr pyramids.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    convert.add_argument(
        "--dataset",
        choices=("era5land", "era5"),
        default="era5land",
        help="Dataset to process.",
    )
    convert.add_argument(
        "--var",
        dest="variables",
        default=None,
        help="Comma-separated variables.",
    )
    convert.add_argument(
        "--freq",
        default="all",
        help="Comma-separated frequencies: 1hr,day,mon,fx.",
    )
    convert.add_argument(
        "--interval",
        default=None,
        help=(
            "Date interval (START,END) where each date may be YYYY, YYYYMM, YYYYMMDD "
            "(hyphens optional). Empty END means today."
        ),
    )
    convert.add_argument(
        "--batches",
        type=int,
        default=None,
        metavar="MONTHS",
        help=(
            "Split the requested interval into sequential batches of N months "
            "and process each batch in a loop."
        ),
    )
    convert.add_argument(
        "--batch-mode",
        choices=BATCH_EXECUTION_MODES,
        default="subprocess",
        help=(
            "Execution mode for --batches. "
            "'subprocess' runs each batch in a fresh child process to release "
            "memory between intervals, while 'inprocess' keeps the legacy "
            "single-process loop."
        ),
    )
    convert.add_argument(
        "--root",
        default=None,
        help="Override /pool/data/ERA5 for tests or alternate mounts.",
    )
    convert.add_argument(
        "--output-path",
        default=None,
        help=(
            "Override the published HEALPix output root directory. "
            "Useful for test runs that should write outside the default location."
        ),
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
        default=True,
        help="Disable cached GRIB inventories.",
    )
    convert.add_argument(
        "--cache-input-datasets",
        action="store_true",
        dest="use_input_cache",
        default=False,
        help="Enable cached multi-file input dataset pickles.",
    )
    convert.add_argument(
        "--record-threads",
        action="store_true",
        dest="use_record_threads",
        default=False,
        help="Open source records in parallel within each frequency merge.",
    )
    convert.add_argument(
        "--weights-dir",
        default=str(source_mapper["weights_path"]),
        help="Directory where HEALPix weight files are stored and reused.",
    )
    convert.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Overwrite existing Zarr outputs instead of updating them incrementally.",
    )
    convert.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help="Delete the whole dataset output root before writing any new stores.",
    )
    convert.add_argument(
        "--truncate-after",
        default=None,
        metavar="DATE",
        help=(
            "Before updating an existing time-based Zarr store, remove timestamps "
            "strictly AFTER DATE. Accepts YYYY, YYYYMM, YYYYMMDD and hyphenated "
            "equivalents. Useful for stitching a rerun cleanly from a cutoff date. "
            "Does not work stand-alone."
        ),
    )
    convert.add_argument(
        "-ao",
        "--attrs-only",
        action="store_true",
        default=False,
        help="Refresh variable attrs on existing Zarr outputs without remapping data.",
    )

    mode_group = convert.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-hlo",
        "--highest-level-only",
        action="store_true",
        default=False,
        help="Only remap and write the finest HEALPix zoom level for each frequency.",
    )
    mode_group.add_argument(
        "-co",
        "--coarsen-only",
        nargs="?",
        const="all",
        default=None,
        metavar="LEVELS",
        help=(
            "Skip GRIB remapping and derive lower zoom levels from an existing "
            "higher-level Zarr store. Optionally provide comma-separated target "
            "levels such as 8,0, ranges such as 8-5, or a combination of both, "
            "such as 8-5,3-0. Without a value, all lower levels are rebuilt."
        ),
    )

    convert.add_argument(
        "-ps",
        "--pyramid-strategy",
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
    """Split one inclusive interval into inclusive month-sized batches."""

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


def build_batch_command(
    args: argparse.Namespace,
    *,
    interval: Tuple[Optional[date], Optional[date]],
    clean: bool,
) -> list[str]:
    """Build one isolated child-process command for a single batch interval.

    The child inherits the current Python interpreter and script path so it
    runs inside the same job allocation and environment while still releasing
    all batch-local memory on process exit.
    """

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "convert-healpix",
        "--dataset",
        args.dataset,
        "--freq",
        args.freq,
        "--interval",
        format_interval(interval),
        "--zarr-format",
        str(args.zarr_format),
        "--weights-dir",
        str(args.weights_dir),
        "--batch-mode",
        "inprocess",
        "--pyramid-strategy",
        args.pyramid_strategy,
    ]

    if args.variables is not None:
        command.extend(["--var", args.variables])
    if args.root is not None:
        command.extend(["--root", args.root])
    if args.output_path is not None:
        command.extend(["--output-path", args.output_path])
    if args.truncate_after is not None:
        command.extend(["--truncate-after", args.truncate_after])
    if not args.use_inventory_cache:
        command.append("--no-inventory-cache")
    if args.use_input_cache:
        command.append("--cache-input-datasets")
    if args.use_record_threads:
        command.append("--record-threads")
    if clean:
        command.append("--clean")
    if args.attrs_only:
        command.append("--attrs-only")
    if args.highest_level_only:
        command.append("--highest-level-only")
    if args.coarsen_only is not None:
        command.append("--coarsen-only")
        if args.coarsen_only != "all":
            command.append(args.coarsen_only)

    return command


def _batch_state_path() -> Path:
    """Return the writable per-job path used to persist active batch state."""

    global _ACTIVE_BATCH_STATE_PATH
    if _ACTIVE_BATCH_STATE_PATH is not None:
        return _ACTIVE_BATCH_STATE_PATH

    job_token = hashlib.sha256(
        f"{os.getpid()}:{Path.cwd()}:{SCRIPT_DIR}".encode("utf-8")
    ).hexdigest()[:12]
    filename = f".current_batch_pid.{job_token}.json"

    for candidate_dir in (Path.cwd(), SCRIPT_DIR):
        candidate_path = candidate_dir / filename
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            with candidate_path.open("w", encoding="utf-8") as handle:
                json.dump({}, handle)
            candidate_path.unlink()
            _ACTIVE_BATCH_STATE_PATH = candidate_path
            return candidate_path
        except OSError:
            continue

    fallback_path = SCRIPT_DIR / filename
    _ACTIVE_BATCH_STATE_PATH = fallback_path
    return fallback_path


def write_batch_state(state: dict[str, Any]) -> Path:
    """Persist the current batch state for external inspection or manual kill."""

    state_path = _batch_state_path()
    with state_path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return state_path


def clear_batch_state() -> None:
    """Remove the persisted batch state file when no child batch is active."""

    state_path = _batch_state_path()
    state_path.unlink(missing_ok=True)


def run_batched_subprocesses(
    args: argparse.Namespace,
    intervals: Sequence[Tuple[Optional[date], Optional[date]]],
) -> int:
    """Run each batch interval in a fresh child process on the same node."""

    logger = logging.getLogger(__name__)
    active_process: subprocess.Popen[str] | None = None
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def forward_signal(signum: int, _frame: object) -> None:
        """Forward termination signals to the active batch process group."""

        nonlocal active_process
        if active_process is not None and active_process.poll() is None:
            try:
                process_group_id = os.getpgid(active_process.pid)
                logger.warning(
                    "Forwarding signal %s to batch_pid=%s batch_pgid=%s",
                    signum,
                    active_process.pid,
                    process_group_id,
                )
                os.killpg(process_group_id, signum)
            except ProcessLookupError:
                logger.warning(
                    "Batch process already exited before signal %s could be forwarded.",
                    signum,
                )
        raise KeyboardInterrupt

    def _install_handler(
        signum: int,
        handler: Callable[[int, object], None],
    ) -> None:
        """Install one signal handler for the batched parent process."""

        signal.signal(signum, handler)

    _install_handler(signal.SIGINT, forward_signal)
    _install_handler(signal.SIGTERM, forward_signal)

    try:
        for index, current_interval in enumerate(intervals, start=1):
            logger.info(
                "Starting batch %s/%s for interval %s",
                index,
                len(intervals),
                format_interval(current_interval),
            )
            command = build_batch_command(
                args,
                interval=current_interval,
                clean=(args.clean and index == 1),
            )
            active_process = subprocess.Popen(
                command,
                start_new_session=True,
                text=True,
            )
            state_path = write_batch_state(
                {
                    "batch_index": index,
                    "batch_count": len(intervals),
                    "batch_interval": format_interval(current_interval),
                    "batch_mode": "subprocess",
                    "batch_pgid": active_process.pid,
                    "batch_pid": active_process.pid,
                    "command": command,
                    "parent_pid": os.getpid(),
                }
            )
            logger.info(
                "Launched isolated batch process %s/%s batch_pid=%s batch_pgid=%s state_file=%s",
                index,
                len(intervals),
                active_process.pid,
                active_process.pid,
                state_path,
            )
            return_code = active_process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            active_process = None
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        if active_process is not None and active_process.poll() is None:
            try:
                os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
        clear_batch_state()

    return 0


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
        effective_frequencies,
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
    truncate_after = parse_truncate_after(args.truncate_after)
    coarsen_levels = parse_coarsen_levels(args.coarsen_only)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, _ = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )

    if args.from_scratch and args.attrs_only:
        raise ValueError("--from-scratch cannot be combined with --attrs-only.")
    if truncate_after is not None and args.attrs_only:
        raise ValueError("--truncate-after cannot be combined with --attrs-only.")

    if args.highest_level_only and args.pyramid_strategy != "stepwise":
        logger.info(
            "Forcing pyramid strategy to 'stepwise' because --highest-level-only was requested."
        )
        args.pyramid_strategy = "stepwise"

    if args.coarsen_only is not None and args.pyramid_strategy != "lazy":
        logger.info(
            "Ignoring --pyramid-strategy=%s because --coarsen-only does not remap.",
            args.pyramid_strategy,
        )
        args.pyramid_strategy = "lazy"

    if args.from_scratch:
        root_path = dataset_output_root(args.dataset, output_path=args.output_path)
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
                output_path=args.output_path,
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
            coarsen_only=(args.coarsen_only is not None),
            coarsen_levels=coarsen_levels,
            output_path=args.output_path,
            coarsen_interval=interval,
            truncate_after=truncate_after,
        )

    intervals = batched_intervals(interval, batch_months=args.batches)

    if len(intervals) > 1:
        logger.info(
            "Processing %s interval batches of %s month(s) each using %s mode.",
            len(intervals),
            args.batches,
            args.batch_mode,
        )

    if len(intervals) > 1 and args.batch_mode == "subprocess":
        return run_batched_subprocesses(args, intervals)

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
