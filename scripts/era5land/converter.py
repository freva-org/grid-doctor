#! /usr/bin/env python3
"""Unified entry point for the ERA5/ERA5-Land conversion workflow."""

import argparse
from glob import glob, has_magic
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
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple

from helpers.file_fetcher import (
    batched_source_record_files,
    load_json,
    load_variable_requests,
    parse_interval,
    resolve_records,
    selected_variables,
    split_csv_list,
    unresolved_records,
)
from helpers.formatter import dataset_output_root, normalise_frequencies
from helpers.zarr_publisher import merge_zarr_store_directories

VERSION_SERIES = "2026.07"
VERSION_MAJOR = 5
VERSION_MINOR = 3
BETA_REVISION = 1
__version__ = f"{VERSION_SERIES}.{VERSION_MAJOR}.{VERSION_MINOR}b{BETA_REVISION}"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_VAR_TABLE = SCRIPT_DIR / "assets" / "default_variables.csv"
DEFAULT_SOURCE_MAPPER = SCRIPT_DIR / "assets" / "source_mapper.json"
DEFAULT_CMOR_TABLES = SCRIPT_DIR / "tables" / "era5-cmor-tables" / "Tables"
DEFAULT_CHUNK_SIZE: int = 16
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
    "remap_start": "\033[1;36m",
    "frequency_start": "\033[1;94m",
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
_BATCH_FILES_CHILD_INDEX_ENV = "ERA5_BATCH_FILES_CHILD_INDEX"
_BATCH_FILES_CHILD_COUNT_ENV = "ERA5_BATCH_FILES_CHILD_COUNT"
_BATCH_FILES_INCLUDE_SPECIAL_ENV = "ERA5_BATCH_FILES_INCLUDE_SPECIAL"


class BatchFileChildState(NamedTuple):
    """Describe the selected file batch when running inside a child process."""

    index: int | None
    count: int | None
    include_special: bool


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


def parse_level_selection(value: Optional[str]) -> Optional[Tuple[int, ...]]:
    """Parse optional HEALPix level selections from CLI arguments."""

    return parse_coarsen_levels(value)


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
        help="Show the remapper version and exit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    fetch_cmd = subparsers.add_parser(
        "fetch",
        help="Resolve source GRIB files from the CMOR tables.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    fetch_cmd.add_argument(
        "--dataset",
        choices=("era5land", "era5"),
        default="era5land",
        help="Dataset to process.",
    )
    fetch_cmd.add_argument(
        "--var",
        dest="variables",
        default=None,
        help="Comma-separated variables.",
    )
    fetch_cmd.add_argument(
        "--freq",
        default="all",
        help="Comma-separated frequencies: 1hr,day,mon,fx.",
    )
    fetch_cmd.add_argument(
        "--interval",
        default=None,
        help=(
            "Date interval (START,END) where each date may be YYYY, YYYYMM, YYYYMMDD "
            "(hyphens optional). Empty END means today."
        ),
    )
    fetch_cmd.add_argument(
        "--root",
        default=None,
        help="Override /pool/data/ERA5 for tests or alternate mounts.",
    )
    fetch_cmd.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print records, missing matches, and unresolved requests as JSON.",
    )
    fetch_cmd.add_argument(
        "--show-patterns",
        action="store_true",
        default=False,
        help="Print resolved glob patterns instead of matching files.",
    )
    fetch_cmd.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero if any resolved source has no matching files.",
    )

    remap_cmd = subparsers.add_parser(
        "remap",
        help="Resolve GRIB files and remap them to HEALPix Zarr pyramids.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    remap_cmd.add_argument(
        "--dataset",
        choices=("era5land", "era5"),
        default="era5land",
        help="Dataset to process.",
    )
    remap_cmd.add_argument(
        "--var",
        dest="variables",
        default=None,
        help="Comma-separated variables.",
    )
    remap_cmd.add_argument(
        "--freq",
        default="all",
        help="Comma-separated frequencies: 1hr,day,mon,fx.",
    )
    remap_cmd.add_argument(
        "--interval",
        default=None,
        help=(
            "Date interval (START,END) where each date may be YYYY, YYYYMM, YYYYMMDD "
            "(hyphens optional). Empty END means today."
        ),
    )
    remap_cmd.add_argument(
        "--batch-months",
        dest="batch_months",
        type=int,
        default=None,
        metavar="MONTHS",
        help=(
            "Split the requested interval into sequential batches of N months "
            "and process each batch in a loop."
        ),
    )
    remap_cmd.add_argument(
        "--batch-files",
        type=int,
        default=None,
        metavar="FILES",
        help=(
            "Split the requested interval into sequential batches of N files"
            "and process each batch in a loop. When set, this replaces "
            "calendar-month batching for remap execution."
        ),
    )
    remap_cmd.add_argument(
        "--root",
        default=None,
        help="Override /pool/data/ERA5 for tests or alternate mounts.",
    )
    remap_cmd.add_argument(
        "--output-path",
        default=None,
        help=(
            "Override the published HEALPix output root directory. "
            "Useful for test runs that should write outside the default location."
        ),
    )
    remap_cmd.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        help="Zarr format version for the output pyramid.",
    )
    remap_cmd.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="MB",
        help=(
            "Approximate Zarr chunk-size target in megabytes for newly written "
            "or fully rewritten stores."
        ),
    )
    remap_cmd.add_argument(
        "--no-cache",
        "--no-inventory-cache",
        action="store_false",
        dest="use_inventory_cache",
        default=True,
        help="Disable cached GRIB inventories.",
    )
    remap_cmd.add_argument(
        "--cache-input-datasets",
        action="store_true",
        dest="use_input_cache",
        default=False,
        help="Enable cached multi-file input dataset pickles.",
    )
    remap_cmd.add_argument(
        "-fdt","--fail-on-duplicate-times",
        action="store_true",
        dest="fail_on_duplicate_times",
        default=False,
        help=(
            "Raise an error when exact duplicate GRIB time rows are found during "
            "time normalization instead of dropping them. This finishes the run."
        ),
    )
    remap_cmd.add_argument(
        "--weights-dir",
        default=str(source_mapper["weights_path"]),
        help="Directory where HEALPix weight files are stored and reused.",
    )
    remap_cmd.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help=(
            "Overwrite existing Zarr stores with new outputs instead of updating "
            "them incrementally. It wipes the store before starting."
              ),
    )
    remap_cmd.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help="Delete the whole dataset output root before writing any new stores.",
    )
    remap_cmd.add_argument(
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
    remap_cmd.add_argument(
        "--rechunk-only",
        action="store_true",
        default=False,
        help=(
            "Rewrite matching existing Zarr stores using the current "
            "--chunk-size target and then exit without remaping."
        ),
    )
    remap_cmd.add_argument(
        "-ao",
        "--attrs-only",
        action="store_true",
        default=False,
        help="Refresh variable attrs on existing Zarr outputs without remapping data.",
    )

    mode_group = remap_cmd.add_mutually_exclusive_group()
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
            "such as 8-5,3-0. Without a value, all lower levels are rebuilt. "
            "This can be combined with --interval."
        ),
    )

    reflow_cmd = subparsers.add_parser(
        "remap-reflow",
        help="Run the scheduler-backed Reflow remap workflow.",
        formatter_class=RichDefaultsHelpFormatter,
        description=(
            "Forward commands to the ERA5/ERA5-Land Reflow workflow. "
            "Examples: `remap-reflow submit ...`, `remap-reflow runs`, "
            "`remap-reflow status <run-id>`."
        ),
    )
    reflow_cmd.add_argument(
        "reflow_args",
        nargs=argparse.REMAINDER,
        help=(
            "Arguments forwarded to scripts/era5land/helpers/reflow_workflow.py. "
            "For example: `submit --run-dir /scratch/$USER/era5land ...`."
        ),
    )

    clean_cmd = subparsers.add_parser(
        "clean",
        help="Remove variables, levels, frequencies, or the whole HEALPix output root.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    clean_cmd.add_argument(
        "--dataset",
        choices=("era5land", "era5"),
        default="era5land",
        help="Dataset to clean.",
    )
    clean_cmd.add_argument(
        "--var",
        dest="variables",
        default=None,
        help="Comma-separated variables to remove from matching stores.",
    )
    clean_cmd.add_argument(
        "--freq",
        default=None,
        help="Comma-separated frequencies to target: 1hr,day,mon,fx.",
    )
    clean_cmd.add_argument(
        "--levels",
        default=None,
        metavar="LEVELS",
        help=(
            "Optional comma-separated or descending-range level selection such as "
            "8,0 or 8-5. When omitted, all existing levels for each selected "
            "frequency are targeted."
        ),
    )
    clean_cmd.add_argument(
        "--output-path",
        default=None,
        help="Override the published HEALPix output root directory.",
    )
    clean_cmd.add_argument(
        "--truncate-after",
        default=None,
        metavar="DATE",
        help=(
            "Truncate existing time-based stores, removing timestamps strictly "
            "after DATE."
        ),
    )
    clean_cmd.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be removed without changing anything.",
    )

    merge_cmd = subparsers.add_parser(
        "merge",
        help="Merge one or more frequency directories into a target frequency directory.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    merge_cmd.add_argument(
        "--source",
        dest="source_dirs",
        nargs="+",
        required=True,
        help=(
            "Source directories or glob patterns that directly contain "
            "`level_*.zarr` stores. Values may be comma-separated or repeated."
        ),
    )
    merge_cmd.add_argument(
        "--output-path",
        required=True,
        help=(
            "Target directory that directly contains the merged `level_*.zarr` stores."
        ),
    )
    merge_cmd.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        help="Zarr format version for the destination stores.",
    )
    merge_cmd.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="MB",
        help=(
            "Approximate Zarr chunk-size target in megabytes for rewritten "
            "destination stores."
        ),
    )
    merge_cmd.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help=(
            "Recreate each touched destination store on the first merge instead "
            "of updating it incrementally."
        ),
    )
    merge_cmd.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help=(
            "Delete the complete target directory before merging. This is "
            "broader than --clean."
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
        raise ValueError("--batch-months must be a positive integer number of months.")

    start, end = interval
    if start is None or end is None:
        raise ValueError(
            "--batch-months requires a bounded --interval with a start and end date."
        )

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
        "remap",
        "--dataset",
        args.dataset,
        "--freq",
        args.freq,
        "--interval",
        format_interval(interval),
        "--zarr-format",
        str(args.zarr_format),
        "--chunk-size",
        str(args.chunk_size),
        "--weights-dir",
        str(args.weights_dir),
    ]

    if args.variables is not None:
        command.extend(["--var", args.variables])
    if args.root is not None:
        command.extend(["--root", args.root])
    if args.output_path is not None:
        command.extend(["--output-path", args.output_path])
    if args.batch_files is not None:
        command.extend(["--batch-files", str(args.batch_files)])
    if not args.use_inventory_cache:
        command.append("--no-inventory-cache")
    if args.use_input_cache:
        command.append("--cache-input-datasets")
    if args.fail_on_duplicate_times:
        command.append("--fail-on-duplicate-times")
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


def run_batched_months(
    args: argparse.Namespace,
    intervals: Sequence[Tuple[Optional[date], Optional[date]]],
) -> int:
    """Run each interval batch in a fresh child process on the same node."""

    batch_months = args.batch_months if args.batch_months is not None else 1
    batch_plans = [
        {
            "batch_index": index,
            "batch_count": len(intervals),
            "batch_label": interval_batch_label(
                current_interval,
                batch_index=index,
                batch_count=len(intervals),
                batch_months=batch_months,
            ),
            "batch_interval": format_interval(current_interval),
            "command_interval": current_interval,
            "env": None,
        }
        for index, current_interval in enumerate(intervals, start=1)
    ]
    return _run_subprocess(args, batch_plans)


def _run_subprocess(
    args: argparse.Namespace,
    batch_plans: Sequence[dict[str, Any]],
) -> int:
    """Run one or more prepared batch plans in isolated child subprocesses."""

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
        for batch_plan in batch_plans:
            batch_label = str(batch_plan["batch_label"])
            logger.info(
                "🚀 Starting batch %s",
                batch_label,
            )
            command = build_batch_command(
                args,
                interval=batch_plan["command_interval"],
                clean=(args.clean and int(batch_plan["batch_index"]) == 1),
            )
            child_env = (
                os.environ.copy()
                if batch_plan.get("env")
                else None
            )
            if child_env is not None:
                child_env.update(
                    {str(key): str(value) for key, value in dict(batch_plan["env"]).items()}
                )
            active_process = subprocess.Popen(
                command,
                start_new_session=True,
                text=True,
                env=child_env,
            )
            state = dict(batch_plan)
            state.pop("command_interval", None)
            state.pop("env", None)
            state.update(
                {
                    "batch_pgid": active_process.pid,
                    "batch_pid": active_process.pid,
                    "command": command,
                    "parent_pid": os.getpid(),
                }
            )
            state_path = write_batch_state(state)
            logger.info(
                "Launched isolated batch process %s batch_pid=%s batch_pgid=%s state_file=%s",
                batch_label,
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


def run_batched_files(
    args: argparse.Namespace,
    *,
    current_interval: Tuple[Optional[date], Optional[date]],
    batch_labels: Sequence[str],
) -> int:
    """Run each file-count batch in a fresh child process on the same node."""

    batch_count = len(batch_labels)
    batch_plans = [
        {
            "batch_index": index,
            "batch_count": batch_count,
            "batch_label": batch_label,
            "batch_interval": format_interval(current_interval),
            "batch_mode": "files",
            "command_interval": current_interval,
            "env": {
                _BATCH_FILES_CHILD_INDEX_ENV: index,
                _BATCH_FILES_CHILD_COUNT_ENV: batch_count,
                _BATCH_FILES_INCLUDE_SPECIAL_ENV: 1 if index == batch_count else 0,
            },
        }
        for index, batch_label in enumerate(batch_labels, start=1)
    ]
    return _run_subprocess(args, batch_plans)


def parse_cli_freqs(value: str) -> Tuple[str, ...]:
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

    from helpers.special import split_special_variables

    _, special_variables = split_special_variables(requested_variables)
    if not special_variables or "fx" in frequencies:
        return frequencies
    return tuple((*frequencies, "fx"))


def selected_requests(
    *,
    dataset: str,
    variables: Optional[Tuple[str, ...]],
    var_table: Path = DEFAULT_VAR_TABLE,
):
    """Resolve the requested variables for one dataset selection."""

    source_mapper = load_json(DEFAULT_SOURCE_MAPPER)
    dataset_codes = tuple(
        str(code) for code in source_mapper["datasets"][dataset]["priority"]
    )
    requests = selected_variables(
        load_variable_requests(var_table),
        allowed_codes=dataset_codes,
        variables=variables,
    )
    return source_mapper, requests


def parse_cli_args(value: Optional[str]) -> Optional[Tuple[str, ...]]:
    """Parse a comma-separated CLI option."""

    if value is None:
        return None
    return split_csv_list(value)


def expand_source_dirs(values: Sequence[str]) -> Tuple[str, ...]:
    """Expand comma-separated source paths and glob patterns for ``merge``."""

    sources: list[str] = []
    for value in values:
        parsed = parse_cli_args(value)
        if parsed:
            sources.extend(parsed)

    expanded: list[str] = []
    for source in sources:
        matches = sorted(glob(source)) if has_magic(source) else []
        if has_magic(source) and not matches:
            raise ValueError(f"Source pattern did not match any paths: {source}")
        expanded.extend(matches or [source])

    return tuple(dict.fromkeys(expanded))


def validate_remap_args(args: argparse.Namespace) -> None:
    """Validate remap options that cannot be expressed by argparse alone."""

    if args.from_scratch and args.attrs_only:
        raise ValueError("--from-scratch cannot be combined with --attrs-only.")
    if args.truncate_after is not None and args.attrs_only:
        raise ValueError("--truncate-after cannot be combined with --attrs-only.")
    if args.rechunk_only and args.attrs_only:
        raise ValueError("--rechunk-only cannot be combined with --attrs-only.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")
    if args.batch_months is not None and args.batch_files is not None:
        raise ValueError("--batch-months and --batch-files are mutually exclusive.")
    if args.batch_files is not None and args.batch_files <= 0:
        raise ValueError("--batch-files must be a positive integer.")
    if args.batch_files is not None and args.attrs_only:
        raise ValueError("--batch-files cannot be combined with --attrs-only.")
    if args.batch_files is not None and args.coarsen_only is not None:
        raise ValueError("--batch-files cannot be combined with --coarsen-only.")
    if args.batch_files is not None and args.rechunk_only:
        raise ValueError("--batch-files cannot be combined with --rechunk-only.")


def batch_file_child_state() -> BatchFileChildState:
    """Read and validate the environment used to select one file-batch child."""

    index_text = os.environ.get(_BATCH_FILES_CHILD_INDEX_ENV)
    count_text = os.environ.get(_BATCH_FILES_CHILD_COUNT_ENV)
    if (index_text is None) != (count_text is None):
        raise ValueError(
            "Internal file-batch child environment is incomplete; "
            "expected both child index and child count."
        )

    return BatchFileChildState(
        index=int(index_text) if index_text is not None else None,
        count=int(count_text) if count_text is not None else None,
        include_special=(
            os.environ.get(_BATCH_FILES_INCLUDE_SPECIAL_ENV, "0") == "1"
        ),
    )


def map_records(
    records: Sequence[Any],
    *,
    args: argparse.Namespace,
    frequencies: Tuple[str, ...],
    requested_variables: Tuple[str, ...],
    interval: Tuple[Optional[date], Optional[date]],
    clean: bool,
    coarsen_only: bool = False,
    coarsen_levels: Optional[Tuple[int, ...]] = None,
    use_input_cache: Optional[bool] = None,
    drop_duplicate_time_rows: Optional[bool] = None,
    coarsen_interval: Optional[Tuple[Optional[date], Optional[date]]] = None,
) -> None:
    """Map records using the common remap settings from the CLI namespace.

    Batch and special-variable paths use this adapter with narrower frequency or
    variable selections while inheriting the same output and caching settings.
    """

    from helpers.mapper import map_grib_to_healpix

    map_grib_to_healpix(
        list(records),
        dataset=args.dataset,
        frequencies=frequencies,
        requested_variables=requested_variables,
        interval=interval,
        zarr_format=args.zarr_format,
        use_inventory_cache=args.use_inventory_cache,
        use_input_cache=(
            args.use_input_cache if use_input_cache is None else use_input_cache
        ),
        drop_duplicate_time_rows=(
            not args.fail_on_duplicate_times
            if drop_duplicate_time_rows is None
            else drop_duplicate_time_rows
        ),
        weights_dir=args.weights_dir,
        clean=clean,
        target_chunk_mb=args.chunk_size,
        highest_level_only=args.highest_level_only,
        coarsen_only=coarsen_only,
        coarsen_levels=coarsen_levels,
        output_path=args.output_path,
        coarsen_interval=interval if coarsen_interval is None else coarsen_interval,
        truncate_after=None,
    )


def build_file_batch_plan(
    records: Sequence[Any],
    *,
    batch_files: int,
    fallback_interval: Tuple[Optional[date], Optional[date]],
) -> list[tuple[Any, Tuple[Optional[date], Optional[date]], str]]:
    """Expand resolved records into a stable global file-batch execution plan."""

    plan: list[tuple[Any, Tuple[Optional[date], Optional[date]], str]] = []
    for record in records:
        record_batches = batched_source_record_files(
            record,
            batch_files=batch_files,
            fallback_interval=fallback_interval,
        )
        batch_count = len(record_batches)
        for batch_index, (batched_record, batch_interval) in enumerate(
            record_batches,
            start=1,
        ):
            label = (
                f"{batched_record.variable} {batched_record.frequency} "
                f"{batch_index}/{batch_count} {format_interval(batch_interval)} "
                f"({len(batched_record.files)} file{'s' if len(batched_record.files) != 1 else ''})"
            )
            plan.append((batched_record, batch_interval, label))
    return plan


def interval_batch_label(
    interval: Tuple[Optional[date], Optional[date]],
    *,
    batch_index: int,
    batch_count: int,
    batch_months: int,
) -> str:
    """Return one human-readable label for an interval batch."""

    month_text = "month" if batch_months == 1 else "months"
    return (
        f"interval {batch_index}/{batch_count} "
        f"{format_interval(interval)} ({batch_months} {month_text})"
    )


def run_fetch(args: argparse.Namespace) -> int:
    """Resolve source files and print either JSON records or paths."""

    from helpers.special import split_special_variables

    variables = parse_cli_args(args.variables)
    frequencies = parse_cli_freqs(args.freq)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, special_variables = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )

    records = resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
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


def run_remap(args: argparse.Namespace) -> int:
    """Resolve source files, remap them with grid_doctor, and write Zarr output."""

    from helpers.cleanup import truncate_existing_healpix_stores
    from helpers.mapper import (
        rechunk_existing_healpix_stores,
        update_healpix_attrs_only,
    )
    from helpers.special import split_special_variables

    logger = logging.getLogger(__name__)
    variables = parse_cli_args(args.variables)
    frequencies = parse_cli_freqs(args.freq)
    interval = parse_interval(args.interval)
    truncate_after = parse_truncate_after(args.truncate_after)
    coarsen_levels = parse_coarsen_levels(args.coarsen_only)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)
    requested_variable_names = tuple(request.name for request in requests)
    source_variables, special_variables = split_special_variables(requested_variable_names)
    effective_frequencies = extend_frequencies_for_special_variables(
        frequencies,
        requested_variable_names,
    )

    validate_remap_args(args)
    file_child = batch_file_child_state()

    if args.from_scratch:
        root_path = dataset_output_root(args.dataset, output_path=args.output_path)
        if root_path.exists():
            logger.warning("Deleting dataset output root %s", root_path)
            shutil.rmtree(root_path)
        else:
            logger.info("Dataset output root %s does not exist; nothing to delete.", root_path)

    if truncate_after is not None:
        truncated_count = truncate_existing_healpix_stores(
            dataset=args.dataset,
            frequencies=effective_frequencies,
            zarr_format=args.zarr_format,
            cutoff=truncate_after,
            highest_level_only=args.highest_level_only,
            output_path=args.output_path,
        )
        logger.info(
            "Completed pre-run truncation after %s for %s existing store(s).",
            truncate_after,
            truncated_count,
        )

    if args.rechunk_only:
        rechunked_count = rechunk_existing_healpix_stores(
            dataset=args.dataset,
            frequencies=effective_frequencies,
            zarr_format=args.zarr_format,
            target_chunk_mb=args.chunk_size,
            highest_level_only=args.highest_level_only,
            output_path=args.output_path,
        )
        logger.info(
            "Completed standalone rechunking with target chunk size %s MB for %s existing store(s).",
            args.chunk_size,
            rechunked_count,
        )
        return 0

    def run_single_interval(
        current_interval: Tuple[Optional[date], Optional[date]],
        *,
        clean: bool,
    ) -> None:
        """Process one interval with the existing record-resolution pipeline."""

        records = resolve_records(
            var_table=DEFAULT_VAR_TABLE,
            cmor_tables_dir=DEFAULT_CMOR_TABLES,
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

        if args.batch_files is not None:
            batched_records = build_file_batch_plan(
                records,
                batch_files=args.batch_files,
                fallback_interval=current_interval,
            )

            if not batched_records:
                map_records(
                    records,
                    args=args,
                    frequencies=effective_frequencies,
                    requested_variables=requested_variable_names,
                    interval=current_interval,
                    clean=clean,
                    coarsen_interval=interval,
                )
                return

            if (
                file_child.index is None
                and file_child.count is None
            ):
                logger.info(
                    "📦 Processing %s file batches of up to %s file(s) each using isolated subprocesses.",
                    len(batched_records),
                    args.batch_files,
                )
                return_code = run_batched_files(
                    args,
                    current_interval=current_interval,
                    batch_labels=[label for _, _, label in batched_records],
                )
                if return_code != 0:
                    raise SystemExit(return_code)
                return

            selected_batches = batched_records
            if file_child.index is not None:
                batch_count = len(batched_records)
                if file_child.count != batch_count:
                    raise ValueError(
                        "Resolved file-batch count does not match the parent batch plan: "
                        f"expected {file_child.count}, got {batch_count}."
                    )
                child_index = file_child.index
                if child_index < 1 or child_index > batch_count:
                    raise ValueError(
                        f"--batch-files-child-index must be between 1 and {batch_count}."
                    )
                selected_batches = [batched_records[child_index - 1]]

            clean_remaining = clean
            for batched_record, batch_interval, batch_label in selected_batches:
                logger.info(
                    "📦 Processing batch %s",
                    batch_label,
                )
                map_records(
                    [batched_record],
                    args=args,
                    frequencies=(batched_record.frequency,),
                    requested_variables=(batched_record.variable,),
                    interval=batch_interval,
                    clean=clean_remaining,
                    coarsen_interval=interval,
                )
                clean_remaining = False

            if special_variables and (
                file_child.index is None or file_child.include_special
            ):
                map_records(
                    [],
                    args=args,
                    frequencies=("fx",),
                    requested_variables=special_variables,
                    interval=current_interval,
                    clean=False,
                    use_input_cache=False,
                    drop_duplicate_time_rows=True,
                    coarsen_interval=interval,
                )
            return

        map_records(
            records,
            args=args,
            frequencies=effective_frequencies,
            requested_variables=requested_variable_names,
            interval=current_interval,
            clean=clean,
            coarsen_only=(args.coarsen_only is not None),
            coarsen_levels=coarsen_levels,
            coarsen_interval=interval,
        )

    intervals = (
        (interval,)
        if args.batch_files is not None
        else batched_intervals(interval, batch_months=args.batch_months)
    )

    if len(intervals) > 1:
        logger.info(
            "📦 Processing %s batches using isolated subprocesses.",
            len(intervals),
        )
        return run_batched_months(args, intervals)

    run_single_interval(intervals[0], clean=args.clean)

    return 0


def run_clean(args: argparse.Namespace) -> int:
    """Clean existing HEALPix outputs at variable, level, frequency, or root scope."""

    from helpers.cleanup import (
        delete_dataset_root,
        delete_frequency_directory,
        delete_frequency_level_stores,
        remove_variables_from_frequency_stores,
        truncate_existing_healpix_stores,
    )

    logger = logging.getLogger(__name__)
    variables = parse_cli_args(args.variables)
    levels = parse_level_selection(args.levels)
    frequencies = parse_cli_freqs(args.freq or "all")

    if args.truncate_after is not None:
        if variables is not None or levels is not None:
            raise ValueError(
                "--truncate-after cannot be combined with --var or --levels."
            )
        if args.dry_run:
            raise ValueError("--truncate-after does not support --dry-run.")

        cutoff = parse_truncate_after(args.truncate_after)
        if cutoff is None:
            raise ValueError("--truncate-after requires a bounded date value.")
        truncated_count = truncate_existing_healpix_stores(
            dataset=args.dataset,
            frequencies=frequencies,
            cutoff=cutoff,
            output_path=args.output_path,
        )
        logger.info(
            "Truncated %s existing store(s) after %s.",
            truncated_count,
            cutoff,
        )
        return 0

    actions: list[str] = []

    if variables is None and levels is None and args.freq is None:
        actions.extend(
            delete_dataset_root(
                dataset=args.dataset,
                output_path=args.output_path,
                dry_run=args.dry_run,
            )
        )
    elif variables is None and levels is None:
        for frequency in frequencies:
            actions.extend(
                delete_frequency_directory(
                    dataset=args.dataset,
                    frequency=frequency,
                    output_path=args.output_path,
                    dry_run=args.dry_run,
                )
            )
    else:
        for frequency in frequencies:
            if variables:
                actions.extend(
                    remove_variables_from_frequency_stores(
                        dataset=args.dataset,
                        frequency=frequency,
                        variable_names=variables,
                        levels=levels,
                        output_path=args.output_path,
                        dry_run=args.dry_run,
                    )
                )
            elif levels is not None:
                actions.extend(
                    delete_frequency_level_stores(
                        dataset=args.dataset,
                        frequency=frequency,
                        levels=levels,
                        output_path=args.output_path,
                        dry_run=args.dry_run,
                    )
                )

    if not actions:
        logger.info("No matching HEALPix outputs found for the requested cleanup.")
        return 0

    for action in actions:
        logger.info(action)
    return 0


def run_merge(args: argparse.Namespace) -> int:
    """Merge one or more frequency directories into a target frequency directory."""

    logger = logging.getLogger(__name__)
    source_dirs = expand_source_dirs(args.source_dirs)
    if not source_dirs:
        raise ValueError("Expected at least one comma-separated value.")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    target_dir = Path(args.output_path)
    if args.from_scratch and target_dir.exists():
        logger.warning("Deleting merge target directory %s", target_dir)
        shutil.rmtree(target_dir)

    merged_destinations = merge_zarr_store_directories(
        source_dirs=source_dirs,
        target_dir=target_dir,
        clean=args.clean,
        zarr_format=args.zarr_format,
        target_chunk_mb=args.chunk_size,
    )

    if not merged_destinations:
        logger.info("No matching temporary Zarr stores were found in the requested source directories.")
        return 0

    for destination in sorted(set(merged_destinations)):
        logger.info("🔗 Merged into %s", destination)
    return 0


def run_reflow(reflow_args: Sequence[str]) -> int:
    """Forward one Reflow workflow command through the main ERA5-Land CLI."""

    workflow_script = SCRIPT_DIR / "helpers" / "reflow_workflow.py"
    command = [sys.executable, str(workflow_script), *reflow_args]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the ERA5/ERA5-Land remapper."""

    configure_logging()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    if raw_argv and raw_argv[0] == "remap-reflow":
        return run_reflow(raw_argv[1:])

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "fetch":
        return run_fetch(args)

    if args.command == "remap":
        return run_remap(args)

    if args.command == "clean":
        return run_clean(args)

    if args.command == "merge":
        return run_merge(args)

    if args.command == "remap-reflow":
        return run_reflow(args.reflow_args)

    parser.error(f"Unsupported command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
