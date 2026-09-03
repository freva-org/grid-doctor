#! /usr/bin/env python3
"""Unified entry point for the ERA5/ERA5-Land conversion workflow."""

import argparse
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta
from glob import glob, has_magic
from pathlib import Path
from typing import Any, NamedTuple

from rich_argparse import RichHelpFormatter

from . import __version__
from .cli.arguments import (
    add_cache_arguments,
    add_clean_options,
    add_dataset_argument,
    add_frequency_argument,
    add_interval_argument,
    add_publication_arguments,
    add_root_argument,
    add_variable_argument,
)
from .helpers.file_fetcher import (
    batched_source_record_files,
    file_interval,
    load_json,
    load_variable_requests,
    overlaps_interval,
    parse_interval,
    resolve_records,
    selected_variables,
    split_csv_list,
    unresolved_records,
)
from .helpers.formatter import (
    dataset_output_root,
    existing_destinations_for_frequency,
    merge_dataset_root,
    normalise_frequencies,
)
from .helpers.metadata import LAST_PERMANENT_UPDATE_ATTR
from .helpers.zarr_publisher import merge_zarr_stores, sync_named_variable_attrs
from .resources import ASSETS_DIR, CMOR_TABLES_DIR, PACKAGE_DIR

# Keep runtime state next to the legacy launcher rather than in site-packages.
SCRIPT_DIR = PACKAGE_DIR.parents[1]
DEFAULT_VAR_TABLE = ASSETS_DIR / "default_variables.csv"
DEFAULT_SOURCE_MAPPER = ASSETS_DIR / "source_mapper.json"
DEFAULT_CMOR_TABLES = CMOR_TABLES_DIR
PERMANENT_DATA_LAG_MONTHS = 3
FREQUENCIES = ("1hr", "day", "mon", "fx")
UNRESOLVED_REASON = "not found in CMOR table, unsupported stream/frequency, or has no DKRZ_ID/grib_paramID"
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
    "remap_start": "\033[1;95m",
    "frequency_start": "\033[1;94m",
    "grib_merge_done": "\033[36m",
    "weight_calculation": "\033[93m",
    "remap_materialize_done": "\033[95m",
    "coarsen_source_open": "\033[36m",
    "zarr_write_start": "\033[32m",
    "frequency_done": "\033[1;32m",
    "frequency_skip_empty": "\033[90m",
    "attrs_only": "\033[32m",
    "update_skip": "\033[90m",
    "update_permanent": "\033[1;38;5;208m",
    "update_forward": "\033[1;38;5;208m",
    "update_batch": "\033[1;33m",
    "update_frequency": "\033[1;94m",
}
_ACTIVE_BATCH_STATE_PATH: Path | None = None
_BATCH_FILES_CHILD_INDEX_ENV = "ERA5_BATCH_FILES_CHILD_INDEX"
_BATCH_FILES_CHILD_COUNT_ENV = "ERA5_BATCH_FILES_CHILD_COUNT"
_BATCH_FILES_INCLUDE_SPECIAL_ENV = "ERA5_BATCH_FILES_INCLUDE_SPECIAL"


class BatchFileChildState(NamedTuple):
    """Describe the selected file batch when running inside a child process."""

    batch_index: int | None
    batch_count: int | None
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
        color = STAGE_COLORS.get(stage_name or "")
        if color is None:
            color = LEVEL_COLORS.get(record.levelno)
        if color is None:
            return message
        return f"{color}{message}{RESET_COLOR}"

    @staticmethod
    def _stage_name(record: logging.LogRecord) -> str | None:
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


def parse_coarsen_levels(value: str | None) -> tuple[int, ...] | None:
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
                raise ValueError(f"Unsupported coarsen level range {token!r}; use values like 8-0.") from exc
            if start_level < 0 or end_level < 0:
                raise ValueError("Coarsen levels must be non-negative integers.")
            if start_level < end_level:
                raise ValueError(f"Unsupported ascending coarsen range {token!r}; use descending ranges like 8-0.")
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


def parse_truncate_after(value: str | None) -> str | None:
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


def parse_level_selection(value: str | None) -> tuple[int, ...] | None:
    """Parse optional HEALPix level selections from CLI arguments."""

    return parse_coarsen_levels(value)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    source_mapper = load_json(DEFAULT_SOURCE_MAPPER)

    parser = argparse.ArgumentParser(
        description=f"ERA5/ERA5-Land source discovery and remapping tools (v{__version__})",
        formatter_class=RichDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
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
    fetch_cmd.set_defaults(_command_parser=fetch_cmd)
    add_dataset_argument(fetch_cmd)
    add_frequency_argument(fetch_cmd)
    add_variable_argument(fetch_cmd)
    add_interval_argument(fetch_cmd)
    add_root_argument(fetch_cmd)
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

    update_cmd = subparsers.add_parser(
        "update",
        help="Append newly available data and refresh the three-month permanent batch.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    update_cmd.set_defaults(_command_parser=update_cmd)
    add_dataset_argument(update_cmd)
    add_frequency_argument(update_cmd)
    add_variable_argument(update_cmd)
    add_root_argument(update_cmd)
    add_publication_arguments(update_cmd)
    update_cmd.add_argument(
        "--batch-months",
        type=int,
        default=None,
        metavar="MONTHS",
        help=(
            "Process update data in sequential calendar-month batches instead of "
            "file batches. When supplied, this takes precedence over --batch-files."
        ),
    )
    update_cmd.add_argument(
        "--batch-files",
        type=int,
        default=None,
        metavar="FILES",
        help=(
            "Process permanent and forward update data in isolated file batches "
            "of this size. When omitted, process each update directly."
        ),
    )
    update_cmd.add_argument(
        "--preview",
        action="store_true",
        help="Resolve and summarize the update without changing the output stores.",
    )
    add_cache_arguments(
        update_cmd,
        weights_dir=str(source_mapper["weights_path"]),
        highest_level_help="Only update and write the finest HEALPix zoom level for each frequency.",
    )

    remap_cmd = subparsers.add_parser(
        "remap",
        help="Resolve GRIB files and remap them to HEALPix Zarr pyramids.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    remap_cmd.set_defaults(_command_parser=remap_cmd)
    add_dataset_argument(remap_cmd)
    add_frequency_argument(remap_cmd)
    add_variable_argument(remap_cmd)
    add_interval_argument(remap_cmd)
    remap_cmd.add_argument(
        "-pl",
        "--pressure-levels",
        default=argparse.SUPPRESS,
        metavar="HPA",
        help=(
            "Comma-separated pressure levels in hPa to remap for pressure-level variables. "
            "Use 'all' to retain every available level; when omitted, use the configured selection "
            f"({','.join(str(level) for level in source_mapper['remap_defaults']['pressure_levels_hpa'])} hPa)."
        ),
    )
    remap_cmd.add_argument(
        "--batch-months",
        dest="batch_months",
        type=int,
        default=None,
        metavar="MONTHS",
        help=("Split the requested interval into sequential batches of N months and process each batch in a loop."),
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
    add_root_argument(remap_cmd)
    add_publication_arguments(remap_cmd)
    add_cache_arguments(
        remap_cmd,
        weights_dir=str(source_mapper["weights_path"]),
        highest_level_help="Only remap and write the finest HEALPix zoom level for each frequency.",
        include_highest_level=False,
    )
    add_clean_options(remap_cmd)
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

    subparsers.add_parser(
        "remap-reflow",
        help="Run the scheduler-backed Reflow remap workflow.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    subparsers.add_parser(
        "reflow-queue",
        help="Orchestrate and submit reflow runs in a controlled sequence of time intervals.",
        formatter_class=RichDefaultsHelpFormatter,
    )

    clean_cmd = subparsers.add_parser(
        "clean",
        help="Remove variables, levels, frequencies, or the whole HEALPix output root.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    clean_cmd.set_defaults(_command_parser=clean_cmd)
    add_dataset_argument(clean_cmd, help_text="Dataset to clean.")
    add_frequency_argument(clean_cmd, default=None)
    add_variable_argument(clean_cmd)
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
        help=("Truncate existing time-based stores, removing timestamps strictly after DATE."),
    )
    clean_cmd.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be removed without changing anything.",
    )

    merge_cmd = subparsers.add_parser(
        "merge",
        help="Merge one or more (frequency) directories into a target (frequency) directory.",
        formatter_class=RichDefaultsHelpFormatter,
    )
    merge_cmd.set_defaults(_command_parser=merge_cmd)
    merge_cmd.add_argument(
        "--source",
        dest="source_dirs",
        nargs="+",
        required=True,
        help=("Source directories, worker-output roots, or glob patterns. Values may be comma-separated or repeated."),
    )
    add_dataset_argument(merge_cmd)
    add_frequency_argument(
        merge_cmd,
        default=None,
        help_text="Optional comma-separated frequencies (1hr,day,mon,fx).",
    )
    add_variable_argument(merge_cmd, default=None)
    add_interval_argument(merge_cmd)
    merge_cmd.add_argument(
        "--levels",
        default=None,
        metavar="LEVELS",
        help=(
            "Optional comma-separated or descending-range HEALPix levels to "
            "merge, such as 7 or 6-0. When omitted, all levels are merged."
        ),
    )
    add_publication_arguments(
        merge_cmd,
        output_help=(
            "Target directory that (if --dataset, --freq are omitted) "
            "directly contains the merged `level_*.zarr` stores."
        ),
        output_required=True,
    )
    add_clean_options(merge_cmd)

    subparsers._name_parser_map = dict(sorted(subparsers._name_parser_map.items()))
    subparsers.metavar = "{" + ",".join(subparsers._name_parser_map) + "}"
    subparsers._choices_actions.sort(key=lambda action: action.dest)
    return parser


def add_months(current: date, months: int) -> date:
    """Return ``current`` shifted forward by ``months`` calendar months."""

    year = current.year + (current.month - 1 + months) // 12
    month = (current.month - 1 + months) % 12 + 1
    return date(year, month, 1)


def batched_intervals(
    interval: tuple[date | None, date | None],
    *,
    batch_months: int | None,
) -> tuple[tuple[date | None, date | None], ...]:
    """Split one inclusive interval into inclusive month-sized batches."""

    if batch_months is None:
        return (interval,)
    if batch_months <= 0:
        raise ValueError("--batch-months must be a positive integer number of months.")

    start, end = interval
    if start is None or end is None:
        raise ValueError("--batch-months requires a bounded --interval with a start and end date.")

    intervals: list[tuple[date, date]] = []
    current_start = start
    while current_start <= end:
        next_start = add_months(date(current_start.year, current_start.month, 1), batch_months)
        current_end = min(end, next_start - timedelta(days=1))
        intervals.append((current_start, current_end))
        current_start = current_end + timedelta(days=1)
    return tuple(intervals)


def format_interval(interval: tuple[date | None, date | None]) -> str:
    """Render one interval tuple for logs."""

    start, end = interval
    start_text = start.isoformat() if start is not None else ""
    end_text = end.isoformat() if end is not None else ""
    return f"{start_text},{end_text}"


def build_batch_command(
    args: argparse.Namespace,
    *,
    interval: tuple[date | None, date | None],
    clean: bool,
) -> list[str]:
    """Build one isolated child-process command for a single batch interval.

    The child inherits the current Python interpreter and runs the installed
    package module, preserving its job allocation and environment while still
    releasing all batch-local memory on process exit.
    """

    command = [
        sys.executable,
        "-m",
        "heal_era5.main",
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

    job_token = hashlib.sha256(f"{os.getpid()}:{Path.cwd()}:{SCRIPT_DIR}".encode()).hexdigest()[:12]
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
    intervals: Sequence[tuple[date | None, date | None]],
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
            child_env = os.environ.copy() if batch_plan.get("env") else None
            if child_env is not None:
                child_env.update({str(key): str(value) for key, value in dict(batch_plan["env"]).items()})
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
    current_interval: tuple[date | None, date | None],
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


def parse_cli_freqs(value: str) -> tuple[str, ...]:
    """Parse and validate a frequency CLI option."""

    frequencies = normalise_frequencies(FREQUENCIES) if value == "all" else normalise_frequencies(split_csv_list(value))
    unknown_freqs = sorted(set(frequencies) - set(FREQUENCIES))
    if unknown_freqs:
        raise ValueError(f"Unsupported frequencies: {', '.join(unknown_freqs)}")
    return tuple(frequencies)


def extend_frequencies_for_special_variables(
    frequencies: tuple[str, ...],
    requested_variables: tuple[str, ...],
) -> tuple[str, ...]:
    """Add the `fx` publication pass when special variables are requested."""

    from .helpers.special import split_special_variables

    _, special_variables = split_special_variables(requested_variables)
    if not special_variables or "fx" in frequencies:
        return frequencies
    return (*frequencies, "fx")


def selected_requests(
    *,
    dataset: str,
    variables: tuple[str, ...] | None,
    var_table: Path = DEFAULT_VAR_TABLE,
):
    """Resolve the requested variables for one dataset selection."""

    source_mapper = load_json(DEFAULT_SOURCE_MAPPER)
    dataset_codes = tuple(str(code) for code in source_mapper["datasets"][dataset]["priority"])
    requests = selected_variables(
        load_variable_requests(var_table),
        allowed_codes=dataset_codes,
        variables=variables,
    )
    return source_mapper, requests


def parse_cli_args(value: str | None) -> tuple[str, ...] | None:
    """Parse a comma-separated CLI option."""

    if value in (None, "all"):
        return None
    return split_csv_list(value)


def parse_pressure_levels(
    value: str | None,
    *,
    source_mapper: dict[str, Any],
) -> tuple[int, ...] | None:
    """Resolve a pressure-level selection in hPa, with ``all`` disabling filtering."""

    if value == "all":
        return None
    if value is None:
        value = ",".join(str(level) for level in source_mapper["remap_defaults"]["pressure_levels_hpa"])

    try:
        levels = tuple(int(token) for token in split_csv_list(value))
    except ValueError as exc:
        raise ValueError("--pressure-levels must be comma-separated integer values in hPa, or 'all'.") from exc
    if not levels or any(level <= 0 for level in levels):
        raise ValueError("--pressure-levels must contain one or more positive integer values in hPa.")
    return tuple(dict.fromkeys(levels))


def expand_source_dirs(values: Sequence[str]) -> tuple[str, ...]:
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
            "Internal file-batch child environment is incomplete; expected both child index and child count."
        )

    return BatchFileChildState(
        batch_index=int(index_text) if index_text is not None else None,
        batch_count=int(count_text) if count_text is not None else None,
        include_special=(os.environ.get(_BATCH_FILES_INCLUDE_SPECIAL_ENV, "0") == "1"),
    )


def map_records(
    records: Sequence[Any],
    *,
    args: argparse.Namespace,
    frequencies: tuple[str, ...],
    requested_variables: tuple[str, ...],
    interval: tuple[date | None, date | None],
    clean: bool,
    coarsen_only: bool = False,
    coarsen_levels: tuple[int, ...] | None = None,
    use_input_cache: bool | None = None,
    drop_duplicate_time_rows: bool | None = None,
    coarsen_interval: tuple[date | None, date | None] | None = None,
) -> None:
    """Map records using the common remap settings from the CLI namespace.

    Batch and special-variable paths use this adapter with narrower frequency or
    variable selections while inheriting the same output and caching settings.
    """

    from .helpers.mapper import map_grib_to_healpix

    map_grib_to_healpix(
        list(records),
        dataset=args.dataset,
        frequencies=frequencies,
        requested_variables=requested_variables,
        interval=interval,
        zarr_format=args.zarr_format,
        use_inventory_cache=args.use_inventory_cache,
        use_input_cache=(args.use_input_cache if use_input_cache is None else use_input_cache),
        drop_duplicate_time_rows=(
            not args.fail_on_duplicate_times if drop_duplicate_time_rows is None else drop_duplicate_time_rows
        ),
        pressure_levels=args.pressure_levels,
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
    fallback_interval: tuple[date | None, date | None],
) -> list[tuple[Any, tuple[date | None, date | None], str]]:
    """Expand resolved records into a stable global file-batch execution plan."""

    plan: list[tuple[Any, tuple[date | None, date | None], str]] = []
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
    interval: tuple[date | None, date | None],
    *,
    batch_index: int,
    batch_count: int,
    batch_months: int,
) -> str:
    """Return one human-readable label for an interval batch."""

    month_text = "month" if batch_months == 1 else "months"
    return f"interval {batch_index}/{batch_count} {format_interval(interval)} ({batch_months} {month_text})"


def run_fetch(args: argparse.Namespace) -> int:
    """Resolve source files and print either JSON records or paths."""

    from .helpers.special import split_special_variables

    if args.dataset is None:
        args._command_parser.print_help()
        return 2

    variables = parse_cli_args(args.variables)
    frequencies = parse_cli_freqs(args.freq)
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
        for missing_record in missing:
            print(
                f"missing: {missing_record.variable} {missing_record.frequency} {missing_record.pattern}",
                file=sys.stderr,
            )
        for unresolved_record in unresolved:
            print(
                f"unresolved: {unresolved_record.variable} {unresolved_record.frequency}: {unresolved_record.reason}",
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

    from .helpers.cleanup import truncate_existing_healpix_stores
    from .helpers.mapper import (
        rechunk_existing_healpix_stores,
        update_healpix_attrs_only,
    )
    from .helpers.special import split_special_variables

    logger = logging.getLogger(__name__)
    if args.dataset is None:
        args._command_parser.print_help()
        return 2

    variables = parse_cli_args(args.variables)
    frequencies = parse_cli_freqs(args.freq)
    interval = parse_interval(args.interval)
    truncate_after = parse_truncate_after(args.truncate_after)
    coarsen_levels = parse_coarsen_levels(args.coarsen_only)
    source_mapper, requests = selected_requests(dataset=args.dataset, variables=variables)
    args.pressure_levels = parse_pressure_levels(
        getattr(args, "pressure_levels", None),
        source_mapper=source_mapper,
    )
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
        current_interval: tuple[date | None, date | None],
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

            if file_child.batch_index is None and file_child.batch_count is None:
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
            if file_child.batch_index is not None:
                batch_count = len(batched_records)
                if file_child.batch_count != batch_count:
                    raise ValueError(
                        "Resolved file-batch count does not match the parent batch plan: "
                        f"expected {file_child.batch_count}, got {batch_count}."
                    )
                child_index = file_child.batch_index
                if child_index < 1 or child_index > batch_count:
                    raise ValueError(f"--batch-files-child-index must be between 1 and {batch_count}.")
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

            if special_variables and (file_child.batch_index is None or file_child.include_special):
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
        (interval,) if args.batch_files is not None else batched_intervals(interval, batch_months=args.batch_months)
    )

    if len(intervals) > 1:
        logger.info(
            "📦 Processing %s batches using isolated subprocesses.",
            len(intervals),
        )
        return run_batched_months(args, intervals)

    run_single_interval(intervals[0], clean=args.clean)

    return 0


def _existing_variable_last_date(
    dataset: str,
    frequency: str,
    variable: str,
    *,
    zarr_format: int,
    output_path: str | Path | None,
) -> tuple[date | None, date | None]:
    """Return consistent coverage and watermark metadata across all levels.

    A preview must not be distorted by a stale attribute on one HEALPix level.
    If the permanent watermark is missing or differs between levels, it is
    treated as missing so the caller uses the safe inferred refresh interval.
    """

    import xarray as xr

    destinations = existing_destinations_for_frequency(
        dataset,
        frequency,
        output_path=output_path,
    )
    data_dates: list[date] = []
    permanent_dates: list[date] = []
    variable_destinations = 0
    for destination in destinations:
        opened = xr.open_zarr(destination, consolidated=(zarr_format == 2))
        try:
            if variable not in opened or "time" not in opened[variable].dims:
                continue
            variable_destinations += 1
            data_date = date.fromisoformat(str(opened[variable]["time"].values[-1])[:10])
            data_dates.append(data_date)
            permanent_attr = opened[variable].attrs.get(LAST_PERMANENT_UPDATE_ATTR)
            if permanent_attr:
                permanent_dates.append(date.fromisoformat(str(permanent_attr)[:10]))
        finally:
            opened.close()

    if not data_dates:
        return None, None

    permanent_date = None
    if variable_destinations > 0 and len(permanent_dates) == variable_destinations and len(set(permanent_dates)) == 1:
        permanent_date = permanent_dates[0]

    return max(data_dates), permanent_date


def _local_modification_date(source_file: str) -> date:
    """Return a source file's modification date in the machine's local timezone."""

    timestamp = Path(source_file).stat().st_mtime
    return datetime.fromtimestamp(timestamp, UTC).astimezone().date()


def _is_final_source_file(
    source_file: str,
    *,
    dataset: str,
    frequency: str,
) -> bool:
    """Return whether a source file is eligible for the permanent pass.

    ERA5's ET files are explicitly provisional, so only E5/E1 files qualify.
    ERA5-Land is supplied as the merged EL collection; there the filesystem
    modification date is used as the replacement marker. A file qualifies
    only when it was modified at least one calendar month after the date in
    its filename.
    """

    name = Path(source_file).name
    if dataset == "era5":
        return not name.startswith("ET")

    coverage = file_interval(source_file, frequency)
    if coverage is None:
        return False

    file_start, _ = coverage
    modified_date = _local_modification_date(source_file)
    return modified_date >= add_months(file_start, 1)


def _map_update_records(
    records: Sequence[Any],
    *,
    args: argparse.Namespace,
    remap_args: argparse.Namespace,
    interval: tuple[date | None, date | None],
    frequency: str,
    variable: str,
    logger: logging.Logger,
    phase: str,
    batch_months: int | None,
    batch_files: int | None,
) -> None:
    """Map update records directly or in bounded file/month batches.

    Batches are processed sequentially so the mapper never has to load a long
    update interval for a vertical variable at once. The existing mapper
    cleanup runs after each batch; callers can additionally use a small batch
    size when node memory is constrained.
    """

    if batch_months is not None:
        intervals = batched_intervals(interval, batch_months=batch_months)
        batches = []
        for current_interval in intervals:
            for record in records:
                current_files = tuple(
                    source_file
                    for source_file in record.files
                    if (
                        file_interval(source_file, frequency) is not None
                        and overlaps_interval(
                            source_file,
                            frequency,
                            current_interval[0],
                            current_interval[1],
                        )
                    )
                )
                if current_files:
                    batches.append(
                        (
                            record._replace(files=current_files),
                            current_interval,
                        )
                    )
    elif batch_files is not None:
        batches = []
        for record in records:
            batches.extend(
                batched_source_record_files(
                    record,
                    batch_files=batch_files,
                    fallback_interval=interval,
                )
            )
    else:
        map_records(
            records,
            args=remap_args,
            frequencies=(frequency,),
            requested_variables=(variable,),
            interval=interval,
            clean=False,
        )
        return

    logger.info(
        "stage=update_batch 📦 %s %s %s batches=%s",
        phase,
        frequency,
        variable,
        len(batches),
    )
    for batch_index, (batch_record, batch_interval) in enumerate(batches, start=1):
        logger.info(
            "stage=update_batch 📦 %s batch=%s/%s dates=%s..%s files=%s",
            phase,
            batch_index,
            len(batches),
            batch_interval[0],
            batch_interval[1],
            len(batch_record.files),
        )
        map_records(
            [batch_record],
            args=remap_args,
            frequencies=(frequency,),
            requested_variables=(variable,),
            interval=batch_interval,
            clean=False,
        )


class UpdateSelection(NamedTuple):
    """Describe source records selected for one update phase."""

    records: list[Any]
    interval: tuple[date, date] | None
    file_count: int


class UpdatePreviewRow(NamedTuple):
    """Summarize the planned permanent and forward updates for one variable."""

    frequency: str
    variable: str
    stored_end: date
    permanent: str
    permanent_files: int
    forward: str
    forward_files: int


def _update_remap_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the remapping arguments used by an incremental update."""

    return argparse.Namespace(
        dataset=args.dataset,
        zarr_format=args.zarr_format,
        use_inventory_cache=args.use_inventory_cache,
        use_input_cache=args.use_input_cache,
        fail_on_duplicate_times=args.fail_on_duplicate_times,
        weights_dir=args.weights_dir,
        chunk_size=args.chunk_size,
        highest_level_only=args.highest_level_only,
        output_path=args.output_path,
    )


def _resolve_update_records(
    *,
    args: argparse.Namespace,
    variable: str,
    frequency: str,
    interval: tuple[date, date],
) -> list[Any]:
    """Resolve source records for one variable, frequency, and date interval."""

    return resolve_records(
        var_table=DEFAULT_VAR_TABLE,
        cmor_tables_dir=DEFAULT_CMOR_TABLES,
        dataset=args.dataset,
        variables=(variable,),
        frequencies=(frequency,),
        interval=interval,
        root=args.root,
        glob_files=True,
    )


def _select_permanent_records(
    records: Sequence[Any],
    *,
    dataset: str,
    frequency: str,
    latest_date: date,
    permanent_watermark: date | None,
) -> UpdateSelection:
    """Keep source files eligible for the permanent refresh.

    When no permanent watermark exists, the file modification date is used as
    a bootstrap boundary: files must have arrived or changed on or after the
    latest date already stored in the output. Finality is then checked using
    the dataset-specific source-file policy.
    """

    selected_files: set[str] = set()
    selected_intervals: list[tuple[date, date]] = []
    for record in records:
        for source_file in record.files:
            coverage = file_interval(source_file, frequency)
            if coverage is None:
                continue
            file_start, _ = coverage
            if permanent_watermark and file_start < permanent_watermark:
                continue
            modified_date = _local_modification_date(source_file)
            if permanent_watermark is None and modified_date < latest_date:
                continue
            if not _is_final_source_file(
                source_file,
                dataset=dataset,
                frequency=frequency,
            ):
                continue
            selected_files.add(source_file)
            selected_intervals.append(coverage)

    if not selected_intervals:
        return UpdateSelection([], None, 0)

    selected_records = [
        record._replace(files=tuple(source_file for source_file in record.files if source_file in selected_files))
        for record in records
    ]
    selected_records = [record for record in selected_records if record.files]
    interval = (
        min(start for start, _ in selected_intervals),
        max(end for _, end in selected_intervals),
    )
    return UpdateSelection(selected_records, interval, len(selected_files))


def _apply_permanent_update(
    selection: UpdateSelection,
    *,
    args: argparse.Namespace,
    remap_args: argparse.Namespace,
    frequency: str,
    variable: str,
    logger: logging.Logger,
) -> None:
    """Map final records and persist the resulting permanent watermark."""

    if selection.interval is None:
        return

    logger.info(
        "stage=update_permanent 🔁 Refreshing %s permanent source file(s) for %s %s: dates=%s..%s",
        selection.file_count,
        frequency,
        variable,
        selection.interval[0],
        selection.interval[1],
    )
    _map_update_records(
        selection.records,
        args=args,
        remap_args=remap_args,
        interval=selection.interval,
        frequency=frequency,
        variable=variable,
        logger=logger,
        phase="permanent",
        batch_months=args.batch_months,
        batch_files=args.batch_files,
    )

    permanent_starts = []
    for record in selection.records:
        for source_file in record.files:
            coverage = file_interval(source_file, frequency)
            if coverage is not None:
                permanent_starts.append(coverage[0])
    permanent_watermark = max(permanent_starts)
    watermark_attrs = {variable: {LAST_PERMANENT_UPDATE_ATTR: permanent_watermark.isoformat()}}
    for destination in existing_destinations_for_frequency(
        args.dataset,
        frequency,
        output_path=args.output_path,
    ):
        sync_named_variable_attrs(watermark_attrs, destination)


def _apply_forward_update(
    records: Sequence[Any],
    *,
    args: argparse.Namespace,
    remap_args: argparse.Namespace,
    interval: tuple[date, date],
    frequency: str,
    variable: str,
    latest_date: date,
    logger: logging.Logger,
) -> int:
    """Map forward records and return the number of source files processed."""

    file_count = sum(len(record.files) for record in records)
    if file_count == 0:
        return 0

    logger.info(
        "stage=update_forward ➕ Updating forward data for %s %s: dates=%s..%s",
        frequency,
        variable,
        latest_date,
        interval[1],
    )
    _map_update_records(
        records,
        args=args,
        remap_args=remap_args,
        interval=interval,
        frequency=frequency,
        variable=variable,
        logger=logger,
        phase="forward",
        batch_months=args.batch_months,
        batch_files=args.batch_files,
    )
    return file_count


def _preview_update_row(
    *,
    frequency: str,
    variable: str,
    latest_date: date,
    permanent: UpdateSelection,
    forward_files: int,
    today: date,
) -> UpdatePreviewRow:
    """Build one row for the update preview report."""

    permanent_range = f"{permanent.interval[0]}..{permanent.interval[1]}" if permanent.interval is not None else "-"
    forward_range = f"{latest_date}..{today}" if forward_files else "-"
    return UpdatePreviewRow(
        frequency,
        variable,
        latest_date,
        permanent_range,
        permanent.file_count,
        forward_range,
        forward_files,
    )


def _log_update_preview(
    rows: Sequence[UpdatePreviewRow],
    *,
    batch_mode: str,
    logger: logging.Logger,
) -> None:
    """Log the planned update operations in tabular form."""

    logger.info("stage=update_preview 📋 Update preview (batch_mode=%s)", batch_mode)
    logger.info(
        "stage=update_preview %-10s %-18s %-12s %-25s %s %-25s %s",
        "frequency",
        "variable",
        "stored_end",
        "permanent dates",
        "perm_files",
        "forward dates",
        "fwd_files",
    )
    for row in rows:
        logger.info(
            "stage=update_preview %-10s %-18s %-12s %-25s %10s %-25s %s",
            row.frequency,
            row.variable,
            row.stored_end,
            row.permanent,
            row.permanent_files,
            row.forward,
            row.forward_files,
        )


def run_update(args: argparse.Namespace) -> int:
    """Update each existing variable/frequency with permanent and new source data.

    The permanent pass selects final source files from the permanent watermark
    through the command date. For ERA5-Land, finality is inferred from the
    source file's modification date being more than one calendar month after
    the date encoded in its filename. The forward pass starts at the latest
    stored date, allowing the publisher to replace provisional data and append
    newer timestamps. Each variable is resolved separately so unrelated
    variables are not remapped.
    """

    if args.dataset is None:
        args._command_parser.print_help()
        return 2

    variables = parse_cli_args(args.variables)
    frequencies = parse_cli_freqs(args.freq)
    _, requests = selected_requests(dataset=args.dataset, variables=variables)
    requested_variables = tuple(request.name for request in requests)
    today = datetime.now().astimezone().date()
    logger = logging.getLogger(__name__)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")
    if args.batch_files is not None and args.batch_files <= 0:
        raise ValueError("--batch-files must be a positive integer.")
    if args.batch_months is not None and args.batch_months <= 0:
        raise ValueError("--batch-months must be a positive integer.")

    preview_rows: list[UpdatePreviewRow] = []
    for frequency in frequencies:
        if frequency == "fx":
            continue
        logger.info(
            "stage=update_frequency ─────────────── frequency=%s ───────────────",
            frequency,
        )
        for variable in requested_variables:
            latest_date, permanent_watermark = _existing_variable_last_date(
                args.dataset,
                frequency,
                variable,
                zarr_format=args.zarr_format,
                output_path=args.output_path,
            )
            if latest_date is None:
                logger.info(
                    "stage=update_skip ⏭️  Skipping %s %s: no existing time series found",
                    frequency,
                    variable,
                )
                continue

            remap_args = _update_remap_args(args)

            permanent_date = today
            # A store without a permanent watermark may have been published
            # long enough ago for a multi-month permanent refresh to be due.
            # Infer the missing watermark from the final stored coordinate.
            permanent_start = permanent_watermark or add_months(
                latest_date,
                -PERMANENT_DATA_LAG_MONTHS,
            )
            permanent_records: list[Any] = []
            if permanent_start <= permanent_date:
                permanent_records = _resolve_update_records(
                    args=args,
                    variable=variable,
                    frequency=frequency,
                    interval=(permanent_start, permanent_date),
                )
            permanent = _select_permanent_records(
                permanent_records,
                dataset=args.dataset,
                frequency=frequency,
                latest_date=latest_date,
                permanent_watermark=permanent_watermark,
            )
            if permanent.interval is not None and not args.preview:
                _apply_permanent_update(
                    permanent,
                    args=args,
                    remap_args=remap_args,
                    frequency=frequency,
                    variable=variable,
                    logger=logger,
                )

            forward_interval = (latest_date, today)
            forward_records: list[Any] = []
            if latest_date <= today:
                forward_records = _resolve_update_records(
                    args=args,
                    variable=variable,
                    frequency=frequency,
                    interval=forward_interval,
                )
            forward_file_count = sum(len(record.files) for record in forward_records)
            if args.preview:
                preview_rows.append(
                    _preview_update_row(
                        frequency=frequency,
                        variable=variable,
                        latest_date=latest_date,
                        permanent=permanent,
                        forward_files=forward_file_count,
                        today=today,
                    )
                )
                continue
            if forward_file_count:
                _apply_forward_update(
                    forward_records,
                    args=args,
                    remap_args=remap_args,
                    interval=forward_interval,
                    frequency=frequency,
                    variable=variable,
                    latest_date=latest_date,
                    logger=logger,
                )

    if args.preview:
        if args.batch_months is not None:
            batch_mode = f"months={args.batch_months}"
        elif args.batch_files is not None:
            batch_mode = f"files={args.batch_files}"
        else:
            batch_mode = "direct"
        _log_update_preview(preview_rows, batch_mode=batch_mode, logger=logger)

    return 0


def run_clean(args: argparse.Namespace) -> int:
    """Clean existing HEALPix outputs at variable, level, frequency, or root scope."""

    from .helpers.cleanup import (
        delete_dataset_root,
        delete_frequency_directory,
        delete_frequency_level_stores,
        remove_variables_from_frequency_stores,
        truncate_existing_healpix_stores,
    )

    logger = logging.getLogger(__name__)
    variables = parse_cli_args(args.variables)
    levels = parse_level_selection(args.levels)
    frequencies = parse_cli_freqs(args.freq) if args.freq is not None else ()

    if args.truncate_after is not None:
        if variables is not None or levels is not None:
            raise ValueError("--truncate-after cannot be combined with --var or --levels.")
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
    variables = parse_cli_args(args.variables)
    selectors = (args.dataset, args.freq, variables)
    if args.dataset is None and any(value is not None for value in selectors[1:]):
        raise ValueError("--dataset is required when --freq or --var is provided.")

    source_dirs = expand_source_dirs(args.source_dirs)
    if not source_dirs:
        raise ValueError("No matching merge source directories were found.")

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be a positive integer.")

    levels = parse_level_selection(args.levels)
    interval = parse_interval(args.interval)

    target_dir = Path(args.output_path)
    if args.dataset is not None:
        target_dir = merge_dataset_root(
            args.dataset,
            output_path=target_dir,
            frequencies=args.freq,
        )
    if args.from_scratch and target_dir.exists():
        logger.warning("Deleting merge target directory %s", target_dir)
        shutil.rmtree(target_dir)

    merged_destinations = merge_zarr_stores(
        sources=source_dirs,
        target_dir=target_dir,
        dataset=args.dataset,
        frequency=args.freq,
        variable=variables,
        levels=levels,
        interval=interval,
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

    from .cli.reflow_workflow import main as reflow_main

    return int(reflow_main(reflow_args))


def run_reflow_queue(queue_args: Sequence[str]) -> int:
    """Run the Reflow campaign queue through the unified CLI."""

    from .cli.reflow_queue import main as queue_main

    return int(queue_main(queue_args, prog="heal-era5 reflow-queue"))


def main(argv: list[str] | None = None) -> int:
    """Run the ERA5/ERA5-Land remapper."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    delegated_handlers = {
        "remap-reflow": run_reflow,
        "reflow-queue": run_reflow_queue,
    }
    if raw_argv and raw_argv[0] in delegated_handlers:
        if len(raw_argv) == 1:
            return delegated_handlers[raw_argv[0]](["-h"])
        return delegated_handlers[raw_argv[0]](raw_argv[1:])

    parser = build_parser()
    normal_commands = {"fetch", "remap", "update", "clean", "merge"}
    if len(raw_argv) == 1 and raw_argv[0] in normal_commands:
        try:
            parser.parse_args([raw_argv[0], "-h"])
        except SystemExit as exc:
            return int(exc.code or 0)

    args = parser.parse_args(raw_argv)

    if args.command is None:
        parser.print_help()
        return 2

    configure_logging()
    handlers = {
        "fetch": run_fetch,
        "remap": run_remap,
        "update": run_update,
        "clean": run_clean,
        "merge": run_merge,
    }
    return int(handlers[args.command](args))


if __name__ == "__main__":
    raise SystemExit(main())
