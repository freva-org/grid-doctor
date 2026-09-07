"""Reusable argument definitions for the ERA5/ERA5-Land CLI."""

import argparse
from typing import Protocol, cast

DATASET_CHOICES = ("era5land", "era5")
DATASET_HELP = "Dataset to process: era5land or era5."
VARIABLE_HELP = "Comma-separated variables."
FREQUENCY_HELP = "Comma-separated frequencies: 1hr,day,mon,fx."
INTERVAL_HELP = (
    "Date interval (START,END) where each date may be YYYY, YYYYMM, YYYYMMDD (hyphens optional). Empty END means today."
)
OUTPUT_PUBLICATION_HELP = (
    "Override the published HEALPix output root directory. "
    "Useful for test runs that should write outside the default location."
)
DEFAULT_CHUNK_SIZE: int = 16


class _DisplayDefaultAction(Protocol):
    """An argparse action carrying help-only default text."""

    display_default: str


def _set_display_default(action: argparse.Action, value: str) -> None:
    """Attach display-only default text used by the Rich help formatter."""

    cast(_DisplayDefaultAction, action).display_default = value


def add_dataset_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = None,
    help_text: str = DATASET_HELP,
) -> None:
    """Add the common dataset selector to a command parser."""

    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=default,
        help=help_text,
    )


def add_variable_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = "all",
    display_default: str | None = None,
    help_text: str | None = VARIABLE_HELP,
) -> None:
    """Add the common variable selector to a command parser."""

    action = parser.add_argument("--var", dest="variables", default=default, help=help_text)
    if display_default is not None:
        _set_display_default(action, display_default)


def add_frequency_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str | None = "all",
    help_text: str = FREQUENCY_HELP,
) -> None:
    """Add the common frequency selector to a command parser."""

    parser.add_argument("--freq", default=default, help=help_text)


def add_interval_argument(
    parser: argparse.ArgumentParser,
    *,
    display_default: str | None = None,
) -> None:
    """Add the common inclusive date interval option to a command parser."""

    action = parser.add_argument("--interval", default=None, help=INTERVAL_HELP)
    if display_default is not None:
        _set_display_default(action, display_default)


def add_root_argument(
    parser: argparse.ArgumentParser,
    *,
    display_default: str | None = None,
) -> None:
    """Add the optional source-data root override."""

    action = parser.add_argument(
        "--root",
        default=None,
        help="Override GRIB root folder for tests or alternate mounts.",
    )
    if display_default is not None:
        _set_display_default(action, display_default)


def add_publication_arguments(
    parser: argparse.ArgumentParser,
    *,
    output_help: str = OUTPUT_PUBLICATION_HELP,
    output_required: bool = False,
    output_display_default: str | None = None,
) -> None:
    """Add output path, Zarr format, and chunk-size options."""

    action = parser.add_argument(
        "--output-path",
        required=output_required,
        default=None,
        help=output_help,
    )
    if output_display_default is not None:
        _set_display_default(action, output_display_default)
    parser.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        help="Zarr format version for the output pyramid.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="MB",
        help=("Approximate Zarr chunk-size target in megabytes for rewritten destination stores."),
    )


def add_cache_arguments(
    parser: argparse.ArgumentParser,
    *,
    weights_dir: str,
    highest_level_help: str,
    include_highest_level: bool = True,
) -> None:
    """Add shared GRIB-cache, duplicate-time, weights, and level options."""

    parser.add_argument(
        "--no-cache",
        "--no-inventory-cache",
        action="store_false",
        dest="use_inventory_cache",
        default=True,
        help="Disable cached GRIB inventories.",
    )
    parser.add_argument(
        "--cache-input-datasets",
        action="store_true",
        dest="use_input_cache",
        default=False,
        help="Enable cached multi-file input dataset pickles.",
    )
    parser.add_argument(
        "-fdt",
        "--fail-on-duplicate-times",
        action="store_true",
        dest="fail_on_duplicate_times",
        default=False,
        help=(
            "Raise an error when exact duplicate GRIB time rows are found during "
            "time normalization instead of dropping them."
        ),
    )
    parser.add_argument(
        "--weights-dir",
        default=weights_dir,
        help="Directory where HEALPix weight files are stored and reused.",
    )
    if include_highest_level:
        parser.add_argument(
            "-hlo",
            "--highest-level-only",
            action="store_true",
            default=False,
            help=highest_level_help,
        )


def add_clean_options(parser: argparse.ArgumentParser) -> None:
    """Add options controlling incremental or destructive publication."""

    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help=(
            "Overwrite existing Zarr stores instead of updating them incrementally. It wipes the store before starting."
        ),
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        default=False,
        help="Delete the complete output root before writing.",
    )
