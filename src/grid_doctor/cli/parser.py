"""Argument parser helpers for grid-doctor command-line tools."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from rich_argparse import ArgumentDefaultsRichHelpFormatter as ArgFormatter
except ImportError:  # pragma: no cover - optional dependency
    ArgFormatter = argparse.ArgumentDefaultsHelpFormatter


def get_parser(name: str, description: str | None = None) -> argparse.ArgumentParser:
    """Create a parser preloaded with common grid-doctor options.

    Parameters
    ----------
    name:
        Program name shown in `--help` output.
    description:
        Optional one-line program description.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with common S3 and verbosity arguments.

    Examples
    --------
    ```python
    parser = get_parser("convert-icon", description="Build a HEALPix pyramid")
    args = parser.parse_args()
    ```
    """
    parser = argparse.ArgumentParser(
        prog=name,
        description=description,
        formatter_class=ArgFormatter,
    )
    parser.add_argument("--s3-bucket", help="S3 target bucket.", required=True)
    parser.add_argument(
        "--s3-endpoint",
        default="https://s3.eu-dkrz-3.dkrz.cloud",
        help="S3 endpoint URL.",
    )
    parser.add_argument(
        "--s3-credentials-file",
        default=Path.home() / ".s3-credentials.json",
        help="Path to a JSON file with accessKey/secretKey.",
    )
    parser.add_argument(
        "--log-dir", type=Path, help="Log output to this directory.", default=None
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity with repeated flags such as -v or -vv.",
    )
    return parser


def setup_logging_from_args(args: argparse.Namespace, **kwargs: Any) -> None:
    """Initialise logging from parsed CLI arguments.

    Parameters
    ----------
    args:
        Parsed arguments containing a `verbose` attribute.
    **kwargs:
        Extra keyword arguments forwarded to
        [`grid_doctor.setup_logging`][grid_doctor.log.setup_logging].
    """
    import grid_doctor as gd

    gd.setup_logging(verbosity=args.verbose, log_dir=args.log_dir or None, **kwargs)
