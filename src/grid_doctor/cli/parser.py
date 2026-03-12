"""Argument parser factory for grid-doctor conversion scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

try:
    from rich_argparse import ArgumentDefaultsRichHelpFormatter as ArgFormatter
except ImportError:
    from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter


def get_parser(
    name: str,
    description: Optional[str] = None,
) -> argparse.ArgumentParser:
    """Create a pre-configured argument parser for a conversion script.

    The returned parser already includes the common arguments shared by
    all grid-doctor scripts (S3 target, endpoint, credentials, verbosity).

    Parameters
    ----------
    name : str
        Program name shown in ``--help`` output.
    description : str or None, optional
        One-line description of the script.

    Returns
    -------
    argparse.ArgumentParser
        Parser with common arguments.  Script-specific arguments can be
        added before calling :meth:`~argparse.ArgumentParser.parse_args`.
    """
    parser = argparse.ArgumentParser(
        prog=name,
        description=description,
        formatter_class=ArgFormatter,
    )
    parser.add_argument(
        "--s3-bucket",
        help="S3 target bucket.",
        required=True,
    )
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (repeat for more: -v, -vv, -vvv).",
    )
    return parser


def setup_logging_from_args(
    args: argparse.Namespace,
    **kwargs: Union[str, int],
) -> None:
    """Initialise logging from parsed CLI arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments (must contain ``verbose``).
    **kwargs : str or int
        Forwarded to :func:`grid_doctor.setup_logging`.
    """
    import grid_doctor as gd

    gd.setup_logging(verbosity=args.verbose, **kwargs)
