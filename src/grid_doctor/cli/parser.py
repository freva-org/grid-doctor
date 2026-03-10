"""Cli helper functions for parsing command line arguments."""

import argparse
from pathlib import Path
from typing import Optional, Union

try:
    from rich_argparse import ArgumentDefaultsRichHelpFormatter as ArgFormatter
except ImportError:
    from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter


def get_parser(name: str, description: Optional[str] = None) -> argparse.ArgumentParser:
    """Create a agrument parser.

    Parameters
    ----------
    prog: str
    description: str

    """
    parser = argparse.ArgumentParser(
        prog=name,
        description=description,
        formatter_class=ArgFormatter,
    )
    parser.add_argument("s3_bucket", help="S3 target bucket.", metavar="s3-bucket")
    parser.add_argument(
        "--s3-endpoint",
        default="https://s3.eu-dkrz-3.dkrz.cloud",
        help="S3 enpoint url",
    )
    parser.add_argument(
        "--s3-credentials-file",
        default=Path.home() / ".s3-credentials.json",
        help="Where to read secrets from.",
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
    args: argparse.Namespace, **kwargs: Union[str, int]
) -> None:
    """Call :func:`setup_logging` using the parsed *args* from
    :func:`add_log_args`.

    Any extra *kwargs* are forwarded to :func:`setup_logging`.
    """
    import grid_doctor as gd

    gd.setup_logging(verbosity=args.verbose, **kwargs)
