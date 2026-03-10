"""Shared utilities for grid-doctor conversion scripts."""

from __future__ import annotations

import logging
from getpass import getuser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)


def get_scratch(*args: str) -> Path:
    """Return a scratch directory, falling back to ``/tmp``.

    On DKRZ's Levante the path is ``/scratch/<initial>/<user>/<args>``.

    Parameters
    ----------
    *args : str
        Additional path components appended to the scratch root.

    Returns
    -------
    Path
        Writable scratch directory.
    """
    scratch = Path("/scratch/{0:.1}/{0}".format(getuser()))
    if scratch.is_dir():
        return scratch.joinpath(*args)
    return Path("/tmp").joinpath(*args)


class AutoRaiseSession(requests.Session):
    """A :class:`requests.Session` that raises on HTTP errors automatically."""

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        """Send a request and raise :class:`~requests.HTTPError` on failure.

        Parameters
        ----------
        *args : Any
            Positional arguments forwarded to
            :meth:`requests.Session.request`.
        **kwargs : Any
            Keyword arguments forwarded to
            :meth:`requests.Session.request`.

        Returns
        -------
        requests.Response
            The HTTP response.

        Raises
        ------
        requests.HTTPError
            On any 4xx / 5xx status code.
        """
        response = super().request(*args, **kwargs)
        response.raise_for_status()
        return response


def download_file(
    url: str,
    target_dir: str | Path,
    timeout: int = 60,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download a single URL to *target_dir*.

    Downloads are written to a ``.part`` temporary file and atomically
    renamed on success.  Existing files are skipped unless *overwrite*
    is ``True``.

    Parameters
    ----------
    url : str
        URL to download.
    target_dir : str or Path
        Directory where the file is written.
    timeout : int, optional
        Per-request timeout in seconds (default ``60``).
    overwrite : bool, optional
        Re-download even if the target file already exists.
    chunk_size : int, optional
        Read buffer size in bytes (default 1 MiB).

    Returns
    -------
    str
        Absolute path to the downloaded file.

    Raises
    ------
    ValueError
        If a filename cannot be determined from the URL.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    filename = Path(urlsplit(url).path).name
    if not filename:
        raise ValueError(f"Could not determine filename from URL: {url}")

    output_file = target_path / filename
    if output_file.exists() and not overwrite:
        logger.debug("Skipping existing download %s", output_file)
        return str(output_file)

    tmp_file = output_file.with_suffix(output_file.suffix + ".part")

    with AutoRaiseSession() as session:
        with session.get(url, stream=True, timeout=timeout) as response:
            logger.debug("Downloading %s → %s", url, tmp_file)
            with tmp_file.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

    tmp_file.replace(output_file)
    return str(output_file)
