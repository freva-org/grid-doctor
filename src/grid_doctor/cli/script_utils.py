"""Shared helpers for grid-doctor conversion scripts."""

from __future__ import annotations

import logging
from getpass import getuser
from pathlib import Path
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)


def get_scratch(*parts: str) -> Path:
    """Return a writable scratch directory.

    Parameters
    ----------
    *parts:
        Extra path components appended to the scratch root.

    Returns
    -------
    pathlib.Path
        Scratch directory on `/scratch/<initial>/<user>` when available,
        otherwise below `/tmp`.
    """
    scratch = Path("/scratch/{0:.1}/{0}".format(getuser()))
    if scratch.is_dir():
        return scratch.joinpath(*parts)
    return Path("/tmp").joinpath(*parts)


class AutoRaiseSession(requests.Session):
    """`requests.Session` variant that raises on HTTP errors automatically."""

    def request(self, *args: object, **kwargs: object) -> requests.Response:
        """Send one request and raise for unsuccessful HTTP status codes.

        Parameters
        ----------
        *args:
            Positional arguments forwarded to
            `requests.Session.request`.
        **kwargs:
            Keyword arguments forwarded to
            `requests.Session.request`.

        Returns
        -------
        requests.Response
            HTTP response object with `raise_for_status()` already applied.
        """
        response = super().request(*args, **kwargs)  # type: ignore
        response.raise_for_status()
        return response


def download_file(
    url: str,
    target_dir: str | Path,
    *,
    timeout: int = 60,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download one URL into `target_dir`.

    Parameters
    ----------
    url:
        Source URL.
    target_dir:
        Output directory.
    timeout:
        Per-request timeout in seconds.
    overwrite:
        Replace an existing file when `True`.
    chunk_size:
        Streaming chunk size in bytes.

    Returns
    -------
    str
        Absolute path of the downloaded file.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    filename = Path(urlsplit(url).path).name
    if not filename:
        raise ValueError(f"Could not determine a filename from URL: {url}")

    output_file = target_path / filename
    if output_file.exists() and not overwrite:
        logger.debug("Skipping existing download %s", output_file)
        return str(output_file)

    temp_file = output_file.with_suffix(output_file.suffix + ".part")
    with AutoRaiseSession() as session:
        with session.get(url, stream=True, timeout=timeout) as response:
            with temp_file.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)

    temp_file.replace(output_file)
    return str(output_file)
