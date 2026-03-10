"""Utilities for running scripts."""

import logging
from getpass import getuser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)


def get_scratch(*args: str) -> Path:
    """Define the scratch dir."""
    scratch = Path("/scratch/{0:.1}/{0}".format(getuser()))
    if scratch.is_dir():
        return scratch.joinpath(*args)
    return Path("/tmp").joinpath(*args)


class AutoRaiseSession(requests.Session):
    """A requests.Session that always raises for HTTP errors."""

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        response = super().request(*args, **kwargs)
        response.raise_for_status()
        return response


def download_file(
    url: str,
    target_dir: str,
    timeout: int = 60,
    overwrite: bool = False,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Download one URL to target_dir and return the output path as a string."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    filename = Path(urlsplit(url).path).name

    if not filename:
        raise ValueError(f"Could not determine filename from URL: {url}")
    output_file = target_path / filename
    if output_file.exists() and not overwrite:
        logger.debug("Skipping existing download path %s", output_file)
        return str(output_file)

    tmp_file = output_file.with_suffix(output_file.suffix + ".part")

    with AutoRaiseSession() as session:
        with session.get(url, stream=True, timeout=timeout) as response:
            logger.debug("Downloading file to %s", tmp_file)
            with tmp_file.open("wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)

    tmp_file.replace(output_file)
    return str(output_file)
