"""Store options: S3 credentials and chunk sizes for HEALPix stores."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


def chunk_for_target_store_size(
    *,
    level: int,
    dtype: npt.DTypeLike = "float32",
    target_stored_mib: float = 16.0,
    compression_ratio: float = 2.0,
    access: Literal["time_series", "map"] = "map",
    ntime: int | None = None,
    max_time_chunk: int | None = None,
    max_cell_chunk: int | None = None,
) -> dict[str, int]:
    """
    Compute (time, cell) chunks for a HEALPix dataset.

    Parameters
    ----------
    level
        HEALPix order / level.
    dtype
        Variable dtype.
    target_stored_mib
        Desired approximate compressed chunk size on disk.
    compression_ratio
        Estimated ratio:
            uncompressed_bytes / compressed_bytes
    access
        "map" or "time_series".
    ntime
        Total time size. Needed for time_series mode unless max_time_chunk is given.
    max_time_chunk
        Optional cap for time chunk.
    max_cell_chunk
        Optional cap for cell chunk.

    Returns
    -------
    dict[str, int]
        Chunk sizes, e.g. {"time": 5, "cell": 786432}.
    """
    nside = 2**level
    ncell = 12 * nside * nside
    itemsize = np.dtype(dtype).itemsize

    target_stored_bytes = int(target_stored_mib * 1024 * 1024)
    target_uncompressed_bytes = int(target_stored_bytes * compression_ratio)

    if access == "map":
        cell_chunk = ncell if max_cell_chunk is None else min(ncell, max_cell_chunk)
        time_chunk = max(1, target_uncompressed_bytes // (itemsize * cell_chunk))
        return {"time": int(time_chunk), "cell": int(cell_chunk)}

    if access == "time_series":
        if max_time_chunk is not None:
            time_chunk = max_time_chunk
        elif ntime is not None:
            time_chunk = ntime
        else:
            raise ValueError("For access='time_series', provide either ntime or max_time_chunk.")

        if ntime is not None:
            time_chunk = min(time_chunk, ntime)

        cell_chunk = max(1, target_uncompressed_bytes // (itemsize * time_chunk))
        cell_chunk = min(cell_chunk, ncell)

        if max_cell_chunk is not None:
            cell_chunk = min(cell_chunk, max_cell_chunk)

        return {"time": int(time_chunk), "cell": int(cell_chunk)}

    raise ValueError(f"Unsupported access mode: {access!r}")


def get_s3_options(
    endpoint_url: str,
    secrets_file: str | Path,
    **kwargs: str,
) -> dict[str, str]:
    """Build an S3 options dictionary from an endpoint and credentials file.

    Parameters
    ----------
    endpoint_url:
        S3-compatible endpoint URL.
    secrets_file:
        JSON file containing `accessKey` and `secretKey`.
    **kwargs:
        Additional options merged into the returned dictionary.

    Returns
    -------
    dict[str, str]
        Options for `s3fs.S3FileSystem`.
    """
    secrets: dict[str, str] = json.loads(Path(secrets_file).read_text())
    return {
        "endpoint_url": endpoint_url,
        "secret": secrets["secretKey"],
        "key": secrets["accessKey"],
        **kwargs,
    }
