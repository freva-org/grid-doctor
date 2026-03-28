"""Caching utilities for datasets and HEALPix weight files."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import tempfile
from os import environ
from pathlib import Path
from typing import Any, Collection, Literal, cast

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def chunk_for_target_store_size(
    *,
    level: int,
    dtype: str | np.dtype = "float32",
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
            raise ValueError(
                "For access='time_series', provide either ntime or max_time_chunk."
            )

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


def cache_dir() -> Path:
    """Return the writable cache directory used by grid-doctor.

    The function prefers `/scratch/<initial>/<user>/tmp*` on systems such as
    Levante and falls back to `tempfile.gettempdir`.
    """
    scratch = Path("/scratch/{0:.1}/{0}".format(environ["USER"]))
    if scratch.is_dir():
        scratch_temp = scratch / tempfile.gettempprefix()
        try:
            scratch_temp.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            logger.warning(
                "Could not create %s, falling back to the default temp dir.",
                scratch_temp,
            )
        else:
            environ["TMPDIR"] = str(scratch_temp)
            return scratch_temp
    return Path(tempfile.gettempdir())


def cached_open_dataset(files: Collection[str], **kwargs: Any) -> xr.Dataset:
    """Open multiple files and cache the merged dataset as a pickle.

    Parameters
    ----------
    files:
        Input file paths or glob-expanded file names.
    **kwargs:
        Extra keyword arguments for `xarray.open_mfdataset`.

    Returns
    -------
    xarray.Dataset
        The opened dataset.
    """
    digest = hashlib.sha256()
    normalised = sorted({str(path) for path in files})
    digest.update("\0".join(normalised).encode())
    pickle_file = cache_dir() / f"{digest.hexdigest()}.pickle"

    if pickle_file.exists():
        try:
            with pickle_file.open("rb") as handle:
                return cast(xr.Dataset, pickle.load(handle))  # nosec B301  # noqa: S301
        except Exception as exc:  # pragma: no cover - defensive cache cleanup
            logger.warning("Could not read cached dataset %s: %s", pickle_file, exc)
            pickle_file.unlink(missing_ok=True)

    from dask.diagnostics.progress import ProgressBar

    merged_kwargs: dict[str, Any] = {"parallel": True, "chunks": "auto"} | kwargs
    with ProgressBar():
        dataset = xr.open_mfdataset(normalised, **merged_kwargs)

    with pickle_file.open("wb") as handle:
        pickle.dump(dataset, handle)
    return dataset


def cached_weights(
    ds: xr.Dataset,
    level: int | None = None,
    *,
    method: Literal["nearest", "conservative"] = "conservative",
    nest: bool = True,
    source_units: Literal["auto", "rad", "deg"] = "auto",
    cache_path: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Compute or load a cached HEALPix weight file.

    Parameters
    ----------
    ds:
        Source dataset whose grid geometry defines the weights.
    level:
        HEALPix level.
    method:
        Weight-generation method. Supported values are `"nearest"` and
        `"conservative"`.
    nest:
        Use nested HEALPix ordering when `True`.
    source_units:
        Unit convention of the source coordinates.
    cache_path:
        Cache directory or explicit file name. When omitted,
        [`cache_dir`][grid_doctor.utils.cache_dir] is used.
    **kwargs:
        Any additional keyword arguments for
        [`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights]

    Returns
    -------
    pathlib.Path
        Path to the cached NetCDF weight file.

    Examples
    --------
    ```python
    weight_file = cached_weights(ds, level=8, method="conservative")
    ```
    """
    from .remap import compute_healpix_weights
    from .helpers import resolution_to_healpix_level, get_latlon_resolution

    digest = hashlib.sha256()
    for candidate in (
        "clon_vertices",
        "clat_vertices",
        "lon_vertices",
        "lat_vertices",
        "clon",
        "clat",
        "lon",
        "lat",
        "longitude",
        "latitude",
        "rlon",
        "rlat",
        "X",
        "Y",
    ):
        if candidate in ds:
            digest.update(
                np.ascontiguousarray(np.asarray(ds[candidate].values)).tobytes()
            )
    digest.update(
        f"level={level};method={method};nest={nest};units={source_units}".encode()
    )
    key = digest.hexdigest()[:16]

    if cache_path is None:
        weight_file = cache_dir() / f"weights_{key}.nc"
    else:
        path = Path(cache_path)
        weight_file = path / f"weights_{key}.nc" if path.is_dir() else path
    if weight_file.exists():
        logger.info("Using cached weight file %s", weight_file)
        return weight_file

    logger.info("Generating HEALPix weight file %s", weight_file)
    if level is None:
        level = resolution_to_healpix_level(get_latlon_resolution(ds))

    return compute_healpix_weights(
        ds,
        level,
        method=method,
        nest=nest,
        source_units=source_units,
        weights_path=weight_file,
        **kwargs,
    )


__all__ = [
    "cache_dir",
    "cached_open_dataset",
    "chunk_for_target_store_size",
    "cached_weights",
    "get_s3_options",
]
