"""Caching utilities for datasets and interpolation weights."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import tempfile
from os import environ
from pathlib import Path
from typing import Any, Collection, Optional, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def get_s3_options(
    endpoint_url: str,
    secrets_file: Union[str, Path],
    **kwargs: str,
) -> dict[str, str]:
    """Build an S3 options dictionary from an endpoint and a secrets file.

    Parameters
    ----------
    endpoint_url : str
        S3-compatible endpoint URL (e.g.
        ``"https://s3.eu-dkrz-3.dkrz.cloud"``).
    secrets_file : str or Path
        Path to a JSON file containing ``accessKey`` and ``secretKey``.
    **kwargs : str
        Additional keyword arguments merged into the returned dict.

    Returns
    -------
    dict[str, str]
        Dictionary suitable for :func:`save_pyramid_to_s3`'s
        *s3_options* parameter.
    """
    secrets: dict[str, str] = json.loads(Path(secrets_file).read_text())
    return {
        "endpoint_url": endpoint_url,
        "secret": secrets["secretKey"],
        "key": secrets["accessKey"],
        **kwargs,
    }


def cache_dir() -> Path:
    """Return the directory used for caching.

    Tries ``/scratch/$USER`` first (common on HPC systems like Levante)
    and falls back to :func:`tempfile.gettempdir`.

    Returns
    -------
    Path
        Writable cache directory.
    """
    scratch = Path("/scratch/{0:.1}/{0}".format(environ["USER"]))
    if scratch.is_dir():
        scratch_temp = scratch / tempfile.gettempprefix()
        try:
            scratch_temp.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            logger.error(
                "Unable to create cache in /scratch, continuing with default"
            )
        else:
            environ["TMPDIR"] = str(scratch_temp)
            return scratch_temp
    return Path(tempfile.gettempdir())


def cached_open_dataset(
    files: Collection[str],
    **kwargs: Any,
) -> xr.Dataset:
    """Open files as a single dataset, caching the result as a pickle.

    On subsequent calls with the same file list the cached pickle is
    loaded directly, skipping the (potentially slow) ``open_mfdataset``
    call.

    Parameters
    ----------
    files : Collection[str]
        Paths (or glob patterns) to open.
    **kwargs : Any
        Forwarded to :func:`xarray.open_mfdataset`.  Defaults to
        ``parallel=True, chunks="auto"`` which can be overridden.

    Returns
    -------
    xr.Dataset
        The opened (and cached) dataset.
    """
    h = hashlib.sha256()
    h.update(np.sort(np.unique(list(files))).astype(bytes))
    pickle_file = cache_dir() / f"{h.hexdigest()}.pickle"

    if pickle_file.exists():
        logger.debug("Loading dataset from %s", pickle_file)
        try:
            with pickle_file.open("rb") as f:
                dset: xr.Dataset = pickle.load(f)  # nosec B301  # noqa: S301
                return dset
        except Exception as error:
            logger.warning("Failed to load pickle file: %s", error)
            pickle_file.unlink()

    logger.info("Opening dataset with %d files …", len(files))
    from dask.diagnostics.progress import ProgressBar

    with ProgressBar():
        merged_kwargs: dict[str, Any] = {
            "parallel": True,
            "chunks": "auto",
        } | kwargs
        ds: xr.Dataset = xr.open_mfdataset(list(files), **merged_kwargs)

    with open(pickle_file, "wb") as f:
        logger.debug("Saving dataset in %s", pickle_file)
        pickle.dump(ds, file=f)
    return ds


def cached_weights(
    ds: xr.Dataset,
    level: int,
    nest: bool = True,
    cache_path: Optional[Union[str, Path]] = None,
) -> xr.Dataset:
    """Compute (or load cached) Delaunay interpolation weights.

    Weights are expensive to compute for large grids.  This helper
    transparently caches them as NetCDF files so subsequent runs reuse
    previously computed weights.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset (any grid type supported by grid-doctor).
    level : int
        Target HEALPix level (``nside = 2**level``).
    nest : bool, optional
        HEALPix pixel ordering (default ``True`` = NESTED).
    cache_path : str, Path or None, optional
        Explicit path (file or directory) for the weight cache.
        When ``None`` the default :func:`cache_dir` is used.

    Returns
    -------
    xr.Dataset
        Weight dataset for use with
        :func:`~grid_doctor.helpers.regrid_unstructured_to_healpix`.

    See Also
    --------
    compute_weights_delaunay : Low-level weight computation.
    """
    from .helpers import _get_latlon_arrays, compute_weights_delaunay

    lat, lon = _get_latlon_arrays(ds)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(lat).ravel()).data)
    h.update(np.ascontiguousarray(np.asarray(lon).ravel()).data)
    h.update(f"level={level},nest={nest}".encode())
    key = h.hexdigest()[:16]

    if cache_path is None:
        weight_file = cache_dir() / f"weights_{key}.nc"
    else:
        p = Path(cache_path)
        weight_file = p / f"weights_{key}.nc" if p.is_dir() else p

    if weight_file.exists():
        logger.info("Loading cached weights from %s", weight_file)
        return xr.open_dataset(weight_file)

    logger.info("Computing Delaunay weights (level=%d) …", level)
    weights = compute_weights_delaunay(ds, level, nest=nest)

    weight_file.parent.mkdir(parents=True, exist_ok=True)
    weights.to_netcdf(weight_file)
    logger.info("Weights cached at %s", weight_file)
    return weights
