import hashlib
import json
import logging
import pickle
import tempfile
from os import environ
from pathlib import Path
from typing import Any, Collection, Dict, Union

import numpy as np
import xarray as xr


def get_s3_options(
    endpoint_url: str, secrets_files: Union[str, Path], **kwargs: Any
) -> Dict[str, str]:
    """Construct the s3 options parameters."""
    secrets = json.loads(Path(secrets_files).read_text())
    return {
        **{
            "endpoint_url": endpoint_url,
            "secret": secrets["secretKey"],
            "key": secrets["accessKey"],
        },
        **kwargs,
    }


def cache_dir() -> Path:
    """
    Returns the Path used for caching.
    Tries setting a cache directory in `/scratch` setting `TMPDIR` accordingly.
    Otherwise fallsback to `tempfile.gettempdir()`.
    """
    scratch = Path("/scratch/{0:.1}/{0}".format(environ["USER"]))
    if scratch.is_dir():
        scratch_temp = scratch / tempfile.gettempprefix()
        try:
            scratch_temp.mkdir(exist_ok=True, parents=True)
        except PermissionError:
            logging.error("Unable to create cache in /scratch, continuing with default")
            pass
        else:
            environ["TMPDIR"] = str(scratch_temp)
            return scratch_temp
    return Path(tempfile.gettempdir())


def cached_open_dataset(files: Collection[str], **kwargs: Any):
    """
    Opens specified files as single dataset, caching the resulting xarray.Dataset using pickle.
    """
    h = hashlib.sha256()
    h.update(np.sort(np.unique(files)).astype(bytes))
    pickle_file = cache_dir() / f"{h.hexdigest()}.pickle"

    if pickle_file.exists():
        logging.debug("Loading dataset from %s", pickle_file)
        with open(pickle_file, "rb") as f:
            _ds = pickle.load(f)
            return _ds

    print(
        f"Opening dataset with {len(files)} files ..." + (" " * 40),
        flush=True,
        end="\r",
    )
    from dask.diagnostics import ProgressBar

    with ProgressBar():
        _kwargs = {"parallel": True, "chunks": "auto"} | kwargs
        _ds = xr.open_mfdataset(files, **_kwargs)

    with open(pickle_file, "wb") as f:
        logging.debug("Saving dataset in %s", pickle_file)
        pickle.dump(_ds, file=f)
    return _ds


def cached_weights(
    ds,
    level: int,
    nest: bool = True,
    cache_path=None,
):
    """Compute (or load cached) Delaunay interpolation weights.

    Weights are expensive to compute for large grids.  This helper
    transparently caches them as NetCDF files so subsequent runs reuse
    previously computed weights.

    Parameters
    ----------
    ds:
        Input dataset (any grid type supported by grid-doctor).
    level:
        Target HEALPix level.
    nest:
        HEALPix ordering.
    cache_path:
        Explicit path (file or directory) for the weight cache.  When
        ``None`` the default :func:`cache_dir` is used.

    Returns
    -------
    xr.Dataset:
        Weight dataset for use with :func:`helpers.regrid_unstructured_to_healpix`.
    """
    from .helpers import _get_latlon_arrays, compute_weights_delaunay

    # Build a deterministic cache key from the source grid coordinates
    # and target level so weights are only recomputed when the grid changes.
    lat, lon = _get_latlon_arrays(ds)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(lat).ravel()).data)
    h.update(np.ascontiguousarray(np.asarray(lon).ravel()).data)
    h.update(f"level={level},nest={nest}".encode())
    key = h.hexdigest()[:16]

    if cache_path is None:
        weight_file = cache_dir() / f"weights_{key}.nc"
    else:
        cache_path = Path(cache_path)
        if cache_path.is_dir():
            weight_file = cache_path / f"weights_{key}.nc"
        else:
            weight_file = cache_path

    if weight_file.exists():
        logging.info("Loading cached weights from %s", weight_file)
        return xr.open_dataset(weight_file)

    logging.info("Computing Delaunay weights (level=%d) ...", level)
    weights = compute_weights_delaunay(ds, level, nest=nest)

    weight_file.parent.mkdir(parents=True, exist_ok=True)
    weights.to_netcdf(weight_file)
    logging.info("Weights cached at %s", weight_file)

    return weights
