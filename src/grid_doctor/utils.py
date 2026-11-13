"""Caching utilities for datasets and HEALPix weight files."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import tempfile
import warnings
from os import environ
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Literal, Unpack, cast

import numpy as np
import numpy.typing as npt
import xarray as xr

from .types import RemapMethod, SourceUnits, WeightOptions

if TYPE_CHECKING:
    from .weights import WeightsStore

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

    # Passed through to xarray unchanged, so typed like its ``**kwargs``.
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
    method: RemapMethod = "conservative",
    nest: bool = True,
    source_units: SourceUnits = "auto",
    store: WeightsStore | str | Path | None = None,
    **kwargs: Unpack[WeightOptions],
) -> Path:
    """Return a local HEALPix weight file, reusing it through a store.

    Weights are identified by their
    [`WeightsDescription`][grid_doctor.weights.WeightsDescription]: the
    normalised source geometry, the HEALPix target and the method. The
    lookup is done by
    [`WeightsStore.fetch_or_compute`][grid_doctor.weights.WeightsStore.fetch_or_compute].

    Parameters
    ----------
    ds:
        Source dataset whose grid geometry defines the weights.
    level:
        HEALPix level. Derived from the grid resolution when omitted.
    method:
        Weight-generation method. Supported values are `"nearest"` and
        `"conservative"`.
    nest:
        Deprecated, has no effect: weight files are nested. Choose ring
        order where the weights are applied, e.g.
        ``apply_weight_file(..., nest=False)`` or
        ``create_healpix_pyramid(..., nest=False)``.
    source_units:
        Unit convention of the source coordinates.
    store:
        Weights store, or its location (see
        [`WeightsStore`][grid_doctor.weights.WeightsStore]). Defaults to the
        ``GRID_DOCTOR_WEIGHTS_STORE`` environment variable, else the local
        weights directory.
    **kwargs:
        Any additional keyword arguments for
        [`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights]

    Returns
    -------
    pathlib.Path
        Path to the local NetCDF weight file.

    Examples
    --------
    ```python
    weight_file = cached_weights(ds, level=8, store="/work/ab1234/weights")
    ```
    """
    from .helpers import get_latlon_resolution, resolution_to_healpix_level
    from .metadata import HealpixGrid
    from .remap import compute_healpix_weights
    from .weights import WeightsDescription
    from .weights import WeightsStore as _WeightsStore
    from .weights import check_coverage, describe_source

    if not nest:
        warnings.warn(
            "`nest=False` has no effect: weight files are nested; pass `nest=False` where they are applied.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not isinstance(store, _WeightsStore):
        store = _WeightsStore(store)
    if level is None:
        level = resolution_to_healpix_level(get_latlon_resolution(ds))

    describe_kwargs = {k: kwargs[k] for k in ("grid", "source_kind", "spectral_transform_command") if k in kwargs}
    source = describe_source(ds, source_units=source_units, **describe_kwargs)  # type: ignore[arg-type]
    desc = WeightsDescription(source, HealpixGrid(level=level), method)  # stored weights are nested

    def compute(path: Path) -> Path:  # the store holds nested weights; ring is derived from them
        return compute_healpix_weights(
            ds,
            level,
            method=method,
            nest=True,
            source_units=source_units,
            weights_path=path,
            source_description=source,
            **kwargs,
        )

    path = store.fetch_or_compute(desc, compute)
    if kwargs.get("ignore_unmapped") is False:
        check_coverage(path)  # the stored weights may have been computed ignoring unmapped cells
    return path


__all__ = [
    "cache_dir",
    "cached_open_dataset",
    "chunk_for_target_store_size",
    "cached_weights",
    "get_s3_options",
]
