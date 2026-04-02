"""High-level helpers for HEALPix pyramids.

The functions in this module cover three tasks:

- estimating source-grid resolution,
- building a HEALPix pyramid with
  [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix],
- and writing the resulting pyramid to Zarr stores.

Remapping itself lives in [`grid_doctor.remap`][grid_doctor.remap].
"""

from __future__ import annotations

import logging
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import s3fs
import xarray as xr

from .remap import (
    _make_crs_variable,
    regrid_to_healpix,
    regrid_unstructured_to_healpix,
)
from .remap_backend import (
    _get_latlon_arrays,
    _get_unstructured_dim,
    _is_unstructured,
)
from .types import FloatArray, ZarrOptions

logger = logging.getLogger(__name__)


# ===================================================================
# Resolution estimation
# ===================================================================


def get_latlon_resolution(ds: xr.Dataset) -> float:
    """Estimate the horizontal resolution of *ds* in degrees.

    Parameters
    ----------
    ds:
        Dataset on a regular lon/lat grid, a curvilinear grid, or an
        unstructured grid.

    Returns
    -------
    float
        Approximate grid spacing in degrees.

    Examples
    --------
    ```python
    resolution = get_latlon_resolution(ds)
    level = resolution_to_healpix_level(resolution)
    ```
    """
    if _is_unstructured(ds):
        cell_dim = _get_unstructured_dim(ds)
        n_cells = ds.sizes[cell_dim]
        return float(np.degrees(np.sqrt(4.0 * np.pi / float(n_cells))))

    lat, lon = _get_latlon_arrays(ds)
    if lat.ndim == 1:
        lat_res = float(np.nanmin(np.abs(np.diff(lat))))
        lon_res = float(np.nanmin(np.abs(np.diff(lon))))
    else:
        lat_res = float(np.nanmean(np.abs(np.diff(lat, axis=0))))
        lon_res = float(np.nanmean(np.abs(np.diff(lon, axis=1))))
    return min(lat_res, lon_res)


def resolution_to_healpix_level(resolution_deg: float) -> int:
    """Convert an approximate source resolution to a HEALPix level.

    The heuristic uses a characteristic HEALPix pixel spacing of about
    ``58.6° / 2**level``.

    Parameters
    ----------
    resolution_deg:
        Approximate source resolution in degrees.

    Returns
    -------
    int
        Suggested HEALPix level.

    Examples
    --------
    ```python
    level = resolution_to_healpix_level(0.25)
    ```
    """
    if resolution_deg <= 0.0:
        raise ValueError("resolution_deg must be positive.")
    level = int(np.floor(np.log2(58.6 / resolution_deg)))
    return max(0, level)


# ===================================================================
# HEALPix coordinate helpers
# ===================================================================


def _healpix_coords(
    level: int,
    *,
    nest: bool,
) -> tuple[FloatArray, FloatArray]:
    """Return HEALPix cell centres for *level*.

    Delegates to
    [`_healpix_centres`][grid_doctor.remap._healpix_centres].

    Args:
        level: HEALPix refinement level.
        nest: Nested ordering when *True*.

    Returns:
        ``(lat_deg, lon_deg)`` arrays.
    """
    from .remap import _healpix_centres

    return _healpix_centres(level, nest=nest)


# ===================================================================
# Coarsening
# ===================================================================


def _coarsen_array(
    values: npt.NDArray[np.floating[Any]],
    *,
    factor: int,
) -> FloatArray:
    """Coarsen a HEALPix array by grouping contiguous nested cells.

    The last dimension is treated as the cell dimension.  All leading
    dimensions are batch dimensions that are preserved.

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of child cells per parent (``4**delta_level``).

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    batch_shape = arr.shape[:-1]
    n_cells = arr.shape[-1]
    n_target = n_cells // factor
    grouped = arr.reshape(*batch_shape, n_target, factor)
    with np.errstate(invalid="ignore"):
        return cast(
            FloatArray, np.nanmean(grouped, axis=-1).astype(np.float64, copy=False)
        )


def coarsen_healpix(
    ds: xr.Dataset,
    target_level: int,
) -> xr.Dataset:
    """Coarsen a HEALPix dataset to a lower-resolution level.

    The coarsening is performed as a single reshape + nanmean over
    all batch dimensions simultaneously — no per-slice Python loops.

    Parameters
    ----------
    ds:
        HEALPix dataset containing a ``cell`` dimension and the
        attributes ``healpix_nside`` and ``healpix_order``.
    target_level:
        Target HEALPix level (must be lower than the current level).

    Returns
    -------
    xarray.Dataset
        Coarsened dataset.

    Notes
    -----
    Nested HEALPix indices have a direct parent-child relationship:
    pixel *i* at level *L* contains children ``4*i`` to ``4*i+3`` at
    level *L+1*.  Coarsening therefore reduces to grouping contiguous
    blocks of ``4**delta_level`` child cells and averaging.

    Ring-ordered datasets do not have contiguous parent-child layout
    and must be remapped directly at each target level.

    Raises
    ------
    ValueError
        When the ordering is not nested, or *target_level* is not
        lower than the current level.
    """
    current_nside = int(ds.attrs["healpix_nside"])
    target_nside = 2**target_level
    if target_nside >= current_nside:
        raise ValueError("target_level must be lower than the current HEALPix level.")

    is_nested = str(ds.attrs.get("healpix_order", "nested")) in {"nested", "nest"}
    if not is_nested:
        raise ValueError(
            "coarsen_healpix only supports nested HEALPix ordering. "
            "Use create_healpix_pyramid(..., nest=False) to "
            "regenerate ring levels directly."
        )

    current_level = int(ds.attrs.get("healpix_level", int(np.log2(current_nside))))
    delta_level = current_level - target_level
    if delta_level <= 0:
        raise ValueError("target_level must be lower than the current HEALPix level.")

    factor = 4**delta_level
    npix_target = ds.sizes["cell"] // factor

    coarsened_vars: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if "cell" not in data.dims:
            coarsened_vars[str(name)] = data
            continue

        coarsened_vars[str(name)] = cast(
            xr.DataArray,
            xr.apply_ufunc(
                _coarsen_array,
                data,
                input_core_dims=[["cell"]],
                output_core_dims=[["cell"]],
                exclude_dims={"cell"},
                dask="parallelized",
                kwargs={"factor": factor},
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"cell": npix_target}},
                keep_attrs=True,
            ),
        )

    result = xr.Dataset(coarsened_vars, attrs=ds.attrs.copy())
    lat_deg, lon_deg = _healpix_coords(target_level, nest=True)
    result = result.assign_coords(
        cell=np.arange(npix_target, dtype=np.int64),
        latitude=("cell", lat_deg),
        longitude=("cell", lon_deg),
        crs=_make_crs_variable(
            level=target_level,
            nside=target_nside,
            order="nested",
        ),
    )

    # Tag every spatially-mapped data variable.
    for name in result.data_vars:
        if "cell" in result[name].dims:
            result[name].attrs["grid_mapping"] = "crs"

    result.attrs["healpix_nside"] = target_nside
    result.attrs["healpix_level"] = target_level
    result.attrs["healpix_order"] = "nested"
    result.attrs["grid_doctor_coarsened_from_level"] = current_level
    return result


# ===================================================================
# Pyramid construction
# ===================================================================


def create_healpix_pyramid(
    ds: xr.Dataset,
    max_level: int | None = None,
    min_level: int = 0,
    **kwargs: Any,
) -> dict[int, xr.Dataset]:
    """Create a multi-resolution HEALPix pyramid.

    The input dataset is first remapped to *max_level* with
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].
    For nested output ordering, lower levels are derived efficiently
    with
    [`coarsen_healpix`][grid_doctor.helpers.coarsen_healpix].
    For ring ordering, each lower level is regenerated directly from
    the source dataset.

    Parameters
    ----------
    ds:
        Source dataset.
    max_level:
        Finest HEALPix level.
    min_level:
        Coarsest HEALPix level to keep.
    **kwargs:
        Forwarded to
        [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

    Returns
    -------
    dict[int, xarray.Dataset]
        Pyramid keyed by level.
    """
    if max_level is None:
        max_level = resolution_to_healpix_level(get_latlon_resolution(ds))

    pyramid: dict[int, xr.Dataset] = {}
    finest = regrid_to_healpix(ds, max_level, **kwargs)
    pyramid[max_level] = finest

    is_nested = bool(kwargs.get("nest", True))
    if is_nested:
        current = finest
        for level in range(max_level - 1, min_level - 1, -1):
            current = coarsen_healpix(current, level)
            pyramid[level] = current
        return pyramid

    for level in range(max_level - 1, min_level - 1, -1):
        pyramid[level] = regrid_to_healpix(ds, level, **kwargs)
    return pyramid


# ===================================================================
# S3 / Zarr output
# ===================================================================


def save_pyramid_to_s3(
    pyramid: dict[int, xr.Dataset],
    s3_path: str,
    s3_options: dict[str, Any],
    *,
    mode: Literal["a", "w", "r+"] = "a",
    compute: bool = True,
    region: Literal["auto"] | dict[str, slice] = "auto",
    zarr_format: Literal[2, 3] = 2,
    encoding: dict[int, dict[str, dict[str, Any]]] | None = None,
) -> None:
    """Write a HEALPix pyramid to S3-backed Zarr stores.

    Each level is stored below ``"<s3_path>/level_<level>.zarr"``.

    Parameters
    ----------
    pyramid:
        Mapping of HEALPix level to dataset.
    s3_path:
        S3 prefix such as ``"s3://bucket/pyramid"``.
    s3_options:
        Options forwarded to :class:`s3fs.S3FileSystem`.
    mode:
        Zarr write mode.
    compute:
        Trigger Dask execution immediately when ``True``.
    region:
        Region writes for partial updates.
    zarr_format:
        Zarr format version.
    encoding:
        Per-level encoding dictionaries.
    """
    fs = s3fs.S3FileSystem(**s3_options)
    for level, dataset in pyramid.items():
        level_path = f"{s3_path}/level_{level}.zarr"
        logger.info("Writing HEALPix level %s to %s", level, level_path)
        store = s3fs.S3Map(root=level_path, s3=fs)
        zarr_options = ZarrOptions(compute=compute, mode=mode, zarr_format=zarr_format)
        if zarr_format == 2:
            zarr_options["consolidated"] = True
        if encoding is not None:
            zarr_options["encoding"] = encoding[level]

        if region == "auto":
            dataset.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
        else:
            region_keys = set(region)
            to_drop = (
                {
                    name
                    for name, var in dataset.data_vars.items()
                    if region_keys.isdisjoint(map(str, var.dims))
                }
                | {str(dim) for dim in dataset.dims}
                | {str(coord) for coord in dataset.coords}
            )
            dataset.drop_vars(to_drop, errors="ignore").isel(region).to_zarr(
                store,
                region=region,
                **zarr_options,
            )  # type: ignore[call-overload]

        if mode == "w" and not compute:
            coord_options = dict(zarr_options)
            coord_options["mode"] = "w"
            dataset[list(dataset.coords)].to_zarr(store, **coord_options)  # type: ignore[call-overload]


# ===================================================================
# Convenience aliases
# ===================================================================


def latlon_to_healpix_pyramid(
    ds: xr.Dataset,
    *,
    min_level: int = 0,
    max_level: int | None = None,
    **kwargs: Any,
) -> dict[int, xr.Dataset]:
    """Convert a source dataset into a HEALPix pyramid.

    Parameters
    ----------
    ds:
        Source dataset.
    min_level:
        Coarsest level to keep.
    max_level:
        Finest level to generate.  When omitted, the level is
        estimated from the source-grid resolution.
    **kwargs:
        Forwarded to
        [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

    Returns
    -------
    dict[int, xarray.Dataset]
        HEALPix pyramid keyed by level.

    Examples
    --------
    ```python
    pyramid = latlon_to_healpix_pyramid(ds, method="nearest")
    ```
    """
    return create_healpix_pyramid(
        ds,
        max_level=max_level,
        min_level=min_level,
        **kwargs,
    )


__all__ = [
    "coarsen_healpix",
    "create_healpix_pyramid",
    "get_latlon_resolution",
    "latlon_to_healpix_pyramid",
    "regrid_to_healpix",
    "regrid_unstructured_to_healpix",
    "resolution_to_healpix_level",
    "save_pyramid_to_s3",
]
