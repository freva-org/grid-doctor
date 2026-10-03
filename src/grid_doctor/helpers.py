"""High-level helpers for HEALPix pyramids.

The functions in this module cover three tasks:

- estimating source-grid resolution,
- building a HEALPix pyramid with
  [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix],
- and writing the resulting pyramid to Zarr stores.

Remapping itself lives in [`grid_doctor.remap`][grid_doctor.remap].
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, Literal, NamedTuple, Union, cast

import dask.array as da
import numpy as np
import numpy.typing as npt
import s3fs
import xarray as xr
import zarr
from xarray.backends.zarr import encode_zarr_variable

from .pyramid import (
    MeanPartials,
    coarsen_mean,
    coarsen_mean_steps,
    coarsen_mode_steps,
)
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
from .types import CoarsenMode, FloatArray, ZarrOptions

logger = logging.getLogger(__name__)

WRITE_COORDS_MAX_LEVEL = 10
"""Highest level for which ``save_pyramid(write_coords="auto")``
materialises coordinate arrays.  At level 10 the two float64 coordinate
arrays cost ~200 MB per store; one level up they double, and by level 16
they would reach hundreds of GB while carrying no information that is not
already implied by the cell index."""

# ``encode_zarr_variable`` gained ``zarr_format`` in recent xarray releases.
_ENCODE_TAKES_FORMAT = (
    "zarr_format" in inspect.signature(encode_zarr_variable).parameters
)


def _encode_zarr_variable(
    var: xr.Variable, *, name: str, zarr_format: Literal[2, 3]
) -> xr.Variable:
    """Encode *var* exactly as ``to_zarr`` would, across xarray versions."""
    kwargs: Dict[str, Union[str, Literal[2, 3]]] = {"name": name}
    if _ENCODE_TAKES_FORMAT:
        kwargs["zarr_format"] = zarr_format
    encoded = encode_zarr_variable(var, **kwargs)  # type: ignore
    return cast(xr.Variable, encoded)


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
    level = int(np.round(np.log2(58.6 / resolution_deg)))
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
    min_valid_fraction: float = 0.5,
) -> FloatArray:
    """Coarsen a HEALPix array by grouping contiguous nested cells.

    The last dimension is treated as the cell dimension.  All leading
    dimensions are batch dimensions that are preserved.

    Thin wrapper around
    [`coarsen_mean`][grid_doctor.pyramid.coarsen_mean], which
    is exact only when ``values`` is the finest level.

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of child cells per parent (``4**delta_level``).
        min_valid_fraction: Minimum fraction of valid (non-NaN)
            children required.  Parent cells with fewer valid
            children are set to NaN.  Default ``0.5`` (at least
            half of the children must be valid).

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    return coarsen_mean(values, factor=factor, min_valid_fraction=min_valid_fraction)


def _coarsen_array_mode(
    values: npt.NDArray[np.floating[Any]],
    *,
    factor: int,
    min_valid_fraction: float = 0.5,
) -> FloatArray:
    """Coarsen a HEALPix array by taking the mode of grouped children.

    Intended for categorical / discrete data (land cover, soil type,
    …) where averaging class labels would be meaningless.

    For each parent cell the most frequent value among its children is
    selected.  Ties are broken by choosing the value that appears
    first (lowest child index).  When fewer than
    ``min_valid_fraction * factor`` children are valid, the parent is
    set to NaN.

    Args:
        values: Input array with shape ``(*batch, n_cells)``.
        factor: Number of child cells per parent (``4**delta_level``).
        min_valid_fraction: Minimum fraction of valid children.

    Returns:
        Array with shape ``(*batch, n_cells // factor)``.
    """
    arr = np.asarray(values, dtype=np.float64)
    batch_shape = arr.shape[:-1]
    n_cells = arr.shape[-1]
    n_target = n_cells // factor
    grouped = arr.reshape(*batch_shape, n_target, factor)

    valid = np.isfinite(grouped)
    valid_count = valid.sum(axis=-1)

    # For each child position, count how many of the other children
    # share the same value.  This vectorises the mode calculation
    # without any Python-level loops over cells.
    counts = np.zeros_like(grouped)
    for i in range(factor):
        for j in range(factor):
            counts[..., i] += (grouped[..., i] == grouped[..., j]) & valid[..., j]
    counts[~valid] = -1  # invalid positions cannot win

    best = np.argmax(counts, axis=-1)
    result = np.take_along_axis(
        grouped,
        best[..., np.newaxis],
        axis=-1,
    ).squeeze(-1)

    min_count = max(1, int(np.ceil(min_valid_fraction * factor)))
    result[valid_count < min_count] = np.nan
    return result


def _resolve_coarsen_mode(ds: xr.Dataset, coarsen_mode: CoarsenMode) -> CoarsenMode:
    """Resolve ``"auto"`` to ``"mean"`` or ``"mode"`` from dataset attributes."""
    if coarsen_mode != "auto":
        return coarsen_mode
    method = str(ds.attrs.get("grid_doctor_method", "conservative"))
    is_categorical = method == "nearest" or method.endswith("-mode")
    return "mode" if is_categorical else "mean"


def coarsen_healpix(
    ds: xr.Dataset,
    target_level: int,
    coarsen_mode: CoarsenMode = "auto",
    min_valid_fraction: float = 0.5,
) -> xr.Dataset:
    """Coarsen a HEALPix dataset to a lower-resolution level.

    The coarsening is performed as a single reshape + reduction over
    all batch dimensions simultaneously — no per-slice Python loops.

    Parameters
    ----------
    ds:
        HEALPix dataset containing a ``cell`` dimension and the
        attributes ``healpix_nside`` and ``healpix_order``.
    target_level:
        Target HEALPix level (must be lower than the current level).
    coarsen_mode:
        ``"mean"`` — NaN-aware averaging for continuous fields.
        ``"mode"`` — most-frequent-value for categorical fields.
        ``"auto"`` — infer from ``grid_doctor_method`` in dataset
        attributes: ``"nearest"`` → mode, everything else → mean.
    min_valid_fraction:
        Minimum fraction of valid (non-NaN) children required to
        produce a valid parent cell.  Parents with fewer valid
        children are set to NaN.  Default ``0.5`` (at least half
        of the children must be valid).

    Returns
    -------
    xarray.Dataset
        Coarsened dataset.

    Notes
    -----
    Nested HEALPix indices have a direct parent-child relationship:
    pixel *i* at level *L* contains children ``4*i`` to ``4*i+3`` at
    level *L+1*.  Coarsening therefore reduces to grouping contiguous
    blocks of ``4**delta_level`` child cells and averaging (or taking
    the mode for categorical data).

    Ring-ordered datasets do not have contiguous parent-child layout
    and must be remapped directly at each target level.

    With ``coarsen_mode="mean"``, always coarsen from the finest level
    rather than chaining level by level.  A chained mean weights every
    valid parent equally, regardless of how many valid finest-level
    cells it was built from, so data with NaNs (land, sea ice,
    observation gaps) drifts between levels.

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

    resolved_mode = _resolve_coarsen_mode(ds, coarsen_mode)
    cell_chunk = _cell_chunk_of(ds)

    coarsened_vars: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if "cell" not in data.dims:
            coarsened_vars[str(name)] = data
            continue
        template = data.transpose(..., "cell")
        if resolved_mode == "mode":
            out = coarsen_mode_steps(
                template.data,
                delta_level,
                kernel=_coarsen_array_mode,
                min_valid_fraction=min_valid_fraction,
                cell_chunk=cell_chunk,
            )
        else:
            out = coarsen_mean_steps(
                template.data,
                delta_level,
                min_valid_fraction=min_valid_fraction,
                cell_chunk=cell_chunk,
            )
        coarsened_vars[str(name)] = _wrap_cells(template, out)

    return _assemble_coarse_level(ds, coarsened_vars, target_level, current_level)


def _cell_chunk_of(ds: xr.Dataset) -> int | None:
    """Cell chunk size of the first dask-backed cell variable, if any."""
    for data in ds.data_vars.values():
        if "cell" in data.dims and data.chunks is not None:
            return int(data.chunks[data.get_axis_num("cell")][0])
    return None


def _wrap_cells(template: xr.DataArray, values: Any) -> xr.DataArray:
    """Wrap coarsened *values* like *template* (cell last), minus cell coords."""
    coords = {str(k): v for k, v in template.coords.items() if "cell" not in v.dims}
    return xr.DataArray(
        values,
        dims=template.dims,
        coords=coords,
        attrs=template.attrs.copy(),
        name=template.name,
    )


def _assemble_coarse_level(
    template: xr.Dataset,
    data_vars: dict[str, xr.DataArray],
    target_level: int,
    source_level: int,
) -> xr.Dataset:
    """Build a coarse-level dataset with HEALPix coordinates and metadata."""
    target_nside = 2**target_level
    npix_target = 12 * target_nside**2
    result = xr.Dataset(data_vars, attrs=template.attrs.copy())
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
    result.attrs["grid_doctor_coarsened_from_level"] = source_level
    return result


# ===================================================================
# Pyramid construction
# ===================================================================


def create_healpix_pyramid(
    ds: xr.Dataset,
    max_level: int | None = None,
    min_level: int = 0,
    *,
    coarsen_mode: CoarsenMode = "auto",
    min_valid_fraction: float = 0.5,
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

    For dask-backed input the result is lazy and chunked along ``cell``
    (``cell_chunks``, forwarded to
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]); coarser
    levels keep that chunk size.  Every level builds on the finest one,
    so write the pyramid with
    [`save_pyramid`][grid_doctor.helpers.save_pyramid], which computes
    all levels in one pass.  Calling ``.compute()`` per level instead
    regrids the source once per level.

    Parameters
    ----------
    ds:
        Source dataset.
    max_level:
        Finest HEALPix level.
    min_level:
        Coarsest HEALPix level to keep.
    coarsen_mode:
        Coarsening strategy for building lower pyramid levels.
        ``"mean"`` for continuous data, ``"mode"`` for categorical,
        ``"auto"`` to infer from the remapping method.
    min_valid_fraction:
        Minimum fraction of valid children for a parent cell to be
        valid.  Default ``0.5``.
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
        pyramid.update(
            _coarse_levels(
                finest,
                max_level=max_level,
                min_level=min_level,
                coarsen_mode=_resolve_coarsen_mode(finest, coarsen_mode),
                min_valid_fraction=min_valid_fraction,
            )
        )
        return pyramid

    for level in range(max_level - 1, min_level - 1, -1):
        pyramid[level] = regrid_to_healpix(ds, level, **kwargs)
    return pyramid


def _coarse_levels(
    finest: xr.Dataset,
    *,
    max_level: int,
    min_level: int,
    coarsen_mode: CoarsenMode,
    min_valid_fraction: float,
) -> dict[int, xr.Dataset]:
    """Coarsen *finest* level by level, sharing work between levels.

    Means carry sums and counts of valid finest-level cells, so every
    level is exact (issue #54) while each step is a chunk-local
    factor-4 reduction. Each level's graph builds on the previous one,
    so computing all levels together regrids every chunk only once.
    """
    cell_chunk = _cell_chunk_of(finest)
    templates = {
        str(name): data.transpose(..., "cell")
        for name, data in finest.data_vars.items()
        if "cell" in data.dims
    }
    state: dict[str, Any] = {name: t.data for name, t in templates.items()}
    levels: dict[int, xr.Dataset] = {}
    for level in range(max_level - 1, min_level - 1, -1):
        data_vars: dict[str, xr.DataArray] = {}
        for name, data in finest.data_vars.items():
            name = str(name)
            if name not in templates:
                data_vars[name] = data
                continue
            if coarsen_mode == "mode":
                state[name] = coarsen_mode_steps(
                    state[name],
                    1,
                    kernel=_coarsen_array_mode,
                    min_valid_fraction=min_valid_fraction,
                    cell_chunk=cell_chunk,
                )
                values = state[name]
            else:
                partials = state[name]
                partials = (
                    partials.coarsen()
                    if isinstance(partials, MeanPartials)
                    else MeanPartials.from_values(partials)
                ).rechunk(cell_chunk)
                state[name] = partials
                values = partials.mean(min_valid_fraction)
            data_vars[name] = _wrap_cells(templates[name], values)
        levels[level] = _assemble_coarse_level(finest, data_vars, level, max_level)
    return levels


# ===================================================================
# S3 / Zarr output
# ===================================================================


def save_pyramid(
    pyramid: dict[int, xr.Dataset],
    path: str,
    s3_options: dict[str, Any] | None = None,
    *,
    mode: Literal["a", "w", "r+"] = "a",
    compute: bool = True,
    region: Literal["auto"] | dict[str, slice] = "auto",
    zarr_format: Literal[2, 3] = 2,
    encoding: dict[int, dict[str, dict[str, Any]]] | None = None,
    write_coords: bool | Literal["auto"] = "auto",
) -> None:
    """Write a HEALPix pyramid to Zarr stores on S3 or local disk.

    Each level is stored below ``"<path>/level_<level>.zarr"``.

    Parameters
    ----------
    pyramid:
        Mapping of HEALPix level to dataset.
    path:
        Target prefix.  An ``"s3://bucket/pyramid"`` URL writes to S3; any
        other value is treated as a local directory path.
    s3_options:
        Options forwarded to :class:`s3fs.S3FileSystem`.  Only used for S3
        targets; ignored (and optional) for local paths.
    mode:
        Zarr write mode.
    compute:
        Write the data when ``True``.  All levels are written by a
        single dask computation, so the shared finest level is regridded
        only once.  ``False`` only initialises the stores (metadata and
        NumPy-backed variables), e.g. as a template for ``region``
        writes.
    region:
        Region writes for partial updates.
    zarr_format:
        Zarr format version.
    encoding:
        Per-level encoding dictionaries.
    write_coords:
        Whether to materialise the ``cell``/``latitude``/``longitude``
        coordinate arrays in the store.  HEALPix coordinates are a pure
        function of the cell index, and above roughly level 10 the
        arrays dwarf regional payloads (hundreds of GB at level 16 —
        they are never fill values, so chunk elision cannot help).
        ``"auto"`` (default) writes coordinates for levels up to
        :data:`WRITE_COORDS_MAX_LEVEL` and omits them above; ``True`` /
        ``False`` force either behaviour for all levels.  Coordinate-
        less stores keep the ``crs`` variable and all ``healpix_*``
        attributes, carry ``grid_doctor_implicit_coords = 1``, and are
        meant to be accessed through the region selectors
        ([`select_bbox`][grid_doctor.select_bbox],
        [`select_cells`][grid_doctor.select_cells]), which reconstruct
        coordinates for exactly the cells they return.
    """
    is_s3 = path.startswith("s3://")
    fs = s3fs.S3FileSystem(**(s3_options or {})) if is_s3 else None
    # xarray creates every store (metadata, encodings, NumPy-backed
    # variables) eagerly; dask-backed data is collected and written by a
    # single ``da.store``.  Separate ``to_zarr(compute=True)`` calls -- or
    # separate Delayed objects -- would each rebuild the shared
    # finest-level graph, regridding every chunk once per level.
    deferred: list[_DeferredWrite] = []
    for level, dataset in pyramid.items():
        include_coords = (
            write_coords
            if isinstance(write_coords, bool)
            else level <= WRITE_COORDS_MAX_LEVEL
        )
        if not include_coords:
            dataset = dataset.drop_vars(
                ["latitude", "longitude", "cell"], errors="ignore"
            )
            dataset.attrs["grid_doctor_implicit_coords"] = 1
        level_path = f"{path}/level_{level}.zarr"
        logger.info("Writing HEALPix level %s to %s", level, level_path)
        store: Any
        if is_s3:
            store = s3fs.S3Map(root=level_path, s3=fs)
        else:
            Path(level_path).parent.mkdir(parents=True, exist_ok=True)
            store = level_path
        zarr_options = ZarrOptions(compute=False, mode=mode, zarr_format=zarr_format)
        if zarr_format == 2:
            zarr_options["consolidated"] = True
        if encoding is not None:
            zarr_options["encoding"] = encoding[level]

        if region == "auto":
            dataset.to_zarr(store, **zarr_options)  # type: ignore[call-overload]
            written, write_region = dataset, None
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
            written = dataset.drop_vars(to_drop, errors="ignore").isel(region)
            written.to_zarr(  # type: ignore[call-overload]
                store,
                region=region,
                **zarr_options,
            )
            write_region = region
        if compute:
            deferred.extend(
                _deferred_writes(
                    written,
                    store,
                    encoding=zarr_options.get("encoding"),
                    region=write_region,
                    zarr_format=zarr_format,
                )
            )

        if mode == "w" and not compute:
            coord_options = dict(zarr_options)
            # "a", not "w": "w" would replace the store just initialised
            # above and drop its data variables.
            coord_options["mode"] = "a"
            dataset[list(dataset.coords)].to_zarr(store, **coord_options)  # type: ignore[call-overload]

    if deferred:
        da.store(
            [w.source for w in deferred],
            [w.target for w in deferred],
            regions=[w.region for w in deferred],
            lock=False,
        )


class _DeferredWrite(NamedTuple):
    source: Any
    target: Any
    region: tuple[slice, ...]


def _deferred_writes(
    dataset: xr.Dataset,
    store: Any,
    *,
    encoding: dict[str, dict[str, Any]] | None,
    region: dict[str, slice] | None,
    zarr_format: Literal[2, 3],
) -> list[_DeferredWrite]:
    """Collect encoded dask sources and zarr targets of an initialised store."""
    # NumPy-backed variables were already written by ``to_zarr(compute=False)``.
    lazy = {
        str(name): var
        for name, var in dataset.variables.items()
        if var.chunks is not None
    }
    if not lazy:
        return []
    group = zarr.open_group(store, mode="r+", zarr_format=zarr_format)
    writes: list[_DeferredWrite] = []
    for name, var in lazy.items():
        var = var.copy(deep=False)
        var.encoding = {**var.encoding, **(encoding or {}).get(name, {})}
        encoded = _encode_zarr_variable(var, name=name, zarr_format=zarr_format)
        target_region = tuple(
            (region or {}).get(str(dim), slice(None)) for dim in var.dims
        )
        writes.append(_DeferredWrite(encoded.data, group[name], target_region))
    return writes


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
    "save_pyramid",
]
