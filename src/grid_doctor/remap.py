"""Remapping helpers for HEALPix targets.

This module contains the weight-generation and weight-application logic used by
[`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

Two remapping paths are supported:

- `nearest` and `conservative` generate a reusable NetCDF weight file with
  ESMPy/ESMF and apply it with SciPy sparse matrices.
- `linear` performs on-the-fly interpolation from source cell centres to
  HEALPix cell centres.

The implementation is designed to work with regular lon/lat grids,
curvilinear lon/lat grids, and unstructured polygon meshes such as ICON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

from .remap_backend import (
    _LAT_NAMES,
    _UNSTRUCTURED_DIMS,
    OfflineWeightConfig,
    SourceKind,
    _get_latlon_arrays,
    _require_healpix_geo_module,
    compute_healpix_weights_backend,
)

logger = logging.getLogger(__name__)

RemapMethod = Literal["nearest", "linear", "conservative"]
MissingPolicy = Literal["renormalize", "propagate"]
SourceUnits = Literal["auto", "deg", "rad"]

FloatArray = npt.NDArray[np.float64]


def _canonical_lon(lon_deg: FloatArray) -> FloatArray:
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _looks_like_radians(values: FloatArray) -> bool:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    max_abs = float(np.nanmax(np.abs(finite)))
    return max_abs <= (2.0 * np.pi + 1.0e-6)


def _normalise_angle_units(values: FloatArray, units: SourceUnits) -> FloatArray:
    if units == "deg":
        return values.astype(np.float64, copy=False)
    if units == "rad":
        return np.rad2deg(values)
    if _looks_like_radians(values):
        return np.rad2deg(values)
    return values.astype(np.float64, copy=False)


def _lonlat_to_xyz(lon_deg: FloatArray, lat_deg: FloatArray) -> FloatArray:
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat)
    return np.column_stack(
        (
            cos_lat * np.cos(lon),
            cos_lat * np.sin(lon),
            np.sin(lat),
        )
    )


def _xyz_to_lonlat(xyz: FloatArray) -> tuple[float, float]:
    norm = np.linalg.norm(xyz)
    if norm == 0.0:
        raise ValueError("Cannot convert zero-length vector to lon/lat coordinates.")
    unit = xyz / norm
    lon = np.rad2deg(np.arctan2(unit[1], unit[0]))
    lat = np.rad2deg(np.arcsin(np.clip(unit[2], -1.0, 1.0)))
    return float(lon), float(lat)


def _spherical_centroid(
    lon_deg: FloatArray, lat_deg: FloatArray
) -> tuple[float, float]:
    xyz = _lonlat_to_xyz(lon_deg, lat_deg)
    return _xyz_to_lonlat(xyz.mean(axis=0))


def _is_unstructured(ds: xr.Dataset) -> bool:
    if _UNSTRUCTURED_DIMS & {str(dim) for dim in ds.dims}:
        return True
    for var in ds.data_vars.values():
        if var.attrs.get("CDI_grid_type") == "unstructured":
            return True
    return False


def _get_unstructured_dim(ds: xr.Dataset) -> str:
    for dim in _UNSTRUCTURED_DIMS:
        if dim in ds.dims:
            return dim
    lat, _ = _get_latlon_arrays(ds)
    if lat.ndim == 1:
        for name in _LAT_NAMES:
            if name in ds and ds[name].ndim == 1:
                return str(ds[name].dims[0])
    raise ValueError(
        "Could not determine the source cell dimension for the unstructured grid."
    )


def _get_spatial_dims(ds: xr.Dataset) -> tuple[str, str]:
    y_candidates = (
        "rlat",
        "lat",
        "latitude",
        "y",
        "j",
        "nj",
        "south_north",
        "south_north_stag",
        "eta_rho",
        "eta_u",
        "eta_v",
        "eta_psi",
        "yh",
        "yq",
        "njp1",
    )
    x_candidates = (
        "rlon",
        "lon",
        "longitude",
        "x",
        "i",
        "ni",
        "west_east",
        "west_east_stag",
        "xi_rho",
        "xi_u",
        "xi_v",
        "xi_psi",
        "xh",
        "xq",
        "nip1",
    )
    y_dim: str | None = None
    x_dim: str | None = None

    for dim in ds.dims:
        dim_name = str(dim)
        dim_lower = dim_name.lower()
        if y_dim is None and dim_lower in y_candidates:
            y_dim = dim_name
        elif x_dim is None and dim_lower in x_candidates:
            x_dim = dim_name

    if y_dim is None or x_dim is None:
        lat, _ = _get_latlon_arrays(ds)
        if lat.ndim == 2:
            for coord in ds.coords.values():
                if coord.ndim == 2 and coord.shape == lat.shape:
                    dims = tuple(map(str, coord.dims))
                    if len(dims) == 2:
                        return dims[0], dims[1]

    if y_dim is None or x_dim is None:
        raise ValueError(
            f"Could not determine spatial dimensions from {list(ds.dims)}."
        )
    return y_dim, x_dim


def _source_centre_arrays(
    ds: xr.Dataset,
    *,
    source_units: SourceUnits,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    lat, lon = _get_latlon_arrays(ds)
    lat = _normalise_angle_units(lat, source_units)
    lon = _canonical_lon(_normalise_angle_units(lon, source_units))

    if _is_unstructured(ds):
        spatial_dim = _get_unstructured_dim(ds)
        return lat.ravel(), lon.ravel(), (spatial_dim,)

    y_dim, x_dim = _get_spatial_dims(ds)
    if lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lat2d.ravel(), lon2d.ravel(), (y_dim, x_dim)
    return lat.ravel(), lon.ravel(), (y_dim, x_dim)


def _infer_bounds_1d(values: FloatArray) -> FloatArray:
    values = values.astype(np.float64, copy=False)
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional coordinate array.")
    if values.size < 2:
        raise ValueError("At least two coordinate values are required to infer bounds.")
    diffs = np.diff(values)
    bounds = np.empty(values.size + 1, dtype=np.float64)
    bounds[1:-1] = values[:-1] + diffs / 2.0
    bounds[0] = values[0] - diffs[0] / 2.0
    bounds[-1] = values[-1] + diffs[-1] / 2.0
    return bounds


def _infer_curvilinear_corners(
    lat_deg: FloatArray, lon_deg: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Infer cell corner coordinates from curvilinear centre coordinates.

    The corner coordinates are estimated on the unit sphere by averaging the
    neighbouring cell-centre vectors around each corner and converting the
    result back to lon/lat. This keeps the interpolation stable across the
    antimeridian.
    """
    if lat_deg.ndim != 2 or lon_deg.ndim != 2:
        raise ValueError("Curvilinear corner inference expects two-dimensional arrays.")
    ny, nx = lat_deg.shape
    xyz = _lonlat_to_xyz(lon_deg.ravel(), lat_deg.ravel()).reshape(ny, nx, 3)
    lat_corner = np.empty((ny + 1, nx + 1), dtype=np.float64)
    lon_corner = np.empty((ny + 1, nx + 1), dtype=np.float64)

    for j in range(ny + 1):
        for i in range(nx + 1):
            neighbour_vectors: list[FloatArray] = []
            for dj in (-1, 0):
                for di in (-1, 0):
                    jj = j + dj
                    ii = i + di
                    if 0 <= jj < ny and 0 <= ii < nx:
                        neighbour_vectors.append(xyz[jj, ii])
            if not neighbour_vectors:
                raise ValueError("Failed to infer curvilinear cell corners.")
            lon_corner[j, i], lat_corner[j, i] = _xyz_to_lonlat(
                np.mean(np.vstack(neighbour_vectors), axis=0)
            )
    lon_corner = _canonical_lon(lon_corner)
    return lat_corner, lon_corner


def _lon_coverage_span(lon_deg: FloatArray) -> float:
    lon = np.mod(lon_deg[np.isfinite(lon_deg)], 360.0)
    if lon.size == 0:
        return 0.0
    lon_sorted = np.sort(lon)
    wrapped = np.concatenate((lon_sorted, lon_sorted[:1] + 360.0))
    gaps = np.diff(wrapped)
    return float(360.0 - np.max(gaps))


def _median_positive_step(values: FloatArray) -> float:
    """Return the median positive step in a sorted coordinate array."""
    if values.size < 2:
        return 0.0
    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _lon_coverage_from_centres(lon_deg: FloatArray) -> float:
    """
    Estimate longitude coverage in degrees from centre coordinates.

    This is dateline-safe and treats the coordinates as cell centres rather than
    bounds. For a regular global grid with centres at
    `0, 15, ..., 345`, the returned coverage is `360`, not `345`.
    """
    lon = np.mod(lon_deg[np.isfinite(lon_deg)], 360.0)
    if lon.size == 0:
        return 0.0

    lon = np.unique(np.sort(lon))
    if lon.size == 1:
        return 0.0

    dlon = _median_positive_step(lon)

    wrapped = np.concatenate([lon, lon[:1] + 360.0])
    gaps = np.diff(wrapped)
    largest_gap = float(np.max(gaps))

    coverage = 360.0 - largest_gap + dlon
    return float(min(360.0, max(0.0, coverage)))


def _lat_coverage_from_centres(lat_deg: FloatArray) -> float:
    """
    Estimate latitude coverage in degrees from centre coordinates.

    The coordinates are interpreted as cell centres, so one representative
    latitude spacing is added to the raw span.
    """
    lat = lat_deg[np.isfinite(lat_deg)]
    if lat.size == 0:
        return 0.0
    dlat = _median_positive_step(lat)
    coverage = float(lat[-1] - lat[0]) + dlat
    return float(min(180.0, max(0.0, coverage)))


def _healpix_centres(level: int, *, nest: bool) -> tuple[FloatArray, FloatArray]:
    """Return HEALPix cell-centre coordinates in degrees.

    Parameters
    ----------
    level:
        HEALPix refinement level.
    nest:
        Select nested ordering when `True`, otherwise ring ordering.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Latitude and longitude arrays in degrees with one entry per HEALPix
        cell.

    Notes
    -----
    The implementation intentionally uses `healpix-geo` only, so the remapping
    stack no longer depends on `healpy`.
    """
    module, kwargs = _require_healpix_geo_module(nest)
    ipix = np.arange(12 * (4**level), dtype=np.int64)
    lon_deg, lat_deg = module.healpix_to_lonlat(ipix, level, **kwargs)
    return (
        np.asarray(lat_deg, dtype=np.float64),
        _canonical_lon(np.asarray(lon_deg, dtype=np.float64)),
    )


def _extract_sparse_weights(weights_ds: xr.Dataset) -> tuple[csr_matrix, int, int]:
    row_name = "row" if "row" in weights_ds else "dst_address"
    col_name = "col" if "col" in weights_ds else "src_address"
    value_name = "S" if "S" in weights_ds else "remap_matrix"

    row = np.asarray(weights_ds[row_name].values, dtype=np.int64).ravel()
    col = np.asarray(weights_ds[col_name].values, dtype=np.int64).ravel()
    values = np.asarray(weights_ds[value_name].values, dtype=np.float64).ravel()

    if row.size != values.size or col.size != values.size:
        raise ValueError(
            "Weight file columns 'row', 'col', and 'S' must have the same length."
        )
    if row.size == 0:
        raise ValueError("Weight file does not contain any interpolation entries.")

    if row.min() >= 1 and col.min() >= 1:
        row = row - 1
        col = col - 1

    n_target = int(row.max()) + 1
    n_source = int(col.max()) + 1
    matrix = coo_matrix(
        (values, (row, col)), shape=(n_target, n_source), dtype=np.float64
    )
    return matrix.tocsr(), n_target, n_source


def compute_healpix_weights(
    ds: xr.Dataset | str | Path,
    level: int,
    *,
    method: Literal["nearest", "conservative"] = "nearest",
    nest: bool = True,
    source_units: SourceUnits = "auto",
    weights_path: str | Path | None = None,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    ignore_unmapped: bool | None = None,
    large_file: bool = True,
    prefer_offline: bool | None = None,
    nproc: int = 1,
    esmf_regrid_weightgen: str = "ESMF_RegridWeightGen",
    keep_intermediates: bool = False,
    workdir: str | Path | None = None,
    spectral_transform_command: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Generate a reusable NetCDF weight file for HEALPix remapping.

    This is the public facade for reusable weight generation. The heavy lifting
    is delegeted to ESMF which can use either an in-memory ESMPy workflow or
    the offline `ESMF_RegridWeightGen` executable under MPI.

    Parameters
    ----------
    ds:
        Source dataset or a source path. Paths are particularly useful for
        spectral inputs that first need an external transform to a temporary
        gridpoint dataset.
    level:
        HEALPix refinement level. The destination `nside` equals `2**level`.
    method:
        Weight-generation method. Supported values are `"nearest"` and
        `"conservative"`.
    nest:
        Use nested HEALPix ordering when `True`.
    source_units:
        Unit convention of the source coordinates. `"auto"` converts from
        radians when the values look radian-like.
    weights_path:
        Output NetCDF file. A temporary file is created when omitted.
    grid:
        Optional external geometry dataset. This is useful when the data file
        itself only contains values on dimensions such as `cell`.
    source_kind:
        Explicit source representation. Supported values are `"auto"`,
        `"regular"`, `"curvilinear"`, `"unstructured"`, and `"spectral"`.
    ignore_unmapped:
        Whether destination cells without source coverage should be ignored.
        When omitted, the backend chooses a sensible default based on whether
        the source appears global.
    large_file:
        Forwarded to the in-memory ESMPy workflow.
    prefer_offline:
        Force or disable the offline ESMF path. When omitted, the backend uses
        the offline path automatically for large conservative jobs when
        `nproc > 1`.
    nproc:
        Number of MPI ranks used by the offline ESMF path.
    esmf_regrid_weightgen:
        Name or path of the offline ESMF executable.
    keep_intermediates:
        Keep intermediate mesh files generated by the offline workflow.
    workdir:
        Optional working directory for offline intermediate files.
    spectral_transform_command:
        External command used when `source_kind="spectral"`. The command may
        use `{input}` and `{output}` placeholders.
    **kwargs:
        Any additional keyword arguments for
        [`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights]

    Returns
    -------
    pathlib.Path
        Path to the generated NetCDF weight file.

    Examples
    --------
    Conservative remapping for an ICON grid with a separate grid file:

    ```python
    weight_file = compute_healpix_weights(
        ds_icon_data,
        level=8,
        method="conservative",
        grid=ds_icon_grid,
        source_units="rad",
        nproc=16,
    )
    ```

    Force the offline MPI path explicitly:

    ```python
    weight_file = compute_healpix_weights(
        ds,
        level=10,
        method="conservative",
        prefer_offline=True,
        nproc=32,
    )
    ```

    Use an external spectral-to-grid transform before weight generation:

    ```python
    weight_file = compute_healpix_weights(
        "spectral_input.nc",
        level=7,
        method="conservative",
        source_kind="spectral",
        spectral_transform_command=["cdo", "-f", "nc4", "sp2gp", "{input}", "{output}"],
    )
    ```

    See Also
    --------
    [`apply_weight_file`][grid_doctor.remap.apply_weight_file]:
        Apply a previously generated weight file.
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]:
        High-level convenience wrapper that generates and applies weights.
    """
    offline = OfflineWeightConfig(
        enabled=bool(prefer_offline) if prefer_offline is not None else False,
        nproc=nproc,
        esmf_regrid_weightgen=esmf_regrid_weightgen,
        keep_intermediates=keep_intermediates,
        workdir=Path(workdir) if workdir is not None else None,
    )
    return compute_healpix_weights_backend(
        ds,
        level,
        method=method,
        nest=nest,
        source_units=source_units,
        weights_path=weights_path,
        grid=grid,
        source_kind=source_kind,
        ignore_unmapped=ignore_unmapped,
        large_file=large_file,
        offline=offline,
        spectral_transform_command=spectral_transform_command,
    )


def _apply_sparse_array(
    values: npt.NDArray[Any],
    *,
    matrix: csr_matrix,
    missing_policy: MissingPolicy,
) -> FloatArray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(flat)

    if missing_policy == "propagate":
        filled = np.where(valid, flat, 0.0)
        out = np.asarray(matrix @ filled, dtype=np.float64)
        missing_weight = np.asarray(
            matrix @ (~valid).astype(np.float64), dtype=np.float64
        )
        out[missing_weight > 0.0] = np.nan
        return out

    filled = np.where(valid, flat, 0.0)
    weighted = np.asarray(matrix @ filled, dtype=np.float64)
    support = np.asarray(matrix @ valid.astype(np.float64), dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted = weighted / support
    weighted[support <= 0.0] = np.nan
    return weighted


def _flattened_size(ds: xr.Dataset, source_dims: tuple[str, ...]) -> int:
    """Return the flattened size of `source_dims` in `ds`.

    Parameters
    ----------
    ds:
        Dataset whose dimensions should be checked.
    source_dims:
        Dimensions that form the source grid ordering used by the weight file.

    Returns
    -------
    int
        Product of the dimension sizes.

    Raises
    ------
    ValueError
        Raised when any requested dimension is missing from `ds`.
    """
    missing = [dim for dim in source_dims if dim not in ds.dims]
    if missing:
        raise ValueError(
            f"source_dims {source_dims!r} are not all present in the dataset. "
            f"Missing dimensions: {missing!r}."
        )

    size = 1
    for dim in source_dims:
        size *= int(ds.sizes[dim])
    return size


def _parse_source_dims_attr(value: object) -> tuple[str, ...] | None:
    """Parse the stored source-dimension metadata from a weight file.

    Parameters
    ----------
    value:
        Attribute value read from the NetCDF file.

    Returns
    -------
    tuple[str, ...] | None
        Parsed source dimensions when available.
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split(",") if part.strip()]
        if isinstance(parsed, str):
            return (parsed,)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return tuple(parsed)
        return None

    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return value

    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return tuple(value)

    return None


def _guess_source_dims_from_size(
    ds: xr.Dataset, n_source: int
) -> tuple[str, ...] | None:
    """Guess source dimensions from dataset sizes when geometry is unavailable.

    This is intentionally conservative. It only guesses a single source
    dimension when that size uniquely matches `n_source`, which is the common
    unstructured-grid case such as ICON data on a `cell` dimension.

    Parameters
    ----------
    ds:
        Source dataset.
    n_source:
        Number of source cells encoded by the weight file.

    Returns
    -------
    tuple[str, ...] | None
        Guessed source dimensions, or `None` when no safe guess is possible.
    """
    matches = [str(dim) for dim, size in ds.sizes.items() if int(size) == n_source]
    preferred = [dim for dim in matches if dim in _UNSTRUCTURED_DIMS]

    if len(preferred) == 1:
        return (preferred[0],)
    if len(matches) == 1:
        return (matches[0],)
    return None


def _resolve_source_dims_for_weight_application(
    ds: xr.Dataset,
    *,
    n_source: int,
    grid: xr.Dataset | None,
    source_dims: tuple[str, ...] | None,
    source_units: SourceUnits,
    stored_source_dims: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Resolve source dimensions for a previously generated weight file.

    Parameters
    ----------
    ds:
        Dataset that should be remapped.
    n_source:
        Number of source cells encoded by the weight file.
    grid:
        Optional grid dataset containing the geometry used to generate the
        weight file.
    source_dims:
        Optional explicit source dimensions. This is the most direct way to
        apply weights when the dataset itself does not contain geometry.
    source_units:
        Coordinate units used when geometry-based inference is required.
    stored_source_dims:
        Optional source dimensions stored in the weight-file metadata.

    Returns
    -------
    tuple[str, ...]
        Source dimensions in the flattening order expected by the weight file.

    Raises
    ------
    ValueError
        Raised when the source geometry or dimensions cannot be matched to the
        weight file.
    """
    if source_dims is not None:
        flattened = _flattened_size(ds, source_dims)
        if flattened != n_source:
            raise ValueError(
                "The provided source_dims do not match the weight file geometry. "
                f"Expected {n_source} source cells, found {flattened}."
            )
        return source_dims

    if stored_source_dims is not None:
        try:
            flattened = _flattened_size(ds, stored_source_dims)
        except ValueError:
            flattened = -1
        if flattened == n_source:
            return stored_source_dims

    if grid is not None:
        grid_lat, _, grid_dims = _source_centre_arrays(grid, source_units=source_units)
        if grid_lat.size != n_source:
            raise ValueError(
                "The provided grid does not match the weight file geometry. "
                f"Expected {n_source} source cells, found {grid_lat.size} in the grid."
            )

        flattened = _flattened_size(ds, grid_dims)
        if flattened != n_source:
            raise ValueError(
                "The provided grid implies source dimensions that do not match the "
                "dataset being remapped. "
                f"Expected {n_source} source cells, found {flattened} in the dataset."
            )
        return grid_dims

    try:
        src_lat, _, inferred_dims = _source_centre_arrays(ds, source_units=source_units)
    except ValueError:
        guessed_dims = _guess_source_dims_from_size(ds, n_source)
        if guessed_dims is not None:
            return guessed_dims
        raise ValueError(
            "Could not infer the source geometry from the dataset. Pass "
            "`source_dims=...` or `grid=...` when the geometry is stored in a "
            "separate file."
        ) from None

    if src_lat.size == n_source:
        return inferred_dims

    guessed_dims = _guess_source_dims_from_size(ds, n_source)
    if guessed_dims is not None:
        return guessed_dims

    raise ValueError(
        "The weight file does not match the provided dataset geometry. "
        f"Expected {n_source} source cells, found {src_lat.size}. "
        "Pass `source_dims=...` or `grid=...` when the geometry is stored in a "
        "separate file."
    )


def apply_weight_file(
    ds: xr.Dataset,
    weights_path: str | Path,
    *,
    missing_policy: MissingPolicy = "renormalize",
    grid: xr.Dataset | None = None,
    source_dims: tuple[str, ...] | None = None,
    source_units: SourceUnits = "auto",
) -> xr.Dataset:
    """Apply a previously generated ESMF weight file to `ds`.

    The weight file is read into a SciPy sparse matrix and applied with
    `xarray.apply_ufunc`. Missing values can either be propagated or ignored
    with per-target renormalisation.

    Unlike weight generation, weight application does not require the full
    source geometry when the source dimensions are already known. This is useful
    for model outputs whose grid coordinates live in a separate grid file.

    Parameters
    ----------
    ds:
        Source dataset to remap.
    weights_path:
        Path to the NetCDF weight file generated by
        [`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights].
    missing_policy:
        Missing-value handling strategy.

        - `"renormalize"`: ignore missing source values and renormalize by the
          sum of valid source weights.
        - `"propagate"`: any missing contributing source value makes the target
          value missing.
    grid:
        Optional grid dataset containing the geometry that was used to generate
        the weight file. Use this when `ds` does not embed latitude/longitude
        coordinates.
    source_dims:
        Optional explicit source dimensions in the flattening order expected by
        the weight file. This is the most direct option for data files that only
        carry dimensions such as `("cell",)`.
    source_units:
        Unit convention used when geometry-based source inference is required.

    Returns
    -------
    xarray.Dataset
        Dataset on the HEALPix target grid with a `cell` dimension.

    Examples
    --------
    Apply weights to a dataset that embeds its own geometry:

    ```python
    ds_hp = apply_weight_file(ds_icon, "icon_to_healpix_d8.nc")
    ```

    Apply weights to an unstructured data file whose geometry lives elsewhere:

    ```python
    ds_hp = apply_weight_file(
        ds_data,
        "icon_to_healpix_d8.nc",
        source_dims=("cell",),
    )
    ```

    Or validate against a separate grid file:

    ```python
    ds_hp = apply_weight_file(
        ds_data,
        "icon_to_healpix_d8.nc",
        grid=ds_grid,
        source_units="rad",
    )
    ```

    See Also
    --------
    [`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights]:
        Generate the reusable weight file.
    """
    if missing_policy not in {"renormalize", "propagate"}:
        raise ValueError("missing_policy must be 'renormalize' or 'propagate'.")

    with xr.open_dataset(weights_path) as weights_ds:
        matrix, n_target, n_source = _extract_sparse_weights(weights_ds)
        level = int(weights_ds.attrs.get("grid_doctor_level", -1))
        order = str(weights_ds.attrs.get("grid_doctor_order", "nested"))
        stored_source_dims = _parse_source_dims_attr(
            weights_ds.attrs.get("grid_doctor_source_dims")
        )

    resolved_source_dims = _resolve_source_dims_for_weight_application(
        ds,
        n_source=n_source,
        grid=grid,
        source_dims=source_dims,
        source_units=source_units,
        stored_source_dims=stored_source_dims,
    )

    regridded_vars: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if not set(resolved_source_dims).issubset(map(str, data.dims)):
            regridded_vars[str(name)] = data
            continue

        regridded_vars[str(name)] = cast(
            xr.DataArray,
            xr.apply_ufunc(
                _apply_sparse_array,
                data,
                input_core_dims=[list(resolved_source_dims)],
                output_core_dims=[["cell"]],
                exclude_dims=set(resolved_source_dims),
                vectorize=True,
                dask="parallelized",
                kwargs={"matrix": matrix, "missing_policy": missing_policy},
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"cell": n_target}},
            ),
        )

    result = xr.Dataset(regridded_vars, attrs=ds.attrs.copy())
    if level >= 0:
        result = _attach_healpix_coords(result, level=level, nest=(order == "nested"))
    return result


def _interpolate_linear_array(
    values: npt.NDArray[Any],
    *,
    points_xyz: FloatArray,
    target_xyz: FloatArray,
) -> FloatArray:
    flat: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(flat)
    if valid.sum() == 0:
        return np.full(target_xyz.shape[0], np.nan, dtype=np.float64)
    if int(valid.sum()) < 4:
        tree = cKDTree(points_xyz[valid])
        _, idx = tree.query(target_xyz, k=1)
        return cast(FloatArray, flat[valid][idx])

    interpolator = LinearNDInterpolator(
        points_xyz[valid], flat[valid], fill_value=np.nan
    )
    out = interpolator(target_xyz)
    return np.asarray(out, dtype=np.float64)


def _attach_healpix_coords(ds_hp: xr.Dataset, *, level: int, nest: bool) -> xr.Dataset:
    lat_deg, lon_deg = _healpix_centres(level, nest=nest)
    ds_hp = ds_hp.assign_coords(
        cell=np.arange(lat_deg.size, dtype=np.int64),
        latitude=("cell", lat_deg),
        longitude=("cell", lon_deg),
    )
    ds_hp.attrs["healpix_level"] = level
    ds_hp.attrs["healpix_nside"] = 2**level
    ds_hp.attrs["healpix_order"] = "nested" if nest else "ring"
    return ds_hp


def regrid_to_healpix(
    ds: xr.Dataset,
    level: int,
    *,
    nest: bool = True,
    method: RemapMethod = "conservative",
    source_units: SourceUnits = "auto",
    weights_path: str | Path | None = None,
    missing_policy: MissingPolicy = "renormalize",
    **kwags: Any,
) -> xr.Dataset:
    """Regrid `ds` to a HEALPix target grid.

    Parameters
    ----------
    ds:
        Source dataset on a regular, curvilinear, or unstructured grid.
    level:
        HEALPix refinement level.
    nest:
        Use nested HEALPix ordering when `True`.
    method:
        Remapping method.

        - `"nearest"` uses reusable ESMF weights when possible.
        - `"conservative"` uses reusable ESMF weights.
        - `"linear"` performs on-the-fly interpolation from source cell centres.
    source_units:
        Coordinate units for the source grid.
    weights_path:
        Optional existing or target weight file for the weight-based methods.
        When omitted, a temporary file is generated automatically.
    missing_policy:
        Missing-value handling for the weight-based methods. See
        [`apply_weight_file`][grid_doctor.remap.apply_weight_file].

    Returns
    -------
    xarray.Dataset
        Regridded dataset on the HEALPix grid.

    Examples
    --------
    Conservative remapping with a reusable weight file:

    ```python
    ds_hp = regrid_to_healpix(
        ds_icon,
        level=8,
        method="conservative",
        source_units="rad",
        weights_path="icon_to_healpix_d8.nc",
    )
    ```

    Linear remapping without a weight file:

    ```python
    ds_hp = regrid_to_healpix(ds, level=7, method="linear")
    ```
    """
    if method in {"nearest", "conservative"}:
        weight_file = Path(weights_path) if weights_path is not None else None
        if weight_file is None or not weight_file.exists():
            weight_file = compute_healpix_weights(
                ds,
                level,
                method=cast(Literal["nearest", "conservative"], method),
                nest=nest,
                source_units=source_units,
                weights_path=weight_file,
            )
        return apply_weight_file(ds, weight_file, missing_policy=missing_policy)

    src_lat, src_lon, source_dims = _source_centre_arrays(ds, source_units=source_units)
    target_lat, target_lon = _healpix_centres(level, nest=nest)
    source_xyz = _lonlat_to_xyz(src_lon, src_lat)
    target_xyz = _lonlat_to_xyz(target_lon, target_lat)

    regridded_vars: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if not set(source_dims).issubset(map(str, data.dims)):
            regridded_vars[str(name)] = data
            continue
        regridded_vars[str(name)] = cast(
            xr.DataArray,
            xr.apply_ufunc(
                _interpolate_linear_array,
                data,
                input_core_dims=[list(source_dims)],
                output_core_dims=[["cell"]],
                exclude_dims=set(source_dims),
                vectorize=True,
                dask="parallelized",
                kwargs={"points_xyz": source_xyz, "target_xyz": target_xyz},
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"cell": target_xyz.shape[0]}},
            ),
        )

    result = xr.Dataset(regridded_vars, attrs=ds.attrs.copy())
    return _attach_healpix_coords(result, level=level, nest=nest)


def regrid_unstructured_to_healpix(
    ds: xr.Dataset,
    level: int,
    *,
    nest: bool = True,
    method: RemapMethod = "conservative",
    source_units: SourceUnits = "auto",
    weights_path: str | Path | None = None,
    missing_policy: MissingPolicy = "renormalize",
) -> xr.Dataset:
    """Compatibility wrapper around [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

    The function keeps the historic public name for callers that only work with
    unstructured grids. The implementation now delegates to the unified
    remapping stack.
    """
    if not _is_unstructured(ds):
        raise ValueError(
            "regrid_unstructured_to_healpix() expects an unstructured source grid."
        )
    return regrid_to_healpix(
        ds,
        level,
        nest=nest,
        method=method,
        source_units=source_units,
        weights_path=weights_path,
        missing_policy=missing_policy,
    )


__all__ = [
    "apply_weight_file",
    "compute_healpix_weights",
    "regrid_to_healpix",
    "regrid_unstructured_to_healpix",
]
