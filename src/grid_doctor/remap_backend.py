"""Backend utilities for reusable HEALPix remapping weights.

This module encapsulates the low-level mechanics needed by
[`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights] while
keeping the public API in `grid_doctor.remap` small.

The backend supports the following source representations:

- regular lon/lat grids
- curvilinear lon/lat grids
- unstructured polygon meshes such as UGRID or ICON
- spectral inputs indirectly, via an external transform command that converts
  them to a temporary gridpoint dataset before weight generation

Two weight-generation paths are available:

- **In-memory** via ESMPy for moderate-size grids.
- **Offline** via `ESMF_RegridWeightGen` under MPI for large conservative
  jobs where parallel overlap calculation is beneficial.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

RemapMethod = Literal["nearest", "conservative"]
"""Supported weight-generation methods."""

SourceUnits = Literal["auto", "deg", "rad"]
"""Angular unit convention for source coordinates."""

SourceKind = Literal["auto", "regular", "curvilinear", "unstructured", "spectral"]
"""Explicit source grid classification."""

FloatArray = npt.NDArray[np.float64]
"""Shorthand for a float64 NumPy array."""

# ---------------------------------------------------------------------------
# Well-known coordinate and dimension names
# ---------------------------------------------------------------------------

_UNSTRUCTURED_DIMS: frozenset[str] = frozenset({"cell", "ncells", "ncell", "nCells"})
"""Dimension names that signal an unstructured source grid."""

_LAT_NAMES: tuple[str, ...] = (
    "clat",
    "lat",
    "latitude",
    "LAT",
    "LATITUDE",
    "Latitude",
    "XLAT",
    "XLAT_M",
    "XLAT_U",
    "XLAT_V",
    "nav_lat",
    "nav_lat_rho",
    "lat_rho",
    "lat_u",
    "lat_v",
    "lat_psi",
    "gridlat_0",
    "g0_lat_0",
    "yt_ocean",
    "yu_ocean",
    "geolat",
    "geolat_t",
    "geolat_c",
)
"""Priority-ordered latitude variable names recognised by the backend."""

_LON_NAMES: tuple[str, ...] = (
    "clon",
    "lon",
    "longitude",
    "LON",
    "LONGITUDE",
    "Longitude",
    "XLONG",
    "XLONG_M",
    "XLONG_U",
    "XLONG_V",
    "nav_lon",
    "nav_lon_rho",
    "lon_rho",
    "lon_u",
    "lon_v",
    "lon_psi",
    "gridlon_0",
    "g0_lon_0",
    "xt_ocean",
    "xu_ocean",
    "geolon",
    "geolon_t",
    "geolon_c",
)
"""Priority-ordered longitude variable names recognised by the backend."""

_Y_CANDIDATES: tuple[str, ...] = (
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
"""Lower-cased dimension names considered as latitude / y axes."""

_X_CANDIDATES: tuple[str, ...] = (
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
"""Lower-cased dimension names considered as longitude / x axes."""


# ===================================================================
# Data-classes
# ===================================================================


@dataclass(frozen=True, slots=True)
class OfflineWeightConfig:
    """Configuration for offline ESMF weight generation.

    Attributes:
        enabled: Enable offline weight generation with
            `ESMF_RegridWeightGen`.
        nproc: Number of MPI ranks used for the offline command.
        mpirun: MPI launcher executable.
        esmf_regrid_weightgen: Name or path of the offline ESMF
            executable.
        weight_only: Request a simplified weight file that contains
            only `row`, `col`, and `S` when the ESMF build supports it.
        netcdf4: Write NetCDF-4 output.
        keep_intermediates: Keep temporary source and destination mesh
            files for inspection.
        workdir: Optional directory for the intermediate mesh files.
    """

    enabled: bool = False
    nproc: int = 1
    esmf_regrid_weightgen: str = "ESMF_RegridWeightGen"
    weight_only: bool = False
    netcdf4: bool = True
    keep_intermediates: bool = False
    workdir: Path | None = None

    @property
    def mpirun(self) -> str:
        """Check if we are in a slurm env to use srun."""
        if os.getenv("SLURM_JOB_ID") is not None:
            return "srun"
        return "mpirun"


@dataclass(frozen=True, slots=True)
class PolygonMesh:
    """Polygon mesh stored as compact NumPy arrays.

    The mesh is the common in-memory representation shared by regular,
    curvilinear, unstructured, and HEALPix grids.  All coordinate values
    are in degrees.

    Attributes:
        node_lon: Flattened node longitudes in degrees, shape
            `(n_node,)`.
        node_lat: Flattened node latitudes in degrees, shape
            `(n_node,)`.
        face_nodes: Face-to-node connectivity with shape
            `(n_face, max_face_nodes)` and `-1` fill values for shorter
            polygons.
        face_lon: Face-centre longitudes in degrees, shape `(n_face,)`.
        face_lat: Face-centre latitudes in degrees, shape `(n_face,)`.
    """

    node_lon: FloatArray
    node_lat: FloatArray
    face_nodes: npt.NDArray[np.int32]
    face_lon: FloatArray
    face_lat: FloatArray

    @property
    def face_count(self) -> int:
        """Return the number of mesh faces."""
        return int(self.face_nodes.shape[0])

    @property
    def max_face_nodes(self) -> int:
        """Return the padded face-node width."""
        return int(self.face_nodes.shape[1])


@dataclass(frozen=True, slots=True)
class SourceDescription:
    """Resolved description of the source representation.

    Produced by [`describe_source`][grid_doctor.remap_backend.describe_source]
    and consumed by the weight-generation entry point.

    Attributes:
        dataset: Materialised gridpoint dataset used for weight
            generation.
        kind: Source representation class.
        source_dims: Source dimensions in flattening order.
        source_mesh: Source mesh stored as compact node/connectivity
            arrays.
        ignore_unmapped: Recommended default for ESMF unmapped
            handling.
        metadata: Metadata that should be persisted into the output
            weight file.
    """

    dataset: xr.Dataset
    kind: SourceKind
    source_dims: tuple[str, ...]
    source_mesh: PolygonMesh
    ignore_unmapped: bool
    metadata: dict[str, str | int | float | bool]


class SpectralTransformError(RuntimeError):
    """Raised when a spectral-to-grid transform command fails."""


# ===================================================================
# Lazy imports for optional heavy dependencies
# ===================================================================


def _require_esmpy() -> Any:
    """Import and return the ``esmpy`` module.

    Raises:
        ImportError: When ESMPy is not installed.
    """
    try:
        import esmpy
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ESMPy is required for in-memory weight generation. "
            "Install ESMPy or enable the offline ESMF path."
        ) from exc
    return esmpy


def _require_healpix_geo_module(nest: bool) -> tuple[Any, dict[str, str]]:
    """Import and return the appropriate ``healpix_geo`` sub-module.

    Args:
        nest: Select `healpix_geo.nested` when *True*, otherwise
            `healpix_geo.ring`.

    Returns:
        The imported sub-module (`healpix_geo.nested` or
        `healpix_geo.ring`).

    Raises:
        ImportError: When `healpix-geo` is not installed.
    """
    kwargs = {}
    try:
        if nest:
            kwargs["ellipsoid"] = "WGS84"
            from healpix_geo import nested as module
        else:
            from healpix_geo import ring as module
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "healpix-geo is required to construct HEALPix polygons."
        ) from exc
    return module, kwargs


# ===================================================================
# Low-level coordinate helpers (fully vectorised)
# ===================================================================


def _to_float64(values: Any) -> FloatArray:
    """Cast *values* to a contiguous float64 array.

    Args:
        values: Anything accepted by :func:`numpy.asarray`.

    Returns:
        Float64 NumPy array.
    """
    return np.asarray(values, dtype=np.float64)


def _canonical_lon(lon_deg: FloatArray) -> FloatArray:
    """Map longitudes into the range ``[-180, 180)``.

    Args:
        lon_deg: Longitude values in degrees.

    Returns:
        Canonicalised longitude array (same shape as input).
    """
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _looks_like_radians(values: FloatArray) -> bool:
    """Heuristic test whether *values* are in radians.

    The check passes when the maximum absolute finite value is at most
    ``2 * pi + 1e-6``.

    Args:
        values: Coordinate array to inspect.

    Returns:
        *True* when the values appear to be in radians.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return False
    return bool(float(np.nanmax(np.abs(finite))) <= (2.0 * np.pi + 1e-6))


def _normalise_angle_units(
    values: FloatArray,
    units: SourceUnits,
) -> FloatArray:
    """Convert *values* to degrees according to *units*.

    When ``units="auto"`` the function applies
    [`_looks_like_radians`][grid_doctor.remap_backend._looks_like_radians]
    and converts if the heuristic fires.

    Args:
        values: Coordinate array.
        units: Unit convention (``"deg"``, ``"rad"``, or ``"auto"``).

    Returns:
        Coordinate array guaranteed to be in degrees.
    """
    if units == "deg":
        return values.astype(np.float64, copy=False)
    if units == "rad":
        return np.rad2deg(values)
    if _looks_like_radians(values):
        return np.rad2deg(values)
    return values.astype(np.float64, copy=False)


def _lonlat_to_xyz(lon_deg: FloatArray, lat_deg: FloatArray) -> FloatArray:
    """Convert lon/lat coordinates to Cartesian unit-sphere vectors.

    Both inputs are broadcast together before conversion, so any
    compatible shapes are accepted.

    Args:
        lon_deg: Longitude in degrees.
        lat_deg: Latitude in degrees.

    Returns:
        Array of shape ``(*broadcast_shape, 3)`` with
        ``(x, y, z)`` columns.
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat)
    return np.stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


@overload
def _xyz_to_lonlat(
    xyz: FloatArray, *, batch: Literal[False] = ...
) -> tuple[float, float]: ...  # noqa: E501


@overload
def _xyz_to_lonlat(
    xyz: FloatArray, *, batch: Literal[True]
) -> tuple[FloatArray, FloatArray]: ...  # noqa: E501


def _xyz_to_lonlat(
    xyz: FloatArray,
    *,
    batch: bool = False,
) -> tuple[float, float] | tuple[FloatArray, FloatArray]:
    """Convert Cartesian unit-sphere vectors back to lon/lat in degrees.

    Args:
        xyz: Either a single ``(3,)`` vector or, when *batch* is
            *True*, an array of shape ``(..., 3)``.
        batch: When *True* the function operates element-wise over the
            leading dimensions and returns arrays instead of scalars.

    Returns:
        ``(lon_deg, lat_deg)`` as scalars (default) or arrays.

    Raises:
        ValueError: When any vector has zero length.
    """
    if batch:
        norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
        if np.any(norm == 0.0):
            raise ValueError("Cannot convert zero-length vector to lon/lat.")
        unit = xyz / norm
        lon = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0]))
        lat = np.rad2deg(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
        return (
            lon.astype(np.float64, copy=False),
            lat.astype(np.float64, copy=False),
        )

    norm = np.linalg.norm(xyz)
    if norm == 0.0:
        raise ValueError("Cannot convert zero-length vector to lon/lat.")
    unit = xyz / norm
    lon = np.rad2deg(np.arctan2(unit[1], unit[0]))
    lat = np.rad2deg(np.arcsin(np.clip(unit[2], -1.0, 1.0)))
    return float(lon), float(lat)


def _spherical_centroid(
    lon_deg: FloatArray,
    lat_deg: FloatArray,
) -> tuple[float, float]:
    """Return the spherical centroid of a set of lon/lat points.

    The centroid is computed as the mean Cartesian vector projected back
    onto the unit sphere.

    Args:
        lon_deg: Longitude values in degrees.
        lat_deg: Latitude values in degrees.

    Returns:
        ``(lon_deg, lat_deg)`` of the centroid.
    """
    xyz = _lonlat_to_xyz(lon_deg, lat_deg)
    return _xyz_to_lonlat(xyz.mean(axis=0))


def _ensure_ccw(
    lon_deg: FloatArray,
    lat_deg: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Ensure a polygon's vertices are wound counter-clockwise.

    The winding test is performed in a local tangent-plane projection
    centred on the polygon's spherical centroid.

    Args:
        lon_deg: Polygon vertex longitudes in degrees.
        lat_deg: Polygon vertex latitudes in degrees.

    Returns:
        ``(lon_deg, lat_deg)`` in counter-clockwise order.
    """
    pts = _lonlat_to_xyz(lon_deg, lat_deg)
    lon0, lat0 = _spherical_centroid(lon_deg, lat_deg)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)

    east = np.array(
        [-np.sin(lon0_rad), np.cos(lon0_rad), 0.0],
        dtype=np.float64,
    )
    north = np.array(
        [
            -np.sin(lat0_rad) * np.cos(lon0_rad),
            -np.sin(lat0_rad) * np.sin(lon0_rad),
            np.cos(lat0_rad),
        ],
        dtype=np.float64,
    )
    x = pts @ east
    y = pts @ north
    signed_area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed_area < 0.0:
        return lon_deg[::-1].copy(), lat_deg[::-1].copy()
    return lon_deg, lat_deg


# ===================================================================
# Dataset coordinate introspection
# ===================================================================


def _get_latlon_arrays(ds: xr.Dataset) -> tuple[FloatArray, FloatArray]:
    """Extract latitude and longitude arrays from *ds*.

    The function searches coordinates and data variables using the
    priority-ordered name lists
    [`_LAT_NAMES`][grid_doctor.remap_backend._LAT_NAMES] and
    [`_LON_NAMES`][grid_doctor.remap_backend._LON_NAMES].

    Args:
        ds: Source dataset.

    Returns:
        ``(lat, lon)`` as float64 NumPy arrays.

    Raises:
        ValueError: When no recognised coordinate names are found.
    """
    lat: FloatArray | None = None
    lon: FloatArray | None = None

    for name in _LAT_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lat = _to_float64(ds[name].values)
            break
    for name in _LON_NAMES:
        if name in ds.coords or name in ds.data_vars:
            lon = _to_float64(ds[name].values)
            break

    if lat is None or lon is None:
        available = sorted({*map(str, ds.coords), *map(str, ds.data_vars)})
        raise ValueError(
            "Could not locate latitude/longitude coordinates. "
            f"Available names are: {available}."
        )
    return lat, lon


def _is_unstructured(ds: xr.Dataset) -> bool:
    """Check whether *ds* looks like an unstructured grid.

    The test succeeds when any dimension name is in
    [`_UNSTRUCTURED_DIMS`][grid_doctor.remap_backend._UNSTRUCTURED_DIMS]
    or when any variable carries the ``CDI_grid_type=unstructured``
    attribute.

    Args:
        ds: Dataset to test.

    Returns:
        *True* when the dataset appears to be unstructured.
    """
    if _UNSTRUCTURED_DIMS & {str(dim) for dim in ds.dims}:
        return True
    return any(
        var.attrs.get("CDI_grid_type") == "unstructured"
        for var in ds.data_vars.values()
    )


def _get_unstructured_dim(ds: xr.Dataset) -> str:
    """Return the name of the unstructured cell dimension in *ds*.

    Args:
        ds: Unstructured dataset.

    Returns:
        Dimension name.

    Raises:
        ValueError: When the cell dimension cannot be determined.
    """
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
    """Return the ``(y_dim, x_dim)`` spatial dimension names.

    For curvilinear grids where the dimension names are not standard,
    the function falls back to inspecting 2-D coordinate shapes.

    Args:
        ds: Source dataset.

    Returns:
        ``(y_dim, x_dim)`` names.

    Raises:
        ValueError: When the spatial dimensions cannot be identified.
    """
    y_dim: str | None = None
    x_dim: str | None = None

    for dim in ds.dims:
        dim_name = str(dim)
        dim_lower = dim_name.lower()
        if y_dim is None and dim_lower in _Y_CANDIDATES:
            y_dim = dim_name
        elif x_dim is None and dim_lower in _X_CANDIDATES:
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


# ===================================================================
# Bounds and coverage inference
# ===================================================================


def _infer_bounds_1d(values: FloatArray) -> FloatArray:
    """Infer cell boundaries from a 1-D centre coordinate array.

    The boundaries are placed at the midpoints between consecutive
    centres, with half-step extrapolation at both ends.

    Args:
        values: Sorted 1-D coordinate centres.

    Returns:
        Array of length ``values.size + 1``.

    Raises:
        ValueError: When *values* is not 1-D or has fewer than two
            elements.
    """
    if values.ndim != 1:
        raise ValueError("Expected a one-dimensional coordinate array.")
    if values.size < 2:
        raise ValueError("At least two coordinate values are required to infer bounds.")
    values = values.astype(np.float64, copy=False)
    diffs = np.diff(values)
    bounds = np.empty(values.size + 1, dtype=np.float64)
    bounds[1:-1] = values[:-1] + diffs / 2.0
    bounds[0] = values[0] - diffs[0] / 2.0
    bounds[-1] = values[-1] + diffs[-1] / 2.0
    return bounds


def _infer_curvilinear_corners(
    lat_deg: FloatArray,
    lon_deg: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Infer cell corner coordinates from curvilinear centres.

    Corner coordinates are estimated on the unit sphere by averaging
    the Cartesian vectors of the (up to four) adjacent cell centres
    around each corner and projecting back to lon/lat.  This keeps the
    interpolation stable across the antimeridian.

    The implementation is fully vectorised: no Python loops are used.

    Args:
        lat_deg: 2-D latitude centres with shape ``(ny, nx)``.
        lon_deg: 2-D longitude centres with shape ``(ny, nx)``.

    Returns:
        ``(lat_corner, lon_corner)`` each with shape
        ``(ny + 1, nx + 1)`` in degrees.  Longitudes are
        canonicalised to ``[-180, 180)``.

    Raises:
        ValueError: When the inputs are not two-dimensional.
    """
    if lat_deg.ndim != 2 or lon_deg.ndim != 2:
        raise ValueError("Curvilinear corner inference expects 2-D arrays.")
    ny, nx = lat_deg.shape

    # Cartesian cell centres: (ny, nx, 3)
    xyz = _lonlat_to_xyz(lon_deg, lat_deg)

    # Pad with zeros on all sides so that corner (j, i) can always
    # address its four potential neighbours at shifted indices.
    # padded shape: (ny + 2, nx + 2, 3)
    padded = np.zeros((ny + 2, nx + 2, 3), dtype=np.float64)
    padded[1:-1, 1:-1] = xyz

    # Neighbour count for proper averaging.  Same padding logic.
    count = np.zeros((ny + 2, nx + 2), dtype=np.float64)
    count[1:-1, 1:-1] = 1.0

    # For corner (j, i) with 0 <= j <= ny, 0 <= i <= nx the four
    # adjacent cell centres (when they exist) sit at grid indices
    # (j-1, i-1), (j-1, i), (j, i-1), (j, i).  In the padded array
    # these correspond to offsets (j, i), (j, i+1), (j+1, i),
    # (j+1, i+1) — i.e. four shifted views of size (ny+1, nx+1).
    corner_sum = (
        padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]
    )  # (ny + 1, nx + 1, 3)

    corner_count = (count[:-1, :-1] + count[:-1, 1:] + count[1:, :-1] + count[1:, 1:])[
        ..., np.newaxis
    ]  # (ny + 1, nx + 1, 1)

    # Average the Cartesian vectors and project back to lon/lat.
    mean_xyz = corner_sum / corner_count  # safe: count >= 1 always
    lon_corner, lat_corner = _xyz_to_lonlat(mean_xyz, batch=True)

    return lat_corner, _canonical_lon(lon_corner)


def _median_positive_step(values: FloatArray) -> float:
    """Return the median positive step size in a sorted array.

    Args:
        values: 1-D coordinate values (need not be sorted).

    Returns:
        Median of the positive finite differences, or ``0.0`` when
        no positive step exists.
    """
    if values.size < 2:
        return 0.0
    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _lon_coverage_from_centres(lon_deg: FloatArray) -> float:
    """Estimate longitude coverage from cell-centre coordinates.

    The estimate is dateline-safe and adds one representative cell
    width so that a global grid with centres at ``0, 15, ..., 345``
    reports a coverage of ``360``.

    Args:
        lon_deg: Longitude centres in degrees.

    Returns:
        Estimated coverage in degrees, clamped to ``[0, 360]``.
    """
    lon = np.mod(lon_deg[np.isfinite(lon_deg)], 360.0)
    if lon.size == 0:
        return 0.0
    lon = np.unique(np.sort(lon))
    if lon.size == 1:
        return 0.0
    dlon = _median_positive_step(lon)
    wrapped = np.concatenate((lon, lon[:1] + 360.0))
    gaps = np.diff(wrapped)
    coverage = 360.0 - float(np.max(gaps)) + dlon
    return float(min(360.0, max(0.0, coverage)))


def _lat_coverage_from_centres(lat_deg: FloatArray) -> float:
    """Estimate latitude coverage from cell-centre coordinates.

    One representative cell height is added to the raw span.

    Args:
        lat_deg: Latitude centres in degrees.

    Returns:
        Estimated coverage in degrees, clamped to ``[0, 180]``.
    """
    lat = lat_deg[np.isfinite(lat_deg)]
    if lat.size == 0:
        return 0.0
    lat = np.unique(np.sort(lat))
    if lat.size == 1:
        return 0.0
    dlat = _median_positive_step(lat)
    coverage = float(lat[-1] - lat[0]) + dlat
    return float(min(180.0, max(0.0, coverage)))


def _looks_global(
    ds: xr.Dataset,
    *,
    source_units: SourceUnits,
) -> bool:
    """Heuristic check whether *ds* covers the full globe.

    The test passes when the estimated longitude coverage is at least
    350 degrees and the latitude coverage is at least 170 degrees.

    Args:
        ds: Source dataset.
        source_units: Angular unit convention.

    Returns:
        *True* when the source appears global.
    """
    lat, lon = _get_latlon_arrays(ds)
    lat_deg = _normalise_angle_units(lat.ravel(), source_units)
    lon_deg = _normalise_angle_units(lon.ravel(), source_units)
    lon_cov = _lon_coverage_from_centres(lon_deg)
    lat_cov = _lat_coverage_from_centres(lat_deg)
    return bool(lon_cov >= 350.0 and lat_cov >= 170.0)


# ===================================================================
# Source classification
# ===================================================================


def _classify_source_kind(
    ds: xr.Dataset,
    *,
    explicit_kind: SourceKind,
) -> SourceKind:
    """Classify the source grid representation.

    Args:
        ds: Source geometry dataset.
        explicit_kind: Caller-specified kind; returned as-is unless
            ``"auto"``.

    Returns:
        One of ``"regular"``, ``"curvilinear"``, or
        ``"unstructured"``.

    Raises:
        ValueError: When automatic classification fails.
    """
    if explicit_kind != "auto":
        return explicit_kind
    if _is_unstructured(ds):
        return "unstructured"
    lat, _ = _get_latlon_arrays(ds)
    if lat.ndim == 1:
        return "regular"
    if lat.ndim == 2:
        return "curvilinear"
    raise ValueError("Could not classify the source representation.")


# ===================================================================
# Polygon-centre computation (vectorised)
# ===================================================================


def _vectorized_polygon_centres(
    cell_lon: FloatArray,
    cell_lat: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute spherical polygon centres from padded corner arrays.

    The centre of each polygon is the normalised mean Cartesian vector
    of its valid corners, projected back to lon/lat.

    Args:
        cell_lon: Corner longitudes, shape ``(n_face, max_corners)``.
            Invalid slots contain ``NaN``.
        cell_lat: Corner latitudes, same shape as *cell_lon*.

    Returns:
        ``(face_lon, face_lat)`` in degrees, each shape ``(n_face,)``.
        Longitudes are canonicalised to ``[-180, 180)``.

    Raises:
        ValueError: When any polygon has no valid corners or the
            mean vector has zero length.
    """
    mask = np.isfinite(cell_lon) & np.isfinite(cell_lat)
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Every polygon must contain at least one valid corner.")

    lon_rad = np.deg2rad(np.where(mask, cell_lon, 0.0))
    lat_rad = np.deg2rad(np.where(mask, cell_lat, 0.0))
    cos_lat = np.cos(lat_rad)

    x = np.where(mask, cos_lat * np.cos(lon_rad), 0.0).sum(axis=1)
    y = np.where(mask, cos_lat * np.sin(lon_rad), 0.0).sum(axis=1)
    z = np.where(mask, np.sin(lat_rad), 0.0).sum(axis=1)

    norm = np.sqrt(x * x + y * y + z * z)
    if np.any(norm == 0.0):
        raise ValueError("Cannot infer polygon centres from zero-length vectors.")

    lon = np.rad2deg(np.arctan2(y, x)).astype(np.float64, copy=False)
    lat = np.rad2deg(np.arcsin(np.clip(z / norm, -1.0, 1.0))).astype(
        np.float64, copy=False
    )
    return _canonical_lon(lon), lat


# ===================================================================
# Mesh construction — one builder per grid type
# ===================================================================


def _regular_grid_mesh(
    lat: FloatArray,
    lon: FloatArray,
) -> PolygonMesh:
    """Construct a polygon mesh from a regular lon/lat grid.

    The cell boundaries are inferred from the 1-D centre arrays using
    [`_infer_bounds_1d`][grid_doctor.remap_backend._infer_bounds_1d].
    No Python-level cell loops are used.

    Args:
        lat: 1-D latitude centres in degrees, shape ``(ny,)``.
        lon: 1-D longitude centres in degrees, shape ``(nx,)``.

    Returns:
        Compact polygon mesh.
    """
    lat_bounds = _infer_bounds_1d(lat)
    lon_bounds = _canonical_lon(_infer_bounds_1d(lon))
    ny, nx = lat.size, lon.size

    node_idx = np.arange((ny + 1) * (nx + 1), dtype=np.int32).reshape(ny + 1, nx + 1)
    face_nodes = np.stack(
        (
            node_idx[:-1, :-1],
            node_idx[:-1, 1:],
            node_idx[1:, 1:],
            node_idx[1:, :-1],
        ),
        axis=-1,
    ).reshape(-1, 4)

    node_lon_2d, node_lat_2d = np.meshgrid(lon_bounds, lat_bounds)
    face_lon_2d, face_lat_2d = np.meshgrid(_canonical_lon(lon), lat)

    return PolygonMesh(
        node_lon=node_lon_2d.ravel().astype(np.float64, copy=False),
        node_lat=node_lat_2d.ravel().astype(np.float64, copy=False),
        face_nodes=face_nodes,
        face_lon=face_lon_2d.ravel().astype(np.float64, copy=False),
        face_lat=face_lat_2d.ravel().astype(np.float64, copy=False),
    )


def _curvilinear_grid_mesh(
    lat: FloatArray,
    lon: FloatArray,
) -> PolygonMesh:
    """Construct a polygon mesh from a curvilinear lon/lat grid.

    Corner nodes are inferred with
    [`_infer_curvilinear_corners`][grid_doctor.remap_backend._infer_curvilinear_corners].

    Args:
        lat: 2-D latitude centres, shape ``(ny, nx)``.
        lon: 2-D longitude centres, shape ``(ny, nx)``.

    Returns:
        Compact polygon mesh.
    """
    lat_corner, lon_corner = _infer_curvilinear_corners(lat, lon)
    ny, nx = lat.shape

    node_idx = np.arange((ny + 1) * (nx + 1), dtype=np.int32).reshape(ny + 1, nx + 1)
    face_nodes = np.stack(
        (
            node_idx[:-1, :-1],
            node_idx[:-1, 1:],
            node_idx[1:, 1:],
            node_idx[1:, :-1],
        ),
        axis=-1,
    ).reshape(-1, 4)

    return PolygonMesh(
        node_lon=_canonical_lon(lon_corner.ravel().astype(np.float64, copy=False)),
        node_lat=lat_corner.ravel().astype(np.float64, copy=False),
        face_nodes=face_nodes,
        face_lon=_canonical_lon(lon.ravel().astype(np.float64, copy=False)),
        face_lat=lat.ravel().astype(np.float64, copy=False),
    )


def _corner_mesh_from_arrays(
    cell_lon: FloatArray,
    cell_lat: FloatArray,
    *,
    node_round_ndigits: int = 12,
) -> PolygonMesh:
    """Construct a mesh from padded per-face corner arrays.

    Node deduplication is performed by rounding coordinates to
    *node_round_ndigits* decimal digits and using
    :func:`numpy.unique` on a structured ``(lon, lat)`` key.

    Args:
        cell_lon: Corner longitudes, shape
            ``(n_face, max_face_nodes)``.  Invalid slots contain
            ``NaN``.
        cell_lat: Corner latitudes, same shape as *cell_lon*.
        node_round_ndigits: Number of decimal digits used for the
            deduplication rounding.

    Returns:
        Compact polygon mesh with deduplicated nodes.

    Raises:
        ValueError: When any face has fewer than three valid vertices.
    """
    if cell_lon.ndim != 2 or cell_lat.ndim != 2 or cell_lon.shape != cell_lat.shape:
        raise ValueError("Expected corner arrays with shape (n_face, max_face_nodes).")

    mask = np.isfinite(cell_lon) & np.isfinite(cell_lat)
    valid_counts = mask.sum(axis=1)
    if np.any(valid_counts < 3):
        bad = int(np.flatnonzero(valid_counts < 3)[0])
        raise ValueError(f"Source cell {bad} has fewer than three valid vertices.")

    canonical = _canonical_lon(cell_lon)

    # Round for deduplication, keeping original values for output.
    rounded_lon = np.round(np.where(mask, canonical, 0.0), node_round_ndigits)
    rounded_lat = np.round(np.where(mask, cell_lat, 0.0), node_round_ndigits)

    flat_mask = mask.ravel()
    flat_lon = rounded_lon.ravel()[flat_mask]
    flat_lat = rounded_lat.ravel()[flat_mask]

    key_dtype = np.dtype([("lon", "f8"), ("lat", "f8")])
    flat_keys = np.empty(flat_lon.shape[0], dtype=key_dtype)
    flat_keys["lon"] = flat_lon
    flat_keys["lat"] = flat_lat

    _, unique_idx, inverse = np.unique(
        flat_keys, return_index=True, return_inverse=True
    )

    original_lon = canonical.ravel()[flat_mask]
    original_lat = cell_lat.ravel()[flat_mask]

    face_nodes = np.full(cell_lon.shape, -1, dtype=np.int32)
    face_nodes.ravel()[flat_mask] = inverse.astype(np.int32, copy=False)

    face_lon, face_lat = _vectorized_polygon_centres(canonical, cell_lat)

    return PolygonMesh(
        node_lon=original_lon[unique_idx].astype(np.float64, copy=False),
        node_lat=original_lat[unique_idx].astype(np.float64, copy=False),
        face_nodes=face_nodes,
        face_lon=face_lon,
        face_lat=face_lat,
    )


# ===================================================================
# Mesh conversion helpers
# ===================================================================


def _mesh_to_polygons(
    mesh: PolygonMesh,
) -> list[tuple[FloatArray, FloatArray]]:
    """Convert a compact mesh to a list of ``(lon, lat)`` polygon tuples.

    This helper is intended for tests and small debugging scenarios.

    Args:
        mesh: Source polygon mesh.

    Returns:
        List of ``(lon_deg, lat_deg)`` vertex arrays, one per face.
    """
    polygons: list[tuple[FloatArray, FloatArray]] = []
    for row in mesh.face_nodes:
        ids = row[row >= 0]
        polygons.append((mesh.node_lon[ids].copy(), mesh.node_lat[ids].copy()))
    return polygons


def _polygons_to_corner_arrays(
    polygons: list[tuple[FloatArray, FloatArray]],
) -> tuple[FloatArray, FloatArray]:
    """Convert polygon tuples into NaN-padded corner arrays.

    Args:
        polygons: List of ``(lon_deg, lat_deg)`` vertex arrays.

    Returns:
        ``(cell_lon, cell_lat)`` each with shape
        ``(n_face, max_face_nodes)``, padded with ``NaN``.
    """
    max_nodes = max(len(lon) for lon, _ in polygons)
    cell_lon = np.full((len(polygons), max_nodes), np.nan, dtype=np.float64)
    cell_lat = np.full((len(polygons), max_nodes), np.nan, dtype=np.float64)
    for idx, (lon, lat) in enumerate(polygons):
        n = len(lon)
        cell_lon[idx, :n] = lon
        cell_lat[idx, :n] = lat
    return cell_lon, cell_lat


# ===================================================================
# Source mesh dispatch
# ===================================================================


def _source_mesh(
    ds: xr.Dataset,
    *,
    source_units: SourceUnits,
) -> tuple[PolygonMesh, tuple[str, ...]]:
    """Construct the compact source mesh from the dataset geometry.

    The function dispatches to
    [`_regular_grid_mesh`][grid_doctor.remap_backend._regular_grid_mesh],
    [`_curvilinear_grid_mesh`][grid_doctor.remap_backend._curvilinear_grid_mesh],
    or
    [`_corner_mesh_from_arrays`][grid_doctor.remap_backend._corner_mesh_from_arrays]
    depending on the coordinate layout.

    Args:
        ds: Source geometry dataset.
        source_units: Angular unit convention.

    Returns:
        ``(mesh, source_dims)`` where *source_dims* gives the
        dimension names in flattening order.

    Raises:
        ValueError: When the grid type cannot be handled or required
            vertex coordinates are missing.
    """
    if _is_unstructured(ds):
        lon_name = "clon_vertices" if "clon_vertices" in ds else "lon_vertices"
        lat_name = "clat_vertices" if "clat_vertices" in ds else "lat_vertices"
        if lon_name not in ds or lat_name not in ds:
            raise ValueError(
                "Unstructured grids require per-cell vertex "
                "coordinates such as "
                "'clon_vertices'/'clat_vertices'."
            )
        lon_v = _canonical_lon(
            _normalise_angle_units(_to_float64(ds[lon_name].values), source_units)
        )
        lat_v = _normalise_angle_units(_to_float64(ds[lat_name].values), source_units)
        mesh = _corner_mesh_from_arrays(lon_v, lat_v)
        return mesh, (_get_unstructured_dim(ds),)

    lat, lon = _get_latlon_arrays(ds)
    lat = _normalise_angle_units(lat, source_units)
    lon = _canonical_lon(_normalise_angle_units(lon, source_units))
    y_dim, x_dim = _get_spatial_dims(ds)

    if lat.ndim == 1:
        return _regular_grid_mesh(lat, lon), (y_dim, x_dim)
    if lat.ndim == 2:
        return _curvilinear_grid_mesh(lat, lon), (y_dim, x_dim)
    raise ValueError("Latitude/longitude coordinates must be 1-D or 2-D.")


def _source_polygons(
    ds: xr.Dataset,
    *,
    source_units: SourceUnits,
) -> tuple[list[tuple[FloatArray, FloatArray]], tuple[str, ...]]:
    """Return source polygons as Python tuples.

    Note:
        Production weight generation uses
        [`_source_mesh`][grid_doctor.remap_backend._source_mesh] to
        avoid per-cell Python loops.  This wrapper is kept for
        compatibility with tests and small-scale debugging.

    Args:
        ds: Source geometry dataset.
        source_units: Angular unit convention.

    Returns:
        ``(polygons, source_dims)``.
    """
    mesh, source_dims = _source_mesh(ds, source_units=source_units)
    return _mesh_to_polygons(mesh), source_dims


# ===================================================================
# HEALPix target mesh
# ===================================================================


def _target_healpix_mesh(
    level: int,
    *,
    nest: bool,
) -> tuple[npt.NDArray[np.int64], PolygonMesh]:
    """Return the HEALPix destination mesh.

    Args:
        level: HEALPix refinement level.
        nest: Use nested ordering when *True*.

    Returns:
        ``(ipix, mesh)`` where *ipix* is the pixel index array.
    """
    module, kwargs = _require_healpix_geo_module(nest)
    ipix = np.arange(12 * (4**level), dtype=np.int64)
    lon_v, lat_v = module.vertices(ipix, level, **kwargs)
    cell_lon = _canonical_lon(np.asarray(lon_v, dtype=np.float64))
    cell_lat = np.asarray(lat_v, dtype=np.float64)
    return ipix, _corner_mesh_from_arrays(cell_lon, cell_lat)


def _target_healpix_polygons(
    level: int,
    *,
    nest: bool,
) -> tuple[npt.NDArray[np.int64], list[tuple[FloatArray, FloatArray]]]:
    """Return HEALPix polygons as Python tuples.

    Note:
        Legacy helper kept for backward compatibility.

    Args:
        level: HEALPix refinement level.
        nest: Use nested ordering when *True*.

    Returns:
        ``(ipix, polygons)``.
    """
    ipix, mesh = _target_healpix_mesh(level, nest=nest)
    return ipix, _mesh_to_polygons(mesh)


# ===================================================================
# ESMPy mesh construction
# ===================================================================


def _mesh_from_polygon_mesh(
    mesh: PolygonMesh,
    *,
    esmpy_mod: Any,
) -> Any:
    """Construct an ESMPy :class:`Mesh` from a compact polygon mesh.

    Args:
        mesh: Source or target polygon mesh.
        esmpy_mod: The imported ``esmpy`` module.

    Returns:
        An ``esmpy.Mesh`` instance ready for use with
        ``esmpy.Regrid``.
    """
    valid_counts = np.sum(mesh.face_nodes >= 0, axis=1)
    element_types = np.where(
        valid_counts == 3,
        int(esmpy_mod.MeshElemType.TRI),
        np.where(
            valid_counts == 4,
            int(esmpy_mod.MeshElemType.QUAD),
            valid_counts,
        ),
    ).astype(np.int32, copy=False)

    element_conn = mesh.face_nodes[mesh.face_nodes >= 0].astype(np.int32, copy=False)

    # Interleaved (lon, lat) element coordinates.
    element_coords = np.empty(2 * mesh.face_count, dtype=np.float64)
    element_coords[0::2] = mesh.face_lon
    element_coords[1::2] = mesh.face_lat

    esmf_mesh = esmpy_mod.Mesh(
        parametric_dim=2,
        spatial_dim=2,
        coord_sys=esmpy_mod.CoordSys.SPH_DEG,
    )

    n_nodes = mesh.node_lon.size
    node_coords = np.empty(2 * n_nodes, dtype=np.float64)
    node_coords[0::2] = mesh.node_lon
    node_coords[1::2] = mesh.node_lat

    esmf_mesh.add_nodes(
        node_count=n_nodes,
        node_ids=np.arange(1, n_nodes + 1, dtype=np.int32),
        node_coords=node_coords,
        node_owners=np.zeros(n_nodes, dtype=np.int32),
    )
    esmf_mesh.add_elements(
        element_count=mesh.face_count,
        element_ids=np.arange(1, mesh.face_count + 1, dtype=np.int32),
        element_types=element_types,
        element_conn=element_conn,
        element_coords=element_coords,
    )
    return esmf_mesh


def _mesh_from_polygons(
    polygons: list[tuple[FloatArray, FloatArray]],
    *,
    esmpy_mod: Any,
) -> Any:
    """Backward-compatible wrapper for ESMPy mesh construction.

    Note:
        Prefer
        [`_mesh_from_polygon_mesh`][grid_doctor.remap_backend._mesh_from_polygon_mesh]
        with a pre-built
        [`PolygonMesh`][grid_doctor.remap_backend.PolygonMesh].

    Args:
        polygons: List of ``(lon_deg, lat_deg)`` vertex arrays.
        esmpy_mod: The imported ``esmpy`` module.

    Returns:
        An ``esmpy.Mesh`` instance.
    """
    arrays = _polygons_to_corner_arrays(polygons)
    mesh = _corner_mesh_from_arrays(*arrays)
    return _mesh_from_polygon_mesh(mesh, esmpy_mod=esmpy_mod)


# ===================================================================
# UGRID file I/O
# ===================================================================


def write_ugrid_mesh_file(
    polygons: PolygonMesh | list[tuple[FloatArray, FloatArray]],
    path: str | Path,
    *,
    mesh_name: str,
) -> Path:
    """Write a polygon mesh as a simple UGRID NetCDF file.

    The output conforms to UGRID-1.0 conventions and is accepted by
    `ESMF_RegridWeightGen --src_type UGRID`.

    Args:
        polygons: Either a
            [`PolygonMesh`][grid_doctor.remap_backend.PolygonMesh] or
            a list of ``(lon_deg, lat_deg)`` polygon tuples.
        path: Output file path.
        mesh_name: Name used for the mesh topology variable.

    Returns:
        Resolved output path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(polygons, PolygonMesh):
        mesh = polygons
    else:
        mesh = _corner_mesh_from_arrays(*_polygons_to_corner_arrays(polygons))

    topo = xr.DataArray(
        0,
        attrs={
            "cf_role": "mesh_topology",
            "topology_dimension": 2,
            "node_coordinates": "node_lon node_lat",
            "face_node_connectivity": f"{mesh_name}_face_nodes",
        },
    )
    ds = xr.Dataset(
        data_vars={
            mesh_name: topo,
            "node_lon": (
                "node",
                mesh.node_lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "node_lat": (
                "node",
                mesh.node_lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            f"{mesh_name}_face_nodes": (
                ("face", "max_face_nodes"),
                mesh.face_nodes,
                {"start_index": 0},
            ),
            "face_lon": (
                "face",
                mesh.face_lon,
                {
                    "standard_name": "longitude",
                    "units": "degrees_east",
                },
            ),
            "face_lat": (
                "face",
                mesh.face_lat,
                {
                    "standard_name": "latitude",
                    "units": "degrees_north",
                },
            ),
        },
        attrs={"Conventions": "UGRID-1.0"},
    )
    face_node_var = f"{mesh_name}_face_nodes"
    encoding: dict[str, dict[str, Any]] = {
        str(var): {"_FillValue": None}
        for var in ds.data_vars
        if str(var) != face_node_var
    }
    encoding[face_node_var] = {
        "dtype": "int32",
        "_FillValue": np.int32(-1),
    }
    ds.to_netcdf(output, encoding=encoding, format="NETCDF3_64BIT")

    return output


# ===================================================================
# Subprocess and external-tool runners
# ===================================================================


def _run_subprocess(cmd: list[str]) -> None:
    """Run *cmd* and raise on non-zero exit.

    Args:
        cmd: Command and arguments.

    Raises:
        RuntimeError: When the command exits with a non-zero code.
    """
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {completed.returncode}): "
            f"{' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}"
        )


def run_esmf_regrid_weightgen(
    *,
    src_mesh: str | Path,
    dst_mesh: str | Path,
    weights_path: str | Path,
    method: RemapMethod,
    config: OfflineWeightConfig,
    ignore_unmapped: bool,
    src_regional: bool,
    dst_regional: bool,
) -> Path:
    """Run ``ESMF_RegridWeightGen`` under MPI.

    Args:
        src_mesh: Source UGRID mesh file.
        dst_mesh: Destination UGRID mesh file.
        weights_path: Output NetCDF weight file.
        method: Weight method (``"nearest"`` or ``"conservative"``).
        config: Offline execution configuration.
        ignore_unmapped: Pass ``--ignore_unmapped`` when *True*.
        src_regional: Pass ``--src_regional`` when *True*.
        dst_regional: Pass ``--dst_regional`` when *True*.

    Returns:
        Path to the generated weight file.
    """
    method_arg = "neareststod" if method == "nearest" else "conserve"
    cmd: list[str] = []

    if config.nproc > 1:
        flag = "-n" if config.mpirun == "srun" else "-np"
        cmd.extend([config.mpirun, flag, str(config.nproc)])

    cmd.extend(
        [
            config.esmf_regrid_weightgen,
            "-s",
            str(src_mesh),
            "-d",
            str(dst_mesh),
            "-w",
            str(weights_path),
            "--src_type",
            "UGRID",
            "--dst_type",
            "UGRID",
            "-m",
            method_arg,
        ]
    )
    if config.weight_only:
        cmd.append("--weight_only")
    if config.netcdf4:
        cmd.append("--netcdf4")
    if ignore_unmapped:
        cmd.append("--ignore_unmapped")
    if src_regional:
        cmd.append("--src_regional")
    if dst_regional:
        cmd.append("--dst_regional")

    _run_subprocess(cmd)
    return Path(weights_path)


# ===================================================================
# Offline decision heuristic
# ===================================================================


def _default_offline_enabled(
    *,
    method: RemapMethod,
    source_cell_count: int,
    target_cell_count: int,
    config: OfflineWeightConfig,
) -> bool:
    """Decide whether to use the offline ESMF path automatically.

    The offline path is chosen when:

    - *config.enabled* is explicitly *True*, **or**
    - the method is ``"conservative"``, ``nproc > 1``, and the grid
      is large (source >= 200 000 cells or target >= 1 000 000 cells).

    Args:
        method: Weight-generation method.
        source_cell_count: Number of source cells.
        target_cell_count: Number of target cells.
        config: Offline configuration.

    Returns:
        *True* when the offline path should be used.
    """
    if config.enabled:
        return True
    if method != "conservative":
        return False
    return bool(
        config.nproc > 1
        and (source_cell_count >= 200_000 or target_cell_count >= 1_000_000)
    )


# ===================================================================
# Spectral-to-gridpoint materialisation
# ===================================================================


def _materialise_spectral_source(
    source: xr.Dataset | str | Path,
    *,
    transform_command: list[str] | tuple[str, ...] | None,
    workdir: Path,
) -> xr.Dataset:
    """Run an external spectral-to-gridpoint transform.

    Args:
        source: Source dataset or path to the spectral input file.
        transform_command: External command with ``{input}`` and
            ``{output}`` placeholders.
        workdir: Working directory for intermediate files.

    Returns:
        The transformed gridpoint dataset.

    Raises:
        ValueError: When *transform_command* is ``None``.
        SpectralTransformError: When the external command fails.
    """
    if transform_command is None:
        raise ValueError(
            "source_kind='spectral' requires an external transform "
            "command, for example a CDO sp2gp invocation."
        )

    if isinstance(source, xr.Dataset):
        spectral_in = workdir / "spectral_input.nc"
        source.to_netcdf(spectral_in)
    else:
        spectral_in = Path(source)

    spectral_out = workdir / "spectral_gridpoint.nc"
    cmd = [
        arg.format(input=str(spectral_in), output=str(spectral_out))
        for arg in transform_command
    ]
    try:
        _run_subprocess(cmd)
    except RuntimeError as exc:  # pragma: no cover
        raise SpectralTransformError(str(exc)) from exc

    return xr.load_dataset(spectral_out)


# ===================================================================
# Source description facade
# ===================================================================


def describe_source(
    source: xr.Dataset | str | Path,
    *,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    source_units: SourceUnits = "auto",
    spectral_transform_command: (list[str] | tuple[str, ...] | None) = None,
    workdir: Path | None = None,
) -> SourceDescription:
    """Resolve the source representation for HEALPix weight generation.

    This function classifies the source grid, constructs the compact
    polygon mesh, and assembles the metadata dictionary that will be
    written into the output weight file.

    Args:
        source: Source dataset or source path.  A path is mainly
            useful for spectral inputs that need an external transform.
        grid: Optional grid dataset that supplies geometry when the
            data file itself only contains values on dimensions such
            as ``cell``.
        source_kind: Explicit source representation.  Use ``"auto"``
            to infer it from the available coordinates.
        source_units: Angular unit convention of the source
            coordinates.
        spectral_transform_command: External command used when
            ``source_kind="spectral"``.  The command may use
            ``{input}`` and ``{output}`` placeholders.
        workdir: Optional working directory for temporary files.

    Returns:
        Fully resolved source description.
    """
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="grid_doctor_source_"))

    if isinstance(source, xr.Dataset):
        dataset = source
    else:
        dataset = xr.load_dataset(source)

    effective_kind = source_kind
    if effective_kind == "spectral":
        dataset = _materialise_spectral_source(
            source,
            transform_command=spectral_transform_command,
            workdir=workdir,
        )
        effective_kind = "auto"

    geometry_ds = grid if grid is not None else dataset
    resolved_kind = _classify_source_kind(geometry_ds, explicit_kind=effective_kind)
    source_mesh, source_dims = _source_mesh(geometry_ds, source_units=source_units)
    ignore_unmapped = not _looks_global(geometry_ds, source_units=source_units)

    metadata: dict[str, str | int | float | bool] = {
        "grid_doctor_source_kind": resolved_kind,
        "grid_doctor_source_dims": json.dumps(list(source_dims)),
        "grid_doctor_source_size": source_mesh.face_count,
        "grid_doctor_source_units": source_units,
    }

    return SourceDescription(
        dataset=dataset,
        kind=resolved_kind,
        source_dims=source_dims,
        source_mesh=source_mesh,
        ignore_unmapped=ignore_unmapped,
        metadata=metadata,
    )


# ===================================================================
# In-memory weight computation
# ===================================================================


def _compute_weights_in_memory(
    *,
    source_mesh: PolygonMesh,
    target_mesh: PolygonMesh,
    weights_path: Path,
    method: RemapMethod,
    ignore_unmapped: bool,
    large_file: bool,
) -> Path:
    """Compute weights with ESMPy using compact mesh arrays.

    Args:
        source_mesh: Source polygon mesh.
        target_mesh: Target polygon mesh.
        weights_path: Output weight file.
        method: Regridding method.
        ignore_unmapped: Ignore unmapped destination cells when *True*.
        large_file: Enable ESMPy large-file support.

    Returns:
        Path to the generated weight file.
    """
    esmpy_mod = _require_esmpy()
    src_esmf = _mesh_from_polygon_mesh(source_mesh, esmpy_mod=esmpy_mod)
    dst_esmf = _mesh_from_polygon_mesh(target_mesh, esmpy_mod=esmpy_mod)

    src_field = esmpy_mod.Field(src_esmf, name="src", meshloc=esmpy_mod.MeshLoc.ELEMENT)
    dst_field = esmpy_mod.Field(dst_esmf, name="dst", meshloc=esmpy_mod.MeshLoc.ELEMENT)
    src_field.data[...] = 0.0
    dst_field.data[...] = 0.0

    regrid_method = (
        esmpy_mod.RegridMethod.NEAREST_STOD
        if method == "nearest"
        else esmpy_mod.RegridMethod.CONSERVE
    )
    kwargs: dict[str, Any] = {
        "filename": str(weights_path),
        "regrid_method": regrid_method,
        "unmapped_action": (
            esmpy_mod.UnmappedAction.IGNORE
            if ignore_unmapped
            else esmpy_mod.UnmappedAction.ERROR
        ),
        "ignore_degenerate": True,
        "large_file": large_file,
    }
    if method == "conservative":
        kwargs["norm_type"] = esmpy_mod.NormType.DSTAREA
        kwargs["line_type"] = esmpy_mod.LineType.GREAT_CIRCLE

    regrid = esmpy_mod.Regrid(src_field, dst_field, **kwargs)
    regrid.destroy()
    return weights_path


# ===================================================================
# Main backend entry point
# ===================================================================


def compute_healpix_weights_backend(
    source: xr.Dataset | str | Path,
    level: int,
    *,
    method: RemapMethod,
    nest: bool,
    source_units: SourceUnits,
    weights_path: str | Path | None,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    ignore_unmapped: bool | None = None,
    large_file: bool = True,
    offline: OfflineWeightConfig | None = None,
    spectral_transform_command: (list[str] | tuple[str, ...] | None) = None,
) -> Path:
    """Compute reusable HEALPix weights.

    This is the main backend entry point.  It resolves the source
    description, builds the HEALPix target mesh, and dispatches to
    either the in-memory ESMPy workflow or the offline
    ``ESMF_RegridWeightGen`` path.

    Args:
        source: Source dataset or source path.
        level: HEALPix refinement level.
        method: Weight method (``"nearest"`` or ``"conservative"``).
        nest: Use nested HEALPix ordering when *True*.
        source_units: Angular unit convention of the source
            coordinates.
        weights_path: Output weight file path.  A temporary file is
            created when omitted.
        grid: Optional external geometry dataset.
        source_kind: Explicit source representation or ``"auto"``.
        ignore_unmapped: Whether destination cells without source
            coverage should be ignored.
        large_file: Forwarded to ESMPy for the in-memory path.
        offline: Offline weight-generation configuration.
        spectral_transform_command: External command used when
            ``source_kind="spectral"``.

    Returns:
        Path to the generated weight file.

    Raises:
        ValueError: When *method* is not supported.
        FileNotFoundError: When the offline ESMF executable or MPI
            launcher cannot be found.
    """
    if method not in {"nearest", "conservative"}:
        raise ValueError("Only 'nearest' and 'conservative' are supported.")

    offline_cfg = offline or OfflineWeightConfig()
    target = (
        Path(weights_path)
        if weights_path is not None
        else Path(tempfile.mkstemp(suffix=".nc")[1])
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    source_desc = describe_source(
        source,
        grid=grid,
        source_kind=source_kind,
        source_units=source_units,
        spectral_transform_command=spectral_transform_command,
        workdir=offline_cfg.workdir,
    )
    _, target_mesh = _target_healpix_mesh(level, nest=nest)

    use_offline = _default_offline_enabled(
        method=method,
        source_cell_count=source_desc.source_mesh.face_count,
        target_cell_count=target_mesh.face_count,
        config=offline_cfg,
    )
    eff_ignore = (
        source_desc.ignore_unmapped if ignore_unmapped is None else ignore_unmapped
    )

    if use_offline:
        generated = _run_offline_esmf(
            source_desc=source_desc,
            target_mesh=target_mesh,
            target_path=target,
            level=level,
            nest=nest,
            method=method,
            config=offline_cfg,
            ignore_unmapped=eff_ignore,
        )
    else:
        generated = _compute_weights_in_memory(
            source_mesh=source_desc.source_mesh,
            target_mesh=target_mesh,
            weights_path=target,
            method=method,
            ignore_unmapped=eff_ignore,
            large_file=large_file,
        )

    # Annotate the weight file with provenance metadata.
    meta: dict[str, str | int | float | bool] = {
        "grid_doctor_method": method,
        "grid_doctor_level": level,
        "grid_doctor_order": "nested" if nest else "ring",
        "grid_doctor_backend": ("offline-esmf" if use_offline else "esmpy"),
        "grid_doctor_ignore_unmapped": int(bool(eff_ignore)),
        **source_desc.metadata,
    }
    with xr.open_dataset(generated) as raw_weights:
        annotated = raw_weights.load()
    annotated.attrs.update(meta)
    annotated.to_netcdf(generated, mode="w")
    return generated


# ===================================================================
# Offline ESMF orchestration (extracted for readability)
# ===================================================================


def _run_offline_esmf(
    *,
    source_desc: SourceDescription,
    target_mesh: PolygonMesh,
    target_path: Path,
    level: int,
    nest: bool,
    method: RemapMethod,
    config: OfflineWeightConfig,
    ignore_unmapped: bool,
) -> Path:
    """Orchestrate the offline ESMF weight-generation workflow.

    This helper validates the external executables, writes temporary
    UGRID mesh files, runs ``ESMF_RegridWeightGen``, and cleans up
    intermediate files.

    Args:
        source_desc: Resolved source description.
        target_mesh: HEALPix target polygon mesh.
        target_path: Output weight file path.
        level: HEALPix refinement level.
        nest: Use nested ordering when *True*.
        method: Weight method.
        config: Offline execution configuration.
        ignore_unmapped: Ignore unmapped destination cells.

    Returns:
        Path to the generated weight file.

    Raises:
        FileNotFoundError: When the ESMF executable or MPI launcher
            is missing.
    """
    if shutil.which(config.esmf_regrid_weightgen) is None:
        raise FileNotFoundError(
            f"Could not find {config.esmf_regrid_weightgen!r} in PATH."
        )
    if config.nproc > 1 and shutil.which(config.mpirun) is None:
        raise FileNotFoundError(
            f"Could not find MPI launcher {config.mpirun!r} in PATH."
        )

    if config.workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="grid_doctor_esmf_"))
    else:
        workdir = config.workdir
        workdir.mkdir(parents=True, exist_ok=True)

    order_tag = "nest" if nest else "ring"
    src_file = write_ugrid_mesh_file(
        source_desc.source_mesh,
        workdir / "source_mesh.nc",
        mesh_name="source_mesh",
    )
    dst_file = write_ugrid_mesh_file(
        target_mesh,
        workdir / f"healpix_level_{level}_{order_tag}.nc",
        mesh_name="target_mesh",
    )

    generated = run_esmf_regrid_weightgen(
        src_mesh=src_file,
        dst_mesh=dst_file,
        weights_path=target_path,
        method=method,
        config=config,
        ignore_unmapped=ignore_unmapped,
        src_regional=source_desc.ignore_unmapped,
        dst_regional=False,
    )

    if not config.keep_intermediates and config.workdir is None:
        shutil.rmtree(workdir, ignore_errors=True)

    return generated
