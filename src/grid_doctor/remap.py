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

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.interpolate import LinearNDInterpolator
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

RemapMethod = Literal["nearest", "linear", "conservative"]
MissingPolicy = Literal["renormalize", "propagate"]
SourceUnits = Literal["auto", "deg", "rad"]

_UNSTRUCTURED_DIMS: frozenset[str] = frozenset({"cell", "ncells", "ncell", "nCells"})
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

FloatArray = npt.NDArray[np.float64]


def _require_esmpy() -> Any:
    try:
        import esmpy
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "ESMPy is required to generate reusable HEALPix weight files for "
            "'nearest' and 'conservative' remapping. Install esmpy/ESMF or use "
            "method='linear' for on-the-fly interpolation."
        ) from exc
    return esmpy


def _require_healpix_geo_module(nest: bool) -> tuple[Any, dict[str, str]]:
    kwargs = {}
    try:
        if nest:
            kwargs["ellipsoid"] = "WGS84"
            from healpix_geo import nested as module

        else:
            from healpix_geo import ring as module
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "healpix-geo is required to construct HEALPix mesh vertices. "
            "Install healpix-geo to use weight-file based remapping."
        ) from exc
    return module, kwargs


def _to_float64(values: Any) -> FloatArray:
    return np.asarray(values, dtype=np.float64)


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


def _ensure_ccw(
    lon_deg: FloatArray, lat_deg: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Return polygon vertices in counter-clockwise order.

    The orientation test is carried out in a local tangent plane at the polygon
    centroid. This avoids artefacts at the antimeridian and near the poles.
    """
    pts = _lonlat_to_xyz(lon_deg, lat_deg)
    lon0, lat0 = _spherical_centroid(lon_deg, lat_deg)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)

    east = np.array([-np.sin(lon0_rad), np.cos(lon0_rad), 0.0], dtype=np.float64)
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


def _get_latlon_arrays(ds: xr.Dataset) -> tuple[FloatArray, FloatArray]:
    """Return latitude and longitude arrays from `ds`.

    Parameters
    ----------
    ds:
        Dataset containing lat/lon coordinates or data variables.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Latitude and longitude arrays. Structured grids return one- or
        two-dimensional arrays. Unstructured grids typically return one
        dimension per source cell.

    Raises
    ------
    ValueError
        Raised when no recognised latitude/longitude variables are found.
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


def _source_polygons(
    ds: xr.Dataset,
    *,
    source_units: SourceUnits,
) -> tuple[list[tuple[FloatArray, FloatArray]], tuple[str, ...]]:
    """Return source cell polygons and their spatial dimensions.

    The polygons are returned as `(lon_vertices, lat_vertices)` pairs in degrees.
    """
    if _is_unstructured(ds):
        lon_name = "clon_vertices" if "clon_vertices" in ds else "lon_vertices"
        lat_name = "clat_vertices" if "clat_vertices" in ds else "lat_vertices"
        if lon_name not in ds or lat_name not in ds:
            raise ValueError(
                "Unstructured grids require per-cell vertex coordinates such as "
                "'clon_vertices'/'clat_vertices'."
            )
        lon_vertices = _canonical_lon(
            _normalise_angle_units(_to_float64(ds[lon_name].values), source_units)
        )
        lat_vertices = _normalise_angle_units(
            _to_float64(ds[lat_name].values), source_units
        )
        polygons: list[tuple[FloatArray, FloatArray]] = []
        for index in range(lon_vertices.shape[0]):
            mask = np.isfinite(lon_vertices[index]) & np.isfinite(lat_vertices[index])
            cell_lon = lon_vertices[index, mask]
            cell_lat = lat_vertices[index, mask]
            if cell_lon.size < 3:
                raise ValueError(
                    f"Source cell {index} has fewer than three valid vertices."
                )
            polygons.append(_ensure_ccw(cell_lon, cell_lat))
        return polygons, (_get_unstructured_dim(ds),)

    lat, lon = _get_latlon_arrays(ds)
    lat = _normalise_angle_units(lat, source_units)
    lon = _canonical_lon(_normalise_angle_units(lon, source_units))
    y_dim, x_dim = _get_spatial_dims(ds)

    polygons = []
    if lat.ndim == 1:
        lat_bounds = _infer_bounds_1d(lat)
        lon_bounds = _infer_bounds_1d(lon)
        for j in range(lat.size):
            for i in range(lon.size):
                cell_lon = np.array(
                    [
                        lon_bounds[i],
                        lon_bounds[i + 1],
                        lon_bounds[i + 1],
                        lon_bounds[i],
                    ],
                    dtype=np.float64,
                )
                cell_lat = np.array(
                    [
                        lat_bounds[j],
                        lat_bounds[j],
                        lat_bounds[j + 1],
                        lat_bounds[j + 1],
                    ],
                    dtype=np.float64,
                )
                polygons.append(_ensure_ccw(_canonical_lon(cell_lon), cell_lat))
    elif lat.ndim == 2:
        lat_corner, lon_corner = _infer_curvilinear_corners(lat, lon)
        for j in range(lat.shape[0]):
            for i in range(lat.shape[1]):
                cell_lon = np.array(
                    [
                        lon_corner[j, i],
                        lon_corner[j, i + 1],
                        lon_corner[j + 1, i + 1],
                        lon_corner[j + 1, i],
                    ],
                    dtype=np.float64,
                )
                cell_lat = np.array(
                    [
                        lat_corner[j, i],
                        lat_corner[j, i + 1],
                        lat_corner[j + 1, i + 1],
                        lat_corner[j + 1, i],
                    ],
                    dtype=np.float64,
                )
                polygons.append(_ensure_ccw(_canonical_lon(cell_lon), cell_lat))
    else:
        raise ValueError(
            "Latitude/longitude coordinates must be one- or two-dimensional."
        )

    return polygons, (y_dim, x_dim)


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


def _looks_global(
    ds: xr.Dataset, *, source_units: SourceUnits = "auto", epsilon_deg: float = 10.0
) -> bool:
    lat, lon = _get_latlon_arrays(ds)
    lat_deg = _normalise_angle_units(lat, source_units)
    lon_deg = _normalise_angle_units(lon, source_units)
    if not np.isfinite(lat_deg).any() or not np.isfinite(lon_deg).any():
        return False
    lon_coverage = _lon_coverage_from_centres(lon_deg)
    lat_coverage = _lat_coverage_from_centres(lat_deg)
    lon_global = lon_coverage >= (360.0 - epsilon_deg)
    lat_global = lat_coverage >= (180.0 - epsilon_deg)
    return bool(lon_global and lat_global)


def _mesh_from_polygons(
    polygons: list[tuple[FloatArray, FloatArray]], *, esmpy_mod: Any
) -> Any:
    node_lookup: dict[tuple[float, float], int] = {}
    node_lon: list[float] = []
    node_lat: list[float] = []
    element_ids: list[int] = []
    element_types: list[int] = []
    element_conn: list[int] = []
    element_centers: list[float] = []

    for elem_index, (lon_deg, lat_deg) in enumerate(polygons, start=1):
        conn: list[int] = []
        for lon, lat in zip(lon_deg, lat_deg, strict=True):
            key = (round(float(lon), 12), round(float(lat), 12))
            local_index = node_lookup.get(key)
            if local_index is None:
                local_index = len(node_lon)
                node_lookup[key] = local_index
                node_lon.append(key[0])
                node_lat.append(key[1])
            conn.append(local_index)

        n_vertices = len(conn)
        if n_vertices == 3:
            element_type = int(esmpy_mod.MeshElemType.TRI)
        elif n_vertices == 4:
            element_type = int(esmpy_mod.MeshElemType.QUAD)
        else:
            element_type = n_vertices

        lon_c, lat_c = _spherical_centroid(lon_deg, lat_deg)
        element_ids.append(elem_index)
        element_types.append(element_type)
        element_conn.extend(conn)
        element_centers.extend((lon_c, lat_c))

    mesh = esmpy_mod.Mesh(
        parametric_dim=2,
        spatial_dim=2,
        coord_sys=esmpy_mod.CoordSys.SPH_DEG,
    )

    node_count = len(node_lon)
    node_ids = np.arange(1, node_count + 1, dtype=np.int32)
    node_coords = np.empty(2 * node_count, dtype=np.float64)
    node_coords[0::2] = np.asarray(node_lon, dtype=np.float64)
    node_coords[1::2] = np.asarray(node_lat, dtype=np.float64)
    node_owner = np.zeros(node_count, dtype=np.int32)
    mesh.add_nodes(
        node_count=node_count,
        node_ids=node_ids,
        node_coords=node_coords,
        node_owners=node_owner,
    )

    mesh.add_elements(
        element_count=len(polygons),
        element_ids=np.asarray(element_ids, dtype=np.int32),
        element_types=np.asarray(element_types, dtype=np.int32),
        element_conn=np.asarray(element_conn, dtype=np.int32),
        element_coords=np.asarray(element_centers, dtype=np.float64),
    )
    return mesh


def _target_healpix_polygons(
    level: int, *, nest: bool
) -> tuple[np.ndarray, list[tuple[FloatArray, FloatArray]]]:
    module, kwargs = _require_healpix_geo_module(nest)
    n_target = 12 * (4**level)
    ipix = np.arange(n_target, dtype=np.int64)
    lon_vertices, lat_vertices = module.vertices(ipix, level, **kwargs)

    polygons: list[tuple[FloatArray, FloatArray]] = []
    for lon, lat in zip(lon_vertices, lat_vertices, strict=True):
        cell_lon = _canonical_lon(np.asarray(lon, dtype=np.float64))
        cell_lat = np.asarray(lat, dtype=np.float64)
        polygons.append(_ensure_ccw(cell_lon, cell_lat))
    return ipix, polygons


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
    ds: xr.Dataset,
    level: int,
    *,
    method: Literal["nearest", "conservative"] = "nearest",
    nest: bool = True,
    source_units: SourceUnits = "auto",
    weights_path: str | Path | None = None,
    ignore_unmapped: bool | None = None,
    large_file: bool = True,
) -> Path:
    """Generate a reusable NetCDF weight file for HEALPix remapping.

    This function builds an ESMF mesh for the source dataset and a HEALPix mesh
    for the destination grid. The expensive geometric work is delegated to ESMF.
    The resulting NetCDF file can be applied later with
    [`apply_weight_file`][grid_doctor.remap.apply_weight_file].

    Parameters
    ----------
    ds:
        Source dataset. Regular lon/lat grids, curvilinear grids, and
        unstructured polygon meshes are supported.
    level:
        HEALPix refinement level. The destination `nside` equals `2**level`.
    method:
        Weight-generation method. Supported values are `"nearest"` and
        `"conservative"`.
    nest:
        Use nested HEALPix ordering when `True`. Ring ordering is also supported
        when `healpix-geo` provides the ring helpers.
    source_units:
        Unit convention of the source coordinates. `"auto"` converts from
        radians when the values look radian-like.
    weights_path:
        Target file. A temporary file is created when omitted.
    ignore_unmapped:
        Whether destination cells without source coverage should be ignored.
        The default is chosen automatically: global sources use `False`,
        limited-area sources use `True`.
    large_file:
        Forwarded to ESMPy. Enable this for very large weight files.

    Returns
    -------
    pathlib.Path
        Path to the generated NetCDF weight file.

    Examples
    --------
    Create a conservative weight file for an ICON grid:

    ```python
    from pathlib import Path
    from grid_doctor import compute_healpix_weights

    weight_file = compute_healpix_weights(
        ds_icon,
        level=8,
        method="conservative",
        source_units="rad",
        weights_path=Path("icon_to_healpix_d8.nc"),
    )
    ```

    Create a nearest-neighbour file for a regular lon/lat dataset:

    ```python
    weight_file = compute_healpix_weights(ds_era5, level=7, method="nearest")
    ```

    See Also
    --------
    [`apply_weight_file`][grid_doctor.remap.apply_weight_file]:
        Apply a previously generated weight file.
    [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]:
        High-level convenience wrapper that generates and applies weights.
    """
    if method not in {"nearest", "conservative"}:
        raise ValueError(
            "Only 'nearest' and 'conservative' support reusable weight files."
        )

    esmpy_mod = _require_esmpy()
    source_polygons, _ = _source_polygons(ds, source_units=source_units)
    source_mesh = _mesh_from_polygons(source_polygons, esmpy_mod=esmpy_mod)
    _, target_polygons = _target_healpix_polygons(level, nest=nest)
    target_mesh = _mesh_from_polygons(target_polygons, esmpy_mod=esmpy_mod)

    src_field = esmpy_mod.Field(
        source_mesh, name="src", meshloc=esmpy_mod.MeshLoc.ELEMENT
    )
    dst_field = esmpy_mod.Field(
        target_mesh, name="dst", meshloc=esmpy_mod.MeshLoc.ELEMENT
    )
    src_field.data[...] = 0.0
    dst_field.data[...] = 0.0

    if ignore_unmapped is None:
        ignore_unmapped = not _looks_global(ds, source_units=source_units)

    target = (
        Path(weights_path)
        if weights_path is not None
        else Path(tempfile.mkstemp(suffix=".nc")[1])
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    regrid_method = (
        esmpy_mod.RegridMethod.NEAREST_STOD
        if method == "nearest"
        else esmpy_mod.RegridMethod.CONSERVE
    )
    kwargs: dict[str, Any] = {
        "filename": str(target),
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

    meta = {
        "grid_doctor_method": method,
        "grid_doctor_level": level,
        "grid_doctor_order": "nested" if nest else "ring",
        "grid_doctor_source_units": source_units,
    }
    with xr.open_dataset(target) as raw_weights:
        annotated = raw_weights.load()
    annotated.attrs.update(meta)
    annotated.to_netcdf(target, mode="w")
    return target


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


def apply_weight_file(
    ds: xr.Dataset,
    weights_path: str | Path,
    *,
    missing_policy: MissingPolicy = "renormalize",
) -> xr.Dataset:
    """Apply a previously generated ESMF weight file to `ds`.

    The weight file is read into a SciPy sparse matrix and applied with
    `xarray.apply_ufunc`. Missing values can either be
    propagated or ignored with per-target renormalisation.

    Parameters
    ----------
    ds:
        Source dataset used with the same grid geometry that produced the weight
        file.
    weights_path:
        Path to the NetCDF weight file generated by
        [`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights].
    missing_policy:
        Missing-value handling strategy.

        - `"renormalize"`: ignore missing source values and renormalize by the
          sum of valid source weights.
        - `"propagate"`: any missing contributing source value makes the target
          value missing.

    Returns
    -------
    xarray.Dataset
        Dataset on the HEALPix target grid with a `cell` dimension.

    Examples
    --------
    ```python
    ds_hp = apply_weight_file(ds_icon, "icon_to_healpix_d8.nc")
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

    src_lat, _, source_dims = _source_centre_arrays(ds, source_units="auto")
    if src_lat.size != n_source:
        raise ValueError(
            "The weight file does not match the provided dataset geometry. "
            f"Expected {n_source} source cells, found {src_lat.size}."
        )

    regridded_vars: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if not set(source_dims).issubset(map(str, data.dims)):
            regridded_vars[str(name)] = data
            continue
        regridded_vars[str(name)] = cast(
            xr.DataArray,
            xr.apply_ufunc(
                _apply_sparse_array,
                data,
                input_core_dims=[list(source_dims)],
                output_core_dims=[["cell"]],
                exclude_dims=set(source_dims),
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
