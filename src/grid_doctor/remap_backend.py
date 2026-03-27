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

The main performance feature is optional offline weight generation with
`ESMF_RegridWeightGen` under MPI. This keeps the Python workflow intact while
moving the expensive conservative overlap calculation into ESMF's compiled,
parallel offline tool.
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
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import xarray as xr

logger = logging.getLogger(__name__)

RemapMethod = Literal["nearest", "conservative"]
SourceUnits = Literal["auto", "deg", "rad"]
SourceKind = Literal["auto", "regular", "curvilinear", "unstructured", "spectral"]
FloatArray = npt.NDArray[np.float64]

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


@dataclass(frozen=True)
class OfflineWeightConfig:
    """Configuration for offline ESMF weight generation.

    Parameters
    ----------
    enabled:
        Enable offline weight generation with `ESMF_RegridWeightGen`.
    nproc:
        Number of MPI ranks used for the offline command.
    esmf_regrid_weightgen:
        Name or path of the offline ESMF executable.
    weight_only:
        Request a simplified weight file that contains only `row`, `col`, and
        `S` when the ESMF build supports it.
    netcdf4:
        Write NetCDF4 output.
    keep_intermediates:
        Keep temporary source and destination mesh files for inspection.
    workdir:
        Optional directory for the intermediate mesh files.
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
        """Use srun or mpirun."""
        job_id = os.environ.get("SLURM_JOB_ID")
        return "srun" if job_id else "mpirun"


@dataclass(frozen=True)
class SourceDescription:
    """Resolved description of the source representation.

    Parameters
    ----------
    dataset:
        Materialised gridpoint dataset used for weight generation.
    kind:
        Source representation class.
    source_dims:
        Source dimensions in flattening order.
    source_polygons:
        Source cell polygons as `(lon_vertices, lat_vertices)` pairs in degrees.
    ignore_unmapped:
        Recommended default for ESMF unmapped handling.
    metadata:
        Metadata that should be persisted into the output weight file.
    """

    dataset: xr.Dataset
    kind: SourceKind
    source_dims: tuple[str, ...]
    source_polygons: list[tuple[FloatArray, FloatArray]]
    ignore_unmapped: bool
    metadata: dict[str, str | int | float | bool]


class SpectralTransformError(RuntimeError):
    """Raised when a spectral-to-grid transform command fails."""


def _require_esmpy() -> Any:
    try:
        import esmpy
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "ESMPy is required for in-memory weight generation. Install ESMPy "
            "or enable the offline ESMF path."
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
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "healpix-geo is required to construct HEALPix polygons."
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
    return bool(float(np.nanmax(np.abs(finite))) <= (2.0 * np.pi + 1.0e-6))


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
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


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


def _infer_bounds_1d(values: FloatArray) -> FloatArray:
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
    return lat_corner, _canonical_lon(lon_corner)


def _median_positive_step(values: FloatArray) -> float:
    if values.size < 2:
        return 0.0
    diffs = np.diff(np.sort(values))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _lon_coverage_from_centres(lon_deg: FloatArray) -> float:
    lon = np.mod(lon_deg[np.isfinite(lon_deg)], 360.0)
    if lon.size == 0:
        return 0.0
    lon = np.unique(np.sort(lon))
    if lon.size == 1:
        return 0.0
    dlon = _median_positive_step(lon)
    wrapped = np.concatenate((lon, lon[:1] + 360.0))
    gaps = np.diff(wrapped)
    largest_gap = float(np.max(gaps))
    coverage = 360.0 - largest_gap + dlon
    return float(min(360.0, max(0.0, coverage)))


def _lat_coverage_from_centres(lat_deg: FloatArray) -> float:
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


def _classify_source_kind(ds: xr.Dataset, *, explicit_kind: SourceKind) -> SourceKind:
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


def _source_polygons(
    ds: xr.Dataset, *, source_units: SourceUnits
) -> tuple[list[tuple[FloatArray, FloatArray]], tuple[str, ...]]:
    polygons: list[tuple[FloatArray, FloatArray]] = []
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


def _target_healpix_polygons(
    level: int, *, nest: bool
) -> tuple[np.ndarray, list[tuple[FloatArray, FloatArray]]]:
    module, kwargs = _require_healpix_geo_module(nest)
    ipix = np.arange(12 * (4**level), dtype=np.int64)
    lon_vertices, lat_vertices = module.vertices(ipix, level, **kwargs)
    polygons: list[tuple[FloatArray, FloatArray]] = []
    for lon_deg, lat_deg in zip(lon_vertices, lat_vertices):
        lon = _canonical_lon(np.asarray(lon_deg, dtype=np.float64))
        lat = np.asarray(lat_deg, dtype=np.float64)
        polygons.append(_ensure_ccw(lon, lat))
    return ipix, polygons


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
        for lon, lat in zip(lon_deg, lat_deg):
            key = (round(float(lon), 12), round(float(lat), 12))
            if key not in node_lookup:
                node_lookup[key] = len(node_lon)
                node_lon.append(key[0])
                node_lat.append(key[1])
            conn.append(node_lookup[key])

        nvert = len(conn)
        if nvert == 3:
            elem_type = int(esmpy_mod.MeshElemType.TRI)
        elif nvert == 4:
            elem_type = int(esmpy_mod.MeshElemType.QUAD)
        else:
            elem_type = nvert

        lon_c, lat_c = _spherical_centroid(lon_deg, lat_deg)
        element_ids.append(elem_index)
        element_types.append(elem_type)
        element_conn.extend(conn)
        element_centers.extend([lon_c, lat_c])

    mesh = esmpy_mod.Mesh(
        parametric_dim=2, spatial_dim=2, coord_sys=esmpy_mod.CoordSys.SPH_DEG
    )
    node_ids = np.arange(1, len(node_lon) + 1, dtype=np.int32)
    node_coords = np.empty(2 * len(node_lon), dtype=np.float64)
    node_coords[0::2] = np.asarray(node_lon, dtype=np.float64)
    node_coords[1::2] = np.asarray(node_lat, dtype=np.float64)
    node_owners = np.zeros(len(node_lon), dtype=np.int32)
    mesh.add_nodes(
        node_count=len(node_lon),
        node_ids=node_ids,
        node_coords=node_coords,
        node_owners=node_owners,
    )
    mesh.add_elements(
        element_count=len(polygons),
        element_ids=np.asarray(element_ids, dtype=np.int32),
        element_types=np.asarray(element_types, dtype=np.int32),
        element_conn=np.asarray(element_conn, dtype=np.int32),
        element_coords=np.asarray(element_centers, dtype=np.float64),
    )
    return mesh


def _ugrid_arrays_from_polygons(
    polygons: list[tuple[FloatArray, FloatArray]],
    *,
    node_round_ndigits: int = 12,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.int32], FloatArray, FloatArray]:
    node_lookup: dict[tuple[float, float], int] = {}
    node_lon: list[float] = []
    node_lat: list[float] = []
    max_face_nodes = max(len(lon) for lon, _ in polygons)
    face_nodes = np.full((len(polygons), max_face_nodes), -1, dtype=np.int32)
    face_lon = np.empty(len(polygons), dtype=np.float64)
    face_lat = np.empty(len(polygons), dtype=np.float64)

    for face_index, (lon_deg, lat_deg) in enumerate(polygons):
        for offset, (lon, lat) in enumerate(zip(lon_deg, lat_deg)):
            key = (
                round(float(lon), node_round_ndigits),
                round(float(lat), node_round_ndigits),
            )
            if key not in node_lookup:
                node_lookup[key] = len(node_lon)
                node_lon.append(key[0])
                node_lat.append(key[1])
            face_nodes[face_index, offset] = node_lookup[key]
        face_lon[face_index], face_lat[face_index] = _spherical_centroid(
            lon_deg, lat_deg
        )

    return (
        np.asarray(node_lon, dtype=np.float64),
        np.asarray(node_lat, dtype=np.float64),
        face_nodes,
        face_lon,
        face_lat,
    )


def write_ugrid_mesh_file(
    polygons: list[tuple[FloatArray, FloatArray]],
    path: str | Path,
    *,
    mesh_name: str,
) -> Path:
    """Write a polygon mesh as a simple UGRID NetCDF file.

    Parameters
    ----------
    polygons:
        Cell polygons as `(lon_vertices, lat_vertices)` pairs in degrees.
    path:
        Output path.
    mesh_name:
        Name of the mesh topology variable.

    Returns
    -------
    pathlib.Path
        Path to the written file.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    node_lon, node_lat, face_nodes, face_lon, face_lat = _ugrid_arrays_from_polygons(
        polygons
    )
    mesh = xr.DataArray(
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
            mesh_name: mesh,
            "node_lon": (
                "node",
                node_lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "node_lat": (
                "node",
                node_lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
            f"{mesh_name}_face_nodes": (
                ("face", "max_face_nodes"),
                face_nodes,
                {"start_index": 0, "_FillValue": np.int32(-1)},
            ),
            "face_lon": (
                "face",
                face_lon,
                {"standard_name": "longitude", "units": "degrees_east"},
            ),
            "face_lat": (
                "face",
                face_lat,
                {"standard_name": "latitude", "units": "degrees_north"},
            ),
        },
        attrs={"Conventions": "UGRID-1.0"},
    )
    ds.to_netcdf(output)
    return output


def _run_subprocess(cmd: list[str]) -> None:
    logger.info("Running command: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = (
            f"Command failed with exit code {completed.returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
        raise RuntimeError(message)


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
    """Run `ESMF_RegridWeightGen` under MPI.

    Parameters
    ----------
    src_mesh:
        Source UGRID mesh file.
    dst_mesh:
        Destination UGRID mesh file.
    weights_path:
        Output NetCDF weight file.
    method:
        Weight method. Supported values are `"nearest"` and `"conservative"`.
    config:
        Offline execution configuration.
    ignore_unmapped:
        Pass `--ignore_unmapped` when `True`.
    src_regional:
        Pass `--src_regional` when `True`.
    dst_regional:
        Pass `--dst_regional` when `True`.

    Returns
    -------
    pathlib.Path
        Path to the generated weight file.
    """
    method_arg = "neareststod" if method == "nearest" else "conserve"
    cmd: list[str] = []
    if config.nproc > 1:
        cmd.extend([config.mpirun, "-np", str(config.nproc)])
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


def _default_offline_enabled(
    *,
    method: RemapMethod,
    source_cell_count: int,
    target_cell_count: int,
    config: OfflineWeightConfig,
) -> bool:
    if config.enabled:
        return True
    if method != "conservative":
        return False
    return bool(
        config.nproc > 1
        and (source_cell_count >= 200_000 or target_cell_count >= 1_000_000)
    )


def _materialise_spectral_source(
    source: xr.Dataset | str | Path,
    *,
    transform_command: list[str] | tuple[str, ...] | None,
    workdir: Path,
) -> xr.Dataset:
    if transform_command is None:
        raise ValueError(
            "source_kind='spectral' requires an external transform command, for "
            "example a CDO sp2gp invocation."
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
    except RuntimeError as exc:  # pragma: no cover - external tool
        raise SpectralTransformError(str(exc)) from exc
    return xr.load_dataset(spectral_out)


def describe_source(
    source: xr.Dataset | str | Path,
    *,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    source_units: SourceUnits = "auto",
    spectral_transform_command: list[str] | tuple[str, ...] | None = None,
    workdir: Path | None = None,
) -> SourceDescription:
    """Resolve the source representation for HEALPix weight generation.

    Parameters
    ----------
    source:
        Source dataset or source path. A path is mainly useful for spectral
        inputs that need an external transform.
    grid:
        Optional grid dataset that supplies geometry when the data file itself
        only contains values on dimensions such as `cell`.
    source_kind:
        Explicit source representation. Use `"auto"` to infer it from the
        available coordinates.
    source_units:
        Angular unit convention of the source coordinates.
    spectral_transform_command:
        External command used when `source_kind="spectral"`. The command may
        use `{input}` and `{output}` placeholders.
    workdir:
        Optional working directory for temporary files.

    Returns
    -------
    SourceDescription
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
    source_polygons, source_dims = _source_polygons(
        geometry_ds, source_units=source_units
    )
    ignore_unmapped = not _looks_global(geometry_ds, source_units=source_units)
    metadata: dict[str, str | int | float | bool] = {
        "grid_doctor_source_kind": resolved_kind,
        "grid_doctor_source_dims": json.dumps(list(source_dims)),
        "grid_doctor_source_size": len(source_polygons),
        "grid_doctor_source_units": source_units,
    }
    return SourceDescription(
        dataset=dataset,
        kind=resolved_kind,
        source_dims=source_dims,
        source_polygons=source_polygons,
        ignore_unmapped=ignore_unmapped,
        metadata=metadata,
    )


def _compute_weights_in_memory(
    *,
    source_polygons: list[tuple[FloatArray, FloatArray]],
    target_polygons: list[tuple[FloatArray, FloatArray]],
    weights_path: Path,
    method: RemapMethod,
    ignore_unmapped: bool,
    large_file: bool,
) -> Path:
    esmpy_mod = _require_esmpy()
    source_mesh = _mesh_from_polygons(source_polygons, esmpy_mod=esmpy_mod)
    target_mesh = _mesh_from_polygons(target_polygons, esmpy_mod=esmpy_mod)

    src_field = esmpy_mod.Field(
        source_mesh, name="src", meshloc=esmpy_mod.MeshLoc.ELEMENT
    )
    dst_field = esmpy_mod.Field(
        target_mesh, name="dst", meshloc=esmpy_mod.MeshLoc.ELEMENT
    )
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
    spectral_transform_command: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Compute reusable HEALPix weights with optional offline MPI acceleration.

    Parameters
    ----------
    source:
        Source dataset or source path.
    level:
        HEALPix refinement level.
    method:
        Weight method. Supported values are `"nearest"` and `"conservative"`.
    nest:
        Use nested HEALPix ordering when `True`.
    source_units:
        Angular unit convention of the source coordinates.
    weights_path:
        Output weight file path. A temporary file is created when omitted.
    grid:
        Optional external geometry dataset.
    source_kind:
        Explicit source representation or `"auto"`.
    ignore_unmapped:
        Whether destination cells without source coverage should be ignored.
    large_file:
        Forwarded to ESMPy for the in-memory path.
    offline:
        Offline weight-generation configuration.
    spectral_transform_command:
        External command used when `source_kind="spectral"`.

    Returns
    -------
    pathlib.Path
        Path to the generated weight file.
    """
    if method not in {"nearest", "conservative"}:
        raise ValueError("Only 'nearest' and 'conservative' are supported here.")

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
    _, target_polygons = _target_healpix_polygons(level, nest=nest)

    use_offline = _default_offline_enabled(
        method=method,
        source_cell_count=len(source_desc.source_polygons),
        target_cell_count=len(target_polygons),
        config=offline_cfg,
    )
    effective_ignore_unmapped = (
        source_desc.ignore_unmapped if ignore_unmapped is None else ignore_unmapped
    )

    if use_offline:
        if shutil.which(offline_cfg.esmf_regrid_weightgen) is None:
            raise FileNotFoundError(
                f"Could not find {offline_cfg.esmf_regrid_weightgen!r} in PATH."
            )
        if offline_cfg.nproc > 1 and shutil.which(offline_cfg.mpirun) is None:
            raise FileNotFoundError(
                f"Could not find MPI launcher {offline_cfg.mpirun!r} in PATH."
            )

        if offline_cfg.workdir is None:
            workdir = Path(tempfile.mkdtemp(prefix="grid_doctor_esmf_"))
        else:
            workdir = offline_cfg.workdir
            workdir.mkdir(parents=True, exist_ok=True)
        src_mesh = write_ugrid_mesh_file(
            source_desc.source_polygons,
            workdir / "source_mesh.nc",
            mesh_name="source_mesh",
        )
        dst_mesh = write_ugrid_mesh_file(
            target_polygons,
            workdir / f"healpix_level_{level}_{'nest' if nest else 'ring'}.nc",
            mesh_name="target_mesh",
        )
        generated = run_esmf_regrid_weightgen(
            src_mesh=src_mesh,
            dst_mesh=dst_mesh,
            weights_path=target,
            method=method,
            config=offline_cfg,
            ignore_unmapped=effective_ignore_unmapped,
            src_regional=source_desc.ignore_unmapped,
            dst_regional=False,
        )
        if not offline_cfg.keep_intermediates and offline_cfg.workdir is None:
            shutil.rmtree(workdir, ignore_errors=True)
    else:
        generated = _compute_weights_in_memory(
            source_polygons=source_desc.source_polygons,
            target_polygons=target_polygons,
            weights_path=target,
            method=method,
            ignore_unmapped=effective_ignore_unmapped,
            large_file=large_file,
        )

    meta: dict[str, str | int | float] = {
        "grid_doctor_method": method,
        "grid_doctor_level": level,
        "grid_doctor_order": "nested" if nest else "ring",
        "grid_doctor_backend": "offline-esmf" if use_offline else "esmpy",
        "grid_doctor_ignore_unmapped": int(bool(effective_ignore_unmapped)),
        **source_desc.metadata,
    }
    with xr.open_dataset(generated) as raw_weights:
        annotated = raw_weights.load()
    annotated.attrs.update(meta)
    annotated.to_netcdf(generated, mode="w")
    return generated
