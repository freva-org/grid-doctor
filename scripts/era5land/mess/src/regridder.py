"""
Reduced Gaussian GRIB -> Gaussian NetCDF remapping workflow.

This is a standalone extraction/adaptation of the Grid Doctor remapping idea:
- represent source and target grids as compact polygon meshes
- create ESMF Mesh objects directly
- generate ESMF sparse weights
- apply the sparse weights with scipy

It avoids xESMF's regular-grid frontend, which tries to meshgrid lon(cell), lat(cell).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix, csr_matrix

try:
    import esmpy as ESMF
except ImportError:  # some environments expose it as ESMF
    import ESMF  # type: ignore


Method = Literal["nearest", "conservative", "conservative_2nd"]
MissingPolicy = Literal["renormalize", "propagate"]
EARTH_RADIUS_M = 6_371_000.0
MESH_CACHE_VERSION = 1


def open_grib_dataset(
    path: str | Path,
    *,
    normalize_time: bool = True,
    use_grib_cache: bool = False,
) -> xr.Dataset:
    """Open one GRIB file, optionally using era5land's valid-time decoder."""
    if normalize_time:
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from era5land.grib import open_dataset as open_era5land_grib_dataset

        return open_era5land_grib_dataset([path], use_cache=use_grib_cache)

    return xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})


@dataclass(frozen=True, slots=True)
class PolygonMesh:
    """Compact polygon mesh in lon/lat degrees."""

    node_lon: np.ndarray          # shape: (n_node,)
    node_lat: np.ndarray          # shape: (n_node,)
    face_nodes: np.ndarray        # shape: (n_face, max_face_nodes), -1 padded
    face_lon: np.ndarray          # shape: (n_face,)
    face_lat: np.ndarray          # shape: (n_face,)

    @property
    def face_count(self) -> int:
        return int(self.face_nodes.shape[0])


def source_mesh_cache_path(sample_source_file: str | Path, mesh_cache_dir: str | Path) -> Path:
    """Stable cache path for a source mesh derived from file metadata."""
    path = Path(sample_source_file)
    stat = path.stat()
    key = "|".join(
        (
            str(MESH_CACHE_VERSION),
            str(path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
        )
    )
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return Path(mesh_cache_dir) / f"{path.name}.{digest}.source_mesh.npz"


def save_polygon_mesh(path: str | Path, mesh: PolygonMesh) -> None:
    """Persist a PolygonMesh without pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mesh_cache_version=np.asarray(MESH_CACHE_VERSION, dtype=np.int16),
        node_lon=mesh.node_lon,
        node_lat=mesh.node_lat,
        face_nodes=mesh.face_nodes,
        face_lon=mesh.face_lon,
        face_lat=mesh.face_lat,
    )


def load_polygon_mesh(path: str | Path) -> PolygonMesh:
    """Load a PolygonMesh saved by save_polygon_mesh."""
    with np.load(path, allow_pickle=False) as data:
        version = int(data["mesh_cache_version"])
        if version != MESH_CACHE_VERSION:
            raise ValueError(f"Unsupported mesh cache version {version}")
        return PolygonMesh(
            node_lon=data["node_lon"],
            node_lat=data["node_lat"],
            face_nodes=data["face_nodes"],
            face_lon=data["face_lon"],
            face_lat=data["face_lat"],
        )


# -----------------------------------------------------------------------------
# Geometry helpers, adapted from Grid Doctor remap_backend.py
# -----------------------------------------------------------------------------

def canonical_lon(lon_deg: np.ndarray) -> np.ndarray:
    """Map longitudes into [-180, 180)."""
    return ((np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0


def lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
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
    return np.stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)), axis=-1)


def xyz_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian unit-sphere vectors back to lon/lat in degrees.

    Args:
        xyz: an array of shape ``(..., 3)``

    Returns:
        ``(lon_deg, lat_deg)``: and array.

    Raises:
        ValueError: When any vector has zero length.
    """    
    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    if np.any(norm == 0.0):
        raise ValueError("Cannot convert zero-length vector to lon/lat")
    unit = xyz / norm
    lon = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0]))
    lat = np.rad2deg(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
    return canonical_lon(lon), lat


def polygon_centres(cell_lon: np.ndarray, cell_lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spherical polygon centres from padded corner arrays."""
    mask = np.isfinite(cell_lon) & np.isfinite(cell_lat)
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Every polygon must contain at least one valid corner")
    xyz = lonlat_to_xyz(np.where(mask, cell_lon, 0.0), np.where(mask, cell_lat, 0.0))
    xyz_sum = np.where(mask[..., None], xyz, 0.0).sum(axis=1)
    return xyz_to_lonlat(xyz_sum)


def corner_mesh_from_arrays(
    cell_lon: np.ndarray,
    cell_lat: np.ndarray,
    *,
    node_round_ndigits: int = 12,
) -> PolygonMesh:
    """Build a deduplicated polygon mesh from per-cell corner arrays.

    Parameters
    ----------
    cell_lon, cell_lat
        Arrays with shape (n_face, max_face_nodes). For quads this is (n, 4).
        Invalid/padded vertices may be NaN.
    """
    cell_lon = np.asarray(cell_lon, dtype=np.float64)
    cell_lat = np.asarray(cell_lat, dtype=np.float64)
    if cell_lon.ndim != 2 or cell_lat.ndim != 2 or cell_lon.shape != cell_lat.shape:
        raise ValueError("Expected corner arrays with shape (n_face, max_face_nodes)")

    mask = np.isfinite(cell_lon) & np.isfinite(cell_lat)
    valid_counts = mask.sum(axis=1)
    if np.any(valid_counts < 3):
        bad = int(np.flatnonzero(valid_counts < 3)[0])
        raise ValueError(f"Cell {bad} has fewer than three valid vertices")

    canonical = canonical_lon(cell_lon)
    rounded_lon = np.round(np.where(mask, canonical, 0.0), node_round_ndigits)
    rounded_lat = np.round(np.where(mask, cell_lat, 0.0), node_round_ndigits)

    flat_mask = mask.ravel()
    key_dtype = np.dtype([("lon", "f8"), ("lat", "f8")])
    flat_keys = np.empty(flat_mask.sum(), dtype=key_dtype)
    flat_keys["lon"] = rounded_lon.ravel()[flat_mask]
    flat_keys["lat"] = rounded_lat.ravel()[flat_mask]

    _, unique_idx, inverse = np.unique(flat_keys, return_index=True, return_inverse=True)

    original_lon = canonical.ravel()[flat_mask]
    original_lat = cell_lat.ravel()[flat_mask]

    face_nodes = np.full(cell_lon.shape, -1, dtype=np.int32)
    face_nodes.ravel()[flat_mask] = inverse.astype(np.int32, copy=False)

    face_lon, face_lat = polygon_centres(canonical, cell_lat)

    return PolygonMesh(
        node_lon=original_lon[unique_idx].astype(np.float64, copy=False),
        node_lat=original_lat[unique_idx].astype(np.float64, copy=False),
        face_nodes=face_nodes,
        face_lon=face_lon.astype(np.float64, copy=False),
        face_lat=face_lat.astype(np.float64, copy=False),
    )


def infer_bounds_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Expected a 1D coordinate with at least two points")
    diffs = np.diff(values)
    bounds = np.empty(values.size + 1, dtype=np.float64)
    bounds[1:-1] = values[:-1] + 0.5 * diffs
    bounds[0] = values[0] - 0.5 * diffs[0]
    bounds[-1] = values[-1] + 0.5 * diffs[-1]
    return bounds


def regular_grid_mesh(lat: np.ndarray, lon: np.ndarray) -> PolygonMesh:
    """Construct polygon mesh from 1D lat/lon centres."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_bounds = infer_bounds_1d(lat)
    lon_bounds = infer_bounds_1d(lon)
    ny, nx = lat.size, lon.size

    # For global periodic lon, drop duplicate seam nodes and wrap connectivity.
    lon_mod = np.unique(np.sort(np.mod(lon, 360.0)))
    gaps = np.diff(np.concatenate([lon_mod, lon_mod[:1] + 360.0]))
    dlon = np.median(gaps[gaps > 0]) if np.any(gaps > 0) else 0.0
    coverage = 360.0 - float(np.max(gaps)) + float(dlon)
    periodic = coverage >= 350.0

    if periodic:
        node_idx_full = np.arange((ny + 1) * (nx + 1), dtype=np.int32).reshape(ny + 1, nx + 1)
        node_idx_full[:, -1] = node_idx_full[:, 0]
        face_nodes = np.stack(
            (node_idx_full[:-1, :-1], node_idx_full[:-1, 1:], node_idx_full[1:, 1:], node_idx_full[1:, :-1]),
            axis=-1,
        ).reshape(-1, 4)
        keep = np.ones((ny + 1) * (nx + 1), dtype=bool)
        keep[np.arange(ny + 1) * (nx + 1) + nx] = False
        old_to_new = np.full((ny + 1) * (nx + 1), -1, dtype=np.int32)
        old_to_new[keep] = np.arange(keep.sum(), dtype=np.int32)
        face_nodes = old_to_new[face_nodes]
        node_lon_2d, node_lat_2d = np.meshgrid(lon_bounds, lat_bounds)
        node_lon = node_lon_2d.ravel()[keep]
        node_lat = node_lat_2d.ravel()[keep]
    else:
        node_idx = np.arange((ny + 1) * (nx + 1), dtype=np.int32).reshape(ny + 1, nx + 1)
        face_nodes = np.stack(
            (node_idx[:-1, :-1], node_idx[:-1, 1:], node_idx[1:, 1:], node_idx[1:, :-1]),
            axis=-1,
        ).reshape(-1, 4)
        node_lon_2d, node_lat_2d = np.meshgrid(lon_bounds, lat_bounds)
        node_lon = node_lon_2d.ravel()
        node_lat = node_lat_2d.ravel()

    face_lon_2d, face_lat_2d = np.meshgrid(canonical_lon(lon), lat)
    return PolygonMesh(
        node_lon=node_lon.astype(np.float64, copy=False),
        node_lat=node_lat.astype(np.float64, copy=False),
        face_nodes=face_nodes.astype(np.int32, copy=False),
        face_lon=face_lon_2d.ravel().astype(np.float64, copy=False),
        face_lat=face_lat_2d.ravel().astype(np.float64, copy=False),
    )


def regular_grid_coordinate_dataset(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    """Build target 1D lon/lat coordinates with CF-style bounds variables."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    lat_bounds = infer_bounds_1d(lat)
    lon_bounds = infer_bounds_1d(lon)
    lat_bnds = np.column_stack((lat_bounds[:-1], lat_bounds[1:]))
    lon_bnds = np.column_stack((lon_bounds[:-1], lon_bounds[1:]))
    area = regular_lonlat_cell_area(lat_bnds, lon_bnds)
    ds = xr.Dataset(
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "bounds": "lat_bnds"}),
            "lon": ("lon", lon, {"units": "degrees_east", "bounds": "lon_bnds"}),
        },
        data_vars={
            "lat_bnds": (("lat", "bnds"), lat_bnds),
            "lon_bnds": (("lon", "bnds"), lon_bnds),
            "areacella": (
                ("lat", "lon"),
                area,
                {
                    "long_name": "Grid-cell area for atmospheric grid variables",
                    "standard_name": "cell_area",
                    "units": "m2",
                },
            ),
        },
    )
    ds["lat_bnds"].attrs["units"] = "degrees_north"
    ds["lon_bnds"].attrs["units"] = "degrees_east"
    return ds


def regular_lonlat_cell_area(
    lat_bnds: np.ndarray,
    lon_bnds: np.ndarray,
    *,
    radius_m: float = EARTH_RADIUS_M,
) -> np.ndarray:
    """Compute spherical lon/lat cell areas from two-column bounds arrays."""
    lat_bnds = np.asarray(lat_bnds, dtype=np.float64)
    lon_bnds = np.asarray(lon_bnds, dtype=np.float64)
    if lat_bnds.ndim != 2 or lon_bnds.ndim != 2 or lat_bnds.shape[1] != 2 or lon_bnds.shape[1] != 2:
        raise ValueError("Expected lat_bnds and lon_bnds with shape (n, 2)")

    sin_lat_delta = np.abs(np.diff(np.sin(np.deg2rad(lat_bnds)), axis=1)).ravel()
    lon_delta = np.diff(lon_bnds, axis=1).ravel()
    lon_delta = np.where(lon_delta < 0.0, lon_delta + 360.0, lon_delta)
    lon_delta_rad = np.deg2rad(lon_delta)
    return (radius_m * radius_m) * sin_lat_delta[:, None] * lon_delta_rad[None, :]


# -----------------------------------------------------------------------------
# Reduced/regular Gaussian grid helpers
# -----------------------------------------------------------------------------

def reduced_gaussian_bounds_from_centres(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer quadrilateral bounds for a reduced Gaussian point list.

    Returns
    -------
    lon_b, lat_b : arrays with shape (ncell, 4), corner order SW, SE, NE, NW.
    """
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    print(lon.shape, lat.shape)
    if lon.shape != lat.shape:
        raise ValueError("lon and lat must have the same shape")

    ncell = lon.size
    lon_b = np.empty((ncell, 4), dtype=np.float64)
    lat_b = np.empty((ncell, 4), dtype=np.float64)

    row_lats = np.unique(lat)[::-1]  # north -> south
    lat_edges = np.empty(row_lats.size + 1, dtype=np.float64)
    lat_edges[0] = 90.0
    lat_edges[-1] = -90.0
    lat_edges[1:-1] = 0.5 * (row_lats[:-1] + row_lats[1:])

    for j, row_lat in enumerate(row_lats):
        idx = np.where(lat == row_lat)[0]
        idx = idx[np.argsort(lon[idx])]
        row_lon = lon[idx]
        nlon = row_lon.size
        dlon = 360.0 / nlon
        west = row_lon - 0.5 * dlon
        east = row_lon + 0.5 * dlon
        north = lat_edges[j]
        south = lat_edges[j + 1]
        lon_b[idx, :] = np.column_stack((west, east, east, west))
        lat_b[idx, :] = np.column_stack((
            np.full(nlon, south),
            np.full(nlon, south),
            np.full(nlon, north),
            np.full(nlon, north),
        ))

    return lon_b, lat_b


def reduced_gaussian_mesh_from_centres(lon: np.ndarray, lat: np.ndarray) -> PolygonMesh:
    lon_b, lat_b = reduced_gaussian_bounds_from_centres(lon, lat)
    return corner_mesh_from_arrays(lon_b, lat_b)


def gaussian_latitudes(nlat: int) -> np.ndarray:
    """Gaussian latitudes north-to-south from Legendre roots."""
    x, _ = np.polynomial.legendre.leggauss(nlat)
    return np.rad2deg(np.arcsin(x))[::-1]


def gaussian_target_mesh(nlat: int, nlon: int) -> tuple[PolygonMesh, xr.Dataset]:
    """Build a regular/full Gaussian target mesh and target coordinate dataset."""
    lat = gaussian_latitudes(nlat)
    lon = np.arange(nlon, dtype=np.float64) * 360.0 / nlon
    mesh = regular_grid_mesh(lat, lon)
    ds_target = regular_grid_coordinate_dataset(lat, lon)
    return mesh, ds_target


def target_mesh_from_file(path: str | Path) -> tuple[PolygonMesh, xr.Dataset]:
    """Build target mesh from a NetCDF template with 1D lat/lon coordinates."""
    ds = xr.open_dataset(path)
    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"
    lat = ds[lat_name].values.astype(np.float64)
    lon = ds[lon_name].values.astype(np.float64)
    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError("This helper expects a target template with 1D lat/lon")
    # ESMF can handle either order, but for writing nice output use north-to-south if already so.
    mesh = regular_grid_mesh(lat, lon)
    ds_target = regular_grid_coordinate_dataset(lat, lon)
    return mesh, ds_target


# -----------------------------------------------------------------------------
# ESMF mesh and weight generation
# -----------------------------------------------------------------------------

def esmf_mesh_from_polygon_mesh(mesh: PolygonMesh) -> Any:
    """Construct an ESMPy Mesh from a PolygonMesh."""
    valid_counts = np.sum(mesh.face_nodes >= 0, axis=1)
    element_types = np.where(
        valid_counts == 3,
        int(ESMF.MeshElemType.TRI),
        np.where(valid_counts == 4, int(ESMF.MeshElemType.QUAD), valid_counts),
    ).astype(np.int32, copy=False)
    element_conn = mesh.face_nodes[mesh.face_nodes >= 0].astype(np.int32, copy=False)

    element_coords = np.empty(2 * mesh.face_count, dtype=np.float64)
    element_coords[0::2] = mesh.face_lon
    element_coords[1::2] = mesh.face_lat

    esmf_mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_DEG)

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


def compute_weights(
    source_mesh: PolygonMesh,
    target_mesh: PolygonMesh,
    weights_path: str | Path,
    *,
    method: Method = "conservative",
    ignore_unmapped: bool = False,
    large_file: bool = True,
) -> Path:
    """Generate ESMF weights with source and target as Mesh ELEMENT fields."""
    if method not in ("nearest", "conservative", "conservative_2nd"):
        raise ValueError(
            f"Unknown remapping method {method!r}; expected 'nearest', "
            "'conservative', or 'conservative_2nd'"
        )

    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    src_esmf = esmf_mesh_from_polygon_mesh(source_mesh)
    dst_esmf = esmf_mesh_from_polygon_mesh(target_mesh)
    src_field = ESMF.Field(src_esmf, name="src", meshloc=ESMF.MeshLoc.ELEMENT)
    dst_field = ESMF.Field(dst_esmf, name="dst", meshloc=ESMF.MeshLoc.ELEMENT)
    src_field.data[...] = 0.0
    dst_field.data[...] = 0.0

    if method == "nearest":
        regrid_method = ESMF.RegridMethod.NEAREST_STOD
    elif method == "conservative_2nd":
        # This enum exists in recent ESMF/ESMPy builds. If not, this raises clearly.
        regrid_method = ESMF.RegridMethod.CONSERVE_2ND
    elif method == "conservative":
        regrid_method = ESMF.RegridMethod.CONSERVE

    kwargs: dict[str, Any] = {
        "filename": str(weights_path),
        "regrid_method": regrid_method,
        "unmapped_action": ESMF.UnmappedAction.IGNORE if ignore_unmapped else ESMF.UnmappedAction.ERROR,
        "ignore_degenerate": True,
        "large_file": large_file,
    }
    if method in {"conservative", "conservative_2nd"}:
        kwargs["norm_type"] = ESMF.NormType.DSTAREA
        kwargs["line_type"] = ESMF.LineType.GREAT_CIRCLE

    regrid = ESMF.Regrid(src_field, dst_field, **kwargs)
    regrid.destroy()
    return weights_path


# -----------------------------------------------------------------------------
# Sparse weight application
# -----------------------------------------------------------------------------

def load_sparse_weights(weights_path: str | Path) -> tuple[csr_matrix, int, int]:
    ds = xr.open_dataset(weights_path)
    row = ds["row"].values.astype(np.int64).ravel()
    col = ds["col"].values.astype(np.int64).ravel()
    values = ds["S"].values.astype(np.float64).ravel()
    if row.size == 0:
        raise ValueError("Weight file contains no weights")
    if row.min() >= 1:
        row = row - 1
    if col.min() >= 1:
        col = col - 1
    n_target = int(row.max()) + 1
    n_source = int(col.max()) + 1
    mat = coo_matrix((values, (row, col)), shape=(n_target, n_source), dtype=np.float64).tocsr()
    return mat, n_target, n_source


def apply_weights_nd(
    values: np.ndarray,
    matrix: csr_matrix,
    *,
    n_source_dims: int = 1,
    missing_policy: MissingPolicy = "renormalize",
) -> np.ndarray:
    """Apply weights to an array whose last source dims are the source grid."""
    arr = np.asarray(values, dtype=np.float64)
    n_target, n_source = matrix.shape
    batch_shape = arr.shape[:-n_source_dims] if n_source_dims > 1 else arr.shape[:-1]
    arr_2d = arr.reshape(int(np.prod(batch_shape)) if batch_shape else 1, -1)
    if arr_2d.shape[-1] != n_source:
        raise ValueError(f"source size {arr_2d.shape[-1]} does not match weights source size {n_source}")

    valid = np.isfinite(arr_2d)
    filled = np.where(valid, arr_2d, 0.0)
    out = np.asarray(matrix @ filled.T, dtype=np.float64).T

    if missing_policy == "propagate":
        missing = np.asarray(matrix @ (~valid).astype(np.float64).T, dtype=np.float64).T
        out[missing > 0.0] = np.nan
    else:
        support = np.asarray(matrix @ valid.astype(np.float64).T, dtype=np.float64).T
        with np.errstate(invalid="ignore", divide="ignore"):
            out = out / support
        out[support <= 0.0] = np.nan

    return out.reshape(*batch_shape, n_target)


# -----------------------------------------------------------------------------
# User-facing workflow class
# -----------------------------------------------------------------------------

class ReducedGaussianToGaussianRegridder:
    """Python-only reduced Gaussian GRIB -> Gaussian NetCDF workflow."""

    def __init__(
        self,
        sample_source_file: str | Path,
        *,
        weights_file: str | Path,
        target_template_file: str | Path | None = None,
        target_nlat: int | None = None,
        target_nlon: int | None = None,
        grid_type: Literal["era5", "era5_land"] | None = None,
        method: Method = "conservative",
        reuse_weights: bool = True,
        ignore_unmapped: bool = False,
        cache_meshes: bool = True,
        mesh_cache_dir: str | Path | None = None,
    ) -> None:
        if method not in ("nearest", "conservative", "conservative_2nd"):
            raise ValueError(
                f"Unknown remapping method {method!r}; expected 'nearest', "
                "'conservative', or 'conservative_2nd'"
            )

        self.sample_source_file = Path(sample_source_file)
        self.weights_file = Path(weights_file)
        self.method = method
        self.ignore_unmapped = ignore_unmapped

        print("--- Step 1: Building/loading source reduced-Gaussian mesh ---")
        if mesh_cache_dir is None:
            mesh_cache_dir = self.weights_file.parent / "mesh_cache"
        cache_path = source_mesh_cache_path(self.sample_source_file, mesh_cache_dir)
        if cache_meshes and cache_path.exists():
            print(f"Loading cached source mesh: {cache_path}")
            self.source_mesh = load_polygon_mesh(cache_path)
            self.source_size = self.source_mesh.face_count
        else:
            src_ds = open_grib_dataset(self.sample_source_file, normalize_time=False)
            src_lon = src_ds["longitude"].values.ravel()
            src_lat = src_ds["latitude"].values.ravel()
            self.source_size = src_lon.size
            self.source_mesh = reduced_gaussian_mesh_from_centres(src_lon, src_lat)
            if cache_meshes:
                print(f"Saving source mesh cache: {cache_path}")
                save_polygon_mesh(cache_path, self.source_mesh)

        print("--- Step 2: Building target Gaussian mesh ---")
        if target_template_file is not None:
            self.target_mesh, self.ds_target = target_mesh_from_file(target_template_file)
        else:
            if target_nlat is None or target_nlon is None:
                if grid_type == "era5_land":
                    target_nlat, target_nlon = 2560, 5136
                elif grid_type == "era5" or grid_type is None:
                    target_nlat, target_nlon = 640, 1280
            self.target_mesh, self.ds_target = gaussian_target_mesh(int(target_nlat), int(target_nlon))

        self.target_size = self.target_mesh.face_count

        print("--- Step 3: Computing/reusing ESMF weights ---")
        print(f"method       : {self.method}")
        print(f"weights file : {self.weights_file}")
        if reuse_weights and self.weights_file.exists():
            print("Reusing existing weights")
        else:
            compute_weights(
                self.source_mesh,
                self.target_mesh,
                self.weights_file,
                method=self.method,
                ignore_unmapped=self.ignore_unmapped,
            )

        print("--- Step 4: Loading sparse weights ---")
        self.matrix, n_target, n_source = load_sparse_weights(self.weights_file)
        if n_source != self.source_size:
            raise RuntimeError(f"weight source size {n_source} != source size {self.source_size}")
        if n_target != self.target_size:
            raise RuntimeError(f"weight target size {n_target} != target size {self.target_size}")

    def remap_dataarray(self, da: xr.DataArray, *, missing_policy: MissingPolicy = "renormalize") -> xr.DataArray:
        """Remap a DataArray with trailing source dimension named values/cell or size source_size."""
        src_dim = None
        for dim in reversed(da.dims):
            if da.sizes[dim] == self.source_size:
                src_dim = dim
                break
        if src_dim is None:
            raise ValueError(f"Could not find source dimension of size {self.source_size} in {da.dims}")

        da_work = da.transpose(..., src_dim)
        out = apply_weights_nd(da_work.values, self.matrix, n_source_dims=1, missing_policy=missing_policy)

        batch_dims = da_work.dims[:-1]
        batch_coords = {d: da_work.coords[d] for d in batch_dims if d in da_work.coords}
        out = out.reshape(*[da_work.sizes[d] for d in batch_dims], self.ds_target.sizes["lat"], self.ds_target.sizes["lon"])
        return xr.DataArray(
            out,
            dims=(*batch_dims, "lat", "lon"),
            coords={**batch_coords, "lat": self.ds_target["lat"], "lon": self.ds_target["lon"]},
            name=da.name,
            attrs=da.attrs,
        )

    def remap_file(
        self,
        input_grib_path: str | Path,
        output_netcdf_path: str | Path,
        *,
        variable: str | None = None,
        compression_level: int = 4,
        missing_policy: MissingPolicy = "renormalize",
        normalize_grib_time: bool = True,
        use_grib_cache: bool = False,
    ) -> Path:
        ds = open_grib_dataset(
            input_grib_path,
            normalize_time=normalize_grib_time,
            use_grib_cache=use_grib_cache,
        )
        if variable is None:
            data_vars = list(ds.data_vars)
            if len(data_vars) != 1:
                raise ValueError(f"Pass variable=...; found variables {data_vars}")
            variable = data_vars[0]

        out_da = self.remap_dataarray(ds[variable], missing_policy=missing_policy)
        ds_out = out_da.to_dataset(name=variable)
        ds_out[variable].attrs.setdefault("cell_measures", "area: areacella")
        for name in ("lat_bnds", "lon_bnds", "areacella"):
            ds_out[name] = self.ds_target[name]
        output_netcdf_path = Path(output_netcdf_path)
        output_netcdf_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {
            variable: {"zlib": True, "complevel": compression_level},
            "lat_bnds": {"_FillValue": None},
            "lon_bnds": {"_FillValue": None},
            "areacella": {"_FillValue": None},
        }
        ds_out.to_netcdf(output_netcdf_path, encoding=encoding)
        return output_netcdf_path
