"""Build polygon meshes for source grids and HEALPix targets.

Classifies a source grid (regular, curvilinear, unstructured), infers
cell bounds where the source has none, and returns a
[`PolygonMesh`][grid_doctor._geometry.mesh.PolygonMesh] that the remap
backends turn into ESMF meshes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import xarray as xr

from ..metadata import HealpixGrid
from ..types import FloatArray, Int64Array, IntArray, SourceKind, SourceUnits
from .coords import (
    _get_latlon_arrays,
    _get_spatial_dims,
    _get_unstructured_dim,
    _get_unstructured_vertices,
    _is_unstructured,
    _to_float64,
)
from .healpix import _require_healpix_geo_module
from .normalise import _canonical_lon, _normalise_angle_units
from .sphere import _lonlat_to_xyz, _vectorized_polygon_centres, _xyz_to_lonlat

logger = logging.getLogger(__name__)


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
    face_nodes: IntArray
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
    # (j+1, i+1), i.e. four shifted views of size (ny+1, nx+1).
    corner_sum = padded[:-1, :-1] + padded[:-1, 1:] + padded[1:, :-1] + padded[1:, 1:]  # (ny + 1, nx + 1, 3)

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
# Mesh construction: one builder per grid type
# ===================================================================


def _regular_grid_mesh(
    lat: FloatArray,
    lon: FloatArray,
) -> PolygonMesh:
    """Construct a polygon mesh from a regular lon/lat grid.

    The cell boundaries are inferred from the 1-D centre arrays using
    [`_infer_bounds_1d`][grid_doctor._geometry.mesh._infer_bounds_1d].
    No Python-level cell loops are used.

    *lon* may be wrapped, e.g. a ``0..360`` axis normalised to
    ``[-180, 180)`` jumps from ``179`` to ``-179``.  It is unwrapped to a
    continuous sequence before the bounds are inferred, keeping the cell
    order.  Canonicalisation of face centres is applied internally.

    Args:
        lat: 1-D latitude centres in degrees, shape ``(ny,)``.
        lon: 1-D longitude centres in degrees, shape ``(nx,)``.

    Returns:
        Compact polygon mesh.
    """
    lon = np.unwrap(lon.astype(np.float64, copy=False), period=360.0)
    lat_bounds = _infer_bounds_1d(lat)
    lon_bounds = _infer_bounds_1d(lon)
    ny, nx = lat.size, lon.size
    lon_cov = _lon_coverage_from_centres(lon)
    periodic = lon_cov >= 350.0

    if periodic:
        logger.debug(
            "Longitude axis detected as globally periodic (coverage=%.1f°); mesh will wrap at the seam.",
            lon_cov,
        )
        # For a global grid the last longitude bound coincides with the
        # first (modulo 360°).  We drop the duplicate column and let the
        # last column of faces wrap its right-hand nodes back to column 0
        # so the mesh is topologically closed.
        node_idx_full = np.arange((ny + 1) * (nx + 1), dtype=np.int32).reshape(ny + 1, nx + 1)
        # Rewrite last column to point to column 0 nodes.
        node_idx_full[:, -1] = node_idx_full[:, 0]
        face_nodes: IntArray = np.stack(
            (
                node_idx_full[:-1, :-1],
                node_idx_full[:-1, 1:],
                node_idx_full[1:, 1:],
                node_idx_full[1:, :-1],
            ),
            axis=-1,
        ).reshape(-1, 4)

        # Keep only the first nx columns of nodes (drop duplicate last column).
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
            (
                node_idx[:-1, :-1],
                node_idx[:-1, 1:],
                node_idx[1:, 1:],
                node_idx[1:, :-1],
            ),
            axis=-1,
        ).reshape(-1, 4)
        node_lon_2d, node_lat_2d = np.meshgrid(lon_bounds, lat_bounds)
        node_lon = node_lon_2d.ravel()
        node_lat = node_lat_2d.ravel()

    face_lon_2d, face_lat_2d = np.meshgrid(_canonical_lon(lon), lat)

    return PolygonMesh(
        node_lon=node_lon.astype(np.float64, copy=False),
        node_lat=node_lat.astype(np.float64, copy=False),
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
    [`_infer_curvilinear_corners`][grid_doctor._geometry.mesh._infer_curvilinear_corners].

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


def _collapse_repeated_corners(face_nodes: IntArray) -> IntArray:
    """Remove cyclically consecutive duplicate node ids within each face.

    ``face_nodes`` has shape ``(n_face, max_corners)`` with ``-1`` padding.
    Cell count and order are preserved (so source indexing stays aligned
    with the data); only repeated corner references are dropped and the
    freed slots re-padded with ``-1``.
    """
    n_face, max_nodes = face_nodes.shape
    out = np.full_like(face_nodes, -1)
    for f in range(n_face):
        row = face_nodes[f]
        ids = row[row >= 0]
        if ids.size == 0:
            continue
        kept: list[int] = [int(ids[0])]
        for node in ids[1:]:
            if int(node) != kept[-1]:
                kept.append(int(node))
        # Drop a closing duplicate (last vertex repeats the first).
        if len(kept) > 1 and kept[-1] == kept[0]:
            kept.pop()
        out[f, : len(kept)] = kept
    return out


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
    # At |lat| == 90 the longitude is degenerate: every (lon, 90) pair denotes
    # the same point.  Without this, the cells meeting at each pole dedup to
    # distinct nodes and the mesh is disconnected there (ESMF: "gid missing
    # from search list").
    pole = np.isclose(np.abs(cell_lat), 90.0, atol=1e-9) & mask
    canonical = np.where(pole, 0.0, canonical)

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

    _, unique_idx, inverse = np.unique(flat_keys, return_index=True, return_inverse=True)

    original_lon = canonical.ravel()[flat_mask]
    original_lat = cell_lat.ravel()[flat_mask]

    face_nodes = np.full(cell_lon.shape, -1, dtype=np.int32)
    face_nodes.ravel()[flat_mask] = inverse.astype(np.int32, copy=False)

    # Unstructured grids (e.g. FESOM) often pad each cell to a fixed vertex
    # count by repeating vertices. After node de-duplication those repeats
    # become the *same* node id within a face, which ESMF rejects at mesh
    # creation ("repeated points" in triangulate_warea). Collapse cyclically
    # consecutive duplicate node ids per face, preserving cell order/count so
    # the source indexing still matches the data array.
    face_nodes = _collapse_repeated_corners(face_nodes)
    remaining = (face_nodes >= 0).sum(axis=1)
    if np.any(remaining < 3):
        n_bad = int(np.count_nonzero(remaining < 3))
        first = int(np.flatnonzero(remaining < 3)[0])
        raise ValueError(
            f"{n_bad} source cell(s) have fewer than three distinct vertices "
            f"after de-duplication (first: cell {first}). The source grid has "
            f"degenerate cells that cannot form polygons for conservative "
            f"remapping."
        )

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
    [`_regular_grid_mesh`][grid_doctor._geometry.mesh._regular_grid_mesh],
    [`_curvilinear_grid_mesh`][grid_doctor._geometry.mesh._curvilinear_grid_mesh],
    or
    [`_corner_mesh_from_arrays`][grid_doctor._geometry.mesh._corner_mesh_from_arrays]
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
        lon_name, lat_name = _get_unstructured_vertices(ds)
        if lon_name is None or lat_name is None:
            raise ValueError(
                "Unstructured grids require per-cell vertex coordinates, for "
                "example 'clon_vertices'/'clat_vertices', "
                "'lon_vertices'/'lat_vertices', or CF bounds variables "
                "referenced by lon:bounds and lat:bounds."
            )
        lon_v = _canonical_lon(_normalise_angle_units(_to_float64(ds[lon_name].values), source_units))
        lat_v = _normalise_angle_units(_to_float64(ds[lat_name].values), source_units)
        mesh = _corner_mesh_from_arrays(lon_v, lat_v)
        return mesh, (_get_unstructured_dim(ds),)

    lat, lon = _get_latlon_arrays(ds)
    lat = _normalise_angle_units(lat, source_units)
    lon = _normalise_angle_units(lon, source_units)
    y_dim, x_dim = _get_spatial_dims(ds)

    if lat.ndim == 1:
        return _regular_grid_mesh(lat, lon), (y_dim, x_dim)
    if lat.ndim == 2:
        return _curvilinear_grid_mesh(lat, _canonical_lon(lon)), (y_dim, x_dim)
    raise ValueError("Latitude/longitude coordinates must be 1-D or 2-D.")


# ===================================================================
# HEALPix target mesh
# ===================================================================


def _target_healpix_mesh(grid: HealpixGrid) -> tuple[Int64Array, PolygonMesh]:
    """Return the HEALPix destination mesh.

    Args:
        grid: Target HEALPix grid.

    Returns:
        ``(ipix, mesh)`` where *ipix* is the pixel index array.
    """
    module, kwargs = _require_healpix_geo_module(grid)
    ipix = np.arange(grid.npix, dtype=np.int64)
    lon_v, lat_v = module.vertices(ipix, grid.level, **kwargs)
    cell_lon = _canonical_lon(np.asarray(lon_v, dtype=np.float64))
    cell_lat = np.asarray(lat_v, dtype=np.float64)
    return ipix, _corner_mesh_from_arrays(cell_lon, cell_lat)
