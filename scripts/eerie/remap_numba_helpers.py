from __future__ import annotations

import json
import re
import time
import hashlib
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import uxarray as ux
from numba import njit, prange
from tqdm import tqdm


DEFAULT_CACHE_ROOT = Path("/scratch/k/k202181/remap_eerie")
REGRID_IMPL_VERSION = "healpix_prepare_v6_stable_target_hash"
SOURCE_GRID_HASH_VERSION = "uxgrid_hash_v1"
HEALPIX_TARGET_HASH_VERSION = "healpix_target_spec_v1"
PAIR_HASH_VERSION = "grid_pair_v2"


# =============================================================================
# lazy ESMF import
# =============================================================================

def _get_ESMF():
    import ESMF
    return ESMF


# =============================================================================
# dataclass
# =============================================================================

@dataclass
class PreparedHealpixRegridder:
    src_grid: Optional[ux.Grid]
    dst_grid: ux.Grid

    src_n_face: int

    src_grid_hash: str
    dst_grid_hash: str
    pair_hash: str

    source_info: dict
    src_meta: dict
    dst_meta: dict
    timings: dict

    src_cache_dir: str
    dst_cache_dir: str
    weight_file: str
    weight_meta_file: str

    # fallback Voronoi mode:
    #   tri_parent maps source triangles -> original source faces
    # native grid mode:
    #   tri_parent maps source elements -> kept source faces
    tri_parent: np.ndarray

    # fallback Voronoi mode: full original source-face areas
    # native grid mode: kept source-face areas aligned with tri_parent
    src_face_areas: np.ndarray
    dst_face_areas: np.ndarray

    src_mesh: object
    dst_mesh: object
    src_field: object
    dst_field: object
    regrid: object

    zoom: int
    nest: bool


# =============================================================================
# timing
# =============================================================================

@contextmanager
def _timed_step(timings: dict, key: str, verbose: bool = True):
    t0 = time.perf_counter()
    if verbose:
        print(f"[start] {key}")
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        timings[key] = dt
        if verbose:
            print(f"[done ] {key}: {dt:.2f}s")


# =============================================================================
# small utils
# =============================================================================

def _sanitize_key(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("._")
    return s or "source"


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def _load_npy(path: Path, use_memmap: bool = False):
    if use_memmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def _hash_array_blake2b(h, arr) -> None:
    a = np.asarray(arr)
    a = np.ascontiguousarray(a)
    h.update(np.array([a.ndim], dtype=np.int64).tobytes())
    h.update(np.array(a.shape, dtype=np.int64).tobytes())
    h.update(str(a.dtype).encode("utf-8"))
    h.update(memoryview(a).cast("B"))


def hash_uxgrid(uxgrid: ux.Grid) -> str:
    """
    Geometry/topology hash of a concrete UXarray grid.

    Good for:
      - native source grids loaded from ICON grid files
      - explicit source grids provided by the user

    Not ideal for procedural HEALPix targets, because node numbering/order may
    vary across builds even when the mathematical grid is identical.
    """
    h = hashlib.blake2b(digest_size=20)
    h.update(SOURCE_GRID_HASH_VERSION.encode("utf-8"))

    _hash_array_blake2b(h, uxgrid.node_lon.values)
    _hash_array_blake2b(h, uxgrid.node_lat.values)
    _hash_array_blake2b(h, uxgrid.face_lon.values)
    _hash_array_blake2b(h, uxgrid.face_lat.values)
    _hash_array_blake2b(h, uxgrid.n_nodes_per_face.values)
    _hash_array_blake2b(h, uxgrid.face_node_connectivity.values)

    return h.hexdigest()


def hash_healpix_target_spec(*, zoom: int, nest: bool) -> str:
    """
    Stable identity hash for the HEALPix target grid.

    This deliberately hashes the grid specification, not the realized UXarray
    node/face ordering. Same zoom + nest must yield the same cache/weight key.
    """
    h = hashlib.blake2b(digest_size=20)
    h.update(HEALPIX_TARGET_HASH_VERSION.encode("utf-8"))
    h.update(f"|zoom={int(zoom)}|nest={int(bool(nest))}|pixels_only=0|".encode("utf-8"))
    return h.hexdigest()


def hash_grid_pair(src_grid_hash: str, dst_grid_hash: str) -> str:
    """
    Stable weight/operator hash from source-grid identity and target-grid identity.
    """
    h = hashlib.blake2b(digest_size=20)
    h.update(REGRID_IMPL_VERSION.encode("utf-8"))
    h.update(PAIR_HASH_VERSION.encode("utf-8"))
    h.update(b"|")
    h.update(src_grid_hash.encode("utf-8"))
    h.update(b"|")
    h.update(dst_grid_hash.encode("utf-8"))
    return h.hexdigest()


def _should_use_native_source_path(src_grid: ux.Grid) -> bool:
    """
    Use native-cell path for already-native source grids, especially ICON triangles.
    """
    try:
        n_max = int(src_grid.n_max_face_nodes)
    except Exception:
        return False

    has_conn = hasattr(src_grid, "face_node_connectivity")
    return has_conn and n_max == 3


# =============================================================================
# source-grid loading
# =============================================================================

def load_source_grid(
    *,
    src_grid: Optional[ux.Grid] = None,
    gridfile: Optional[str | Path] = None,
    data: Optional[xr.Dataset] = None,
    lon_deg: Optional[np.ndarray] = None,
    lat_deg: Optional[np.ndarray] = None,
    grid_chunks=-1,
    verbose: bool = True,
):
    """
    Precedence:
      1. explicit src_grid
      2. gridfile via ux.open_grid(..., chunks=grid_chunks)
      3. fallback: spherical_voronoi from lon/lat centers
         (provided directly or extracted from data['lon']/data['lat'])
    """
    if src_grid is not None:
        return src_grid, {
            "source_mode": "provided_grid",
            "gridfile": None,
            "grid_chunks": None,
        }

    if gridfile is not None:
        gridfile = str(gridfile)
        if verbose:
            print(f"loading source grid from gridfile: {gridfile} (chunks={grid_chunks})")
        src_grid = ux.open_grid(gridfile, chunks=grid_chunks)
        return src_grid, {
            "source_mode": "gridfile",
            "gridfile": gridfile,
            "grid_chunks": grid_chunks,
        }

    if lon_deg is None or lat_deg is None:
        if data is None:
            raise ValueError(
                "Need one of: src_grid, gridfile, or (lon_deg, lat_deg), or data with lon/lat."
            )

        lat = np.asarray(data["lat"].values)
        lon = np.asarray(data["lon"].values)

        # ICON often stores radians
        if np.abs(lat).max() < np.pi + 0.01:
            lat_deg = np.rad2deg(lat)
            lon_deg = np.rad2deg(lon)
        else:
            lat_deg = lat
            lon_deg = lon

    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)

    if verbose:
        print("building source grid from point centers with spherical_voronoi fallback")

    src_grid = ux.Grid.from_points((lon_deg, lat_deg), method="spherical_voronoi")
    return src_grid, {
        "source_mode": "voronoi_from_points",
        "gridfile": None,
        "grid_chunks": None,
        "n_points": int(lon_deg.size),
    }


# =============================================================================
# numba internals
# =============================================================================

@njit(cache=True, inline="always")
def _has_duplicate_prefix(arr: np.ndarray, n: int) -> bool:
    for i in range(n):
        ai = arr[i]
        for j in range(i + 1, n):
            if arr[j] == ai:
                return True
    return False


@njit(cache=True, inline="always")
def _count_clean_vertices(raw_row: np.ndarray, nraw: int, n_node: int) -> int:
    """
    Count valid vertices after:
      - range check
      - removing consecutive duplicates
      - removing repeated closing node
      - rejecting non-consecutive duplicates
    Returns 0 if invalid.
    """
    if nraw < 3:
        return 0

    max_n = raw_row.shape[0]
    tmp = np.empty(max_n, dtype=np.int64)

    m = 0
    prev = -9223372036854775807
    first = True

    for k in range(nraw):
        v = int(raw_row[k])
        if v < 0 or v >= n_node:
            return 0
        if first or v != prev:
            tmp[m] = v
            m += 1
            prev = v
            first = False

    if m > 1 and tmp[0] == tmp[m - 1]:
        m -= 1

    if m < 3:
        return 0

    if _has_duplicate_prefix(tmp, m):
        return 0

    return m


@njit(cache=True)
def _prepare_face_ordered(
    raw_row: np.ndarray,
    nraw: int,
    n_node: int,
    node_lon_deg: np.ndarray,
    node_lat_deg: np.ndarray,
    clon_deg: float,
    clat_deg: float,
    out_nodes: np.ndarray,
) -> int:
    """
    Clean a face and write ordered CCW node ids into out_nodes[:m].
    Returns m, or 0 if invalid.
    """
    if nraw < 3:
        return 0

    m = 0
    prev = -9223372036854775807
    first = True

    for k in range(nraw):
        v = int(raw_row[k])
        if v < 0 or v >= n_node:
            return 0
        if first or v != prev:
            out_nodes[m] = v
            m += 1
            prev = v
            first = False

    if m > 1 and out_nodes[0] == out_nodes[m - 1]:
        m -= 1

    if m < 3:
        return 0

    if _has_duplicate_prefix(out_nodes, m):
        return 0

    deg2rad = np.pi / 180.0
    clon = clon_deg * deg2rad
    clat = clat_deg * deg2rad

    ang = np.empty(m, dtype=np.float64)
    ordered = np.empty(m, dtype=np.int64)

    for i in range(m):
        idx = out_nodes[i]
        lonr = node_lon_deg[idx] * deg2rad
        latr = node_lat_deg[idx] * deg2rad

        x = np.cos(latr) * np.sin(lonr - clon)
        y = (
            np.cos(clat) * np.sin(latr)
            - np.sin(clat) * np.cos(latr) * np.cos(lonr - clon)
        )
        ang[i] = np.arctan2(y, x)

    order = np.argsort(ang)
    for i in range(m):
        ordered[i] = out_nodes[order[i]]

    area2 = 0.0
    x_prev = 0.0
    y_prev = 0.0

    for i in range(m):
        idx = ordered[i]
        lonr = node_lon_deg[idx] * deg2rad
        latr = node_lat_deg[idx] * deg2rad

        x = np.cos(latr) * np.sin(lonr - clon)
        y = (
            np.cos(clat) * np.sin(latr)
            - np.sin(clat) * np.cos(latr) * np.cos(lonr - clon)
        )

        if i > 0:
            area2 += x_prev * y - y_prev * x

        x_prev = x
        y_prev = y

    idx0 = ordered[0]
    lonr0 = node_lon_deg[idx0] * deg2rad
    latr0 = node_lat_deg[idx0] * deg2rad
    x0 = np.cos(latr0) * np.sin(lonr0 - clon)
    y0 = (
        np.cos(clat) * np.sin(latr0)
        - np.sin(clat) * np.cos(latr0) * np.cos(lonr0 - clon)
    )
    area2 += x_prev * y0 - y_prev * x0

    if area2 < 0.0:
        for i in range(m):
            out_nodes[i] = ordered[m - 1 - i]
    else:
        for i in range(m):
            out_nodes[i] = ordered[i]

    return m


@njit(cache=True, parallel=True)
def _count_source_triangles_per_face(
    conn: np.ndarray,
    n_nodes_per_face: np.ndarray,
    n_node: int,
) -> np.ndarray:
    n_face = n_nodes_per_face.shape[0]
    tri_counts = np.zeros(n_face, dtype=np.int64)

    for i in prange(n_face):
        m = _count_clean_vertices(conn[i], int(n_nodes_per_face[i]), n_node)
        tri_counts[i] = m  # triangle fan count == cleaned vertex count

    return tri_counts


@njit(cache=True, parallel=True)
def _fill_source_triangles(
    conn: np.ndarray,
    n_nodes_per_face: np.ndarray,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    face_lon: np.ndarray,
    face_lat: np.ndarray,
    n_node: int,
    offsets: np.ndarray,
    tri_conn: np.ndarray,
    tri_parent: np.ndarray,
) -> None:
    n_face = n_nodes_per_face.shape[0]
    max_n = conn.shape[1]

    for i in prange(n_face):
        start = offsets[i]
        end = offsets[i + 1]
        m_expected = end - start

        if m_expected == 0:
            continue

        ordered = np.empty(max_n, dtype=np.int64)
        m = _prepare_face_ordered(
            conn[i],
            int(n_nodes_per_face[i]),
            n_node,
            node_lon,
            node_lat,
            float(face_lon[i]),
            float(face_lat[i]),
            ordered,
        )

        if m != m_expected:
            continue

        cidx = n_node + i
        for j in range(m):
            tri_conn[start + j, 0] = cidx
            tri_conn[start + j, 1] = ordered[j]
            tri_conn[start + j, 2] = ordered[(j + 1) % m]
            tri_parent[start + j] = i


@njit(cache=True, parallel=True)
def _count_target_vertices_per_face(
    conn: np.ndarray,
    n_nodes_per_face: np.ndarray,
    n_node: int,
) -> np.ndarray:
    n_face = n_nodes_per_face.shape[0]
    counts = np.zeros(n_face, dtype=np.int64)

    for i in prange(n_face):
        counts[i] = _count_clean_vertices(conn[i], int(n_nodes_per_face[i]), n_node)

    return counts


@njit(cache=True, parallel=True)
def _fill_target_polygons(
    conn: np.ndarray,
    n_nodes_per_face: np.ndarray,
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    face_lon: np.ndarray,
    face_lat: np.ndarray,
    n_node: int,
    kept: np.ndarray,
    offsets: np.ndarray,
    elem_types: np.ndarray,
    elem_conn: np.ndarray,
) -> None:
    max_n = conn.shape[1]
    n_keep = kept.shape[0]

    for kk in prange(n_keep):
        i = int(kept[kk])
        ordered = np.empty(max_n, dtype=np.int64)

        m = _prepare_face_ordered(
            conn[i],
            int(n_nodes_per_face[i]),
            n_node,
            node_lon,
            node_lat,
            float(face_lon[i]),
            float(face_lat[i]),
            ordered,
        )

        elem_types[kk] = m
        start = offsets[kk]

        for j in range(m):
            elem_conn[start + j] = ordered[j]


# =============================================================================
# source cache builders
# =============================================================================

def _build_source_triangle_cache(
    src_grid: ux.Grid,
    src_cache_dir: Path,
    *,
    show_progress: bool = False,
    verbose: bool = True,
) -> None:
    """
    Fallback path:
    Build source cache from a Voronoi UX grid by splitting each face into a triangle fan.
    """
    src_cache_dir.mkdir(parents=True, exist_ok=True)

    node_lon = np.asarray(src_grid.node_lon.values, dtype=np.float64)
    node_lat = np.asarray(src_grid.node_lat.values, dtype=np.float64)
    face_lon = np.asarray(src_grid.face_lon.values, dtype=np.float64)
    face_lat = np.asarray(src_grid.face_lat.values, dtype=np.float64)
    conn = np.asarray(src_grid.face_node_connectivity.values)
    n_nodes_per_face = np.asarray(src_grid.n_nodes_per_face.values, dtype=np.int64)

    n_face = face_lon.size
    n_node = node_lon.size

    all_lon = np.concatenate([node_lon, face_lon])
    all_lat = np.concatenate([node_lat, face_lat])

    pbar = tqdm(total=2, desc="source cache", disable=not show_progress)

    tri_counts = _count_source_triangles_per_face(conn, n_nodes_per_face, n_node)
    pbar.update(1)

    offsets = np.empty(n_face + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(tri_counts, dtype=np.int64)

    n_tri = int(offsets[-1])
    tri_conn = np.empty((n_tri, 3), dtype=np.int32)
    tri_parent = np.empty(n_tri, dtype=np.int64)

    _fill_source_triangles(
        conn,
        n_nodes_per_face,
        node_lon,
        node_lat,
        face_lon,
        face_lat,
        n_node,
        offsets,
        tri_conn,
        tri_parent,
    )
    pbar.update(1)
    pbar.close()

    _save_npy(src_cache_dir / "all_lon.npy", all_lon)
    _save_npy(src_cache_dir / "all_lat.npy", all_lat)
    _save_npy(src_cache_dir / "tri_conn.npy", tri_conn)
    _save_npy(src_cache_dir / "tri_parent.npy", tri_parent)

    meta = {
        "source_mode": "voronoi_triangle_fan",
        "n_node_original": int(n_node),
        "n_face_original": int(n_face),
        "n_node_triangulated": int(all_lon.size),
        "n_triangles": int(n_tri),
        "n_bad_faces_skipped": int(np.count_nonzero(tri_counts == 0)),
    }
    with open(src_cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"source cache written: {src_cache_dir}")
        print(json.dumps(meta, indent=2))


def _load_or_build_source_triangle_cache(
    src_grid: ux.Grid,
    cache_root: Path,
    src_key: str,
    *,
    rebuild: bool = False,
    use_memmap: bool = False,
    show_progress: bool = False,
    verbose: bool = True,
):
    src_key = _sanitize_key(src_key)
    src_cache_dir = cache_root / "source_voronoi_trifan" / src_key

    needed = [
        src_cache_dir / "all_lon.npy",
        src_cache_dir / "all_lat.npy",
        src_cache_dir / "tri_conn.npy",
        src_cache_dir / "tri_parent.npy",
        src_cache_dir / "meta.json",
    ]

    if rebuild or not all(p.exists() for p in needed):
        _build_source_triangle_cache(
            src_grid,
            src_cache_dir,
            show_progress=show_progress,
            verbose=verbose,
        )

    all_lon = _load_npy(src_cache_dir / "all_lon.npy", use_memmap=use_memmap)
    all_lat = _load_npy(src_cache_dir / "all_lat.npy", use_memmap=use_memmap)
    tri_conn = _load_npy(src_cache_dir / "tri_conn.npy", use_memmap=use_memmap)
    tri_parent = _load_npy(src_cache_dir / "tri_parent.npy", use_memmap=use_memmap)

    with open(src_cache_dir / "meta.json") as f:
        meta = json.load(f)

    return src_cache_dir, all_lon, all_lat, tri_conn, tri_parent, meta


def _build_native_ux_source_cache(
    src_grid: ux.Grid,
    src_cache_dir: Path,
    *,
    show_progress: bool = False,
    verbose: bool = True,
) -> None:
    """
    Native-grid path:
    Build source cache directly from a native UX grid.
    No Voronoi reconstruction. No triangle fan.
    """
    src_cache_dir.mkdir(parents=True, exist_ok=True)

    node_lon = np.asarray(src_grid.node_lon.values, dtype=np.float64)
    node_lat = np.asarray(src_grid.node_lat.values, dtype=np.float64)
    face_lon = np.asarray(src_grid.face_lon.values, dtype=np.float64)
    face_lat = np.asarray(src_grid.face_lat.values, dtype=np.float64)
    conn = np.asarray(src_grid.face_node_connectivity.values)
    n_nodes_per_face = np.asarray(src_grid.n_nodes_per_face.values, dtype=np.int64)
    face_areas = np.asarray(src_grid.face_areas.values, dtype=np.float64)

    n_face = face_lon.size
    n_node = node_lon.size

    pbar = tqdm(total=2, desc="native UX source cache", disable=not show_progress)

    counts = _count_target_vertices_per_face(conn, n_nodes_per_face, n_node)
    kept = np.flatnonzero(counts > 0).astype(np.int64)
    elem_types = counts[kept].astype(np.int32)
    pbar.update(1)

    offsets = np.empty(kept.size + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(elem_types, dtype=np.int64)

    elem_conn = np.empty(int(offsets[-1]), dtype=np.int32)

    _fill_target_polygons(
        conn,
        n_nodes_per_face,
        node_lon,
        node_lat,
        face_lon,
        face_lat,
        n_node,
        kept,
        offsets,
        elem_types,
        elem_conn,
    )
    pbar.update(1)
    pbar.close()

    tri_parent = kept.copy()

    _save_npy(src_cache_dir / "node_lon.npy", node_lon)
    _save_npy(src_cache_dir / "node_lat.npy", node_lat)
    _save_npy(src_cache_dir / "elem_types.npy", elem_types)
    _save_npy(src_cache_dir / "elem_conn.npy", elem_conn)
    _save_npy(src_cache_dir / "tri_parent.npy", tri_parent)
    _save_npy(src_cache_dir / "kept.npy", kept)
    _save_npy(src_cache_dir / "face_areas.npy", face_areas[kept])

    meta = {
        "source_mode": "gridfile_native_uxgrid",
        "n_face_original": int(n_face),
        "n_face_kept": int(kept.size),
        "n_node": int(n_node),
        "n_max_face_nodes": int(src_grid.n_max_face_nodes),
        "n_bad_faces_skipped": int(np.count_nonzero(counts == 0)),
    }
    with open(src_cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"native UX source cache written: {src_cache_dir}")
        print(json.dumps(meta, indent=2))


def _load_or_build_native_ux_source_cache(
    src_grid: ux.Grid,
    cache_root: Path,
    src_key: str,
    *,
    rebuild: bool = False,
    use_memmap: bool = False,
    show_progress: bool = False,
    verbose: bool = True,
):
    src_key = _sanitize_key(src_key)
    src_cache_dir = cache_root / "source_native_ux" / src_key

    needed = [
        src_cache_dir / "node_lon.npy",
        src_cache_dir / "node_lat.npy",
        src_cache_dir / "elem_types.npy",
        src_cache_dir / "elem_conn.npy",
        src_cache_dir / "tri_parent.npy",
        src_cache_dir / "kept.npy",
        src_cache_dir / "face_areas.npy",
        src_cache_dir / "meta.json",
    ]

    if rebuild or not all(p.exists() for p in needed):
        _build_native_ux_source_cache(
            src_grid,
            src_cache_dir,
            show_progress=show_progress,
            verbose=verbose,
        )

    node_lon = _load_npy(src_cache_dir / "node_lon.npy", use_memmap=use_memmap)
    node_lat = _load_npy(src_cache_dir / "node_lat.npy", use_memmap=use_memmap)
    elem_types = _load_npy(src_cache_dir / "elem_types.npy", use_memmap=use_memmap)
    elem_conn = _load_npy(src_cache_dir / "elem_conn.npy", use_memmap=use_memmap)
    tri_parent = _load_npy(src_cache_dir / "tri_parent.npy", use_memmap=use_memmap)
    kept = _load_npy(src_cache_dir / "kept.npy", use_memmap=use_memmap)
    face_areas = _load_npy(src_cache_dir / "face_areas.npy", use_memmap=use_memmap)

    with open(src_cache_dir / "meta.json") as f:
        meta = json.load(f)

    return src_cache_dir, node_lon, node_lat, elem_types, elem_conn, tri_parent, kept, face_areas, meta


# =============================================================================
# target cache builder
# =============================================================================

def _build_target_polygon_cache(
    dst_grid: ux.Grid,
    dst_cache_dir: Path,
    *,
    show_progress: bool = False,
    verbose: bool = True,
) -> None:
    dst_cache_dir.mkdir(parents=True, exist_ok=True)

    node_lon = np.asarray(dst_grid.node_lon.values, dtype=np.float64)
    node_lat = np.asarray(dst_grid.node_lat.values, dtype=np.float64)
    face_lon = np.asarray(dst_grid.face_lon.values, dtype=np.float64)
    face_lat = np.asarray(dst_grid.face_lat.values, dtype=np.float64)
    conn = np.asarray(dst_grid.face_node_connectivity.values)
    n_nodes_per_face = np.asarray(dst_grid.n_nodes_per_face.values, dtype=np.int64)
    face_areas_full = np.asarray(dst_grid.face_areas.values, dtype=np.float64)

    n_face = face_lon.size
    n_node = node_lon.size

    pbar = tqdm(total=2, desc="target cache", disable=not show_progress)

    counts = _count_target_vertices_per_face(conn, n_nodes_per_face, n_node)
    kept = np.flatnonzero(counts > 0).astype(np.int64)
    elem_types = counts[kept].astype(np.int32)
    pbar.update(1)

    offsets = np.empty(kept.size + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(elem_types, dtype=np.int64)

    elem_conn = np.empty(int(offsets[-1]), dtype=np.int32)

    _fill_target_polygons(
        conn,
        n_nodes_per_face,
        node_lon,
        node_lat,
        face_lon,
        face_lat,
        n_node,
        kept,
        offsets,
        elem_types,
        elem_conn,
    )
    pbar.update(1)
    pbar.close()

    _save_npy(dst_cache_dir / "node_lon.npy", node_lon)
    _save_npy(dst_cache_dir / "node_lat.npy", node_lat)
    _save_npy(dst_cache_dir / "elem_types.npy", elem_types)
    _save_npy(dst_cache_dir / "elem_conn.npy", elem_conn)
    _save_npy(dst_cache_dir / "kept.npy", kept)
    _save_npy(dst_cache_dir / "face_areas.npy", face_areas_full[kept])

    meta = {
        "n_face_full": int(n_face),
        "n_face_kept": int(kept.size),
        "n_node_all": int(n_node),
        "n_bad_faces_skipped": int(np.count_nonzero(counts == 0)),
    }
    with open(dst_cache_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if verbose:
        print(f"target cache written: {dst_cache_dir}")
        print(json.dumps(meta, indent=2))


def _load_or_build_target_polygon_cache(
    dst_grid: ux.Grid,
    cache_root: Path,
    dst_key: str,
    *,
    rebuild: bool = False,
    use_memmap: bool = False,
    show_progress: bool = False,
    verbose: bool = True,
):
    dst_key = _sanitize_key(dst_key)
    dst_cache_dir = cache_root / "target" / dst_key

    needed = [
        dst_cache_dir / "node_lon.npy",
        dst_cache_dir / "node_lat.npy",
        dst_cache_dir / "elem_types.npy",
        dst_cache_dir / "elem_conn.npy",
        dst_cache_dir / "kept.npy",
        dst_cache_dir / "face_areas.npy",
        dst_cache_dir / "meta.json",
    ]

    if rebuild or not all(p.exists() for p in needed):
        _build_target_polygon_cache(
            dst_grid,
            dst_cache_dir,
            show_progress=show_progress,
            verbose=verbose,
        )

    node_lon = _load_npy(dst_cache_dir / "node_lon.npy", use_memmap=use_memmap)
    node_lat = _load_npy(dst_cache_dir / "node_lat.npy", use_memmap=use_memmap)
    elem_types = _load_npy(dst_cache_dir / "elem_types.npy", use_memmap=use_memmap)
    elem_conn = _load_npy(dst_cache_dir / "elem_conn.npy", use_memmap=use_memmap)
    kept = _load_npy(dst_cache_dir / "kept.npy", use_memmap=use_memmap)
    face_areas = _load_npy(dst_cache_dir / "face_areas.npy", use_memmap=use_memmap)

    with open(dst_cache_dir / "meta.json") as f:
        meta = json.load(f)

    return dst_cache_dir, node_lon, node_lat, elem_types, elem_conn, kept, face_areas, meta


# =============================================================================
# ESMF mesh builders
# =============================================================================

def _build_interleaved_lonlat_coords(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> np.ndarray:
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)

    if lon_deg.shape != lat_deg.shape:
        raise ValueError(
            f"lon/lat shape mismatch for element coords: {lon_deg.shape} vs {lat_deg.shape}"
        )

    coords = np.empty(lon_deg.size * 2, dtype=np.float64)
    coords[0::2] = lon_deg
    coords[1::2] = lat_deg
    return coords
    

def _build_triangle_mesh_from_arrays(
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    tri_conn: np.ndarray,
    *,
    element_lon: np.ndarray | None = None,
    element_lat: np.ndarray | None = None,
    element_area: np.ndarray | None = None,
):
    ESMF = _get_ESMF()

    mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_DEG)

    node_ids = np.arange(node_lon.size, dtype=np.int32)
    node_coords = np.empty(node_lon.size * 2, dtype=np.float64)
    node_coords[0::2] = np.asarray(node_lon, dtype=np.float64)
    node_coords[1::2] = np.asarray(node_lat, dtype=np.float64)
    node_owner = np.zeros(node_lon.size, dtype=np.int32)

    mesh.add_nodes(node_lon.size, node_ids, node_coords, node_owner)

    tri_conn = np.asarray(tri_conn, dtype=np.int32)
    elem_count = tri_conn.shape[0]
    elem_ids = np.arange(elem_count, dtype=np.int32)
    elem_types = np.full(elem_count, ESMF.MeshElemType.TRI, dtype=np.int32)

    kwargs = {}

    if element_area is not None:
        area = np.asarray(element_area, dtype=np.float64)
        if area.shape != (elem_count,):
            raise ValueError(
                f"triangle element_area shape mismatch: expected {(elem_count,)}, got {area.shape}"
            )
        kwargs["element_area"] = area

    if element_lon is not None or element_lat is not None:
        if element_lon is None or element_lat is None:
            raise ValueError("element_lon and element_lat must be provided together")
        if np.asarray(element_lon).shape != (elem_count,) or np.asarray(element_lat).shape != (elem_count,):
            raise ValueError(
                f"triangle element coords shape mismatch: expected {(elem_count,)}, "
                f"got lon={np.asarray(element_lon).shape}, lat={np.asarray(element_lat).shape}"
            )
        kwargs["element_coords"] = _build_interleaved_lonlat_coords(element_lon, element_lat)

    mesh.add_elements(
        elem_count,
        elem_ids,
        elem_types,
        tri_conn.ravel().astype(np.int32),
        **kwargs,
    )
    return mesh


def _build_polygon_mesh_from_arrays(
    node_lon: np.ndarray,
    node_lat: np.ndarray,
    elem_types: np.ndarray,
    elem_conn: np.ndarray,
    face_areas: np.ndarray | None = None,
    *,
    element_lon: np.ndarray | None = None,
    element_lat: np.ndarray | None = None,
):
    ESMF = _get_ESMF()

    mesh = ESMF.Mesh(parametric_dim=2, spatial_dim=2, coord_sys=ESMF.CoordSys.SPH_DEG)

    node_ids = np.arange(node_lon.size, dtype=np.int32)
    node_coords = np.empty(node_lon.size * 2, dtype=np.float64)
    node_coords[0::2] = np.asarray(node_lon, dtype=np.float64)
    node_coords[1::2] = np.asarray(node_lat, dtype=np.float64)
    node_owner = np.zeros(node_lon.size, dtype=np.int32)

    mesh.add_nodes(node_lon.size, node_ids, node_coords, node_owner)

    elem_types = np.asarray(elem_types, dtype=np.int32)
    elem_count = elem_types.size
    elem_ids = np.arange(elem_count, dtype=np.int32)

    kwargs = {}
    if face_areas is not None:
        area = np.asarray(face_areas, dtype=np.float64)
        if area.shape != (elem_count,):
            raise ValueError(
                f"polygon face_areas shape mismatch: expected {(elem_count,)}, got {area.shape}"
            )
        kwargs["element_area"] = area

    if element_lon is not None or element_lat is not None:
        if element_lon is None or element_lat is None:
            raise ValueError("element_lon and element_lat must be provided together")
        if np.asarray(element_lon).shape != (elem_count,) or np.asarray(element_lat).shape != (elem_count,):
            raise ValueError(
                f"polygon element coords shape mismatch: expected {(elem_count,)}, "
                f"got lon={np.asarray(element_lon).shape}, lat={np.asarray(element_lat).shape}"
            )
        kwargs["element_coords"] = _build_interleaved_lonlat_coords(element_lon, element_lat)

    mesh.add_elements(
        elem_count,
        elem_ids,
        elem_types,
        np.asarray(elem_conn, dtype=np.int32),
        **kwargs,
    )
    return mesh


# =============================================================================
# public api: prepare / apply
# =============================================================================

def prepare_healpix_regridder(
    *,
    zoom: int,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    nest: bool = True,
    use_memmap: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    rebuild_source_cache: bool = False,
    rebuild_target_cache: bool = False,
    rebuild_weights: bool = False,
    src_grid: Optional[ux.Grid] = None,
    gridfile: Optional[str | Path] = None,
    data: Optional[xr.Dataset] = None,
    lon_deg: Optional[np.ndarray] = None,
    lat_deg: Optional[np.ndarray] = None,
    grid_chunks=-1,
) -> PreparedHealpixRegridder:
    """
    Prepare everything needed for repeated conservative remaps to HEALPix.

    Usage:
      - native ICON gridfile:
            g = ux.open_grid(gridfile, chunks=-1)
            prepared = prepare_healpix_regridder(src_grid=g, ...)
      - direct gridfile is also supported if your session handles it
      - no gridfile:
            prepared = prepare_healpix_regridder(data=data, ...)
    """
    timings = {}
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # source grid
    # ------------------------------------------------------------------
    with _timed_step(timings, "load/build source grid", verbose=verbose):
        src_grid, source_info = load_source_grid(
            src_grid=src_grid,
            gridfile=gridfile,
            data=data,
            lon_deg=lon_deg,
            lat_deg=lat_deg,
            grid_chunks=grid_chunks,
            verbose=verbose,
        )

    with _timed_step(timings, "hash source grid", verbose=verbose):
        src_grid_hash = hash_uxgrid(src_grid)

    native_source = _should_use_native_source_path(src_grid)

    # ------------------------------------------------------------------
    # source cache + source mesh
    # ------------------------------------------------------------------
    if native_source:
        with _timed_step(timings, "load/build native source cache", verbose=verbose):
            (
                src_cache_dir,
                src_node_lon,
                src_node_lat,
                src_elem_types,
                src_elem_conn,
                tri_parent,
                kept_src,
                src_face_areas,
                src_meta,
            ) = _load_or_build_native_ux_source_cache(
                src_grid=src_grid,
                cache_root=cache_root,
                src_key=src_grid_hash,
                rebuild=rebuild_source_cache,
                use_memmap=use_memmap,
                show_progress=show_progress,
                verbose=verbose,
            )

        src_n_face = int(src_grid.n_face)

        with _timed_step(timings, "build source mesh", verbose=verbose):
            src_elem_lon = np.asarray(src_grid.face_lon.values, dtype=np.float64)[np.asarray(kept_src, dtype=np.int64)]
            src_elem_lat = np.asarray(src_grid.face_lat.values, dtype=np.float64)[np.asarray(kept_src, dtype=np.int64)]

            if np.all(np.asarray(src_elem_types) == 3):
                src_tri_conn = np.asarray(src_elem_conn, dtype=np.int32).reshape(-1, 3)
                src_mesh = _build_triangle_mesh_from_arrays(
                    src_node_lon,
                    src_node_lat,
                    src_tri_conn,
                    element_lon=src_elem_lon,
                    element_lat=src_elem_lat,
                    element_area=src_face_areas,
                )
            else:
                src_mesh = _build_polygon_mesh_from_arrays(
                    src_node_lon,
                    src_node_lat,
                    src_elem_types,
                    src_elem_conn,
                    face_areas=src_face_areas,
                    element_lon=src_elem_lon,
                    element_lat=src_elem_lat,
                )

        source_info["source_path_used"] = "native"

    else:
        with _timed_step(timings, "load/build source cache", verbose=verbose):
            (
                src_cache_dir,
                all_lon,
                all_lat,
                tri_conn,
                tri_parent,
                src_meta,
            ) = _load_or_build_source_triangle_cache(
                src_grid=src_grid,
                cache_root=cache_root,
                src_key=src_grid_hash,
                rebuild=rebuild_source_cache,
                use_memmap=use_memmap,
                show_progress=show_progress,
                verbose=verbose,
            )

        src_n_face = int(src_grid.n_face)
        src_face_areas = np.asarray(src_grid.face_areas.values, dtype=np.float64)

        with _timed_step(timings, "build source mesh", verbose=verbose):
            src_mesh = _build_triangle_mesh_from_arrays(all_lon, all_lat, tri_conn)

        source_info["source_path_used"] = "voronoi_triangle_fan"

    # ------------------------------------------------------------------
    # target grid/cache
    # ------------------------------------------------------------------
    with _timed_step(timings, "build target grid", verbose=verbose):
        dst_grid = ux.Grid.from_healpix(zoom=zoom, pixels_only=False, nest=nest)

    with _timed_step(timings, "hash target grid", verbose=verbose):
        # Stable cache identity for HEALPix target
        dst_grid_hash = hash_healpix_target_spec(zoom=zoom, nest=nest)

        # Optional debug hash of the realized UXarray grid geometry.
        # Useful for diagnostics, but NOT used for cache keys.
        dst_grid_geom_hash = hash_uxgrid(dst_grid)

    with _timed_step(timings, "load/build target cache", verbose=verbose):
        (
            dst_cache_dir,
            dst_node_lon,
            dst_node_lat,
            dst_elem_types,
            dst_elem_conn,
            kept_dst,
            dst_face_areas,
            dst_meta,
        ) = _load_or_build_target_polygon_cache(
            dst_grid=dst_grid,
            cache_root=cache_root,
            dst_key=dst_grid_hash,
            rebuild=rebuild_target_cache,
            use_memmap=use_memmap,
            show_progress=show_progress,
            verbose=verbose,
        )

    with _timed_step(timings, "hash grid pair", verbose=verbose):
        pair_hash = hash_grid_pair(src_grid_hash, dst_grid_hash)

    with _timed_step(timings, "build target mesh", verbose=verbose):
        dst_elem_lon = np.asarray(dst_grid.face_lon.values, dtype=np.float64)[np.asarray(kept_dst, dtype=np.int64)]
        dst_elem_lat = np.asarray(dst_grid.face_lat.values, dtype=np.float64)[np.asarray(kept_dst, dtype=np.int64)]

        dst_mesh = _build_polygon_mesh_from_arrays(
            dst_node_lon,
            dst_node_lat,
            dst_elem_types,
            dst_elem_conn,
            face_areas=dst_face_areas,
            element_lon=dst_elem_lon,
            element_lat=dst_elem_lat,
        )

    ESMF = _get_ESMF()

    with _timed_step(timings, "build fields", verbose=verbose):
        src_field = ESMF.Field(src_mesh, name="src_field", meshloc=ESMF.MeshLoc.ELEMENT)
        dst_field = ESMF.Field(dst_mesh, name="dst_field", meshloc=ESMF.MeshLoc.ELEMENT)

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------
    weight_dir = cache_root / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    weight_file = weight_dir / f"{pair_hash}.nc"
    weight_meta_file = weight_dir / f"{pair_hash}.json"

    weight_status = {
        "mode": None,                  # "loaded_existing" | "built_new"
        "reason": None,                # "existing_ok" | "missing" | "rebuild_requested" | "load_failed"
        "load_error": None,
    }

    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _build_regrid_and_write_weights(reason: str):
        weight_status["mode"] = "built_new"
        weight_status["reason"] = reason

        if verbose:
            print(f"building new weights: {weight_file}")

        regrid_kwargs = dict(
            srcfield=src_field,
            dstfield=dst_field,
            regrid_method=ESMF.RegridMethod.CONSERVE,
            line_type=ESMF.LineType.GREAT_CIRCLE,
            norm_type=ESMF.NormType.DSTAREA,
            unmapped_action=ESMF.UnmappedAction.ERROR,
            ignore_degenerate=True,
            filename=str(weight_file),
        )

        try:
            regrid_obj = ESMF.Regrid(large_file=True, **regrid_kwargs)
        except TypeError:
            regrid_obj = ESMF.Regrid(**regrid_kwargs)

        if not weight_file.exists():
            raise RuntimeError(
                f"ESMF.Regrid completed but weight file was not created: {weight_file}"
            )

        try:
            size = weight_file.stat().st_size
        except FileNotFoundError:
            size = 0

        if size <= 0:
            raise RuntimeError(
                f"ESMF.Regrid completed but weight file is empty: {weight_file}"
            )

        return regrid_obj

    if rebuild_weights:
        if verbose and weight_file.exists():
            print(f"rebuild_weights=True -> deleting existing weight file: {weight_file}")
        _safe_unlink(weight_file)
        _safe_unlink(weight_meta_file)

    with _timed_step(timings, "build/load regrid operator", verbose=verbose):
        if weight_file.exists():
            if verbose:
                print(f"using existing weights: {weight_file}")
            try:
                regrid = ESMF.RegridFromFile(
                    src_field,
                    dst_field,
                    filename=str(weight_file),
                )
                weight_status["mode"] = "loaded_existing"
                weight_status["reason"] = "existing_ok"
            except Exception as e:
                weight_status["load_error"] = repr(e)

                if verbose:
                    print(f"failed to load existing weights: {e}")
                    print(f"deleting unreadable weight file and rebuilding: {weight_file}")

                _safe_unlink(weight_file)
                _safe_unlink(weight_meta_file)

                regrid = _build_regrid_and_write_weights(reason="load_failed")
        else:
            reason = "rebuild_requested" if rebuild_weights else "missing"
            regrid = _build_regrid_and_write_weights(reason=reason)

    meta = {
        "impl_version": REGRID_IMPL_VERSION,
        "src_grid_hash": src_grid_hash,
        "dst_grid_hash": dst_grid_hash,
        "dst_grid_hash_mode": "healpix_spec",
        "dst_grid_geom_hash_debug": dst_grid_geom_hash,
        "pair_hash": pair_hash,
        "zoom": int(zoom),
        "nest": bool(nest),
        "source_info": source_info,
        "src_cache_dir": str(src_cache_dir),
        "dst_cache_dir": str(dst_cache_dir),
        "weight_file": str(weight_file),
        "src_meta": src_meta,
        "dst_meta": dst_meta,
         "weight_status": weight_status,
    }
    with open(weight_meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    dst_face_areas = np.asarray(dst_face_areas, dtype=np.float64)

    if verbose:
        print("\nPrepare timings:")
        for k, v in timings.items():
            print(f"  {k:30s} {v:10.2f}s")
        print(f"  src_grid_hash        : {src_grid_hash}")
        print(f"  dst_grid_hash        : {dst_grid_hash}  (stable HEALPix spec hash)")
        print(f"  dst_grid_geom_hash   : {dst_grid_geom_hash}  (debug only)")
        print(f"  pair_hash            : {pair_hash}")

    return PreparedHealpixRegridder(
        src_grid=src_grid,
        dst_grid=dst_grid,
        src_n_face=src_n_face,
        src_grid_hash=src_grid_hash,
        dst_grid_hash=dst_grid_hash,
        pair_hash=pair_hash,
        source_info=source_info,
        src_meta=src_meta,
        dst_meta=dst_meta,
        timings=timings,
        src_cache_dir=str(src_cache_dir),
        dst_cache_dir=str(dst_cache_dir),
        weight_file=str(weight_file),
        weight_meta_file=str(weight_meta_file),
        tri_parent=np.asarray(tri_parent, dtype=np.int64),
        src_face_areas=np.asarray(src_face_areas, dtype=np.float64),
        dst_face_areas=dst_face_areas,
        src_mesh=src_mesh,
        dst_mesh=dst_mesh,
        src_field=src_field,
        dst_field=dst_field,
        regrid=regrid,
        zoom=int(zoom),
        nest=bool(nest),
    )


def apply_prepared_healpix_regridder(
    prepared: PreparedHealpixRegridder,
    src_values,
    *,
    output_name: str = "var",
    return_uxda: bool = True,
    check_conservation: bool = False,
    verbose: bool = True,
):
    """
    Apply a previously prepared regridder to one 1D face-centered field.

    Notes
    -----
    - Do not force-cast the input field here.
    - ESMF field buffers are typically float64 internally.
    - Conservation checks are skipped if non-finite values are present.
    """
    if isinstance(src_values, xr.DataArray):
        attrs = dict(src_values.attrs)
        da_name = src_values.name
        src_arr = np.asarray(src_values.values)   # no explicit dtype cast
        if output_name == "var" and da_name:
            output_name = str(da_name)
    else:
        attrs = {}
        src_arr = np.asarray(src_values)          # no explicit dtype cast

    if src_arr.ndim != 1:
        raise ValueError(f"src_values must be 1D, got shape {src_arr.shape}")

    n_face_src = int(prepared.src_n_face)
    if src_arr.size != n_face_src:
        raise ValueError(
            f"src_values length ({src_arr.size}) does not match prepared source n_face ({n_face_src})"
        )

    src_element_vals = np.asarray(src_arr[prepared.tri_parent], dtype=np.float64)
    
    with np.errstate(invalid="ignore"):
        prepared.src_field.data[:] = src_element_vals
        prepared.dst_field.data[:] = 0.0

    prepared.regrid(prepared.src_field, prepared.dst_field)

    # keep ESMF/native dtype here; no explicit cast
    dst_vals = np.asarray(prepared.dst_field.data).copy()

    xr_da = xr.DataArray(
        dst_vals,
        dims=("n_face",),
        name=output_name,
        attrs=attrs,
    )
    xr_da.attrs["healpix_zoom"] = prepared.zoom
    xr_da.attrs["healpix_npix"] = int(12 * (4 ** prepared.zoom))
    xr_da.attrs["healpix_nest"] = prepared.nest
    xr_da.attrs["src_grid_hash"] = prepared.src_grid_hash
    xr_da.attrs["dst_grid_hash"] = prepared.dst_grid_hash
    xr_da.attrs["pair_hash"] = prepared.pair_hash
    xr_da.attrs["weight_file"] = prepared.weight_file
    xr_da.attrs["source_mode"] = prepared.source_info["source_mode"]
    xr_da.attrs["source_path_used"] = prepared.source_info.get("source_path_used", "unknown")

    uxda = ux.UxDataArray(xr_da, uxgrid=prepared.dst_grid) if return_uxda else None

    result = {
        "values": dst_vals,
        "xr_da": xr_da,
        "uxda": uxda,
        "dst_grid": prepared.dst_grid,
        "weight_file": prepared.weight_file,
        "weight_meta_file": prepared.weight_meta_file,
        "src_grid_hash": prepared.src_grid_hash,
        "dst_grid_hash": prepared.dst_grid_hash,
        "pair_hash": prepared.pair_hash,
    }

    if check_conservation:
        src64 = np.asarray(src_arr, dtype=np.float64)
        dst64 = np.asarray(dst_vals, dtype=np.float64)

        if prepared.source_info.get("source_path_used") == "native":
            src_check_vals = src64[prepared.tri_parent]
        else:
            src_check_vals = src64

        finite_src = np.isfinite(src_check_vals).all()
        finite_dst = np.isfinite(dst64).all()

        if not (finite_src and finite_dst):
            result["src_integral"] = np.nan
            result["dst_integral"] = np.nan
            result["rel_diff"] = np.nan
            if verbose:
                print("non-finite values detected; skipping conservation check")
        else:
            src_integral = float(np.sum(src_check_vals * prepared.src_face_areas))
            dst_integral = float(np.sum(dst64 * prepared.dst_face_areas))
            rel_diff = float((dst_integral - src_integral) / src_integral)

            result["src_integral"] = src_integral
            result["dst_integral"] = dst_integral
            result["rel_diff"] = rel_diff

            if verbose:
                print(f"src integral = {src_integral:.15e}")
                print(f"dst integral = {dst_integral:.15e}")
                print(f"rel diff     = {rel_diff:.3e}")

    return result


# =============================================================================
# one-shot wrapper
# =============================================================================

def remap_face_values_to_healpix_conservative(
    src_values,
    *,
    zoom: int,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
    nest: bool = True,
    use_memmap: bool = False,
    show_progress: bool = True,
    verbose: bool = True,
    rebuild_source_cache: bool = False,
    rebuild_target_cache: bool = False,
    rebuild_weights: bool = False,
    src_grid: Optional[ux.Grid] = None,
    gridfile: Optional[str | Path] = None,
    data: Optional[xr.Dataset] = None,
    lon_deg: Optional[np.ndarray] = None,
    lat_deg: Optional[np.ndarray] = None,
    grid_chunks=-1,
    output_name: str = "var",
    return_uxda: bool = True,
    check_conservation: bool = True,
):
    prepared = prepare_healpix_regridder(
        zoom=zoom,
        cache_root=cache_root,
        nest=nest,
        use_memmap=use_memmap,
        show_progress=show_progress,
        verbose=verbose,
        rebuild_source_cache=rebuild_source_cache,
        rebuild_target_cache=rebuild_target_cache,
        rebuild_weights=rebuild_weights,
        src_grid=src_grid,
        gridfile=gridfile,
        data=data,
        lon_deg=lon_deg,
        lat_deg=lat_deg,
        grid_chunks=grid_chunks,
    )

    return apply_prepared_healpix_regridder(
        prepared,
        src_values,
        output_name=output_name,
        return_uxda=return_uxda,
        check_conservation=check_conservation,
        verbose=verbose,
    )


# =============================================================================
# healpix degrade helper
# =============================================================================

def degrade_healpix_nested_mean(values, zoom_in: int, zoom_out: int) -> np.ndarray:
    """
    Downscale NESTED HEALPix by arithmetic mean over children.
    Good for intensive quantities like temperature.
    """
    values = np.asarray(values, dtype=np.float64)

    if zoom_out > zoom_in:
        raise ValueError("zoom_out must be <= zoom_in")

    npix_in = 12 * (4 ** zoom_in)
    npix_out = 12 * (4 ** zoom_out)

    if values.size != npix_in:
        raise ValueError(f"Expected {npix_in} input pixels for zoom {zoom_in}, got {values.size}")

    factor = 4 ** (zoom_in - zoom_out)
    out = values.reshape(npix_out, factor).mean(axis=1)
    return out