"""Apply an ESMF weight file to an xarray dataset.

Reads the weight file and its grid-doctor metadata, works out which
source dimensions the weights cover, and applies the sparse matrix
(row block by row block for dask inputs) through the engine in
[`_sparse`][grid_doctor.remap._sparse].
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

import dask.array as da
import numpy as np
import xarray as xr
from dask.base import tokenize

if TYPE_CHECKING:
    from scipy.sparse import csr_array, csr_matrix

from .._geometry.coords import (
    _UNSTRUCTURED_DIMS,
    _get_latlon_arrays,
    _get_spatial_dims,
    _get_unstructured_dim,
    _is_unstructured,
)
from .._geometry.healpix import _attach_healpix_coords
from .._geometry.normalise import _canonical_lon, _normalise_angle_units
from ..metadata import ELLIPSOID_ATTR, ELLIPSOIDS, HealpixGrid
from ..metadata import _keys as keys
from ..types import (
    ApplyBackend,
    Ellipsoid,
    FloatArray,
    MissingPolicy,
    SourceUnits,
)
from ._sparse import (
    apply_weights_nd,
    extract_sparse_weights,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Source-centre coordinate extraction
# ===================================================================


def _source_centre_arrays(
    ds: xr.Dataset,
    *,
    source_units: SourceUnits,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    """Extract flattened source-centre coordinates and dimension names.

    Args:
        ds: Source dataset.
        source_units: Angular unit convention.

    Returns:
        ``(lat, lon, source_dims)``.
    """
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


# ===================================================================
# Weight-file reading (xarray layer on top of _sparse)
# ===================================================================


class _WeightFile(NamedTuple):
    """Contents of a weight file, as returned by `_read_weight_file`."""

    matrix: csr_array
    n_target: int
    n_source: int
    level: int
    order: str
    method: str | None
    stored_source_dims: tuple[str, ...] | None
    ellipsoid: str | None
    """As stored; ``None`` if the file records none (see `_weights_ellipsoid`)."""


def _read_weight_file(weights_path: str | Path, nest: bool | None = None) -> _WeightFile:
    """Read a weight file and return matrix plus metadata.

    With *nest*, the target rows are renumbered into that HEALPix order
    when the file is in the other one; ``None`` keeps the file's order.
    """
    from ..weights import nested_to_ring, ring_to_nested

    with xr.open_dataset(weights_path) as wds:
        row_name = "row" if "row" in wds else "dst_address"
        col_name = "col" if "col" in wds else "src_address"
        val_name = "S" if "S" in wds else "remap_matrix"

        level = int(wds.attrs.get(keys.LEVEL_ATTR, -1))
        source_size = wds.attrs.get(keys.SOURCE_SIZE_ATTR)
        order = str(wds.attrs.get(keys.ORDER_ATTR, "nested"))
        rows = wds[row_name].values
        if nest is not None and nest != (order == "nested"):
            renumber = nested_to_ring if order == "nested" else ring_to_nested
            rows = renumber(rows.astype(np.int64) - 1, level) + 1  # rows are 1-based
            order = "nested" if nest else "ring"
        matrix, n_target, n_source = extract_sparse_weights(
            row=rows,
            col=wds[col_name].values,
            values=wds[val_name].values,
            n_source=int(source_size) if source_size is not None else None,
        )
        method_raw = wds.attrs.get(keys.METHOD_ATTR)
        method = str(method_raw) if method_raw is not None else None
        stored_source_dims = _parse_source_dims_attr(wds.attrs.get(keys.SOURCE_DIMS_ATTR))
        ellipsoid = wds.attrs.get(ELLIPSOID_ATTR)
    return _WeightFile(
        matrix,
        n_target,
        n_source,
        level,
        order,
        method,
        stored_source_dims,
        None if ellipsoid is None else str(ellipsoid),
    )


def _weights_ellipsoid(requested: Ellipsoid | None, stored: str | None, path: str | Path) -> Ellipsoid:
    """Ellipsoid of a weight file; files without the attribute predate WGS84 support and are spherical.

    Raises
    ------
    ValueError
        If the file records an unsupported ellipsoid, or *requested* is
        not ``None`` and differs from the file's.
    """
    if stored is None:
        logger.warning("Weight file %s records no ellipsoid (%s); assuming 'sphere'.", path, ELLIPSOID_ATTR)
    elif stored not in ELLIPSOIDS:
        raise ValueError(f"{path} records an unsupported ellipsoid {stored!r}.")
    resolved = cast(Ellipsoid, stored or "sphere")
    if requested is not None and requested != resolved:
        raise ValueError(f"{path} was computed on {resolved!r}, not {requested!r}.")
    return resolved


# ===================================================================
# Source-dimension resolution for weight application
# ===================================================================


def _flattened_size(
    ds: xr.Dataset,
    source_dims: tuple[str, ...],
) -> int:
    """Return the product of ``source_dims`` sizes in *ds*."""
    missing = [d for d in source_dims if d not in ds.dims]
    if missing:
        raise ValueError(f"source_dims {source_dims!r} not in dataset. Missing: {missing!r}.")
    size = 1
    for d in source_dims:
        size *= int(ds.sizes[d])
    return size


def _parse_source_dims_attr(value: object) -> tuple[str, ...] | None:
    """Parse stored source-dimension metadata from a weight file."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [p.strip() for p in text.split(",") if p.strip()]
        if isinstance(parsed, str):
            return (parsed,)
        if isinstance(parsed, list) and all(isinstance(i, str) for i in parsed):
            return tuple(parsed)
        return None
    if isinstance(value, (tuple, list)) and all(isinstance(i, str) for i in value):
        return tuple(value)
    return None


def _guess_source_dims_from_size(
    ds: xr.Dataset,
    n_source: int,
) -> tuple[str, ...] | None:
    """Guess source dimensions when geometry is unavailable."""
    matches = [str(d) for d, s in ds.sizes.items() if int(s) == n_source]
    preferred = [d for d in matches if d in _UNSTRUCTURED_DIMS]
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
    """Resolve source dimensions for weight application."""
    if source_dims is not None:
        flat = _flattened_size(ds, source_dims)
        if flat != n_source:
            raise ValueError(f"source_dims mismatch: expected {n_source}, got {flat}.")
        return source_dims

    if stored_source_dims is not None:
        try:
            flat = _flattened_size(ds, stored_source_dims)
        except ValueError:
            flat = -1
        if flat == n_source:
            return stored_source_dims

    if grid is not None:
        grid_lat, _, grid_dims = _source_centre_arrays(grid, source_units=source_units)
        if grid_lat.size != n_source:
            raise ValueError(f"Grid mismatch: expected {n_source}, got {grid_lat.size}.")
        flat = _flattened_size(ds, grid_dims)
        if flat != n_source:
            raise ValueError(f"Grid dims mismatch: expected {n_source}, got {flat}.")
        return grid_dims

    try:
        src_lat, _, inferred = _source_centre_arrays(ds, source_units=source_units)
    except ValueError:
        guessed = _guess_source_dims_from_size(ds, n_source)
        if guessed is not None:
            return guessed
        raise ValueError("Could not infer source geometry. Pass source_dims= or grid=.") from None

    if src_lat.size == n_source:
        return inferred

    guessed = _guess_source_dims_from_size(ds, n_source)
    if guessed is not None:
        return guessed

    raise ValueError(
        f"Weight file expects {n_source} source cells, dataset has {src_lat.size}. Pass source_dims= or grid=."
    )


# ===================================================================
# Weight-file application
# ===================================================================


DEFAULT_CELL_CHUNK = 4**10
"""Default output chunk along ``cell`` (1 Mi cells, 8 MiB as float64).

Dask-backed regridding runs one serial kernel per chunk, so the chunk
count is also the parallelism of a single time step (12 tasks at
level 10)."""


def _resolve_cell_chunk(cell_chunks: int | Literal["auto"] | None, n_target: int) -> int | None:
    """Validate the cell chunk size; ``None`` when no chunking is needed."""
    if cell_chunks is None:
        return None
    size = DEFAULT_CELL_CHUNK if cell_chunks == "auto" else int(cell_chunks)
    if size < 4:
        raise ValueError("cell_chunks must be at least 4.")
    size -= size % 4  # nested parents must never straddle a chunk
    return size if size < n_target else None


def _row_blocks(matrix: csr_array, size: int) -> list[csr_matrix]:
    """Split a CSR matrix into row blocks that share its data arrays.

    ``matrix[a:b]`` would copy ``data`` and ``indices`` (seconds and
    gigabytes for large weight files); views keep a single copy.
    """
    from scipy.sparse import csr_matrix

    blocks = []
    for start in range(0, matrix.shape[0], size):
        stop = min(start + size, matrix.shape[0])
        lo, hi = matrix.indptr[start], matrix.indptr[stop]
        # The constructor copies even with copy=False; assigning the
        # arrays to an empty matrix of the right shape keeps them as views.
        block = csr_matrix((stop - start, matrix.shape[1]), dtype=matrix.dtype)
        block.data = matrix.data[lo:hi]
        block.indices = matrix.indices[lo:hi]
        block.indptr = matrix.indptr[start : stop + 1] - lo
        blocks.append(block)
    return blocks


def _apply_weights_rowblocked(
    data: xr.DataArray,
    source_dims: tuple[str, ...],
    row_blocks: list[csr_matrix],
    *,
    missing_policy: MissingPolicy,
    backend: ApplyBackend,
    parallel: bool,
) -> xr.DataArray:
    """Apply weights to dask-backed *data*, one output chunk per row block.

    Rows of the weight matrix are independent, so output chunk ``k`` is
    ``W[rows_k] @ x``.  The source still has to be complete per batch
    chunk, but every task only produces (and carries the weights for)
    its own slice of HEALPix cells.
    """
    kwargs: dict[str, str | bool] = {
        "missing_policy": missing_policy,
        "backend": backend,
        "parallel": parallel,
    }
    batch_dims = [str(d) for d in data.dims if d not in source_dims]
    ordered = data.transpose(*batch_dims, *source_dims)
    n_batch = len(batch_dims)
    arr = ordered.data.rechunk({i: -1 for i in range(n_batch, ordered.ndim)})
    flat = arr.reshape(*arr.shape[:n_batch], -1)
    # Name tasks after the source and the block position instead of
    # letting dask hash every weight block (slow for large matrices).
    # The blocks are views of one data buffer that stays alive as long as
    # any task needs it, so its id cannot be reused by another matrix.
    weights = row_blocks[0].data
    buffer = weights.base if weights.base is not None else weights
    matrix_id = (id(buffer), buffer.size, sum(b.nnz for b in row_blocks))

    parts = [
        flat.map_blocks(
            apply_weights_nd,
            matrix=block,
            n_source_dims=1,
            chunks=(*flat.chunks[:-1], (block.shape[0],)),
            dtype=np.float64,
            meta=np.empty((0,) * flat.ndim, dtype=np.float64),
            name="regrid-" + tokenize(flat.name, matrix_id, index, sorted(kwargs.items())),
            **kwargs,
        )
        for index, block in enumerate(row_blocks)
    ]
    coords = {str(k): v for k, v in ordered.coords.items() if set(v.dims) <= set(batch_dims)}
    return xr.DataArray(
        da.concatenate(parts, axis=-1),
        dims=(*batch_dims, "cell"),
        coords=coords,
        name=data.name,
    )


def apply_weight_file(
    ds: xr.Dataset,
    weights_path: str | Path,
    *,
    missing_policy: MissingPolicy = "renormalize",
    grid: xr.Dataset | None = None,
    source_dims: tuple[str, ...] | None = None,
    source_units: SourceUnits = "auto",
    backend: ApplyBackend = "auto",
    cell_chunks: int | Literal["auto"] | None = "auto",
    ellipsoid: Ellipsoid | None = None,
    nest: bool | None = None,
) -> xr.Dataset:
    """Apply a previously generated ESMF weight file to *ds*.

    Uses the batched engine in
    `grid_doctor.remap._sparse` which replaces per-slice
    ``vectorize=True`` with a single batched sparse matmul for a
    typical 10–50× speedup.

    Parameters
    ----------
    ds:
        Source dataset to remap.
    weights_path:
        Path to the NetCDF weight file.
    missing_policy:
        Strategy for handling NaN (missing) values in the source
        grid during weight application.

        ``"renormalize"`` (default)
            Source cells that are NaN are excluded from the
            weighted sum and the weights of the remaining valid
            source cells are rescaled so they sum to 1.  A target
            cell receives a valid value as long as **at least one**
            contributing source cell is valid.  This is the right
            choice for most climate fields: for instance, a
            HEALPix cell on a coastline that partly overlaps land
            (NaN) and partly ocean still receives a valid SST from
            the ocean fraction.

        ``"propagate"``
            If **any** contributing source cell is NaN, the target
            cell is set to NaN: no partial averages.  Use this
            when completeness of the source coverage matters more
            than spatial coverage of the output, for example when
            computing area-integrated flux budgets where a partial
            average would bias the integral.  Note that this is
            very aggressive: a single NaN source cell at the edge
            of a target cell's footprint is enough to discard it.
            For conservative remapping where a HEALPix cell can
            overlap dozens of source cells, large portions of the
            output along coastlines, swath edges, and orbit gaps
            will be NaN.
    grid:
        Optional grid dataset with source geometry.
    source_dims:
        Optional explicit source dimensions.
    source_units:
        Unit convention for geometry-based inference.
    backend:
        Application backend (``"auto"``, ``"scipy"``, ``"numba"``).

    cell_chunks:
        Chunk size of the output along ``cell`` for dask-backed input.
        Each output chunk is computed from its own slice of weight-matrix
        rows, so no task holds the whole HEALPix field.  ``"auto"`` uses
        ``DEFAULT_CELL_CHUNK`` cells (rounded to a multiple of 4), ``None``
        produces a single chunk.  NumPy-backed input is always computed
        in one piece.
    ellipsoid:
        Ellipsoid the weights must have been generated on.  ``None``
        (default) takes the one recorded in the weight file (a sphere
        for files that record none); the HEALPix cell centres are
        computed on it.
    nest:
        HEALPix order of the output: nested when ``True``, ring when
        ``False``. ``None`` (default) keeps the order of the weight file.
        Stored weight files are nested; their rows are renumbered for
        ring output, so one file serves both orders.

    Returns
    -------
    xarray.Dataset
        Dataset on the HEALPix target grid with a ``cell`` dimension.

    Examples
    --------
    ```python
    ds_hp = apply_weight_file(ds_icon, "icon_to_healpix_d8.nc")
    ```
    """
    if missing_policy not in {"renormalize", "propagate"}:
        raise ValueError("missing_policy must be 'renormalize' or 'propagate'.")

    matrix, n_target, n_source, level, order, method, stored_sd, stored_ellipsoid = _read_weight_file(
        weights_path, nest=nest
    )
    # Checked before regridding, so a mismatch fails fast.
    ellipsoid = _weights_ellipsoid(ellipsoid, stored_ellipsoid, weights_path)

    resolved_sd = _resolve_source_dims_for_weight_application(
        ds,
        n_source=n_source,
        grid=grid,
        source_dims=source_dims,
        source_units=source_units,
        stored_source_dims=stored_sd,
    )
    n_src_dims = len(resolved_sd)
    cell_chunk = _resolve_cell_chunk(cell_chunks, n_target)
    row_blocks: list[csr_matrix] | None = None

    regridded: dict[str, xr.DataArray] = {}
    for name, data in ds.data_vars.items():
        if not set(resolved_sd).issubset(map(str, data.dims)):
            regridded[str(name)] = data
            continue

        if cell_chunk is not None and data.chunks is not None:
            if row_blocks is None:
                row_blocks = _row_blocks(matrix, cell_chunk)
            regridded[str(name)] = _apply_weights_rowblocked(
                data,
                resolved_sd,
                row_blocks,
                missing_policy=missing_policy,
                backend=backend,
                parallel=False,
            )
            continue

        regridded[str(name)] = cast(
            xr.DataArray,
            xr.apply_ufunc(
                apply_weights_nd,
                data,
                input_core_dims=[list(resolved_sd)],
                output_core_dims=[["cell"]],
                exclude_dims=set(resolved_sd),
                dask="parallelized",
                kwargs={
                    "matrix": matrix,
                    "n_source_dims": n_src_dims,
                    "missing_policy": missing_policy,
                    "backend": backend,
                    # dask already runs tasks in parallel threads; nested
                    # thread teams exhaust the process thread limit.
                    "parallel": data.chunks is None,
                },
                output_dtypes=[np.float64],
                dask_gufunc_kwargs={"output_sizes": {"cell": n_target}},
            ),
        )

    result = xr.Dataset(regridded, attrs=ds.attrs.copy())
    if level >= 0:
        target_grid = HealpixGrid.from_nest(
            level,
            nest=(order == "nested"),
            ellipsoid=ellipsoid,
        )
        result = _attach_healpix_coords(result, target_grid, method=method)
    return result
