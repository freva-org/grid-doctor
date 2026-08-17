r"""Region selection for (high-level) HEALPix datasets.

At high HEALPix levels the global cell dimension becomes enormous
(``12 * 4**16`` ≈ 5.2e10 cells at ~100 m resolution) while regional
datasets — a city, an island, a catchment — occupy a vanishing fraction
of it.  Such stores are written without materialised coordinate arrays
(see the ``write_coords`` parameter of
[`save_pyramid`][grid_doctor.save_pyramid]), and *all* access is
region-driven: nobody ever loads the full cell dimension, the same way
nobody downloads a full XYZ tile set.

The selectors in this module exploit the nested-ordering locality
property: every cell at query level :math:`L - \\Delta` corresponds to
exactly one *contiguous* index range :math:`[p \\cdot 4^\\Delta,
(p+1) \\cdot 4^\\Delta)` at level :math:`L`.  A bounding-box query is
answered at the coarse level (via ``healpix_geo``'s coverage searches),
merged into contiguous runs, and read as a handful of contiguous slices
— which map one-to-one onto Zarr chunks when the store is chunked at a
power of four.

The returned subset is the compact ("sparse") representation also
produced by ``bin_to_healpix(..., dense=False)``: the ``cell``
coordinate holds the actual global HEALPix indices, cell-centre
``latitude``/``longitude`` are computed on the fly for exactly the
selected cells, and ``grid_doctor_sparse = 1`` is set.

Open high-level stores with ``chunks=None``: with the default dask
chunking, ``xarray`` eagerly builds a chunk-grid description
proportional to the *global* chunk count, which is prohibitive at
level 16 and impossible at level 20.  Plain lazy Zarr-backed arrays
slice in O(selection).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from .cf import HealpixNested
from .remap import _make_crs_variable
from .remap_backend import _canonical_lon, _require_healpix_geo_module
from .types import HEALPIX_LEVEL, HEALPIX_NSIDE, HEALPIX_ORDER, Int64Array

logger = logging.getLogger(__name__)

_DEFAULT_QUERY_DELTA = 8
"""Default level offset for coverage queries (ranges of ``4**8`` cells)."""


# ===================================================================
# Range machinery
# ===================================================================


def _contiguous_runs(ids: Int64Array) -> list[tuple[int, int]]:
    """Merge sorted cell IDs into ``[start, stop)`` runs of consecutive IDs."""
    if ids.size == 0:
        return []
    ids = np.sort(np.asarray(ids, dtype=np.int64))
    breaks = np.nonzero(np.diff(ids) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks, [ids.size - 1]))
    return [(int(ids[a]), int(ids[b]) + 1) for a, b in zip(starts, stops)]


def _parents_to_ranges(parents: Int64Array, delta: int) -> list[tuple[int, int]]:
    """Convert coarse parent IDs into fine-level ``[start, stop)`` ranges."""
    factor = 4**delta
    return [
        (start * factor, stop * factor) for start, stop in _contiguous_runs(parents)
    ]


def _dataset_level(ds: xr.Dataset, level: int | None) -> int:
    """Resolve the HEALPix level of *ds*."""
    if level is not None:
        return level
    try:
        return int(ds.attrs[HEALPIX_LEVEL])
    except KeyError:
        raise ValueError(
            "Dataset has no `healpix_level` attribute; pass `level=` explicitly."
        ) from None


def _require_nested(ds: xr.Dataset) -> None:
    """Range-based selection relies on nested ordering."""
    order = str(ds.attrs.get(HEALPIX_ORDER, HealpixNested))
    if order not in {HealpixNested, "nest"}:
        raise ValueError(f"Region selection requires nested ordering, got {order!r}.")


# ===================================================================
# Subset extraction
# ===================================================================


def select_cells(
    ds: xr.Dataset,
    cells: Int64Array | list[int],
    *,
    level: int | None = None,
    load: bool = True,
) -> xr.Dataset:
    """Extract the given HEALPix cells from a dataset.

    Works on dense datasets (positional index equals cell ID, with or
    without materialised coordinates) and on compact subsets (``cell``
    coordinate holds actual IDs).  Contiguous ID runs are read as
    contiguous slices, so a spatially compact selection touches only
    the chunks it needs.

    Parameters
    ----------
    ds:
        HEALPix dataset, typically opened with
        ``xr.open_zarr(store, chunks=None)``.
    cells:
        Global HEALPix cell IDs to extract (any order; duplicates are
        dropped).
    level:
        HEALPix level override when the ``healpix_level`` attribute is
        missing.
    load:
        Load the selected data into memory (default).  The selection is
        small by construction — that is the point of selecting.

    Returns
    -------
    xarray.Dataset
        Compact subset: ``cell`` coordinate holds the requested IDs,
        cell-centre ``latitude``/``longitude`` are attached, and
        ``grid_doctor_sparse = 1`` is set.
    """
    _require_nested(ds)
    resolved_level = _dataset_level(ds, level)
    wanted = np.unique(np.asarray(cells, dtype=np.int64))
    if wanted.size == 0:
        raise ValueError("No cells requested.")
    npix = 12 * 4**resolved_level
    if int(wanted[0]) < 0 or int(wanted[-1]) >= npix:
        raise ValueError(
            f"Cell IDs must be within [0, {npix}) for level {resolved_level}."
        )

    if "cell" in ds.coords:
        # Compact dataset (or legacy dense store with coordinates):
        # positions are found through the coordinate values.
        coord = np.asarray(ds["cell"].values, dtype=np.int64)
        pos = np.searchsorted(coord, wanted)
        pos = np.clip(pos, 0, coord.size - 1)
        present = coord[pos] == wanted
        if not present.all():
            missing = wanted[~present]
            raise KeyError(
                f"{missing.size} requested cells are not present in the "
                f"dataset (first missing: {int(missing[0])})."
            )
        pieces = [ds.isel(cell=slice(a, b)) for a, b in _contiguous_runs(pos)]
    else:
        # Dense store without materialised coordinates: positional
        # index *is* the cell ID.
        pieces = [ds.isel(cell=slice(a, b)) for a, b in _contiguous_runs(wanted)]

    subset = (
        pieces[0]
        if len(pieces) == 1
        else xr.concat(pieces, dim="cell", data_vars="minimal", coords="minimal")
    )
    if load:
        subset = subset.load()
    return attach_cell_coords(subset, wanted, level=resolved_level, attrs=ds.attrs)


def select_bbox(
    ds: xr.Dataset,
    *,
    lon: tuple[float, float],
    lat: tuple[float, float],
    level: int | None = None,
    query_delta: int = _DEFAULT_QUERY_DELTA,
    load: bool = True,
) -> xr.Dataset:
    """Extract all cells covering a geographic bounding box.

    The box is rasterised at the coarse level ``level - query_delta``;
    every coarse parent expands to one contiguous fine-level range of
    ``4**query_delta`` cells.  The result therefore *covers* the box
    (cells straddling the edge are included) with read granularity set
    by *query_delta* — align it with the store's chunk exponent for
    reads with zero waste.

    Parameters
    ----------
    ds:
        HEALPix dataset (see [`select_cells`][grid_doctor.select_cells]).
    lon:
        ``(west, east)`` longitude bounds in degrees.
    lat:
        ``(south, north)`` latitude bounds in degrees.
    level:
        HEALPix level override.
    query_delta:
        Level offset for the coverage query.  Larger values mean fewer,
        larger contiguous reads; smaller values follow the box outline
        more tightly.
    load:
        Load the selection into memory (default).

    Returns
    -------
    xarray.Dataset
        Compact subset covering the box.
    """
    resolved_level = _dataset_level(ds, level)
    query_level = max(0, resolved_level - query_delta)
    module, kwargs = _require_healpix_geo_module(nest=True)
    coverage = np.asarray(
        module.zone_coverage(
            (lon[0], lat[0], lon[1], lat[1]),
            query_level,
            flat=True,
            **kwargs,
        )
    )
    parents = coverage[0].astype(np.int64)
    ranges = _parents_to_ranges(parents, resolved_level - query_level)
    logger.info(
        "Bounding box resolves to %d parent cells at level %d "
        "(%d contiguous ranges at level %d).",
        parents.size,
        query_level,
        len(ranges),
        resolved_level,
    )
    cells = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in ranges])
    return select_cells(ds, cells, level=resolved_level, load=load)


def select_cone(
    ds: xr.Dataset,
    *,
    lon: float,
    lat: float,
    radius: float,
    level: int | None = None,
    query_delta: int = _DEFAULT_QUERY_DELTA,
    load: bool = True,
) -> xr.Dataset:
    """Extract all cells within *radius* degrees of a centre point.

    Parameters
    ----------
    ds:
        HEALPix dataset (see [`select_cells`][grid_doctor.select_cells]).
    lon, lat:
        Centre coordinates in degrees.
    radius:
        Angular radius in degrees.
    level:
        HEALPix level override.
    query_delta:
        Level offset for the coverage query.
    load:
        Load the selection into memory (default).

    Returns
    -------
    xarray.Dataset
        Compact subset covering the cone.
    """
    resolved_level = _dataset_level(ds, level)
    query_level = max(0, resolved_level - query_delta)
    module, kwargs = _require_healpix_geo_module(nest=True)
    coverage = np.asarray(
        module.cone_coverage((lon, lat), radius, query_level, flat=True, **kwargs)
    )
    parents = coverage[0].astype(np.int64)
    ranges = _parents_to_ranges(parents, resolved_level - query_level)
    cells = np.concatenate([np.arange(a, b, dtype=np.int64) for a, b in ranges])
    return select_cells(ds, cells, level=resolved_level, load=load)


# ===================================================================
# Coordinate reconstruction
# ===================================================================


def attach_cell_coords(
    ds: xr.Dataset,
    cells: Int64Array,
    *,
    level: int,
    attrs: Any = None,
) -> xr.Dataset:
    """Attach computed cell coordinates to a compact subset.

    HEALPix coordinates are a pure function of the cell index; stores
    written with ``write_coords=False`` carry none, and this function
    reconstructs them for exactly the cells at hand.

    Parameters
    ----------
    ds:
        Subset whose ``cell`` dimension corresponds to *cells*.
    cells:
        Global HEALPix cell IDs, one per position along ``cell``.
    level:
        HEALPix level of the IDs.
    attrs:
        Optional attribute mapping to merge (e.g. the source store's
        attributes).

    Returns
    -------
    xarray.Dataset
        Subset with ``cell``, ``latitude``, ``longitude``, and ``crs``
        coordinates, ``grid_mapping`` tags, and ``grid_doctor_sparse``
        set.
    """
    cells = np.asarray(cells, dtype=np.int64)
    if ds.sizes.get("cell") != cells.size:
        raise ValueError(
            f"Dataset has {ds.sizes.get('cell')} cells but {cells.size} "
            "IDs were provided."
        )
    module, kwargs = _require_healpix_geo_module(nest=True)
    lon_deg, lat_deg = module.healpix_to_lonlat(cells, level, **kwargs)

    result = ds.assign_coords(
        cell=cells,
        latitude=("cell", np.asarray(lat_deg, dtype=np.float64)),
        longitude=(
            "cell",
            _canonical_lon(np.asarray(lon_deg, dtype=np.float64)),
        ),
        crs=_make_crs_variable(level=level, nside=2**level, order=HealpixNested),
    )
    for name in result.data_vars:
        if "cell" in result[name].dims:
            result[name].attrs["grid_mapping"] = "crs"
    if attrs:
        merged = dict(attrs)
        merged.update(result.attrs)
        result.attrs = merged
    result.attrs[HEALPIX_LEVEL] = level
    result.attrs[HEALPIX_NSIDE] = 2**level
    result.attrs[HEALPIX_ORDER] = HealpixNested
    result.attrs["grid_doctor_sparse"] = 1
    return result


__all__ = [
    "attach_cell_coords",
    "select_bbox",
    "select_cells",
    "select_cone",
]
