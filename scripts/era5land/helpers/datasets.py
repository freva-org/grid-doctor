"""Dataset opening and reshaping helpers for ERA5/ERA5-Land."""

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date, timedelta
import logging
from typing import Any, Optional

import numpy as np
import xarray as xr

from .file_fetcher import SourceRecord
from .grib import get_vars, open_dataset
from .logging_utils import log_stage
from .metadata import clean_output_attrs

LOGGER = logging.getLogger(__name__)
LAT_COORD_NAMES = ("latitude", "lat", "Latitude", "LATITUDE", "y", "Y")
LON_COORD_NAMES = ("longitude", "lon", "Longitude", "LONGITUDE", "x", "X")
STATIC_COORD_NAMES = ("cell", "time", "crs", "surface")


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first matching coordinate name from *candidates*."""

    for name in candidates:
        if name in ds.coords:
            return name
    return None


def normalise_published_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Keep published coord/data-variable classification stable across writes."""

    coord_names = set(ds.coords)
    coord_names.update(
        name
        for name in (*STATIC_COORD_NAMES, *LAT_COORD_NAMES, *LON_COORD_NAMES)
        if name in ds
    )
    if not coord_names:
        return ds
    return ds.set_coords(sorted(coord_names))


def _circular_lon_bounds(lon_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer west/east bounds for one reduced-Gaussian latitude ring."""

    lon_mod = np.mod(np.asarray(lon_values, dtype=np.float64), 360.0)
    order = np.argsort(lon_mod)
    sorted_lon = lon_mod[order]

    prev_lon = np.roll(sorted_lon, 1)
    next_lon = np.roll(sorted_lon, -1)
    prev_lon = np.where(prev_lon > sorted_lon, prev_lon - 360.0, prev_lon)
    next_lon = np.where(next_lon < sorted_lon, next_lon + 360.0, next_lon)

    west_sorted = np.mod((prev_lon + sorted_lon) * 0.5, 360.0)
    east_sorted = np.mod((sorted_lon + next_lon) * 0.5, 360.0)

    west = np.empty_like(west_sorted)
    east = np.empty_like(east_sorted)
    west[order] = west_sorted
    east[order] = east_sorted
    return west, east


def _ring_slices(latitudes: np.ndarray) -> list[slice]:
    """Return contiguous slices for repeated reduced-Gaussian latitude rings."""

    rings: list[slice] = []
    start = 0
    latitudes = np.asarray(latitudes, dtype=np.float64)
    for index in range(1, latitudes.size + 1):
        if index == latitudes.size or not np.isclose(
            latitudes[index], latitudes[start], atol=1e-10, rtol=0.0
        ):
            rings.append(slice(start, index))
            start = index
    return rings


def normalise_reduced_gaussian_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Convert flattened reduced-Gaussian GRIB output to an unstructured form."""

    if "values" not in ds.dims:
        return ds
    if "lon_vertices" in ds or "clon_vertices" in ds:
        return ds

    lat_name = _find_coord_name(ds, LAT_COORD_NAMES)
    lon_name = _find_coord_name(ds, LON_COORD_NAMES)
    if lat_name is None or lon_name is None:
        return ds

    lat_coord = ds[lat_name]
    lon_coord = ds[lon_name]
    if lat_coord.ndim != 1 or lon_coord.ndim != 1:
        return ds
    if lat_coord.dims != ("values",) or lon_coord.dims != ("values",):
        return ds

    ds = ds.rename({"values": "cell"})

    latitudes = np.asarray(ds[lat_name].values, dtype=np.float64)
    longitudes = np.asarray(ds[lon_name].values, dtype=np.float64)
    rings = _ring_slices(latitudes)

    ring_centres = np.asarray([latitudes[ring.start] for ring in rings], dtype=np.float64)
    ring_edges = np.empty(ring_centres.size + 1, dtype=np.float64)
    if ring_centres.size > 1:
        ring_edges[1:-1] = 0.5 * (ring_centres[:-1] + ring_centres[1:])

    ring_edges[0] = 90.0
    ring_edges[-1] = -90.0

    n_cells = ds.sizes["cell"]
    lon_vertices = np.empty((n_cells, 4), dtype=np.float64)
    lat_vertices = np.empty((n_cells, 4), dtype=np.float64)

    for ring_index, ring in enumerate(rings):
        west, east = _circular_lon_bounds(longitudes[ring])
        lat_a = ring_edges[ring_index]
        lat_b = ring_edges[ring_index + 1]
        south = min(lat_a, lat_b)
        north = max(lat_a, lat_b)

        lon_vertices[ring, 0] = west
        lon_vertices[ring, 1] = east
        lon_vertices[ring, 2] = east
        lon_vertices[ring, 3] = west

        lat_vertices[ring, 0] = south
        lat_vertices[ring, 1] = south
        lat_vertices[ring, 2] = north
        lat_vertices[ring, 3] = north

    ds = ds.assign_coords(cell=np.arange(n_cells, dtype=np.int64))
    ds["lon_vertices"] = (("cell", "nv"), lon_vertices)
    ds["lat_vertices"] = (("cell", "nv"), lat_vertices)
    for name in ds.data_vars:
        if "cell" in ds[name].dims:
            ds[name].attrs["CDI_grid_type"] = "unstructured"
    return ds


def open_source_record_dataset(
    record: SourceRecord,
    *,
    use_cache: bool,
) -> xr.Dataset:
    """Open one source record and rename the payload to the requested variable."""
    ds = open_dataset(record.files, use_cache=use_cache)
    if record.variable in ds.data_vars:
        ds_var = ds[[record.variable]]
    else:
        data_vars = get_vars(ds)
        if len(data_vars) != 1:
            raise ValueError(
                f"Expected exactly one GRIB payload variable for {record.variable!r}, "
                f"found {data_vars!r}"
            )
        ds_var = ds.rename({data_vars[0]: record.variable})[[record.variable]]

    data = ds_var[record.variable]
    if record.conversion_factor != 1.0:
        data = data * record.conversion_factor

    data.attrs = clean_output_attrs(dict(record.output_attrs))
    ds_var[record.variable] = data
    return ds_var


def merge_frequency_dataset(
    records: list[SourceRecord],
    *,
    use_cache: bool,
) -> xr.Dataset:
    """Open and merge all resolved variables for one output frequency.

    Source records are opened in parallel at the variable level, with one
    worker per resolved variable record for the current frequency. The merge
    order still follows the input record order.
    """

    resolved_records = [record for record in records if record.files]
    if not resolved_records:
        raise ValueError("No source files were resolved for this frequency.")

    if len(resolved_records) == 1:
        datasets = [
            open_source_record_dataset(resolved_records[0], use_cache=use_cache)
        ]
    else:
        total_files = sum(len(record.files) for record in resolved_records)
        max_workers = len(resolved_records)
        log_stage(
            LOGGER,
            "grib_read_parallel",
            frequency=resolved_records[0].frequency,
            record_tasks=len(resolved_records),
            total_files=total_files,
            workers=max_workers,
        )
        datasets_by_index: list[Optional[xr.Dataset]] = [None] * len(resolved_records)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Future[xr.Dataset], int] = {
                executor.submit(
                    open_source_record_dataset,
                    record,
                    use_cache=use_cache,
                ): index
                for index, record in enumerate(resolved_records)
            }
            for future, index in futures.items():
                datasets_by_index[index] = future.result()

        datasets = [dataset for dataset in datasets_by_index if dataset is not None]

    return xr.merge(
        datasets,
        compat="override",
        join="outer",
        combine_attrs="drop_conflicts",
    )


def select_time_interval(
    ds: xr.Dataset,
    interval: tuple[Optional[date], Optional[date]],
) -> xr.Dataset:
    """Restrict a dataset to the requested inclusive date interval."""

    if "time" not in ds.coords:
        return ds

    start, end = interval
    time_values = ds["time"].values
    mask = np.ones(time_values.shape, dtype=bool)

    if start is not None:
        mask &= time_values >= np.datetime64(start.isoformat())
    if end is not None:
        exclusive_end = end + timedelta(days=1)
        mask &= time_values < np.datetime64(exclusive_end.isoformat())

    return ds.isel(time=mask)
