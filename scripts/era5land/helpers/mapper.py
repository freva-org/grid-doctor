"""GRIB to HEALPix mapping helpers for ERA5/ERA5-Land."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import xarray as xr

import grid_doctor as gd

from .file_fetcher import SourceRecord
from .formatter import destination_for_level, group_records_by_frequency
from .grib import get_vars, open_dataset


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


def _ring_slices(latitudes: np.ndarray) -> List[slice]:
    """Return contiguous slices for repeated reduced-Gaussian latitude rings."""

    rings: List[slice] = []
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
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        return ds
    if "lon_vertices" in ds or "clon_vertices" in ds:
        return ds

    lat_coord = ds["latitude"]
    lon_coord = ds["longitude"]
    if lat_coord.ndim != 1 or lon_coord.ndim != 1:
        return ds
    if lat_coord.dims != ("values",) or lon_coord.dims != ("values",):
        return ds

    ds = ds.rename({"values": "cell"})

    latitudes = np.asarray(ds["latitude"].values, dtype=np.float64)
    longitudes = np.asarray(ds["longitude"].values, dtype=np.float64)
    rings = _ring_slices(latitudes)

    ring_centres = np.asarray([latitudes[ring.start] for ring in rings], dtype=np.float64)
    ring_edges = np.empty(ring_centres.size + 1, dtype=np.float64)
    if ring_centres.size == 1:
        ring_edges[0] = max(-90.0, ring_centres[0] - 0.5)
        ring_edges[1] = min(90.0, ring_centres[0] + 0.5)
    else:
        ring_edges[1:-1] = 0.5 * (ring_centres[:-1] + ring_centres[1:])
        ring_edges[0] = ring_centres[0] + 0.5 * (ring_centres[0] - ring_centres[1])
        ring_edges[-1] = ring_centres[-1] + 0.5 * (
            ring_centres[-1] - ring_centres[-2]
        )
        ring_edges = np.clip(ring_edges, -90.0, 90.0)

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
        return ds[[record.variable]]

    data_vars = get_vars(ds)
    if len(data_vars) != 1:
        raise ValueError(
            f"Expected exactly one GRIB payload variable for {record.variable!r}, "
            f"found {data_vars!r}"
        )
    return ds.rename({data_vars[0]: record.variable})[[record.variable]]


def merge_frequency_dataset(
    records: List[SourceRecord],
    *,
    use_cache: bool,
) -> xr.Dataset:
    """Open and merge all resolved variables for one output frequency."""

    datasets = [
        open_source_record_dataset(record, use_cache=use_cache)
        for record in records
        if record.files
    ]
    if not datasets:
        raise ValueError("No source files were resolved for this frequency.")
    return xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")


def map_grib_to_healpix(
    records: List[SourceRecord],
    *,
    frequencies: Tuple[str, ...],
    time_chunk: int = 48,
    zarr_format: int = 2,
    use_cache: bool = False,
) -> None:
    """Convert resolved GRIB records to per-frequency HEALPix Zarr pyramids."""

    grouped_records = group_records_by_frequency(records)
    if not grouped_records:
        raise ValueError("No matching source files were found for conversion.")

    for frequency in frequencies:
        freq_records = grouped_records.get(frequency, [])
        if not freq_records:
            continue

        ds = merge_frequency_dataset(freq_records, use_cache=use_cache)
        ds = normalise_reduced_gaussian_dataset(ds)
        if time_chunk and "time" in ds.dims:
            ds = ds.chunk({"time": time_chunk})

        pyramid = gd.latlon_to_healpix_pyramid(
            ds,
        )
        for zoom_number, dataset in pyramid.items():
            destination = destination_for_level(frequency, zoom_number)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            dataset.to_zarr(
                destination,
                mode="w",
                zarr_format=zarr_format,
                consolidated=(zarr_format == 2),
            )
