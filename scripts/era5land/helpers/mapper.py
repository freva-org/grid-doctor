"""GRIB to HEALPix mapping helpers for ERA5/ERA5-Land."""

from datetime import date, timedelta
from pathlib import Path
import shutil
import uuid
from typing import Any, List, Optional, Tuple

import numpy as np
import xarray as xr

import grid_doctor as gd

from .file_fetcher import SourceRecord
from .formatter import destination_for_level, group_records_by_frequency
from .grib import get_vars, open_dataset

LAT_COORD_NAMES = ("latitude", "lat", "Latitude", "LATITUDE", "y", "Y")
LON_COORD_NAMES = ("longitude", "lon", "Longitude", "LONGITUDE", "x", "X")


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> Optional[str]:
    """Return the first matching coordinate name from *candidates*."""

    for name in candidates:
        if name in ds.coords:
            return name
    return None


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

    # Reduced Gaussian latitude rings cover the full sphere, so the
    # outermost cells should close exactly at the poles.
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


def _select_time_interval(
    ds: xr.Dataset,
    interval: Tuple[Optional[date], Optional[date]],
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


def _fill_value_for_dtype(dtype: np.dtype[Any]) -> Any:
    """Return a sensible missing value for a dtype."""

    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).min
    return None


def _make_fill_variable(template: xr.DataArray, name: str, time_size: int) -> xr.DataArray:
    """Create a placeholder variable for appended time slices."""

    if "time" not in template.dims:
        values = np.asarray(template.values)
        coords = template.coords
    else:
        values = np.asarray(template.isel(time=slice(0, time_size)).values)
        coords = template.isel(time=slice(0, time_size)).coords

    fill = _fill_value_for_dtype(template.dtype)
    data = np.full(values.shape, fill, dtype=template.dtype)
    array = xr.DataArray(data, dims=template.dims, coords=coords, attrs=template.attrs)
    array.name = name
    return array


def _pad_missing_existing_vars_for_append(
    candidate: xr.Dataset,
    existing: xr.Dataset,
) -> xr.Dataset:
    """Pad appended time slices with placeholder arrays for existing variables."""

    padded = candidate.copy()
    time_size = padded.sizes.get("time", 0)
    for name, data in existing.data_vars.items():
        if name not in padded.data_vars:
            padded[name] = _make_fill_variable(data, name, time_size)

    ordered = [name for name in existing.data_vars if name in padded.data_vars]
    ordered.extend(name for name in padded.data_vars if name not in ordered)
    return padded[ordered]


def _write_dataset(
    dataset: xr.Dataset,
    destination: str,
    *,
    mode: str,
    zarr_format: int,
    append_dim: Optional[str] = None,
) -> None:
    """Write one dataset to a Zarr store with consistent options."""

    options: dict[str, Any] = {
        "mode": mode,
        "zarr_format": zarr_format,
        "consolidated": (zarr_format == 2),
    }
    if append_dim is not None:
        options["append_dim"] = append_dim
    dataset.to_zarr(destination, **options)


def _rewrite_dataset_via_temp(
    dataset: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> None:
    """Rewrite a store via a temporary path to avoid reading and writing it in place."""

    destination_path = Path(destination)
    temp_path = destination_path.parent / (
        f".{destination_path.name}.tmp-{uuid.uuid4().hex}"
    )

    try:
        _write_dataset(
            dataset,
            str(temp_path),
            mode="w",
            zarr_format=zarr_format,
        )
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.move(str(temp_path), str(destination_path))
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


def _merge_time_updates(existing: xr.Dataset, candidate: xr.Dataset) -> xr.Dataset:
    """Merge a candidate dataset into an existing one, preferring candidate values."""

    merged = candidate.combine_first(existing)
    if "time" in merged.coords:
        merged = merged.sortby("time")
    return merged


def _contiguous_slices(indices: np.ndarray) -> List[slice]:
    """Convert sorted integer indices into contiguous slices."""

    if indices.size == 0:
        return []

    slices: List[slice] = []
    start = int(indices[0])
    previous = start
    for value in indices[1:]:
        current = int(value)
        if current != previous + 1:
            slices.append(slice(start, previous + 1))
            start = current
        previous = current
    slices.append(slice(start, previous + 1))
    return slices


def _can_append_new_times(existing: xr.Dataset, candidate: xr.Dataset) -> bool:
    """Return True when candidate times can be appended at the end of existing time."""

    if "time" not in existing.dims or "time" not in candidate.dims:
        return False
    if existing.sizes.get("time", 0) == 0 or candidate.sizes.get("time", 0) == 0:
        return False

    existing_times = existing.indexes["time"]
    candidate_times = candidate.indexes["time"]
    overlap = candidate_times.intersection(existing_times)
    if len(overlap) > 0:
        return False

    try:
        return bool(candidate_times.min() > existing_times.max())
    except TypeError:
        return False


def _write_missing_variables(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> xr.Dataset:
    """Add variables missing from an existing store across the current time axis."""

    missing = [name for name in candidate.data_vars if name not in existing.data_vars]
    if not missing:
        return existing

    add_ds = candidate[missing].reindex(time=existing["time"].values)
    _write_dataset(add_ds, destination, mode="a", zarr_format=zarr_format)
    return xr.merge(
        [existing, add_ds],
        compat="override",
        combine_attrs="drop_conflicts",
    )


def _append_new_times(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> xr.Dataset:
    """Append strictly newer time slices to an existing store."""

    existing_time_strings = set(map(str, existing["time"].values))
    new_time_values = [
        value for value in candidate["time"].values if str(value) not in existing_time_strings
    ]
    if not new_time_values:
        return existing

    append_ds = _pad_missing_existing_vars_for_append(
        candidate.sel(time=new_time_values),
        existing,
    )
    _write_dataset(
        append_ds,
        destination,
        mode="a",
        append_dim="time",
        zarr_format=zarr_format,
    )
    return xr.concat([existing, append_ds], dim="time")


def _rewrite_overlapping_times(
    existing: xr.Dataset,
    candidate: xr.Dataset,
    destination: str,
    *,
    zarr_format: int,
) -> None:
    """Rewrite only the overlapping time regions in an existing store."""

    overlap = candidate.indexes["time"].intersection(existing.indexes["time"])
    if len(overlap) == 0:
        return

    time_positions = existing.indexes["time"].get_indexer(overlap)
    time_slices = _contiguous_slices(np.asarray(time_positions, dtype=np.int64))
    for time_slice in time_slices:
        region_times = existing["time"].values[time_slice]
        region_ds = candidate.sel(time=region_times)
        to_drop = {
            name
            for name, var in region_ds.variables.items()
            if "time" not in var.dims
        }
        region_ds = region_ds.drop_vars(to_drop, errors="ignore")
        region = {"time": time_slice}
        region_ds.to_zarr(
            destination,
            mode="r+",
            region=region,
            zarr_format=zarr_format,
            consolidated=(zarr_format == 2),
        )


def _update_zarr_store(
    dataset: xr.Dataset,
    destination: str,
    *,
    clean: bool,
    zarr_format: int,
) -> None:
    """Incrementally update or recreate one destination Zarr store."""

    path = Path(destination)
    if clean or not path.exists():
        _write_dataset(dataset, destination, mode="w", zarr_format=zarr_format)
        return

    existing = xr.open_zarr(destination, consolidated=(zarr_format == 2))

    if "time" not in dataset.dims or "time" not in existing.dims:
        _write_dataset(dataset, destination, mode="a", zarr_format=zarr_format)
        return

    existing = _write_missing_variables(
        existing,
        dataset,
        destination,
        zarr_format=zarr_format,
    )

    _rewrite_overlapping_times(
        existing,
        dataset,
        destination,
        zarr_format=zarr_format,
    )

    candidate_times = dataset.indexes["time"]
    existing_times = existing.indexes["time"]
    new_times = candidate_times.difference(existing_times)
    if len(new_times) == 0:
        return

    appendable_candidate = dataset.sel(time=new_times.values)
    if _can_append_new_times(existing, appendable_candidate):
        _append_new_times(
            existing,
            appendable_candidate,
            destination,
            zarr_format=zarr_format,
        )
        return

    merged = _merge_time_updates(existing, dataset)
    _rewrite_dataset_via_temp(
        merged,
        destination,
        zarr_format=zarr_format,
    )


def map_grib_to_healpix(
    records: List[SourceRecord],
    *,
    frequencies: Tuple[str, ...],
    interval: Tuple[Optional[date], Optional[date]] = (None, None),
    time_chunk: int = 48,
    zarr_format: int = 2,
    use_cache: bool = False,
    weights_dir: Optional[str] = None,
    clean: bool = False,
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
        ds = _select_time_interval(ds, interval)
        if "time" in ds.dims and ds.sizes.get("time", 0) == 0:
            continue
        ds = normalise_reduced_gaussian_dataset(ds)
        if "cell" in ds.dims:
            ds = ds.chunk({"cell": -1})
        if time_chunk and "time" in ds.dims:
            ds = ds.chunk({"time": time_chunk})

        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
        weight_file = gd.cached_weights(
            ds,
            level=max_level,
            cache_path=weights_dir,
        )
        pyramid = gd.latlon_to_healpix_pyramid(
            ds,
            max_level=max_level,
            weights_path=weight_file,
        )
        for zoom_number, dataset in pyramid.items():
            destination = destination_for_level(frequency, zoom_number)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            _update_zarr_store(
                dataset,
                destination,
                clean=clean,
                zarr_format=zarr_format,
            )
