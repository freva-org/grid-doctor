"""Dataset opening and reshaping helpers for ERA5/ERA5-Land."""

import hashlib
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import xarray as xr
from grid_doctor.utils import cache_dir

from .file_fetcher import SourceRecord
from .grib import get_vars, open_dataset
from .metadata import clean_output_attrs

LOGGER = logging.getLogger(__name__)
TIME_ALIGNMENT_MISMATCH_LOG = Path.cwd() / "time_alignment_mismatches.log"
LAT_COORD_NAMES = ("latitude", "lat", "Latitude", "LATITUDE", "y", "Y")
LON_COORD_NAMES = ("longitude", "lon", "Longitude", "LONGITUDE", "x", "X")
STATIC_COORD_NAMES = ("cell", "time", "crs", "surface")
_REDUCED_GAUSSIAN_GEOMETRY_CACHE: dict[str, dict[str, np.ndarray]] = {}


class EmptySourceDataError(ValueError):
    """Raised when a resolved source contains no usable payload data."""


def _find_coord_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    """Return the first matching coordinate name from *candidates*."""

    for name in candidates:
        if name in ds.coords:
            return name
    return None


def normalise_published_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Keep published coord/data-variable classification stable across writes."""

    coord_names: set[str] = {str(name) for name in ds.coords}
    coord_names.update(name for name in (*STATIC_COORD_NAMES, *LAT_COORD_NAMES, *LON_COORD_NAMES) if name in ds)
    if not coord_names:
        return ds
    return ds.set_coords(sorted(coord_names))


def _format_timestamp_dates(values: pd.Index) -> str:
    """Summarize a timestamp index as sorted ISO calendar dates."""

    if len(values) == 0:
        return "-"
    dates = pd.Index(pd.to_datetime(values).strftime("%Y-%m-%d")).unique()
    return ",".join(str(value) for value in dates.tolist())


def _record_time_alignment_mismatch(
    records: list[SourceRecord],
    datasets: list[xr.Dataset],
) -> None:
    """Log non-shared per-variable timestamps for trimming diagnostics."""

    timed_records = [(record, ds) for record, ds in zip(records, datasets) if "time" in ds.indexes]
    if len(timed_records) < 2:
        return

    shared_time = timed_records[0][1].indexes["time"]
    for _record, ds in timed_records[1:]:
        shared_time = shared_time.intersection(ds.indexes["time"])

    if all(ds.indexes["time"].equals(shared_time) for _record, ds in timed_records):
        return

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    lines: list[str] = []
    for record, ds in timed_records:
        non_shared = ds.indexes["time"].difference(shared_time)
        if len(non_shared) == 0:
            continue
        lines.append(
            "\t".join(
                [
                    timestamp,
                    f"frequency={record.frequency}",
                    f"variable={record.variable}",
                    f"count={len(non_shared)}",
                    f"dates={_format_timestamp_dates(non_shared)}",
                    f"first={non_shared.min()}",
                    f"last={non_shared.max()}",
                ]
            )
        )

    if not lines:
        return

    try:
        with TIME_ALIGNMENT_MISMATCH_LOG.open("a", encoding="utf-8") as handle:
            for line in lines:
                handle.write(f"{line}\n")
    except OSError as exc:
        LOGGER.warning(
            "Could not append time-alignment mismatch log %s: %s",
            TIME_ALIGNMENT_MISMATCH_LOG,
            exc,
        )

    for record, ds in timed_records:
        non_shared = ds.indexes["time"].difference(shared_time)
        if len(non_shared) == 0:
            continue
        LOGGER.warning(
            "Non-shared %s timestamps for %s %s involve date(s): %s",
            len(non_shared),
            record.frequency,
            record.variable,
            _format_timestamp_dates(non_shared),
        )


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
        if index == latitudes.size or not np.isclose(latitudes[index], latitudes[start], atol=1e-10, rtol=0.0):
            rings.append(slice(start, index))
            start = index
    return rings


def _reduced_gaussian_geometry_cache_key(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> str:
    """Return a stable cache key for one reduced-Gaussian horizontal grid."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(latitudes).tobytes())
    digest.update(np.ascontiguousarray(longitudes).tobytes())
    digest.update(f"n={latitudes.size};version=1".encode())
    return digest.hexdigest()


def _reduced_gaussian_geometry_cache_path(cache_key: str) -> Path:
    """Return the shared cache path for one reduced-Gaussian geometry payload."""

    return cache_dir() / f"reduced_gaussian_geometry_{cache_key}.npz"


def _compute_reduced_gaussian_geometry(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute reduced-Gaussian cell-vertex geometry for one horizontal grid."""

    rings = _ring_slices(latitudes)
    ring_centres = np.asarray([latitudes[ring.start] for ring in rings], dtype=np.float64)
    ring_edges = np.empty(ring_centres.size + 1, dtype=np.float64)
    if ring_centres.size > 1:
        ring_edges[1:-1] = 0.5 * (ring_centres[:-1] + ring_centres[1:])

    ring_edges[0] = 90.0
    ring_edges[-1] = -90.0

    n_cells = latitudes.size
    lon_vertices: np.ndarray = np.empty((n_cells, 4), dtype=np.float64)
    lat_vertices: np.ndarray = np.empty((n_cells, 4), dtype=np.float64)

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

    return {
        "lat_vertices": lat_vertices,
        "lon_vertices": lon_vertices,
    }


def _load_cached_reduced_gaussian_geometry(
    cache_key: str,
) -> dict[str, np.ndarray] | None:
    """Load cached reduced-Gaussian geometry from memory or shared disk cache."""

    geometry = _REDUCED_GAUSSIAN_GEOMETRY_CACHE.get(cache_key)
    if geometry is not None:
        return geometry

    cache_path = _reduced_gaussian_geometry_cache_path(cache_key)
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            geometry = {
                "lat_vertices": payload["lat_vertices"],
                "lon_vertices": payload["lon_vertices"],
            }
    except (BadZipFile, EOFError, KeyError, OSError, ValueError) as exc:
        LOGGER.warning(
            "Could not read cached reduced-Gaussian geometry %s: %s",
            cache_path,
            exc,
        )
        cache_path.unlink(missing_ok=True)
        return None

    _REDUCED_GAUSSIAN_GEOMETRY_CACHE[cache_key] = geometry
    LOGGER.info("Using cached reduced-Gaussian geometry %s", cache_path)
    return geometry


def _store_reduced_gaussian_geometry(
    cache_key: str,
    geometry: dict[str, np.ndarray],
) -> None:
    """Persist one reduced-Gaussian geometry payload for reuse by later runs."""

    cache_path = _reduced_gaussian_geometry_cache_path(cache_key)
    temp_path = cache_path.with_name(f"{cache_path.name}.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                lat_vertices=geometry["lat_vertices"],
                lon_vertices=geometry["lon_vertices"],
            )
        temp_path.replace(cache_path)
    except OSError as exc:
        LOGGER.warning(
            "Could not write reduced-Gaussian geometry cache %s: %s",
            cache_path,
            exc,
        )
        temp_path.unlink(missing_ok=True)


def _reduced_gaussian_geometry(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    *,
    use_cache: bool,
) -> dict[str, np.ndarray]:
    """Return reduced-Gaussian geometry, regenerating it if cache entries vanish."""

    cache_key = _reduced_gaussian_geometry_cache_key(latitudes, longitudes)
    if use_cache:
        geometry = _load_cached_reduced_gaussian_geometry(cache_key)
        if geometry is not None:
            return geometry

    geometry = _compute_reduced_gaussian_geometry(latitudes, longitudes)
    _REDUCED_GAUSSIAN_GEOMETRY_CACHE[cache_key] = geometry
    if use_cache:
        LOGGER.info(
            "Caching reduced-Gaussian geometry in %s",
            _reduced_gaussian_geometry_cache_path(cache_key),
        )
        _store_reduced_gaussian_geometry(cache_key, geometry)
    return geometry


def normalise_reduced_gaussian_dataset(
    ds: xr.Dataset,
    *,
    use_cache: bool = True,
) -> xr.Dataset:
    """Convert flattened reduced-Gaussian GRIB output to an unstructured form.

    Parameters
    ----------
    ds
        Dataset that may contain a flattened reduced-Gaussian `values` axis.
    use_cache
        Reuse shared reduced-Gaussian geometry caches when available.

    Returns
    -------
    xarray.Dataset
        Dataset with an unstructured `cell` axis and HEALPix-ready vertices.
    """

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
    geometry = _reduced_gaussian_geometry(
        latitudes,
        longitudes,
        use_cache=use_cache,
    )

    ds = ds.assign_coords(cell=np.arange(ds.sizes["cell"], dtype=np.int64))
    ds["lon_vertices"] = (("cell", "nv"), geometry["lon_vertices"])
    ds["lat_vertices"] = (("cell", "nv"), geometry["lat_vertices"])
    for name in ds.data_vars:
        if "cell" in ds[name].dims:
            ds[name].attrs["CDI_grid_type"] = "unstructured"
    return ds


def validate_one_value_per_day(
    ds: xr.Dataset,
    variable: str,
) -> None:
    """Ensure that a daily variable has at most one value per calendar day."""
    if "time" not in ds.coords or ds.sizes.get("time", 0) < 2:
        return

    days = ds["time"].dt.floor("D").values

    # The GRIB time axis is expected to be ordered, so checking adjacent
    # normalized days is sufficient and avoids the sorting done by np.unique.
    duplicate_positions = np.flatnonzero(days[1:] == days[:-1])
    if duplicate_positions.size:
        index = int(duplicate_positions[0])
        raise ValueError(
            f"{variable!r} contains multiple values for calendar day "
            f"{days[index]!s}: {ds['time'].values[index]!s} and "
            f"{ds['time'].values[index + 1]!s}."
        )


def normalise_time_for_frequency(
    ds: xr.Dataset,
    frequency: str,
) -> xr.Dataset:
    """Assign canonical representative times to aggregated data."""
    if "time" not in ds.coords:
        return ds

    if frequency == "day":
        return ds.assign_coords(time=ds["time"].dt.floor("D") + np.timedelta64(12, "h"))

    if frequency == "mon":
        month_start = ds["time"].astype("datetime64[M]").astype("datetime64[ns]")
        return ds.assign_coords(time=month_start + np.timedelta64(12, "h"))

    return ds


def open_record_dataset(
    record: SourceRecord,
    *,
    interval: tuple[date | None, date | None] = (None, None),
    use_inventory_cache: bool = True,
    use_input_cache: bool = False,
    drop_duplicate_time_rows: bool = True,
    pressure_levels: tuple[int, ...] | None = None,
) -> xr.Dataset:
    """Open one source record, optionally slice it, and rename its payload.

    Parameters
    ----------
    record
        Source metadata describing the files and output variable name.
    interval
        Inclusive start/end date bounds used to trim the dataset immediately
        after opening. ``None`` leaves the corresponding side unbounded.
    use_inventory_cache
        Whether to reuse cached GRIB inventories for this record.
    use_input_cache
        Whether to reuse cached multi-file input datasets for this record.
    drop_duplicate_time_rows
        Whether exact duplicate GRIB time rows should be discarded during time
        normalization instead of raising an error.
    pressure_levels
        Optional pressure levels to retain for pressure-level variables.

    Returns
    -------
    xarray.Dataset
        Dataset containing only the requested output variable.
    """
    ds = open_dataset(
        record.files,
        use_inventory_cache=use_inventory_cache,
        use_input_cache=use_input_cache,
        drop_duplicate_time_rows=drop_duplicate_time_rows,
        pressure_levels=pressure_levels,
    )

    if record.frequency == "day":
        validate_one_value_per_day(ds, record.variable)

    if record.frequency in {"day", "mon"}:
        ds = normalise_time_for_frequency(ds, record.frequency)

    ds = select_time_interval(ds, interval)

    if record.variable in ds.data_vars:
        ds_var = ds[[record.variable]]
    else:
        data_vars = get_vars(ds)

        if not data_vars:
            raise EmptySourceDataError(
                f"No GRIB payload data found for {record.variable!r} "
                f"at frequency {record.frequency!r} in the requested interval."
            )

        if len(data_vars) != 1:
            raise ValueError(f"Expected exactly one GRIB payload variable for {record.variable!r}, found {data_vars!r}")
        ds_var = ds.rename({data_vars[0]: record.variable})[[record.variable]]

    data = ds_var[record.variable]
    if record.conversion_factor != 1.0:
        data = data * record.conversion_factor

    data.attrs = clean_output_attrs(dict(record.output_attrs))
    ds_var[record.variable] = data
    return ds_var


def _align_datasets_on_shared_time(
    records: list[SourceRecord],
    datasets: list[xr.Dataset],
) -> list[xr.Dataset]:
    """Trim per-variable datasets to the timestamps shared by every variable.

    Parameters
    ----------
    records
        Source records paired positionally with ``datasets``.
    datasets
        Datasets opened for each source record.

    Returns
    -------
    list[xarray.Dataset]
        Datasets whose ``time`` indexes match exactly wherever a time axis is
        present.

    Raises
    ------
    ValueError
        If the datasets do not share any timestamps after alignment.
    """
    time_indexes: list[pd.Index] = [ds.indexes["time"] for ds in datasets if "time" in ds.indexes]
    if len(time_indexes) < 2:
        return datasets

    common_time = time_indexes[0]
    for time_index in time_indexes[1:]:
        common_time = common_time.intersection(time_index)

    if common_time.empty:
        variable_names = ",".join(record.variable for record in records)
        raise ValueError(
            f"Resolved variables do not share any timestamps after GRIB time normalization: {variable_names}."
        )

    _record_time_alignment_mismatch(records, datasets)

    aligned_datasets: list[xr.Dataset] = []
    for record, ds in zip(records, datasets):
        if "time" not in ds.indexes:
            aligned_datasets.append(ds)
            continue

        time_index = ds.indexes["time"]
        if time_index.equals(common_time):
            aligned_datasets.append(ds)
            continue

        dropped_count = len(time_index.difference(common_time))
        dropped_dates = _format_timestamp_dates(time_index.difference(common_time))
        LOGGER.warning(
            "Trimming %s timestamp(s) from %s %s data so variables share a common time axis; affected date(s): %s.",
            dropped_count,
            record.frequency,
            record.variable,
            dropped_dates,
        )
        aligned_datasets.append(ds.sel(time=common_time))

    return aligned_datasets


def merge_frequency_dataset(
    records: list[SourceRecord],
    *,
    use_inventory_cache: bool = True,
    use_input_cache: bool = False,
    drop_duplicate_time_rows: bool = True,
    interval: tuple[date | None, date | None] = (None, None),
    pressure_levels: tuple[int, ...] | None = None,
) -> xr.Dataset:
    """Open and merge all resolved variables for one output frequency.

    Parameters
    ----------
    records
        Resolved source records for a single output frequency.
    use_inventory_cache
        Whether to reuse cached GRIB inventories while opening records.
    use_input_cache
        Whether to reuse cached multi-file GRIB input datasets while opening
        records.
    drop_duplicate_time_rows
        Whether exact duplicate GRIB time rows should be discarded during time
        normalization instead of raising an error.
    interval
        Inclusive start/end date bounds applied to each per-record dataset
        immediately after it is opened.
    pressure_levels
        Optional pressure levels to retain for pressure-level variables.

    Returns
    -------
    xarray.Dataset
        Merged dataset containing all variables for the requested frequency.
    """
    resolved_records = [record for record in records if record.files]
    if not resolved_records:
        raise ValueError("No source files were resolved for this frequency.")

    datasets = []
    for record in resolved_records:
        ds = open_record_dataset(
            record,
            interval=interval,
            use_inventory_cache=use_inventory_cache,
            use_input_cache=use_input_cache,
            drop_duplicate_time_rows=drop_duplicate_time_rows,
            pressure_levels=pressure_levels,
        )
        datasets.append(ds)

    for record, ds in zip(resolved_records, datasets):
        if "time" not in ds.indexes:
            continue
        if ds.indexes["time"].has_duplicates:
            raise ValueError(f"{record.variable!r} contains duplicate timestamps after frequency normalization.")

    datasets = _align_datasets_on_shared_time(resolved_records, datasets)

    return xr.merge(
        datasets,
        compat="override",
        join="exact",
        combine_attrs="drop_conflicts",
    )


def select_time_interval(
    ds: xr.Dataset,
    interval: tuple[date | None, date | None],
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
