import hashlib
import json
from collections.abc import Collection
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xarray as xr
from eccodes import codes_get, codes_grib_new_from_file, codes_release

import grid_doctor as gd
from grid_doctor.utils import cache_dir
import logging

LOGGER = logging.getLogger(__name__)
DUPLICATE_TIME_ROWS_LOG = Path(__file__).resolve().parent.parent / "duplicate_grib_time_rows.log"

GRIB_KEYS = [
    "shortName",
    "paramId",
    "typeOfLevel",
    "level",
    "dataDate",
    "dataTime",
    "stepRange",
    "startStep",
    "endStep",
    "P1",
    "P2",
    "timeRangeIndicator",
]

# common cfgrib auxiliary names
EXCLUDE_NAMES = {
    "time",
    "step",
    "valid_time",
    "latitude",
    "longitude",
    "number",
    "surface",
    "heightAboveGround",
    "isobaricInhPa",
}
SINGLETON_AUX_COORDS = {
    "number",
}
TIME_INDEX_COLUMNS = [
    "ref_time",
    "step_timedelta",
    "valid_time",
    "time_bnds_start",
    "time_bnds_end",
]
DUPLICATE_TIME_COLUMNS = [*TIME_INDEX_COLUMNS, "level"]
VERTICAL_COORD_RENAMES = {
    "isobaricInhPa": "plev",
}


def grib_inventory(files: Collection[str | Path]) -> pd.DataFrame:
    """Build a message-level inventory for one or more GRIB files.

    Parameters
    ----------
    files : iterable of path-like
        GRIB files to scan with ecCodes.

    Returns
    -------
    pandas.DataFrame
        One row per GRIB message, containing the source file, message index,
        selected ``GRIB_KEYS``, reference time, valid time, forecast step, and
        start/end time bounds derived from the GRIB step metadata.
    """

    rows: list[dict[str, Any]] = []

    for file in files:
        with open(file, "rb") as f:
            message = 0

            while True:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    break

                row = {"file": file, "message": message}
                row.update({k: codes_get(gid, k) for k in GRIB_KEYS})
                rows.append(row)

                codes_release(gid)
                message += 1

    df = pd.DataFrame(rows)

    df["ref_time"] = pd.to_datetime(
        df["dataDate"].astype(str).str.zfill(8) + df["dataTime"].astype(str).str.zfill(4),
        format="%Y%m%d%H%M",
    )

    df["valid_time"] = df["ref_time"] + pd.to_timedelta(df["endStep"], unit="h")
    df["step_timedelta"] = pd.to_timedelta(df["endStep"], unit="h")

    df["time_bnds_start"] = df["ref_time"] + pd.to_timedelta(df["startStep"], unit="h")
    df["time_bnds_end"] = df["ref_time"] + pd.to_timedelta(df["endStep"], unit="h")

    return df


def cached_grib_inventory(files: Collection[str | Path]) -> pd.DataFrame:
    """Return a cached GRIB inventory keyed by file identity and metadata.

    The cache key includes the absolute file paths, file sizes, modification
    times, and the list of GRIB keys used by :func:`grib_inventory`. Cached
    inventories are stored as pickle files under ``grid_doctor``'s cache
    directory.

    Parameters
    ----------
    files : iterable of path-like
        GRIB files to inventory.

    Returns
    -------
    pandas.DataFrame
        The inventory produced by :func:`grib_inventory`, loaded from cache
        when possible.
    """
    files = [str(Path(file).expanduser().resolve()) for file in files]

    digest = hashlib.sha256()
    digest.update(b"grib_inventory_v1")
    digest.update(json.dumps(GRIB_KEYS, sort_keys=True).encode())

    for file in sorted(files):
        path = Path(file)
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())

    pickle_file = cache_dir() / f"grib_inventory_{digest.hexdigest()}.pickle"

    if pickle_file.exists():
        return cast(pd.DataFrame, pd.read_pickle(pickle_file))

    inv = grib_inventory(files)
    inv.to_pickle(pickle_file)
    return inv


def get_vars(ds: xr.Dataset) -> list[str]:
    """Return likely geophysical data variables in a cfgrib dataset.

    Coordinate-like and auxiliary variables commonly emitted by cfgrib are
    excluded, as are scalar or tiny variables and variables without a spatial
    or flattened-cell dimension.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset opened from cfgrib.

    Returns
    -------
    list of str
        Names of variables that look like real data fields.
    """

    candidates: list[str] = []

    for name, da in ds.data_vars.items():
        if name in EXCLUDE_NAMES:
            continue
        if da.ndim == 0 or da.size < 100:
            continue
        if not any(dim in da.dims for dim in ["values", "lat", "lon", "cell"]):
            continue
        candidates.append(str(name))

    return candidates


def _record_duplicate_time_rows(sample_file: str, duplicate_count: int) -> None:
    """Append one duplicate-time-row event to the local diagnostics log."""

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        with DUPLICATE_TIME_ROWS_LOG.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{timestamp}\tduplicates={duplicate_count}\tfile={sample_file}\n"
            )
    except OSError as exc:
        LOGGER.warning(
            "Could not append duplicate GRIB time-row log %s: %s",
            DUPLICATE_TIME_ROWS_LOG,
            exc,
        )


def time_normalizer(
    ds: xr.Dataset,
    *,
    grib_time_df: pd.DataFrame,
    drop_duplicate_time_rows: bool = True,
    keep_time_bounds: bool = False,
) -> xr.Dataset:
    """Normalize cfgrib time coordinates to valid time.

    cfgrib often represents accumulated or forecast-like GRIB messages with
    separate reference ``time`` and ``step`` coordinates. This function selects
    the messages described by ``grib_time_df``, replaces the message dimension
    with the computed valid time, sorts by time, and optionally attaches
    ``time_bnds``.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing exactly one real data variable.
    grib_time_df : pandas.DataFrame
        Message inventory for the same file as ``ds``. It must contain
        ``ref_time``, ``step_timedelta``, ``valid_time``,
        ``time_bnds_start``, and ``time_bnds_end`` columns as produced by
        :func:`grib_inventory`.
    drop_duplicate_time_rows : bool, optional
        If ``True``, discard exact duplicate GRIB time rows before mapping the
        dataset onto valid time. If ``False``, raise ``ValueError`` instead.
    keep_time_bounds : bool, optional
        If ``True``, add a two-column ``time_bnds`` variable and reference it
        from the ``time`` coordinate's ``bounds`` attribute.

    Returns
    -------
    xarray.Dataset
        Dataset with valid time as the time dimension.

    Raises
    ------
    ValueError
        If ``ds`` does not contain exactly one real data variable, or when
        duplicate GRIB time rows are found while ``drop_duplicate_time_rows``
        is ``False``.
    """
    df = grib_time_df
    duplicate_time_rows = df.duplicated(subset=DUPLICATE_TIME_COLUMNS, keep="first")
    if duplicate_time_rows.any():
        duplicate_count = int(duplicate_time_rows.sum())
        sample_file = str(df.iloc[0]["file"]) if "file" in df.columns and not df.empty else "<unknown>"
        if not drop_duplicate_time_rows:
            raise ValueError(
                f"Found {duplicate_count} duplicate GRIB time row(s) while "
                f"normalizing {sample_file!r}."
            )
        LOGGER.warning(
            "Dropping %s duplicate GRIB time row(s) while normalizing %s",
            duplicate_count,
            sample_file,
        )
        _record_duplicate_time_rows(sample_file, duplicate_count)
        df = df.loc[~duplicate_time_rows].copy()
    else:
        df = df.copy()

    # Multi-level GRIB groups legitimately repeat the same message-time
    # metadata once per vertical level. Collapse those rows here so valid-time
    # normalization preserves the vertical axis instead of duplicating or
    # overwriting slices level-by-level.
    df = df.drop_duplicates(subset=TIME_INDEX_COLUMNS, keep="first").reset_index(drop=True)

    vars_ = get_vars(ds)

    if len(vars_) != 1:
        raise ValueError(f"Expected exactly one data variable, found {vars_}")

    varname = vars_[0]

    ref_times = xr.DataArray(
        df["ref_time"].values.astype("datetime64[ns]"),
        dims="message",
        name="ref_time",
    )

    steps = xr.DataArray(
        df["step_timedelta"].values.astype("timedelta64[ns]"),
        dims="message",
        name="step",
    )

    valid_times = df["valid_time"].values.astype("datetime64[ns]")

    da = ds[varname]

    indexers = {}
    if "time" in da.dims:
        indexers["time"] = ref_times
    if "step" in da.dims:
        indexers["step"] = steps

    if indexers:
        da = da.sel(indexers)

    da = da.assign_coords(time=("message", valid_times))
    da = da.swap_dims({"message": "time"})
    da = da.drop_vars(["message", "step", "valid_time"], errors="ignore")
    da = da.sortby("time")

    ds_out = da.to_dataset(name=varname)
    ds_out.attrs = ds.attrs.copy()
    ds_out[varname].attrs = ds[varname].attrs.copy()

    if keep_time_bounds:
        time_bnds = np.column_stack(
            [
                df["time_bnds_start"].values.astype("datetime64[ns]"),
                df["time_bnds_end"].values.astype("datetime64[ns]"),
            ]
        )

        order = np.argsort(valid_times)

        ds_out = ds_out.assign_coords(bnds=[0, 1])
        ds_out["time_bnds"] = (("time", "bnds"), time_bnds[order])
        ds_out["time"].attrs["bounds"] = "time_bnds"

    return ds_out


def normalise_vertical_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename known GRIB vertical coordinates to stable output names.

    Parameters
    ----------
    ds
        Dataset opened from cfgrib.

    Returns
    -------
    xarray.Dataset
        Dataset with standardized vertical coordinate names where applicable.
    """

    rename_map = {
        source: target
        for source, target in VERTICAL_COORD_RENAMES.items()
        if source in ds.dims or source in ds.coords
    }
    if not rename_map:
        return ds
    return ds.rename(rename_map)


def drop_singleton_auxiliary_coords(ds: xr.Dataset) -> xr.Dataset:
    """Drop scalar auxiliary coordinates that vary inconsistently across files.

    Some ERA5 GRIB groups expose singleton coordinates such as ``number`` in
    only a subset of files. `xarray.open_mfdataset(..., combine="by_coords")`
    then fails because those coordinates are not present everywhere even though
    they do not carry a meaningful dimension for this workflow.
    """

    drop_names = [
        name
        for name in SINGLETON_AUX_COORDS
        if name in ds.coords and name not in ds.dims and ds[name].size == 1
    ]
    if not drop_names:
        return ds
    return ds.drop_vars(drop_names)


def open_dataset(
    files: Collection[str | Path],
    *,
    use_inventory_cache: bool = True,
    use_input_cache: bool = False,
    drop_duplicate_time_rows: bool = True,
    pressure_levels: tuple[int, ...] | None = None,
) -> xr.Dataset:
    """Open GRIB files as a merged xarray dataset.

    Files are inventoried first, then opened one GRIB field group at a time
    using cfgrib filters for ``shortName`` and ``typeOfLevel`` while keeping
    compatible vertical levels together.
    Each per-variable dataset is normalized to valid time before all variables
    are merged.

    Parameters
    ----------
    files : iterable of path-like
        GRIB files to open.
    use_inventory_cache : bool, optional
        If ``True``, reuse cached GRIB inventories. If ``False``, rebuild the
        inventory directly from the source files.
    use_input_cache : bool, optional
        If ``True``, reuse cached multi-file dataset pickles through
        ``grid_doctor.cached_open_dataset``. If ``False``, open the datasets
        directly with ``xarray.open_mfdataset``.
    drop_duplicate_time_rows : bool, optional
        Whether exact duplicate GRIB time rows should be discarded during time
        normalization instead of raising an error.
    pressure_levels : tuple[int, ...] | None, optional
        Optional pressure levels to retain when opening isobaric datasets.

    Returns
    -------
    xarray.Dataset
        Merged dataset containing all discovered GRIB field groups.
    """
    files = [str(Path(file).expanduser().resolve()) for file in files]
    inv = cached_grib_inventory(files) if use_inventory_cache else grib_inventory(files)

    inv["_file_key"] = inv["file"].map(lambda file: str(Path(file).resolve()))

    group_cols = ["shortName", "paramId", "typeOfLevel"]

    datasets: list[xr.Dataset] = []

    for key, g in inv.groupby(group_cols):
        short_name, param_id, type_of_level = key
        if pressure_levels is not None and type_of_level == "isobaricInhPa":
            g = g[g["level"].isin(pressure_levels)]
            if g.empty:
                continue

        files_for_var = [str(file) for file in sorted(g["file"].unique())]
        time_by_file = {
            file: rows.drop(columns="_file_key")
            for file, rows in g.groupby("_file_key", sort=False)
        }

        def preprocess(ds: xr.Dataset) -> xr.Dataset:
            source = ds.encoding.get("source")
            if source is None:
                raise ValueError("cfgrib dataset has no source path in ds.encoding")

            source_key = str(Path(source).expanduser().resolve())
            try:
                grib_time_df = time_by_file[source_key]
            except KeyError as exc:
                raise KeyError(f"No GRIB inventory rows found for {source!r}") from exc

            return normalise_vertical_coords(
                drop_singleton_auxiliary_coords(
                    time_normalizer(
                        ds,
                        grib_time_df=grib_time_df,
                        drop_duplicate_time_rows=drop_duplicate_time_rows,
                        keep_time_bounds=False,
                    )
                )
            )

        open_kwargs: dict[str, Any] = {
            "engine": "cfgrib",
            "backend_kwargs": {
                "indexpath": "",
                "filter_by_keys": {
                    "shortName": short_name,
                    "typeOfLevel": type_of_level,
                },
            },
            "combine": "by_coords",
            "preprocess": preprocess,
        }
        if use_input_cache:
            ds_raw = gd.cached_open_dataset(
                files_for_var,
                **open_kwargs,
            )
        else:
            ds_raw = xr.open_mfdataset(
                files_for_var,
                **open_kwargs,
                parallel=True,
                chunks="auto",
            )

        if pressure_levels is not None and "plev" in ds_raw.coords:
            ds_raw = ds_raw.sel(plev=list(pressure_levels))

        datasets.append(ds_raw)

    ds_all = xr.merge(datasets, compat="override")
    return ds_all
