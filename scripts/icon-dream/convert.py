"""Convert ICON-DREAM GRIB data to HEALPix Zarr without cross-worker Dask.

This script is designed around a three-stage workflow that maps naturally to
Slurm array jobs:

1. ``plan`` builds a manifest of source files that should be processed.
2. ``prepare`` downloads the raw files in parallel and computes the HEALPix
   weights once.
3. ``worker`` converts exactly one manifest entry to temporary HEALPix Zarr
   stores.
4. ``finalize`` merges the temporary stores into the final S3 target and only
   appends time steps that are not already present.

The same script can also be executed outside Slurm with the ``run`` subcommand,
which performs the full workflow locally.
"""

import argparse
import concurrent.futures
import dataclasses
import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Dict, Iterator, List, Literal,
                    Optional, Sequence, Tuple, Union)
from urllib.parse import urlsplit

import numpy as np
import s3fs

if TYPE_CHECKING:
    import xarray as xr

import grid_doctor as gd
import grid_doctor.cli as gd_cli

LOGGER = logging.getLogger(__name__)
UTC = timezone.utc

DATE_TOKEN_RE = re.compile(r"_(\d{6}|\d{8})_")
DEFAULT_SOURCE_ROOT = (
    "https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-Global"
)
DEFAULT_GRID_URL = f"{DEFAULT_SOURCE_ROOT}/invariant/ICON-DREAM-Global_grid.nc"
DEFAULT_INVARIANT_URL = (
    f"{DEFAULT_SOURCE_ROOT}/invariant/ICON-DREAM-Global_constant_fields.grb"
)
TIME_FREQUENCY = Literal["hourly", "daily", "monthly", "fx"]
OUTPUT_MODE = Literal["temp", "direct"]

ICON_DREAM_VARIABLES: Tuple[str, ...] = (
    "aswdifd_s",
    "aswdir_s",
    "clct",
    "den",
    "p",
    "pmsl",
    "ps",
    "qv",
    "qv_s",
    "t",
    "td_2m",
    "tke",
    "tmax_2m",
    "tmin_2m",
    "tot_prec",
    "t_2m",
    "u",
    "u_10m",
    "v",
    "vmax_10m",
    "v_10m",
    "ws",
    "ws_10m",
    "z0",
)


class HrefParser(HTMLParser):
    """Collect href targets ending in a specific suffix.

    Parameters
    ----------
    suffix : str, optional
        File suffix to keep.
    """

    def __init__(self, suffix: str = ".grb") -> None:
        super().__init__()
        self.suffix = suffix
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        """Inspect HTML start tags.

        Parameters
        ----------
        tag : str
            HTML tag name.
        attrs : list[tuple[str, str | None]]
            Tag attributes.
        """
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value and value.endswith(self.suffix):
                self.hrefs.append(value)


@dataclass
class SourceFile:
    """Metadata for one source file to process.

    Parameters
    ----------
    item_index : int
        Sequential index used by array workers.
    variable : str
        ICON-DREAM variable name.
    frequency : str
        Data frequency.
    url : str
        Source URL.
    filename : str
        File name derived from the URL.
    relative_path : str
        File path relative to the run directory.
    date_token : str | None, optional
        Date token extracted from the file name.
    period_start : str | None, optional
        ISO 8601 start of the coarse file coverage.
    period_end : str | None, optional
        ISO 8601 exclusive end of the coarse file coverage.
    """

    item_index: int
    variable: str
    frequency: str
    url: str
    filename: str
    relative_path: str
    date_token: Optional[str] = None
    period_start: Optional[str] = None
    period_end: Optional[str] = None


@dataclass
class RunManifest:
    """Serializable run manifest.

    Parameters
    ----------
    run_dir : str
        Shared working directory for the whole run.
    target_root : str
        Final S3 prefix excluding the level suffix.
    frequency : str
        Data frequency.
    variables : list[str]
        Requested variables.
    requested_time : tuple[str, str]
        User supplied time interval.
    created_at : str
        Manifest creation timestamp.
    grid_url : str
        URL to the ICON-DREAM grid file.
    invariant_url : str
        URL to the invariant file.
    grid_path : str
        Path of the downloaded grid file.
    weights_path : str
        Path of the cached weights file.
    temp_root : str
        Root directory for worker output.
    raw_root : str
        Root directory for downloaded raw files.
    source_items : list[SourceFile]
        Files to process.
    source_engine : str
        Xarray backend engine used for source files.
    source_backend_kwargs : dict[str, Any]
        Backend kwargs forwarded to xarray.
    max_level : int | None, optional
        Highest HEALPix level.
    existing_max_time : str | None, optional
        Maximum time already present in the target store.
    output_mode : str, optional
        Worker write mode.
    """

    run_dir: str
    target_root: str
    frequency: str
    variables: List[str]
    requested_time: Tuple[str, str]
    created_at: str
    grid_url: str
    invariant_url: str
    grid_path: str
    weights_path: str
    temp_root: str
    raw_root: str
    source_items: List[SourceFile]
    source_engine: str
    source_backend_kwargs: Dict[str, Any]
    max_level: Optional[int] = None
    existing_max_time: Optional[str] = None
    output_mode: str = "temp"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunPaths:
    """Convenience container for run directories."""

    run_dir: Path
    raw_root: Path
    temp_root: Path
    metadata_root: Path
    logs_root: Path
    manifest_path: Path
    grid_path: Path
    weights_path: Path


@dataclass
class ExistingTargetInfo:
    """Summary of an already existing target store."""

    has_store: bool
    time_values: Optional[List[str]] = None
    max_time: Optional[str] = None


@dataclass
class WorkerResult:
    """Summary of one worker output."""

    item_index: int
    variable: str
    level_paths: Dict[int, str]
    time_count: int
    time_start: Optional[str]
    time_end: Optional[str]
    has_time: bool


def _parse_datetime(value: str) -> datetime:
    """Parse a user-provided datetime string.

    Parameters
    ----------
    value : str
        Input string.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.

    Raises
    ------
    ValueError
        Raised when parsing fails.
    """
    normalized = value.strip()
    if normalized.lower() == "now":
        return datetime.now(tz=UTC)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        normalized = normalized + "T00:00:00+00:00"
    else:
        normalized = normalized.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"Could not parse datetime string: {value!r}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _isoformat_utc(value: datetime) -> str:
    """Convert a datetime to ISO 8601 in UTC."""
    return value.astimezone(UTC).isoformat()


def _default_run_dir() -> Path:
    """Return a default shared run directory."""
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return gd_cli.get_scratch("grid-doctor", f"icon-dream-healpix-{stamp}")


def _build_paths(run_dir: Union[str, Path]) -> RunPaths:
    """Build the directory layout for one run."""
    root = Path(run_dir).expanduser().resolve()
    raw_root = root / "raw"
    temp_root = root / "tmp-zarr"
    metadata_root = root / "metadata"
    logs_root = root / "logs"
    manifest_path = root / "manifest.json"
    grid_path = root / "cache" / "ICON-DREAM-Global_grid.nc"
    weights_path = root / "cache" / "weights.nc"
    for path in (raw_root, temp_root, metadata_root, logs_root, grid_path.parent):
        path.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_dir=root,
        raw_root=raw_root,
        temp_root=temp_root,
        metadata_root=metadata_root,
        logs_root=logs_root,
        manifest_path=manifest_path,
        grid_path=grid_path,
        weights_path=weights_path,
    )


def _manifest_to_dict(manifest: RunManifest) -> Dict[str, Any]:
    """Convert a manifest dataclass to a JSON-serializable dictionary."""
    return dataclasses.asdict(manifest)


def _manifest_from_dict(data: Dict[str, Any]) -> RunManifest:
    """Create a manifest dataclass from a dictionary."""
    items = [SourceFile(**item) for item in data.pop("source_items")]
    return RunManifest(source_items=items, **data)


def save_manifest(manifest: RunManifest, path: Union[str, Path]) -> None:
    """Write the manifest to disk."""
    target = Path(path)
    target.write_text(json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True))


def load_manifest(path: Union[str, Path]) -> RunManifest:
    """Load a run manifest from disk."""
    payload = json.loads(Path(path).read_text())
    return _manifest_from_dict(payload)


class IconDreamSource:
    """Discover source files on the DWD open data server.

    Parameters
    ----------
    variables : sequence[str]
        Variables to process.
    frequency : {"hourly", "daily", "monthly", "fx"}
        Data frequency.
    requested_time : tuple[str, str]
        User supplied start and end time.
    source_root : str, optional
        Root URL of the DWD dataset.
    grid_url : str, optional
        Grid file URL.
    invariant_url : str, optional
        Invariant file URL.
    """

    def __init__(
        self,
        variables: Sequence[str],
        frequency: TIME_FREQUENCY,
        requested_time: Tuple[str, str],
        source_root: str = DEFAULT_SOURCE_ROOT,
        grid_url: str = DEFAULT_GRID_URL,
        invariant_url: str = DEFAULT_INVARIANT_URL,
    ) -> None:
        self.variables = list(variables)
        self.frequency = frequency
        self.requested_time = requested_time
        self.source_root = source_root.rstrip("/")
        self.grid_url = grid_url
        self.invariant_url = invariant_url
        self.start_time = _parse_datetime(requested_time[0])
        self.end_time = _parse_datetime(requested_time[1])

    def _directory_url(self, variable: str) -> str:
        """Return the DWD directory URL for one variable."""
        return (
            f"{self.source_root}/{self.frequency}/{variable.upper().replace('-', '_')}"
        )

    def _period_from_token(self, token: str) -> Tuple[datetime, datetime]:
        """Infer a coarse file coverage interval from a file name token.

        Parameters
        ----------
        token : str
            Either ``YYYYMM`` or ``YYYYMMDD``.

        Returns
        -------
        tuple[datetime, datetime]
            Start and exclusive end of the period.
        """
        if len(token) == 6:
            start = datetime.strptime(token, "%Y%m").replace(tzinfo=UTC)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
            return start, end
        if len(token) == 8:
            start = datetime.strptime(token, "%Y%m%d").replace(tzinfo=UTC)
            return start, start + timedelta(days=1)
        raise ValueError(f"Unsupported date token: {token!r}")

    def _extract_token(self, href: str) -> Optional[str]:
        """Extract a coarse date token from a file name."""
        match = DATE_TOKEN_RE.search(href)
        if match is None:
            return None
        return match.group(1)

    def _should_keep(
        self, token: Optional[str], existing_max_time: Optional[datetime]
    ) -> bool:
        """Decide whether a coarse file period can still contain new data."""
        if token is None:
            return True
        period_start, period_end = self._period_from_token(token)
        if period_end <= self.start_time or period_start > self.end_time:
            return False
        if existing_max_time is not None and period_end <= existing_max_time:
            return False
        return True

    def _list_variable_urls(
        self,
        variable: str,
        existing_max_time: Optional[datetime],
    ) -> Iterator[SourceFile]:
        """Yield matching source files for one variable."""
        url = self._directory_url(variable)
        parser = HrefParser(suffix=".grb")
        with gd_cli.AutoRaiseSession() as session:
            response = session.get(url, timeout=30)
            parser.feed(response.text)

        for href in sorted(set(parser.hrefs)):
            token = self._extract_token(href)
            if not self._should_keep(token, existing_max_time):
                continue
            period_start: Optional[str] = None
            period_end: Optional[str] = None
            if token is not None:
                start, end = self._period_from_token(token)
                period_start = _isoformat_utc(start)
                period_end = _isoformat_utc(end)
            filename = Path(href).name
            yield SourceFile(
                item_index=-1,
                variable=variable,
                frequency=self.frequency,
                url=f"{url}/{href}",
                filename=filename,
                relative_path=str(Path(variable) / filename),
                date_token=token,
                period_start=period_start,
                period_end=period_end,
            )

    def list_items(
        self, existing_max_time: Optional[datetime] = None
    ) -> List[SourceFile]:
        """List the source files that should be processed.

        Parameters
        ----------
        existing_max_time : datetime | None, optional
            Maximum time already present in the target store. Files whose
            coarse coverage ends before or at this time are skipped.

        Returns
        -------
        list[SourceFile]
            Ordered list of source files.
        """
        items: List[SourceFile] = []
        if self.frequency == "fx":
            filename = Path(urlsplit(self.invariant_url).path).name
            items.append(
                SourceFile(
                    item_index=0,
                    variable="fx",
                    frequency=self.frequency,
                    url=self.invariant_url,
                    filename=filename,
                    relative_path=str(Path("fx") / filename),
                )
            )
            return items

        for variable in self.variables:
            items.extend(self._list_variable_urls(variable, existing_max_time))

        items.sort(
            key=lambda item: (
                item.period_start or "",
                item.variable,
                item.filename,
            )
        )
        for index, item in enumerate(items):
            item.item_index = index
        return items


def _read_json_text(value: str) -> Dict[str, Any]:
    """Parse JSON text from the command line."""
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise TypeError("Backend kwargs JSON must decode to an object.")
    return parsed


def _download_one(
    url: str,
    target_path: Path,
    timeout: int,
    overwrite: bool,
    chunk_size: int,
) -> str:
    """Download one file to a fixed output path.

    Parameters
    ----------
    url : str
        Source URL.
    target_path : Path
        Final output file path.
    timeout : int
        Request timeout in seconds.
    overwrite : bool
        Whether to overwrite an existing file.
    chunk_size : int
        HTTP stream chunk size in bytes.

    Returns
    -------
    str
        Absolute output path.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() and not overwrite:
        LOGGER.debug("Skipping existing download %s", target_path)
        return str(target_path)

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    with gd_cli.AutoRaiseSession() as session:
        with session.get(url, stream=True, timeout=timeout) as response:
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
    tmp_path.replace(target_path)
    return str(target_path)


def download_files(
    items: Sequence[SourceFile],
    raw_root: Path,
    max_workers: int,
    timeout: int,
    overwrite: bool,
    chunk_size: int,
) -> List[Path]:
    """Download source files in parallel using ``concurrent.futures``.

    Parameters
    ----------
    items : sequence[SourceFile]
        Files to download.
    raw_root : Path
        Root directory for downloaded files.
    max_workers : int
        Number of download threads.
    timeout : int
        Per-request timeout in seconds.
    overwrite : bool
        Whether to overwrite existing files.
    chunk_size : int
        HTTP stream chunk size in bytes.

    Returns
    -------
    list[Path]
        Downloaded file paths in manifest order.
    """
    if not items:
        return []

    results: Dict[int, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map: Dict[concurrent.futures.Future[str], int] = {}
        for item in items:
            target = raw_root / item.relative_path
            future = executor.submit(
                _download_one,
                item.url,
                target,
                timeout,
                overwrite,
                chunk_size,
            )
            future_map[future] = item.item_index

        for future in concurrent.futures.as_completed(future_map):
            item_index = future_map[future]
            output = Path(future.result())
            results[item_index] = output
            LOGGER.info("Downloaded item %s -> %s", item_index, output)

    return [results[item.item_index] for item in items]


def _s3_map(path: str, s3_options: Dict[str, str]) -> s3fs.S3Map:
    """Create an S3 mapping for a Zarr store."""
    fs = s3fs.S3FileSystem(**s3_options)
    return s3fs.S3Map(root=path, s3=fs, check=False)


def _open_existing_target(
    level_path: str,
    s3_options: Dict[str, str],
) -> Optional["xr.Dataset"]:
    """Open an existing Zarr store from S3 if it exists."""
    import xarray as xr

    fs = s3fs.S3FileSystem(**s3_options)
    if not fs.exists(level_path):
        return None
    store = s3fs.S3Map(root=level_path, s3=fs, check=False)
    try:
        return xr.open_zarr(store, consolidated=True)
    except Exception:
        return xr.open_zarr(store, consolidated=False)


def _target_root(bucket: str, frequency: str) -> str:
    """Return the final S3 prefix for one frequency."""
    return f"s3://{bucket}/healpix/icon-dream/{frequency}"


def _load_existing_target_info(
    target_root: str,
    s3_options: Dict[str, str],
) -> ExistingTargetInfo:
    """Inspect the existing target store.

    Parameters
    ----------
    target_root : str
        S3 prefix excluding the level suffix.
    s3_options : dict[str, str]
        S3 credentials.

    Returns
    -------
    ExistingTargetInfo
        Existing time coordinate information, if available.
    """
    level_path = f"{target_root}/level_0.zarr"
    existing = _open_existing_target(level_path, s3_options)
    if existing is None:
        return ExistingTargetInfo(has_store=False)

    if "time" not in existing.coords and "time" not in existing.dims:
        return ExistingTargetInfo(has_store=True)

    time_values = _to_time_strings(existing["time"].values)
    return ExistingTargetInfo(
        has_store=True,
        time_values=time_values,
        max_time=time_values[-1] if time_values else None,
    )


def _open_source_dataset(
    path: Union[str, Path],
    engine: str,
    backend_kwargs: Dict[str, Any],
) -> "xr.Dataset":
    """Open one source file as an xarray dataset.

    Parameters
    ----------
    path : str or Path
        Source file path.
    engine : str
        Xarray backend engine.
    backend_kwargs : dict[str, Any]
        Backend kwargs for the engine.

    Returns
    -------
    xr.Dataset
        Opened dataset.
    """
    import xarray as xr

    dataset = xr.open_dataset(
        Path(path),
        engine=engine,
        backend_kwargs=backend_kwargs or None,
        chunks="auto",
    )
    return dataset


def _open_grid_dataset(path: Union[str, Path]) -> "xr.Dataset":
    """Open the ICON-DREAM grid definition."""
    import xarray as xr

    return xr.open_dataset(Path(path), chunks="auto")


def _maybe_start_local_client(n_workers: int) -> Optional[Any]:
    """Start a local distributed client when requested.

    Parameters
    ----------
    n_workers : int
        Number of local workers. A value smaller than one disables the client.

    Returns
    -------
    object | None
        ``distributed.Client`` instance when available, else ``None``.
    """
    if n_workers < 1:
        return None
    try:
        from distributed import Client, LocalCluster
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.warning(
            "distributed is not available; continuing without a local client"
        )
        return None

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=1,
        processes=True,
        dashboard_address=None,
    )
    return Client(cluster)


def _rename_values_dim(ds: "xr.Dataset") -> "xr.Dataset":
    """Rename a ``values`` dimension to ``cell`` when needed."""
    if "values" in ds.dims and "cell" not in ds.dims:
        return ds.rename({"values": "cell"})
    return ds.chunk({"cell": -1})


def _flatten_forecast_time(ds: "xr.Dataset") -> "xr.Dataset":
    """Collapse ``time`` and ``step`` into one valid ``time`` axis.

    Some ICON-DREAM GRIB files expose an analysis/forecast layout with a base
    ``time`` dimension, a forecast ``step`` dimension, and a two-dimensional
    ``valid_time`` coordinate. For downstream HEALPix conversion we want a
    single monotonic time axis.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with a single ``time`` dimension when ``valid_time`` can be
        flattened, otherwise the original dataset.
    """
    if "step" not in ds.dims:
        return ds
    if "valid_time" not in ds.coords:
        return ds
    if ds["valid_time"].ndim != 2:
        return ds

    valid_time_values = np.asarray(ds["valid_time"].values).ravel()
    stacked = ds.stack(_stacked_time=("time", "step"), create_index=False)
    stacked = stacked.drop_vars(["valid_time", "time", "step"], errors="ignore")
    stacked = stacked.assign_coords(_stacked_time=valid_time_values)
    stacked = stacked.rename({"_stacked_time": "time"})
    return stacked.sortby("time")


def _normalise_time_axis(ds: "xr.Dataset") -> "xr.Dataset":
    """Normalise the main time axis to a canonical ``time`` dimension.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset.

    Returns
    -------
    xr.Dataset
        Dataset with a standard ``time`` dimension when a valid time axis can
        be inferred. Datasets without temporal coordinates are returned
        unchanged.
    """
    ds = _flatten_forecast_time(ds)

    if "time" in ds.dims:
        return ds.sortby("time")

    if "valid_time" in ds.coords and ds["valid_time"].ndim == 1:
        source_dim = str(ds["valid_time"].dims[0])
        if source_dim != "valid_time":
            ds = ds.swap_dims({source_dim: "valid_time"})
        ds = ds.rename({"valid_time": "time"})
        if source_dim in ds.coords and source_dim != "time":
            ds = ds.drop_vars(source_dim, errors="ignore")
        return ds.sortby("time")

    for coord_name, coord in ds.coords.items():
        if coord.ndim != 1:
            continue
        if str(coord.dtype).startswith("datetime64"):
            source_dim = str(coord.dims[0])
            if source_dim != coord_name:
                ds = ds.swap_dims({source_dim: coord_name})
            ds = ds.rename({coord_name: "time"})
            if source_dim in ds.coords and source_dim != "time":
                ds = ds.drop_vars(source_dim, errors="ignore")
            return ds.sortby("time")

    return ds


def _prepare_dataset_for_regridding(ds: "xr.Dataset") -> "xr.Dataset":
    """Apply lightweight dataset normalisation before regridding."""
    ds = _rename_values_dim(ds)
    ds = _normalise_time_axis(ds)
    if "time" in ds.dims:
        _, index = np.unique(ds["time"].values, return_index=True)
        index = np.sort(index)
        ds = ds.isel(time=index)
    return ds.chunk({"cell": -1})


def _to_time_strings(values: Union[np.ndarray, Sequence[Any]]) -> List[str]:
    """Convert a time coordinate to sorted ISO strings."""
    array = np.asarray(values)
    if array.size == 0:
        return []
    strings = [
        np.datetime_as_string(value, unit="ns", timezone="UTC") for value in array
    ]
    return sorted(strings)


def _chunk_healpix_dataset(
    ds: "xr.Dataset",
    time_chunk: int,
    cell_chunk: int,
) -> "xr.Dataset":
    """Apply explicit chunks to a HEALPix dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset on the HEALPix grid.
    time_chunk : int
        Time chunk size.
    cell_chunk : int
        Cell chunk size.

    Returns
    -------
    xr.Dataset
        Chunked dataset.
    """
    chunks: Dict[str, int] = {}
    if "time" in ds.dims:
        chunks["time"] = min(time_chunk, int(ds.sizes["time"]))
    if "cell" in ds.dims:
        chunks["cell"] = min(cell_chunk, int(ds.sizes["cell"]))
    if not chunks:
        return ds
    return ds.chunk(chunks)


def _worker_output_root(paths: RunPaths, item_index: int) -> Path:
    """Return the temporary output directory of one worker."""
    return paths.temp_root / f"item-{item_index:06d}"


def _worker_metadata_path(paths: RunPaths, item_index: int) -> Path:
    """Return the metadata JSON path of one worker."""
    return paths.metadata_root / f"item-{item_index:06d}.json"


def _write_worker_result(paths: RunPaths, result: WorkerResult) -> None:
    """Persist worker output metadata."""
    target = _worker_metadata_path(paths, result.item_index)
    target.write_text(json.dumps(dataclasses.asdict(result), indent=2, sort_keys=True))


def _read_worker_result(paths: RunPaths, item_index: int) -> Optional[WorkerResult]:
    """Load one worker result if it exists."""
    path = _worker_metadata_path(paths, item_index)
    if not path.exists():
        return None
    return WorkerResult(**json.loads(path.read_text()))


def _level_output_path(output_root: Path, level: int) -> Path:
    """Return the temporary Zarr store path for one level."""
    return output_root / f"level_{level}.zarr"


def _write_temp_pyramid(
    pyramid: Dict[int, "xr.Dataset"],
    output_root: Path,
    time_chunk: int,
    cell_chunk: int,
    zarr_format: Literal[2, 3] = 2,
    compression_level: int = 4,
    access_pattern: Literal["map", "time_series"] = "map",
    strict_access_pattern: bool = True,
) -> Dict[int, str]:
    """Write one worker's HEALPix pyramid to local temporary Zarr stores."""

    level_paths: Dict[int, str] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    for level, ds in pyramid.items():
        target = _level_output_path(output_root, level)
        if target.exists():
            shutil.rmtree(target)
        chunked = _chunk_healpix_dataset(
            ds, time_chunk=time_chunk, cell_chunk=cell_chunk
        )
        chunked.to_zarr(
            target,
            mode="w",
            consolidated=True,
            zarr_format=zarr_format,
            align_chunks=True,
        )
        level_paths[level] = str(target)
    return level_paths


def _append_missing_times(
    ds: "xr.Dataset",
    target_path: str,
    s3_options: Dict[str, str],
    consolidated: bool = True,
    zarr_format: Literal[2, 3] = 2,
    compression_level: int = 4,
    access_pattern: Literal["map", "time_series"] = "map",
) -> Tuple[int, int]:
    """Append only the missing time steps of a temporary dataset.

    Parameters
    ----------
    ds : "xr.Dataset"
        Temporary worker output.
    target_path : str
        Final S3 Zarr path for one level.
    s3_options : dict[str, str]
        S3 credentials.
    consolidated : bool, optional
        Whether to use consolidated metadata.
    zarr_format: int
        Zarr format version (default ``2``)
    compression_level: int
        Encoding compression level (default ``4``)
    access_patter: str
        - access_pattern="map": optimize for slicing a single primary step (e.g. one time)
          across large secondary axes.
        - access_pattern="time_series": optimize for long runs along the primary axis at
          fixed/small secondary axes.
    strict_access_pattern: bool
        enforces chunk-size 1 of the axes applied to access_pattern.


    Returns
    -------
    tuple[int, int]
        Number of time steps written and total number of time steps considered.

    Notes
    -----
    This helper performs append-only updates. If a candidate store contains
    missing timestamps that would need to be inserted *before* the current
    maximum time already present in the target store, the function raises an
    error instead of silently corrupting the time order.
    """
    store = _s3_map(target_path, s3_options)
    from grid_doctor.helpers import dataset_encoding

    encoding = dataset_encoding(
        ds,
        comp_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=True,
    )
    existing = _open_existing_target(target_path, s3_options)
    if existing is None:
        ds.to_zarr(
            store,
            mode="w",
            consolidated=consolidated,
            zarr_format=zarr_format,
            encoding=encoding,
            align_chunks=True,
        )
        count = int(ds.sizes.get("time", 0))
        return count, count

    existing_time_list = _to_time_strings(existing["time"].values)
    existing_times = set(existing_time_list)
    existing_max_time = existing_time_list[-1] if existing_time_list else None
    candidate_times = _to_time_strings(ds["time"].values)
    missing_index = [
        i for i, value in enumerate(candidate_times) if value not in existing_times
    ]
    if not missing_index:
        return 0, len(candidate_times)

    if existing_max_time is not None:
        older_missing = [
            candidate_times[index]
            for index in missing_index
            if candidate_times[index] <= existing_max_time
        ]
        if older_missing:
            raise RuntimeError(
                "Append-only update cannot insert missing timestamps before the "
                f"current maximum target time {existing_max_time}. "
                f"Offending timestamps: {older_missing[:5]}"
            )

    update = ds.isel(time=missing_index)
    update.to_zarr(
        store,
        mode="a",
        append_dim="time",
        consolidated=consolidated,
        zarr_format=zarr_format,
        align_chunks=True,
    )
    return len(missing_index), len(candidate_times)


def _write_static_dataset(
    ds: "xr.Dataset",
    target_path: str,
    s3_options: Dict[str, str],
    overwrite_static: bool,
    consolidated: bool = True,
    zarr_format: Literal[2, 3] = 2,
    access_pattern: Literal["map", "time_series"] = "map",
    compression_level: int = 4,
    strict_access_pattern: bool = True,
) -> bool:
    """Write a non-temporal dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Static dataset.
    target_path : str
        Final S3 Zarr path for one level.
    s3_options : dict[str, str]
        S3 credentials.
    overwrite_static : bool
        Whether to overwrite an existing static target.
    consolidated : bool, optional
        Whether to use consolidated metadata.
    zarr_format: int
        Zarr format version (default ``2``)
    compression_level: int
        Encoding compression level (default ``4``)
    access_pattern: str
        - access_pattern="map": optimize for slicing a single primary step (e.g. one time)
          across large secondary axes.
        - access_pattern="time_series": optimize for long runs along the primary axis at
          fixed/small secondary axes.
    strict_access_pattern: bool
        enforces chunk-size 1 of the axes applied to access_pattern.

    Returns
    -------
    bool
        ``True`` when data were written.
    """
    from grid_doctor.helpers import dataset_encoding

    fs = s3fs.S3FileSystem(**s3_options)
    existing = _open_existing_target(target_path, s3_options)
    if existing is not None and not overwrite_static:
        LOGGER.info("Static target already exists, skipping %s", target_path)
        return False

    if existing is not None and overwrite_static:
        fs.rm(target_path, recursive=True)
    encoding = dataset_encoding(
        ds,
        comp_level=compression_level,
        access_pattern=access_pattern,
        strict_access_pattern=strict_access_pattern,
    )
    store = s3fs.S3Map(root=target_path, s3=fs, check=False)
    ds.to_zarr(
        store,
        mode="w",
        consolidated=consolidated,
        zarr_format=zarr_format,
        encoding=encoding,
        align_chunks=True,
    )
    return True


def _resolve_item_index(explicit_index: Optional[int]) -> int:
    """Resolve the worker index from CLI or Slurm environment."""
    if explicit_index is not None:
        return explicit_index
    for env_name in ("SLURM_ARRAY_TASK_ID", "ICON_DREAM_ITEM_INDEX"):
        value = os.getenv(env_name)
        if value is not None:
            return int(value)
    return 0


def plan_run(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the static run manifest.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict[str, Any]
        Small summary suitable for machine parsing.
    """
    run_dir = Path(args.run_dir) if args.run_dir else _default_run_dir()
    paths = _build_paths(run_dir)
    s3_options = gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file)
    target_root = _target_root(args.s3_bucket, args.freq)
    target_info = _load_existing_target_info(target_root, s3_options)
    existing_max_time_dt = (
        _parse_datetime(target_info.max_time) if target_info.max_time else None
    )

    source = IconDreamSource(
        variables=args.variables,
        frequency=args.freq,
        requested_time=(args.time[0], args.time[1]),
    )
    items = source.list_items(
        existing_max_time=existing_max_time_dt if args.update_only else None
    )

    manifest = RunManifest(
        run_dir=str(paths.run_dir),
        target_root=target_root,
        frequency=args.freq,
        variables=list(args.variables),
        requested_time=(args.time[0], args.time[1]),
        created_at=_isoformat_utc(datetime.now(tz=UTC)),
        grid_url=DEFAULT_GRID_URL,
        invariant_url=DEFAULT_INVARIANT_URL,
        grid_path=str(paths.grid_path),
        weights_path=str(paths.weights_path),
        temp_root=str(paths.temp_root),
        raw_root=str(paths.raw_root),
        source_items=items,
        source_engine=args.source_engine,
        source_backend_kwargs=_read_json_text(args.source_backend_kwargs_json),
        existing_max_time=target_info.max_time,
        output_mode=args.output_mode,
        metadata={
            "update_only": bool(args.update_only),
            "time_chunk": int(args.time_chunk),
            "cell_chunk": int(args.cell_chunk),
        },
    )
    save_manifest(manifest, paths.manifest_path)

    summary = {
        "run_dir": str(paths.run_dir),
        "manifest": str(paths.manifest_path),
        "item_count": len(items),
        "existing_max_time": target_info.max_time,
        "target_root": target_root,
    }
    return summary


def prepare_run(args: argparse.Namespace) -> Dict[str, Any]:
    """Download raw files and compute weights.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict[str, Any]
        Small execution summary.
    """
    if not args.run_dir:
        raise ValueError("--run-dir is required for the prepare step.")
    paths = _build_paths(args.run_dir)
    manifest = load_manifest(paths.manifest_path)
    downloaded_grid = gd_cli.download_file(
        manifest.grid_url,
        paths.grid_path.parent,
        timeout=args.download_timeout,
        overwrite=args.overwrite_downloads,
        chunk_size=args.download_chunk_size,
    )
    manifest.grid_path = downloaded_grid

    grid_ds = _open_grid_dataset(downloaded_grid)
    if args.max_level is None:
        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
    else:
        max_level = int(args.max_level)
    manifest.max_level = max_level

    gd.cached_weights(grid_ds, level=max_level, cache_path=paths.weights_path)
    manifest.weights_path = str(paths.weights_path)

    download_files(
        manifest.source_items,
        raw_root=paths.raw_root,
        max_workers=args.download_workers,
        timeout=args.download_timeout,
        overwrite=args.overwrite_downloads,
        chunk_size=args.download_chunk_size,
    )
    save_manifest(manifest, paths.manifest_path)

    return {
        "run_dir": str(paths.run_dir),
        "manifest": str(paths.manifest_path),
        "item_count": len(manifest.source_items),
        "max_level": manifest.max_level,
        "weights_path": manifest.weights_path,
    }


def worker_run(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert one manifest entry to a temporary HEALPix pyramid.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict[str, Any]
        Worker summary.
    """
    import xarray as xr

    if not args.run_dir:
        raise ValueError("--run-dir is required for the worker step.")
    paths = _build_paths(args.run_dir)
    manifest = load_manifest(paths.manifest_path)
    if manifest.max_level is None:
        raise RuntimeError("Manifest has no max_level. Run the prepare step first.")

    item_index = _resolve_item_index(args.item_index)
    try:
        item = manifest.source_items[item_index]
    except IndexError as error:
        raise IndexError(
            f"Item index {item_index} is out of range for {len(manifest.source_items)} items"
        ) from error

    client = _maybe_start_local_client(args.local_dask_workers)
    try:
        source_path = Path(manifest.raw_root) / item.relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing raw input file: {source_path}")

        ds = _open_source_dataset(
            source_path,
            engine=manifest.source_engine,
            backend_kwargs=manifest.source_backend_kwargs,
        )
        ds = _prepare_dataset_for_regridding(ds)

        weights = xr.open_dataset(manifest.weights_path)
        pyramid = gd.latlon_to_healpix_pyramid(
            ds,
            max_level=manifest.max_level,
            weights=weights,
        )

        if manifest.output_mode == "direct":
            if len(manifest.source_items) != 1:
                raise RuntimeError(
                    "output_mode='direct' is only safe when exactly one worker writes. "
                    "Use the default temp mode for Slurm arrays."
                )
            s3_options = gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file)
            for level, level_ds in pyramid.items():
                target_path = f"{manifest.target_root}/level_{level}.zarr"
                level_ds = _chunk_healpix_dataset(
                    level_ds, args.time_chunk, args.cell_chunk
                )
                if "time" in level_ds.dims:
                    _append_missing_times(level_ds, target_path, s3_options)
                else:
                    _write_static_dataset(
                        level_ds,
                        target_path,
                        s3_options,
                        overwrite_static=args.overwrite_static,
                    )
            summary = {
                "item_index": item_index,
                "variable": item.variable,
                "mode": "direct",
            }
            return summary

        output_root = _worker_output_root(paths, item_index)
        level_paths = _write_temp_pyramid(
            pyramid,
            output_root=output_root,
            time_chunk=args.time_chunk,
            cell_chunk=args.cell_chunk,
        )
        time_values = _to_time_strings(ds["time"].values) if "time" in ds.dims else []
        result = WorkerResult(
            item_index=item_index,
            variable=item.variable,
            level_paths=level_paths,
            time_count=len(time_values),
            time_start=time_values[0] if time_values else None,
            time_end=time_values[-1] if time_values else None,
            has_time=bool(time_values),
        )
        _write_worker_result(paths, result)
        return {
            "item_index": item_index,
            "variable": item.variable,
            "mode": "temp",
            "levels": sorted(level_paths),
            "time_count": len(time_values),
        }
    finally:
        if client is not None:
            client.close()


def finalize_run(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge worker outputs into the final S3 target.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    dict[str, Any]
        Finalization summary.
    """
    import xarray as xr

    if not args.run_dir:
        raise ValueError("--run-dir is required for the finalize step.")
    paths = _build_paths(args.run_dir)
    manifest = load_manifest(paths.manifest_path)
    if manifest.max_level is None:
        raise RuntimeError("Manifest has no max_level. Run the prepare step first.")

    s3_options = gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file)
    worker_results: List[WorkerResult] = []
    for item in manifest.source_items:
        result = _read_worker_result(paths, item.item_index)
        if result is None:
            LOGGER.warning(
                "Skipping item %s because no worker metadata was found",
                item.item_index,
            )
            continue
        worker_results.append(result)

    worker_results.sort(
        key=lambda result: (
            result.time_start or "",
            result.variable,
            result.item_index,
        )
    )

    total_time_candidates = 0
    total_time_written = 0
    static_written = 0

    for result in worker_results:
        for level, level_path in sorted(result.level_paths.items()):
            ds = xr.open_zarr(level_path, consolidated=True)
            target_path = f"{manifest.target_root}/level_{level}.zarr"
            if "time" in ds.dims:
                written, considered = _append_missing_times(ds, target_path, s3_options)
                total_time_written += written
                total_time_candidates += considered
                LOGGER.info(
                    "Merged item %s level %s: wrote %s/%s time slices",
                    result.item_index,
                    level,
                    written,
                    considered,
                )
            else:
                did_write = _write_static_dataset(
                    ds,
                    target_path,
                    s3_options,
                    overwrite_static=args.overwrite_static,
                )
                static_written += int(did_write)

    return {
        "target_root": manifest.target_root,
        "items_seen": len(worker_results),
        "time_slices_considered": total_time_candidates,
        "time_slices_written": total_time_written,
        "static_writes": static_written,
    }


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full workflow locally in one process."""
    plan_summary = plan_run(args)
    args.run_dir = plan_summary["run_dir"]
    prepare_summary = prepare_run(args)
    item_count = int(plan_summary["item_count"])
    for item_index in range(item_count):
        worker_args = argparse.Namespace(**vars(args))
        worker_args.item_index = item_index
        worker_run(worker_args)
    finalize_summary = finalize_run(args)
    return {
        "plan": plan_summary,
        "prepare": prepare_summary,
        "finalize": finalize_summary,
    }


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common CLI arguments to a parser."""
    parser.add_argument("s3_bucket", help="S3 target bucket.")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Shared working directory used by all stages.",
    )
    parser.add_argument(
        "--s3-endpoint",
        default="https://s3.eu-dkrz-3.dkrz.cloud",
        help="S3 endpoint URL.",
    )
    parser.add_argument(
        "--s3-credentials-file",
        default=str(Path.home() / ".s3-credentials.json"),
        help="Path to a JSON file containing accessKey and secretKey.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=["t_2m", "tot_prec"],
        choices=ICON_DREAM_VARIABLES,
        help="Variables to process.",
    )
    parser.add_argument(
        "--freq",
        default="hourly",
        choices=["hourly", "daily", "monthly", "fx"],
        help="ICON-DREAM data frequency.",
    )
    parser.add_argument(
        "--time",
        nargs=2,
        default=["2010-01-01T00:00:00Z", "now"],
        metavar=("START", "END"),
        help="Requested time interval in UTC.",
    )
    parser.add_argument(
        "--source-engine",
        default="cfgrib",
        help="Xarray backend engine for source files.",
    )
    parser.add_argument(
        "--source-backend-kwargs-json",
        default="{}",
        help="JSON object forwarded as backend_kwargs to xarray.",
    )
    parser.add_argument(
        "--output-mode",
        choices=["temp", "direct"],
        default="temp",
        help="Worker output mode. Temp mode is the safe choice for arrays.",
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Skip coarse source files that end before the current target max time.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Re-download raw files even if they are already present.",
    )
    parser.add_argument(
        "--overwrite-static",
        action="store_true",
        help="Overwrite an existing non-temporal target store.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="Number of parallel download threads in the prepare step.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for downloads.",
    )
    parser.add_argument(
        "--download-chunk-size",
        type=int,
        default=1024 * 1024,
        help="HTTP stream chunk size in bytes.",
    )
    parser.add_argument(
        "--max-level",
        type=int,
        default=None,
        help="Override the automatically selected maximum HEALPix level.",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=168,
        help="Time chunk size for temporary HEALPix Zarr stores.",
    )
    parser.add_argument(
        "--cell-chunk",
        type=int,
        default=262144,
        help="Cell chunk size for temporary HEALPix Zarr stores.",
    )
    parser.add_argument(
        "--local-dask-workers",
        type=int,
        default=0,
        help="Optional local distributed workers inside one process.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase log verbosity.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(
        prog="convert.py",
        description="Convert ICON-DREAM GRIB data to HEALPix Zarr.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("plan", "prepare", "worker", "finalize", "run"):
        subparser = subparsers.add_parser(name)
        _add_common_arguments(subparser)
        if name == "worker":
            subparser.add_argument(
                "--item-index",
                type=int,
                default=None,
                help="Manifest item index processed by this worker.",
            )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv : sequence[str] | None, optional
        Command line arguments. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    gd.setup_logging(verbosity=args.verbose)

    handlers = {
        "plan": plan_run,
        "prepare": prepare_run,
        "worker": worker_run,
        "finalize": finalize_run,
        "run": run_all,
    }
    summary = handlers[args.command](args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
