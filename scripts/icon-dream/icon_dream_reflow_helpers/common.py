#!/usr/bin/env python
"""Shared utilities for the Reflow-based ICON-DREAM pipeline."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence
from urllib.parse import urlsplit

import numpy as np
import s3fs

import grid_doctor as gd
import grid_doctor.cli as gd_cli

if TYPE_CHECKING:
    import xarray as xr

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
ICON_DREAM_VARIABLES: tuple[str, ...] = (
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
    """Collect href targets ending in a specific suffix."""

    def __init__(self, suffix: str = ".grb") -> None:
        super().__init__()
        self.suffix = suffix
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        """Collect matching href attributes from anchor tags."""
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value and value.endswith(self.suffix):
                self.hrefs.append(value)


def parse_datetime(value: str) -> datetime:
    """Parse a flexible ISO-like UTC timestamp."""
    if value == "now":
        return datetime.now(tz=UTC)
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def isoformat_utc(value: datetime) -> str:
    """Serialise a timezone-aware datetime as UTC."""
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def default_run_dir() -> Path:
    """Return the default local run directory."""
    return Path.cwd() / "icon-dream-reflow-run"


def build_paths(run_dir: str | Path) -> dict[str, Path]:
    """Return the standard file layout under a run directory."""
    root = Path(run_dir)
    return {
        "run_dir": root,
        "plan_path": root / "plan.json",
        "grid_path": root / "shared" / "ICON-DREAM-Global_grid.nc",
        "weights_path": root / "shared" / "healpix-weights.nc",
        "temp_root": root / "temp-zarr",
        "raw_root": root / "raw-input",
    }


def save_plan(plan: dict[str, Any], path: Path) -> None:
    """Persist the run plan to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")


def load_plan(run_dir: str | Path) -> dict[str, Any]:
    """Load the persisted run plan for a run directory."""
    return json.loads(build_paths(run_dir)["plan_path"].read_text(encoding="utf-8"))


def read_json_text(value: str) -> dict[str, Any]:
    """Parse a JSON object from text."""
    data = json.loads(value)
    if not isinstance(data, dict):
        raise TypeError("Expected a JSON object.")
    return data


def download_one(
    url: str,
    local_path: Path,
    *,
    timeout: int,
    overwrite: bool,
    chunk_size: int,
) -> str:
    """Download a file once, while retaining the local skip logic."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and not overwrite:
        LOGGER.info("Skipping download because %s already exists", local_path)
        return str(local_path)
    return str(
        gd_cli.download_file(
            url,
            local_path.parent,
            timeout=timeout,
            overwrite=overwrite,
            chunk_size=chunk_size,
            path=local_path,
        )
    )


def s3_map(path: str, s3_options: dict[str, str]) -> s3fs.S3Map:
    """Return an S3-backed mutable mapping for a Zarr store."""
    return s3fs.S3Map(root=path, s3=s3fs.S3FileSystem(**s3_options), check=False)


def open_existing_target(target_path: str, s3_options: dict[str, str]) -> "xr.Dataset | None":
    """Open an existing target Zarr dataset from S3 if it exists."""
    import xarray as xr

    mapper = s3_map(target_path, s3_options)
    try:
        return xr.open_zarr(mapper, consolidated=True)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        return None


def target_root(bucket: str, frequency: str) -> str:
    """Return the S3 root for a given bucket and frequency."""
    return f"{bucket.rstrip('/')}/icon-dream/{frequency}"


def load_existing_target_info(target_root_path: str, s3_options: dict[str, str]) -> dict[str, Any]:
    """Inspect the existing target and return a compact summary."""
    variables: set[str] = set()
    max_time: datetime | None = None
    for level in range(16):
        ds = open_existing_target(f"{target_root_path}/level_{level}.zarr", s3_options)
        if ds is None:
            continue
        variables.update(map(str, ds.data_vars))
        if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
            candidate = parse_datetime(str(ds["time"].values[-1]))
            max_time = candidate if max_time is None else max(max_time, candidate)
    return {
        "exists": bool(variables),
        "variables": sorted(variables),
        "max_time": isoformat_utc(max_time) if max_time else None,
    }


def open_source_dataset(path: str | Path, *, engine: str, backend_kwargs: dict[str, Any]) -> "xr.Dataset":
    """Open one ICON-DREAM source file."""
    import xarray as xr

    return xr.open_dataset(path, engine=engine, backend_kwargs=backend_kwargs)


def open_grid_dataset(path: str | Path) -> "xr.Dataset":
    """Open the ICON-DREAM grid file."""
    import xarray as xr

    return xr.open_dataset(path)


def maybe_start_local_client(n_workers: int) -> Any | None:
    """Start a local Dask client if requested."""
    if n_workers <= 0:
        return None
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
    return Client(cluster)


def to_time_strings(values: np.ndarray | Sequence[Any]) -> list[str]:
    """Convert time-like values to stable strings."""
    result: list[str] = []
    for value in values:
        if isinstance(value, np.datetime64):
            result.append(str(value.astype("datetime64[ns]")))
        else:
            result.append(str(value))
    return result
