"""Download/update and convert ICON-DREAM in GRIB to HEALPix.

Pipeline modes
--------------
This version is designed for HPC/SLURM style execution without dask-distributed:

1. ``prepare``
   - downloads all source GRIB files in parallel into a shared directory
   - downloads the grid file and pre-computes/saves the weights dataset
   - groups files into worker jobs (typically one time chunk per job)
   - writes a JSON manifest consumed by worker/pack jobs

2. ``worker``
   - processes one manifest entry (selected via ``SLURM_ARRAY_TASK_ID`` or
     ``--job-index``)
   - opens all GRIB files for that job, merges them, converts to HEALPix,
     writes a temporary local Zarr pyramid, and copies it to the shared parts
     directory

3. ``pack``
   - runs serially after all workers finish
   - opens each shared temporary pyramid in manifest order and appends it to the
     final S3-backed Zarr pyramid
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
)

import dateparser
import xarray as xr

import grid_doctor.cli as gd_cli

if TYPE_CHECKING:
    import argparse

    import numpy as np

DATE_FORMAT = "%Y%m"
MANIFEST_NAME = "job_manifest.json"
SUCCESS_MARKER = "_SUCCESS.json"

logger = logging.getLogger("icon-dream-catcher")

IconDreamVariable = Literal[
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
]


# -----------------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------------


def flatten_step(ds: "xr.Dataset") -> "xr.Dataset":
    """Merge ``time`` + ``step`` into one valid-time axis."""
    if "step" not in ds.dims:
        return ds

    valid = ds.valid_time.values
    ds = ds.stack(valid=("time", "step"))
    ds = ds.drop_vars(["valid", "time", "step"], errors="ignore")
    ds = ds.assign_coords(valid=valid.ravel())
    ds = ds.rename({"valid": "time"})
    ds = ds.sortby("time")
    return ds


def _open_dataset(
    file: str,
    clat: "np.ndarray",
    clon: "np.ndarray",
) -> "xr.Dataset":
    ds = xr.open_dataset(file, engine="cfgrib", chunks="auto")
    ds = flatten_step(ds.rename_dims({"values": "cell"}))
    ds = ds.assign_coords(clat=("cell", clat), clon=("cell", clon))
    chunk_spec = {"cell": -1}
    if "time" in ds.dims:
        chunk_spec["time"] = -1
    ds = ds.chunk(chunk_spec)
    return ds


def _open_merged_dataset(
    files: list[str],
    clat: "np.ndarray",
    clon: "np.ndarray",
    parallel: bool = True,
) -> "xr.Dataset":
    datasets = [
        _open_dataset(str(path), clat=clat, clon=clon, parallel=parallel)
        for path in files
    ]
    if len(datasets) == 1:
        return datasets[0]

    ds = xr.merge(
        datasets,
        compat="override",
        combine_attrs="drop_conflicts",
        join="outer",
    )
    if "time" in ds.dims:
        ds = ds.sortby("time")
    return ds


# -----------------------------------------------------------------------------
# Download + manifest helpers
# -----------------------------------------------------------------------------


class HrefParser(HTMLParser):
    """Extract ``.grb`` links from an HTML directory listing."""

    file_suffix: str = ".grb"

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, Optional[str]]]
    ) -> None:
        if tag != "a":
            return
        for key, value in attrs:
            if key == "href" and value and value.endswith(self.file_suffix):
                self.hrefs.append(value)


@dataclass
class IconDreamSource:
    """Describe an ICON-DREAM data source on the DWD open-data server."""

    variables: List[IconDreamVariable]
    time_frequency: Literal["hourly", "daily", "monthly", "fx"]
    time: Tuple[str, str]

    def __post_init__(self) -> None:
        self._dir_source = (
            "https://opendata.dwd.de/climate_environment/REA/"
            "ICON-DREAM-Global/{freq}/{var}"
        )
        self.grid_source = (
            "https://opendata.dwd.de/climate_environment/REA/"
            "ICON-DREAM-Global/invariant/ICON-DREAM-Global_grid.nc"
        )
        self._invariants = (
            "https://opendata.dwd.de/climate_environment/REA/"
            "ICON-DREAM-Global/invariant/"
            "ICON-DREAM-Global_constant_fields.grb"
        )

        self._start = self._parse_user_datetime(self.time[0])
        self._end = self._parse_user_datetime(self.time[-1])

    def _parse_user_datetime(self, value: str) -> datetime:
        dt = dateparser.parse(
            value,
            settings={
                "TIMEZONE": "UTC",
                "TO_TIMEZONE": "UTC",
                "RETURN_AS_TIMEZONE_AWARE": True,
            },
        )
        if dt is None:
            raise ValueError(f"Could not parse datetime string: {value!r}")
        return dt

    def _href_to_datetime(self, href: str) -> Tuple[str, datetime]:
        match = re.search(r"_(\d{6})_", href)
        if not match:
            raise ValueError(f"Could not extract YYYYMM from href: {href}")
        dt = datetime.strptime(match.group(1), "%Y%m").replace(
            tzinfo=timezone.utc
        )
        return href, dt

    def _get_download_link(self, variable: str) -> Iterator[str]:
        url = self._dir_source.format(
            freq=self.time_frequency,
            var=variable.upper().replace("-", "_"),
        )
        parser = HrefParser()
        with gd_cli.AutoRaiseSession() as session:
            response = session.get(url, timeout=5)
            parser.feed(response.text)

        for href, date_value in map(self._href_to_datetime, parser.hrefs):
            if self._start <= date_value <= self._end:
                logger.debug("Found %s/%s", url, href)
                yield f"{url}/{href}"

    @property
    def links(self) -> Iterator[str]:
        if self.time_frequency == "fx":
            yield self._invariants
            return
        for var in self.variables:
            yield from self._get_download_link(var)


def extract_job_label(path: str) -> str:
    """Return the manifest grouping label for a downloaded file."""
    if path.endswith("ICON-DREAM-Global_constant_fields.grb"):
        return "fx"
    match = re.search(r"_(\d{6})_", Path(path).name)
    if not match:
        raise ValueError(f"Could not extract YYYYMM from file path: {path}")
    return match.group(1)


def download_files_parallel(
    urls: list[str],
    target_dir: Path,
    timeout: int = 60,
    overwrite: bool = False,
    max_workers: int = 4,
) -> list[Path]:
    """Download files concurrently to a shared directory."""
    target_dir.mkdir(parents=True, exist_ok=True)

    def _download_one(url: str) -> Path:
        result = gd_cli.download_file(
            url,
            target_dir,
            timeout=timeout,
            overwrite=overwrite,
            chunk_size=1024 * 1024,
        )
        return Path(result)

    results: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one, url): url for url in urls}
        for future in as_completed(futures):
            url = futures[future]
            result = future.result()
            logger.info("Downloaded %s -> %s", url, result)
            results.append(result)
    return sorted(results)


def build_jobs(
    downloaded_files: list[Path], time_frequency: str
) -> list[Dict[str, Any]]:
    """Group downloaded GRIB files into worker jobs.

    For hourly/daily/monthly data we group by ``YYYYMM`` so one worker job sees
    all selected variables for the same month.

    For ``fx``/time-invariant data we intentionally emit exactly one job, since
    the resulting datasets may not have a ``time`` dimension and therefore
    cannot be packed via append-along-time semantics.
    """
    files = [str(path) for path in sorted(downloaded_files)]
    if time_frequency == "fx":
        return [
            {
                "job_index": 0,
                "label": "fx",
                "files": files,
                "part_name": "00000_fx",
            }
        ]

    groups: Dict[str, list[str]] = {}
    for path in files:
        label = extract_job_label(path)
        groups.setdefault(label, []).append(path)

    jobs: list[Dict[str, Any]] = []
    for job_index, label in enumerate(sorted(groups)):
        jobs.append(
            {
                "job_index": job_index,
                "label": label,
                "files": groups[label],
                "part_name": f"{job_index:05d}_{label}",
            }
        )
    return jobs


# -----------------------------------------------------------------------------
# Manifest persistence
# -----------------------------------------------------------------------------


def get_run_paths(work_dir: Path) -> Dict[str, Path]:
    return {
        "work_dir": work_dir,
        "raw_dir": work_dir / "raw",
        "shared_dir": work_dir / "shared",
        "parts_dir": work_dir / "parts",
        "manifests_dir": work_dir / "manifests",
        "logs_dir": work_dir / "logs",
        "manifest_path": work_dir / "manifests" / MANIFEST_NAME,
    }


def ensure_run_layout(work_dir: Path) -> Dict[str, Path]:
    paths = get_run_paths(work_dir)
    for key, path in paths.items():
        if key.endswith("_path"):
            continue
        path.mkdir(parents=True, exist_ok=True)
    paths["manifest_path"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("Wrote manifest to %s", path)


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


# -----------------------------------------------------------------------------
# Pyramid local temp store helpers
# -----------------------------------------------------------------------------


def write_pyramid_to_local(
    pyramid: Dict[int, xr.Dataset], target_dir: Path
) -> None:
    """Write one HEALPix pyramid to a local/shared filesystem Zarr directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for level, ds in sorted(pyramid.items()):
        level_path = target_dir / f"level_{level}.zarr"
        logger.info("Writing local part level %d to %s", level, level_path)
        ds.to_zarr(level_path, mode="w", consolidated=True)


def open_local_pyramid(target_dir: Path) -> Dict[int, xr.Dataset]:
    pyramid: Dict[int, xr.Dataset] = {}
    for level_dir in sorted(target_dir.glob("level_*.zarr")):
        match = re.search(r"level_(\d+)\.zarr$", level_dir.name)
        if not match:
            continue
        level = int(match.group(1))
        pyramid[level] = xr.open_zarr(level_dir, consolidated=True)
    if not pyramid:
        raise RuntimeError(f"No pyramid levels found in {target_dir}")
    return pyramid


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def mark_success(part_dir: Path, payload: Dict[str, Any]) -> None:
    (part_dir / SUCCESS_MARKER).write_text(
        json.dumps(payload, indent=2, sort_keys=True)
    )


def check_success(part_dir: Path) -> bool:
    return (part_dir / SUCCESS_MARKER).exists()


# -----------------------------------------------------------------------------
# Prepare / worker / pack stages
# -----------------------------------------------------------------------------


def prepare_shared_grid_and_weights(
    grid_source: str,
    shared_dir: Path,
) -> tuple[Path, Path, int]:
    import grid_doctor as gd

    shared_dir.mkdir(parents=True, exist_ok=True)
    grid_file = Path(
        gd_cli.download_file(grid_source, shared_dir, overwrite=False)
    )

    grid_ds = gd.cached_open_dataset([grid_file])
    max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
    weights = gd.cached_weights(grid_ds, level=max_level)

    weights_file = shared_dir / f"weights_level_{max_level}.nc"
    if not weights_file.exists():
        logger.info("Saving weights to %s", weights_file)
        weights.to_netcdf(weights_file)

    return grid_file, weights_file, max_level


def run_prepare(args: "argparse.Namespace") -> None:
    source = IconDreamSource(
        time_frequency=args.freq,
        variables=args.variables,
        time=tuple(args.time),
    )
    paths = ensure_run_layout(Path(args.work_dir))

    logger.info("Starting prepare stage in %s", paths["work_dir"])
    grid_file, weights_file, max_level = prepare_shared_grid_and_weights(
        source.grid_source,
        paths["shared_dir"],
    )

    urls = list(source.links)
    logger.info(
        "Downloading %d source files into %s", len(urls), paths["raw_dir"]
    )
    downloaded_files = download_files_parallel(
        urls=urls,
        target_dir=paths["raw_dir"],
        timeout=args.download_timeout,
        overwrite=args.overwrite_downloads,
        max_workers=args.download_workers,
    )

    jobs = build_jobs(downloaded_files, time_frequency=args.freq)
    logger.info("Prepared %d worker jobs", len(jobs))

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "freq": args.freq,
        "variables": list(args.variables),
        "time": list(args.time),
        "work_dir": str(paths["work_dir"]),
        "raw_dir": str(paths["raw_dir"]),
        "shared_dir": str(paths["shared_dir"]),
        "parts_dir": str(paths["parts_dir"]),
        "grid_file": str(grid_file),
        "weights_file": str(weights_file),
        "max_level": max_level,
        "jobs": jobs,
    }
    save_manifest(manifest, paths["manifest_path"])
    logger.info(
        "Prepare stage finished. Manifest: %s (jobs=%d)",
        paths["manifest_path"],
        len(jobs),
    )


def resolve_job_index(args: "argparse.Namespace") -> int:
    if args.job_index is not None:
        return args.job_index
    if os.getenv("SLURM_ARRAY_TASK_ID") is not None:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    raise RuntimeError(
        "No job index given. Use --job-index locally or run inside a SLURM array job."
    )


def run_worker(args: "argparse.Namespace") -> None:
    import grid_doctor as gd

    manifest = load_manifest(Path(args.manifest))
    job_index = resolve_job_index(args)
    jobs = manifest["jobs"]
    if job_index < 0:
        raise IndexError(f"job index must be >= 0, got {job_index}")
    if job_index >= len(jobs):
        logger.info(
            (
                "Worker index %d has no assigned work "
                "(manifest contains %d jobs); exiting."
            ),
            job_index,
            len(jobs),
        )
        return

    job = jobs[job_index]
    part_name = job["part_name"]
    shared_part_dir = Path(manifest["parts_dir"]) / part_name
    shared_staging_dir = Path(manifest["parts_dir"]) / f"{part_name}.staging"

    if (
        shared_part_dir.exists()
        and check_success(shared_part_dir)
        and not args.overwrite_parts
    ):
        logger.info(
            "Part already exists and is marked successful, skipping: %s",
            shared_part_dir,
        )
        return

    local_tmp_root = Path(args.local_tmp_dir)
    local_tmp_root.mkdir(parents=True, exist_ok=True)
    local_part_dir = local_tmp_root / part_name
    if local_part_dir.exists():
        shutil.rmtree(local_part_dir)

    logger.info("Worker job %d processing label=%s", job_index, job["label"])

    grid_ds = gd.cached_open_dataset([manifest["grid_file"]])
    clat = grid_ds.clat.values
    clon = grid_ds.clon.values
    weights = xr.open_dataset(manifest["weights_file"])

    try:
        ds = _open_merged_dataset(
            job["files"], clat=clat, clon=clon, parallel=True
        )
        pyramid = gd.latlon_to_healpix_pyramid(
            ds,
            max_level=int(manifest["max_level"]),
            weights=weights,
        )
        write_pyramid_to_local(pyramid, local_part_dir)

        if shared_staging_dir.exists():
            shutil.rmtree(shared_staging_dir)
        copy_tree(local_part_dir, shared_staging_dir)
        mark_success(
            shared_staging_dir,
            {
                "job_index": job_index,
                "label": job["label"],
                "files": job["files"],
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        if shared_part_dir.exists():
            shutil.rmtree(shared_part_dir)
        os.replace(shared_staging_dir, shared_part_dir)
        logger.info("Worker job %d finished -> %s", job_index, shared_part_dir)
    finally:
        try:
            if local_part_dir.exists() and not args.keep_local_parts:
                shutil.rmtree(local_part_dir)
        finally:
            del weights, grid_ds
            gc.collect()


def pyramid_has_time_dimension(pyramid: Dict[int, xr.Dataset]) -> bool:
    return all("time" in ds.dims for ds in pyramid.values())


def append_pyramid_to_s3(
    pyramid: Dict[int, xr.Dataset],
    output_path: str,
    s3_opts: Dict[str, str],
    *,
    mode: Literal["w", "a"],
) -> None:
    import s3fs

    fs = s3fs.S3FileSystem(**s3_opts)
    for level, ds in sorted(pyramid.items()):
        level_path = f"{output_path}/level_{level}.zarr"
        store = s3fs.S3Map(root=level_path, s3=fs)

        kwargs: Dict[str, Any] = {"consolidated": True, "mode": mode}
        if mode == "a":
            if "time" not in ds.dims:
                raise RuntimeError(
                    f"Cannot append non-time dataset for level {level}; "
                    "time-invariant data must be packed from a single part only."
                )
            kwargs["append_dim"] = "time"

        logger.info(
            "Packing level %d to %s with mode=%s",
            level,
            level_path,
            kwargs["mode"],
        )
        ds.to_zarr(store, **kwargs)


def run_pack(args: "argparse.Namespace") -> None:
    import grid_doctor as gd

    manifest = load_manifest(Path(args.manifest))
    jobs = manifest["jobs"]
    parts_dir = Path(manifest["parts_dir"])
    output_path = "s3://{bucket}/healpix/icon-dream/{time_frequency}".format(
        bucket=args.s3_bucket,
        time_frequency=args.freq,
    )
    s3_opts = gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file)

    if not jobs:
        raise RuntimeError("Manifest contains no jobs")

    first_written = False
    first_part_has_time: Optional[bool] = None

    for job in jobs:
        part_dir = parts_dir / job["part_name"]
        if not part_dir.exists():
            raise FileNotFoundError(
                f"Expected part directory missing: {part_dir}"
            )
        if not check_success(part_dir):
            raise RuntimeError(
                f"Part directory is missing success marker: {part_dir}"
            )

        logger.info("Packing part %s", part_dir)
        pyramid = open_local_pyramid(part_dir)
        has_time = pyramid_has_time_dimension(pyramid)

        if not first_written:
            append_pyramid_to_s3(
                pyramid=pyramid,
                output_path=output_path,
                s3_opts=s3_opts,
                mode="w",
            )
            first_written = True
            first_part_has_time = has_time
        else:
            if not has_time:
                raise RuntimeError(
                    "Encountered an additional time-invariant part during pack. "
                    "For datasets without a time dimension, prepare should emit "
                    "exactly one job."
                )
            if first_part_has_time is False:
                raise RuntimeError(
                    "Cannot append time-varying parts after a time-invariant "
                    "first part."
                )
            append_pyramid_to_s3(
                pyramid=pyramid,
                output_path=output_path,
                s3_opts=s3_opts,
                mode="a",
            )

        del pyramid
        gc.collect()

    logger.info("All parts packed to %s", output_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args(
    name: str, argv: Optional[List[str]] = None
) -> "argparse.Namespace":
    parser = gd_cli.get_parser(
        name,
        description="Download and convert ICON-DREAM data to HEALPix.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["prepare", "worker", "pack"],
        help="Pipeline stage to execute.",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Shared pipeline directory containing raw/shared/parts/manifests.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=(
            "Path to the manifest JSON. Defaults to "
            "<work-dir>/manifests/job_manifest.json."
        ),
    )
    parser.add_argument(
        "--job-index",
        type=int,
        default=None,
        help="Worker job index for local execution. If omitted, "
        "uses SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--local-tmp-dir",
        default=os.environ.get("TMPDIR", "/tmp/icon-dream-catcher"),
        help="Node-local temporary directory used by workers to build local parts.",
    )
    parser.add_argument(
        "--keep-local-parts",
        action="store_true",
        help="Do not remove the worker-local temporary part directory after success.",
    )
    parser.add_argument(
        "--overwrite-parts",
        action="store_true",
        help="Recompute a worker part even if the shared part already exists.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help="Number of concurrent downloads in prepare mode.",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=60,
        help="Per-download timeout in seconds for prepare mode.",
    )
    parser.add_argument(
        "--overwrite-downloads",
        action="store_true",
        help="Re-download files even if they already exist in the "
        "shared raw directory.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=["t_2m", "tot_prec"],
        choices=IconDreamVariable.__args__,
    )
    parser.add_argument(
        "--freq",
        "--time-frequency",
        default="hourly",
        choices=["hourly", "daily", "monthly", "fx"],
        help="Time frequency of the data.",
    )
    parser.add_argument(
        "--time",
        nargs=2,
        default=["2010-01-01T00:00", "now"],
    )

    args = parser.parse_args(argv)
    gd_cli.setup_logging_from_args(args)

    if args.manifest is None:
        args.manifest = str(get_run_paths(Path(args.work_dir))["manifest_path"])
    return args


def main(name: str, argv: Optional[List[str]] = None) -> None:
    args = parse_args(name, argv)

    if args.mode == "prepare":
        run_prepare(args)
    elif args.mode == "worker":
        run_worker(args)
    elif args.mode == "pack":
        run_pack(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main("icon-dream-catcher", sys.argv[1:])
