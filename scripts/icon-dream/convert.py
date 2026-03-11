"""Download/update and convert ICON-DREAM in GRIB to HEALPix.

About
-----
ICON-DREAM is a reanalysis product from the German Weather Service (DWD).

With ICON-DREAM, Germany's national meteorological service DWD
(Deutscher Wetterdienst) started a new generation of reanalyses after the
COSMO-R6G2, based on DWD's global and European forecasting system.
The reanalysis uses the ICOsahedral Nonhydrostatic (ICON) modelling framework
(based on the operational system from March 2024), with a resolution of 13 km
globally and a nested Europe domain with a resolution of 6.5 km.
The data assimilation cycle comprises an EnVar scheme at 3-hourly intervals
complemented by a snow analysis every 3 hours, and soil moisture analysis every
24 hours (at 00 UTC), and the sea surface temperature is initialized using the
daily 1/20° OSTIA product.

All observations used in the first version of ICON-DREAM come from the ones
originally archived by DWD for the purpose of its operational numerical weather
prediction.
Flow-dependent background error covariances are provided by a 20-member
ensemble at 40km global and 20km over Europe, which uses an LETKF-based
ensemble data assimilation scheme.

Version 1 of ICON-DREAM covers the period from 2010 onward, and provides hourly
outputs for 111 variables, of which 3 are for the 8 soil depths, 15 are on the
native model levels (120 for the global, and 74 for the European nest), and
13 on the 37 standard pressure levels.
"""

import gc
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import (TYPE_CHECKING, Dict, Iterator, List, Literal, Optional,
                    Tuple, Union)

import dateparser
import xarray as xr
from mpi4py import MPI

import grid_doctor as gd
import grid_doctor.cli as gd_cli

if TYPE_CHECKING:
    import argparse

    import numpy as np
    import xarray as xr

DATE_FORMAT = "%Y%m"

logger = logging.getLogger("icon-dream-catcher")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

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


def flatten_step(ds: "xr.Dataset") -> "xr.Dataset":
    """Merge time + step into a single time dimension via valid_time.

    ICON-DREAM uses a 3-hourly data assimilation cycle with short
    forecasts filling the gaps.  ``step`` is the forecast lead time
    (1 h, 2 h, 3 h).  This function collapses ``(time, step)`` into a
    continuous ``time`` axis using the pre-computed ``valid_time``
    coordinate.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with ``time`` and ``step`` dimensions.

    Returns
    -------
    xr.Dataset
        Dataset with a single, sorted ``time`` dimension.
    """
    if "step" not in ds.dims:
        return ds

    valid = ds.valid_time.values  # shape (time, step)

    ds = ds.stack(valid=("time", "step"))
    ds = ds.drop_vars(["valid", "time", "step"], errors="ignore")
    ds = ds.assign_coords(valid=valid.ravel())
    ds = ds.rename({"valid": "time"})
    ds = ds.sortby("time")

    return ds


def _open_dataset(
    file: str, clat: "np.ndarray", clon: "np.ndarray", parallel: bool = True
) -> "xr.Dataset":
    import xarray as xr

    if parallel:
        ds = xr.open_dataset(file, engine="cfgrib", chunks="auto")
    else:
        ds = xr.open_dataset(file, engine="cfgrib")
    ds = flatten_step(ds.rename_dims({"values": "cell"}))
    ds = ds.assign_coords(clat=("cell", clat), clon=("cell", clon))
    ds = ds.chunk({"cell": -1, "time": -1})
    return ds


def get_max_parallel(
    client: "distributed.Client", mem_per_task_gb: float = 4.0, cap: int = 128
) -> int:
    """Derive max concurrent tasks from cluster memory."""
    info = client.scheduler_info()
    total_mem = sum(w["memory_limit"] for w in info["workers"].values())
    n_workers = len(info["workers"])

    # Leave 20% headroom per worker for overhead
    usable_mem = total_mem * 0.8
    from_mem = int(usable_mem / (mem_per_task_gb * 1e9))

    # Also cap per worker — no point flooding one node
    from_workers = n_workers * 8

    result = min(from_mem, from_workers, cap)
    logger.info(
        "Cluster: %d workers, %.0f GB total → max_parallel=%d",
        n_workers,
        total_mem / 1e9,
        result,
    )
    return max(1, result)


def download_files_distributed(
    client: "distributed.Client",
    urls: list[str],
    target_dir: Union[str, Path],
    timeout: int = 60,
    overwrite: bool = False,
    retries: int = 2,
    pure: bool = False,
    max_concurrent_downloads: int = 4,
) -> list[Path]:
    """Download a list of files in parallel using a Dask distributed client.

    Parameters
    ----------
    client : Client
        Connected ``dask.distributed.Client``.
    urls : list[str]
        URLs to download.
    target_dir : str or Path
        Directory where files should be written.  Prefer a shared
        filesystem when using multiple SLURM workers/nodes.
    timeout : int
        Per-request timeout in seconds.
    overwrite : bool
        Whether to overwrite existing files.
    retries : int
        Number of task retries at the Dask level.
    pure : bool
        Set ``False`` so identical URLs are not deduplicated by Dask.
    max_concurrent_downloads : int
        Maximum number of concurrent HTTP downloads.

    Returns
    -------
    list[Path]
        Paths to the downloaded files.
    """
    import distributed

    target_dir = str(Path(target_dir))
    semaphore_name = "icon-dream-downloads"
    distributed.Semaphore(max_leases=max_concurrent_downloads, name=semaphore_name)

    def _download_one(
        url: str,
        target_dir: str,
        semaphore_name: str,
        timeout: int = 60,
        overwrite: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        sem = distributed.Semaphore(name=semaphore_name)
        with sem:
            return gd_cli.download_file(
                url,
                target_dir,
                timeout=timeout,
                overwrite=overwrite,
                chunk_size=chunk_size,
            )

    futures = client.map(
        _download_one,
        urls,
        target_dir=target_dir,
        timeout=timeout,
        overwrite=overwrite,
        retries=retries,
        pure=pure,
        semaphore_name=semaphore_name,
    )
    results = client.gather(futures)
    return [Path(p) for p in results]


class HrefParser(HTMLParser):
    """Extract ``.grb`` links from an HTML directory listing."""

    file_suffix: str = ".grb"

    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value and value.endswith(self.file_suffix):
                    self.hrefs.append(value)


def process_one_file(
    filepath: str,
    weights: "xr.Dataset",
    clat: "np.ndarray",
    clon: "np.ndarray",
    max_level: int,
    output_path: str,
    s3_opts: Dict[str, str],
    semaphore_name: str = "regrid-tasks",
) -> str:
    """Open one GRIB file, regrid to HEALPix, write to S3."""

    import grid_doctor as gd

    sem = distributed.Semaphore(name=semaphore_name)
    with sem:
        ds = _open_dataset(filepath, clon=clon, clat=clat)
        pyramid = gd.latlon_to_healpix_pyramid(ds, max_level=max_level, weights=weights)

        gd.save_pyramid_to_s3(
            pyramid,
            output_path,
            s3_options=s3_opts,
            mode="a",
        )

        del ds, pyramid
        gc.collect()
        return filepath


def run_pipeline(args: "argparse.Namespace", temp_dir: Path) -> None:

    source = IconDreamSource(
        time_frequency=args.freq,
        variables=args.variables,
        time=tuple(args.time),
    )
    s3_opts = gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file)
    output_path = "s3://{bucket}/healpix/icon-dream/{time_frequency}".format(
            bucket=args.s3_bucket,
            time_frequency=args.freq,
    )
    data_dir = temp_dir / "dsets"
    if rank == 0:
        data_dir.mkdir(parents=True, exist_ok=True)

        # grid & weights (computed once, scattered to all workers)
        grid_file = gd_cli.download_file(source.grid_source, temp_dir / "grid")
        grid_ds = gd.cached_open_dataset([grid_file])
        max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
        weights = gd.cached_weights(grid_ds, level=max_level)

        clat = grid_ds.clat.values
        clon = grid_ds.clon.values

        
               s3_opts = gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file)

        # download all GRIB files
        files = download_files_distributed(
            client=client,
            urls=list(source.links),
            target_dir=data_dir,
        )
        first_file = str(files[0])
    logger.info("Initialising Zarr store from %s", first_file)
    ds_init = _open_dataset(first_file, clon=clon, clat=clat)
    pyramid_init = gd.latlon_to_healpix_pyramid(
        ds_init, max_level=max_level, weights=weights
    )
    pyramid_init = {k: v for k, v in pyramid_init.items()}
    gd.save_pyramid_to_s3(
        pyramid_init,
        output_path,
        s3_options=s3_opts,
        mode="a",
        encoding=gd.make_encoding(pyramid_init),
    )
    del ds_init, pyramid_init

    max_parallel = get_max_parallel(client)
    semaphore_name = "regrid-tasks"
    distributed.Semaphore(max_leases=max_parallel, name=semaphore_name)
    futures = client.map(
        process_one_file,
        [str(f) for f in files[1:]],
        weights=weights_future,
        clat=clat_future,
        clon=clon_future,
        max_level=max_level,
        output_path=output_path,
        s3_opts=s3_opts,
        semaphore_name="regrid-task",
        retries=1,
    )

    for future, result in distributed.as_completed(futures, with_results=True):
        logger.info("Completed: %s", result)

    shutil.rmtree(temp_dir)


@dataclass
class IconDreamSource:
    """Describes an ICON-DREAM data source on the DWD open-data server.

    Parameters
    ----------
    variables : list[IconDreamVariable]
        Variables to download.
    time_frequency : str
        One of ``"hourly"``, ``"daily"``, ``"monthly"``, ``"fx"``.
    time : tuple[str, str]
        Start and end time as free-form strings (parsed by dateparser).
    """

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
        """Parse a free-form datetime string to a timezone-aware datetime."""
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
        """Extract a YYYYMM datetime from a GRIB filename."""
        match = re.search(r"_(\d{6})_", href)
        if not match:
            raise ValueError(f"Could not extract YYYYMM from href: {href}")
        dt = datetime.strptime(match.group(1), "%Y%m").replace(tzinfo=timezone.utc)
        return href, dt

    def _get_download_link(self, variable: str) -> Iterator[str]:
        """Yield download URLs for one variable within the time range."""
        url = self._dir_source.format(
            freq=self.time_frequency,
            var=variable.upper().replace("-", "_"),
        )
        parser = HrefParser()
        with gd_cli.AutoRaiseSession() as session:
            response = session.get(url, timeout=5)
            parser.feed(response.text)

        for href, _date in map(self._href_to_datetime, parser.hrefs):
            if self._start <= _date <= self._end:
                logger.debug("Found %s/%s", url, href)
                yield f"{url}/{href}"

    @property
    def links(self) -> Iterator[str]:
        """Yield download URLs for all requested variables."""
        if self.time_frequency == "fx":
            yield self._invariants
        else:
            for var in self.variables:
                yield from self._get_download_link(var)


def parse_args(name: str, argv: Optional[List[str]] = None) -> "argparse.Namespace":
    """Create the argument parser for ICON-DREAM conversion."""
    parser = gd_cli.get_parser(
        name,
        description="Download and convert ICON-DREAM data to HEALPix.",
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
    return args


def main(name: str, argv: Optional[List[str]] = None) -> None:
    """Entry point for the ICON-DREAM conversion script."""

    args = parse_args(name, argv)
    tmp_dir = gd_cli.get_scratch("grid-doctor")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    run_pipeline(args, tmp_dir)
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main("icon-dream-catcher", sys.argv[1:])
