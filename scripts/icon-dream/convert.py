"""Download/update and convert icron-dream in grib to healpix.

# About
Icon dream is a reanalysis product from the German Weather Service.

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

import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import (TYPE_CHECKING, Iterator, List, Literal, Optional, Tuple,
                    Union)

import dateparser
from dask_jobqueue.slurm import SLURMRunner
from distributed import Client, Semaphore

import grid_doctor.cli as gd_cli

if TYPE_CHECKING:
    import argparse

DATE_FORMAT = "%Y%m"

logger = logging.getLogger(__name__)

IconDreamVaraible = Literal[
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


def download_files_distributed(
    client: Client,
    urls: list[str],
    target_dir: Union[str, Path],
    timeout: int = 60,
    overwrite: bool = False,
    retries: int = 2,
    pure: bool = False,
    max_concurrent_downloads: int = 4,
) -> list[Path]:
    """
    Download a list of files in parallel using a Dask distributed client.

    Parameters
    ----------
    client:
        Connected dask.distributed.Client
    urls:
        URLs to download
    target_dir:
        Directory where files should be written. Prefer a shared filesystem
        when using multiple Slurm workers/nodes.
    timeout:
        Per-request timeout in seconds
    overwrite:
        Whether to overwrite existing files
    retries:
        Number of task retries at the Dask level
    pure:
        Set False so identical URLs are not deduplicated by Dask

    Returns
    -------
    list[Path]
        Paths to the downloaded files
    """
    target_dir = str(Path(target_dir))
    semaphore_name = "icon-dream-downloads"
    Semaphore(max_leases=max_concurrent_downloads, name=semaphore_name)

    def _download_one(
        url: str,
        target_dir: str,
        semaphore_name: str,
        timeout: int = 60,
        overwrite: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> str:
        sem = Semaphore(name=semaphore_name)

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
    file_suffix: str = ".grb"

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for key, value in attrs:
                if key == "href" and value.endswith(self.file_suffix):
                    self.hrefs.append(value)


def run_pipeline(client: Client, args: "argparse.Namespace") -> None:

    import grid_doctor as gd
    from data_portal_worker.rechunker import ChunkOptimizer

    source = IconDreamSource(
        time_frequency=args.freq,
        variables=args.variables,
        time=tuple(args.time),
    )

    temp_dir = gd_cli.get_scratch("grid-doctor")
    data_dir = temp_dir / "dsets"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Create/register the semaphore via the client process
    grid_file = gd_cli.download_file(source.grid_source, temp_dir / "grid")
    grid_ds = gd.cached_open_dataset([grid_file])
    max_level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
    weights = gd.cached_weights(grid_ds, level=max_level)
    files = download_files_distributed(
        client=client,
        urls=list(source.links),
        target_dir=data_dir,
    )

    dset = (
        gd.cached_open_dataset(files, combine_attrs="drop_conflicts")
        .rename_dims({"values": "cell"})
        .chunk({"cell": -1})
    )
    healpix_pyramid = gd.latlon_to_healpix_pyramid(
        dset, max_level=max_level, weights=weights
    )

    opt = ChunkOptimizer()
    chunked_heal_pix = {k: opt.apply(d) for k, d in healpix_pyramid.items()}

    output_path = "s3://{bucket}/healpix/icon-dream/{time_frequency}"

    gd.save_pyramid_to_s3(
        chunked_heal_pix,
        output_path.format(bucket=args.s3_bucket, time_frequency=args.freq),
        s3_options=gd.get_s3_options(args.s3_endpoint, args.s3_credentials_file),
    )
    shutil.rmtree(temp_dir)


@dataclass
class IconDreamSource:
    variables: List[IconDreamVaraible]
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
            "ICON-DREAM-Global/invariant/ICON-DREAM-Global_constant_fields.grb"
        )

        _start, _end = self.time
        self._start = self._parse_user_datetime(self.time[0])
        self._end = self._parse_user_datetime(self.time[-1])

    def _parse_user_datetime(str, value: str) -> datetime:
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

    def _href_to_datetime(self, href: str) -> datetime:
        match = re.search(r"_(\d{6})_", href)
        if not match:
            raise ValueError(f"Could not extract YYYYMM from href: {href}")
        return href, datetime.strptime(match.group(1), "%Y%m").replace(
            tzinfo=timezone.utc
        )

    def _get_download_link(self, variable: str) -> Iterator[str]:

        url = self._dir_source.format(
            freq=self.time_frequency, var=variable.upper().replace("-", "_")
        )
        parser = HrefParser()
        with gd_cli.AutoRaiseSession() as session:
            response = session.get(url, timeout=5)
            parser.feed(response.text)
        for href, _date in map(self._href_to_datetime, parser.hrefs):
            if _date <= self._end and _date >= self._start:
                uri = f"{url}/{href}"
                logger.debug("Found %s", uri)
                yield f"{url}/{href}"

    @property
    def links(self) -> Iterator[str]:
        """Get a collection of download girb file."""
        if self.time_frequency == "fx":
            yield self._invariants
        else:
            for var in self.variables:
                yield from self._get_download_link(var)


def parse_args(name: str, argv: Optional[List[str]] = None) -> "argparse.Namespace":
    """Create a agrument parser."""

    parser = gd_cli.get_parser(
        name,
        description="Download an convert ICON-DREAM data.",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=["t_2m", "tot_prec"],
        choices=IconDreamVaraible.__args__,
    )
    parser.add_argument(
        "--freq",
        "--time-frequency",
        default="hourly",
        choices=["hourly", "daily", "monthly", "fx"],
        help="Time frequency of the data.",
    )
    parser.add_argument("--time", nargs=2, default=["2010-01-01T00:00", "now"])
    args = parser.parse_args(argv)
    gd_cli.setup_logging_from_args(args)
    return args


def main(name: str, argv: Optional[List[None]] = None) -> None:
    args = parse_args(name, argv)
    tmp_dir = gd_cli.get_scratch("grid-doctor")
    tmp_dir.mkdir(exist_ok=True, parents=True)
    if os.getenv("SLURM_PROCID") or os.getenv("SLURM_JOB_ID"):
        job_id = os.getenv("SLURM_JOB_ID")
        scheduler_file = tmp_dir / f"scheduler-{job_id}.json"
        scheduler_file.parent.mkdir(exist_ok=True, parents=True)
        with SLURMRunner(scheduler_file=str(scheduler_file)) as runner:
            client = Client(runner)
            run_pipeline(client, args)
    else:
        # Local debug mode
        client = Client(processes=False, dashboard_address=None)
        try:
            run_pipeline(client, args)
        finally:
            client.close()


if __name__ == "__main__":
    main(os.path.basename(sys.argv[0]), sys.argv[1:])
