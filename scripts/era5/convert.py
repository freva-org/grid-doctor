#! python3
import xarray as xr
import logging

from argparse import ArgumentParser, Namespace
from glob import glob
from itertools import chain
from os import getenv

import grid_doctor as gd

from typing import Iterable
from enum import StrEnum


class Frequency(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"

    def to_cmor(cls):
        if cls == Frequency.HOURLY:
            return "1hr"
        elif cls == Frequency.DAILY:
            return "day"
        else:
            return "mon"


class Era5Layout:
    pattern = "/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/{0}/atmos/{1}/r1i1p1/{1}_*{0}_reanalysis_era5_r1i1p1_*.nc"
    freqMap = {"1hr": "PT1H", "day": "P1D", "mon": "P1M"}  # ISO 8601

    def __init__(
        self,
        frequency: Frequency,
        init: bool,
        base_path: str,
        variables: Iterable[str] = ("tas", "pr"),
        overwrite: bool = False,
    ):
        self.frequency = frequency
        self.iso_frequency = self.freqMap[frequency.to_cmor()]
        self.init = init
        self.base_path = f"{base_path.rstrip('/')}/{self.iso_frequency}"
        self.vars = variables
        self.overwrite = overwrite

    def files(self) -> list[str]:
        return list(
            chain.from_iterable(
                glob(self.pattern.format(self.frequency.to_cmor(), _var))
                for _var in self.vars
            )
        )

    def open(self) -> xr.Dataset:
        files = self.files()
        if len(files) == 0:
            logging.warning("No files found to be written to %s", self.base_path)
            return

        logging.info(
            "Opening dataset from %s files to be regrided to %s",
            len(files),
            self.base_path,
        )
        ds = gd.cached_open_dataset(files, engine="h5netcdf", parallel=False)
        logging.debug("%s", ds)
        return ds

    @classmethod
    def from_args(cls, args: Namespace):
        return cls(
            args.frequency,
            init=args.init,
            base_path=args.location,
            overwrite=args.overwrite,
        )


def convert(era, region={"time": slice(0, 96)}, chunk_size=48):

    ds = era.open()
    level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))
    store = f"{era.base_path}/level_{level}.zarr"

    logging.info("Converting to healpix")
    ds_hp = gd.regrid_to_healpix(ds.chunk({"time": chunk_size}), level=level)
    logging.debug("%s", ds_hp)

    if era.init:
        logging.info("Initializing store %s", era.base_path)
        gd.utils.init_full_zarr_store(ds_hp, store, overwrite=era.overwrite)

    else:
        region = gd.utils.get_slurm_region(ds.time.size, chunk_size)

        logging.info("Writting to existing store on region: %s", str(region))
        ds_hp.isel(region).drop_vars(
            ["lon", "lat", "cell", "latitude", "longitude", "crs"]
        ).to_zarr(store, mode="r+", region=region)


def parse_args():
    start_idx = int(getenv("SLURM_ARRAY_TASK_ID", 0))
    parser = ArgumentParser()
    parser.add_argument(
        "frequency",
        choices=tuple(f for f in Frequency),
        type=Frequency,
    )
    parser.add_argument(
        "location",
        type=str,
        help="Location where to write the dataset pyramid (parent for each zarr store!)",
    )
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--slice-size", default=48, type=int)
    parser.add_argument("--start", default=start_idx, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    if args.slice_size % 48:
        print("slice-size must be a multiple of 48 (time chunk)")
        exit(1)

    era = Era5Layout.from_args(args)
    convert(era)


if __name__ == "__main__":
    main()
