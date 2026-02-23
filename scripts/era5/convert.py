#! python3
import numpy as np
import healpy as hp
import xarray as xr
import logging

from argparse import ArgumentParser
from glob import glob
from itertools import chain
from os import getenv

from grid_doctor import (
    cached_open_dataset,
    save_pyramid_to_s3,
    latlon_to_healpix_pyramid,
)

from typing import Iterable


def regrid(ds) -> xr.Dataset:
    from grid_doctor.helpers import get_latlon_resolution, resolution_to_healpix_level

    res = get_latlon_resolution(ds)
    zoom = resolution_to_healpix_level(res)
    nside = hp.order2nside(zoom)
    pixels = np.arange(hp.nside2npix(nside))

    hp_lon, hp_lat = hp.pix2ang(nside=nside, ipix=pixels, lonlat=True, nest=True)

    crs = xr.DataArray(
        name="crs",
        data=np.nan,
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": nside,
            "healpix_order": "nest",
        },
    )

    return (
        ds.drop_dims("bnds")
        .interp(
            lon=("cell", hp_lon),
            lat=("cell", hp_lat),
        )
        .assign(crs=crs)
        .assign_coords(cell=pixels)
    )


class Era5Layout:
    pattern = "/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/{0}/atmos/{1}/r1i1p1/{1}_*{0}_reanalysis_era5_r1i1p1_*.nc"
    freqMap = {"1hr": "PT1H", "day": "P1D", "mon": "P1M"}  # ISO 8601

    def __init__(self, store_url: str, variables: Iterable[str] = ("tas", "pr")):
        self.store_url = store_url.rstrip("/")
        self.vars = variables

    def __iter__(self) -> Iterator[Iterable[str], str]:
        for _freq, iso_freq in self.freqMap.items():
            files_chain = chain.from_iterable(
                glob(self.pattern.format(_freq, _var)) for _var in self.vars
            )
            yield files_chain, f"{self.store_url}/{iso_freq}"


def convert(init=False, region={"time": slice(0, 96)}):
    import zarr

    zarr.config.set(default_zarr_format=2)
    opts = {
        "endpoint_url": "https://s3.eu-dkrz-1.dkrz.cloud",
        "key": getenv("S3_KEY"),
        "secret": getenv("S3_SECRET"),
    }

    for files_chain, dst_url in Era5Layout("s3://icdc/healpix/era5/"):
        files = list(files_chain)
        if len(files) == 0:
            logging.warning("No files found to be written to %s", dst_url)
            continue

        logging.info(
            "Opening dataset from %s files to be regrided to %s", len(files), dst_url
        )
        ds = cached_open_dataset(files, engine="h5netcdf", parallel=False)
        logging.debug("%s", ds)

        if region["time"].start > ds.time.size:
            logging.warning(
                "Region (%s) not overlapping with dataset (%s), skipping!",
                region,
                {"time": slice(0, ds.time.size)},
            )
            continue

        logging.info("Converting to healpix")
        ds_hp = latlon_to_healpix_pyramid(ds.chunk({"time": 48}))
        logging.debug("%s", ds)

        if init:
            logging.info("Initializing store %s", dst_url)
            save_pyramid_to_s3(ds_hp, dst_url, mode="w", compute=False, s3_options=opts)

        else:
            region = {
                k: slice(v.start, min(v.stop, len(ds.time)), v.step)
                for k, v in region.items()
            }

            logging.info("Writting to existing store on region: %s", str(region))
            save_pyramid_to_s3(
                ds_hp, dst_url, mode="r+", region=region, s3_options=opts
            )


def main():
    start_idx = int(getenv("SLURM_ARRAY_TASK_ID", 0))
    parser = ArgumentParser()
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--slice-size", default=48, type=int)
    parser.add_argument("--start", default=start_idx, type=int)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    if args.slice_size % 48:
        print("slice-size must be a multiple of 48 (time chunk)")
        exit(1)

    region = {
        "time": slice(args.start * args.slice_size, (args.start + 1) * args.slice_size)
    }
    print(f"init={args.init}, region={region}")
    convert(init=args.init, region=region)


if __name__ == "__main__":
    main()
