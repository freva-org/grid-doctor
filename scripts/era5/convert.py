#! python3
import numpy as np
import healpy as hp
import xarray as xr
import logging

from argparse import ArgumentParser
from glob import glob
from itertools import chain
from os import getenv

from grid_doctor import cached_open_dataset


def regrid(ds) -> xr.Dataset:
    from grid_doctor.helpers  import get_latlon_resolution, resolution_to_healpix_level
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
                "healpix_order": "nest"
                },
            )

    
    return ds.drop_dims('bnds') \
            .interp(lon=("cell", hp_lon),lat=("cell", hp_lat),) \
            .assign(crs=crs) \
            .assign_coords(cell=pixels)


def convert(init=False, region={'time': slice(0,96)}):
    pattern = "/work/bm1159/XCES/data4xces/reanalysis/reanalysis/ECMWF/IFS/ERA5/1hr/atmos/{0}/r1i1p1/{0}_1hr_reanalysis_era5_r1i1p1_*.nc"
    files = list(
        chain.from_iterable(
            glob(pattern.format(var)) for var in ("tas", "pr")
        )
    )
    
    logging.info("Opening dataset from %s files", len(files))
    ds = cached_open_dataset(files, engine='h5netcdf', parallel=False)
    logging.debug("%s", ds)

    logging.info("Converting to healpix")
    ds_hp = ds.pipe(regrid).chunk({'time':48})
    logging.debug("%s", ds)

    opts = {
        'zarr_format': 2,
        'storage_options': {
            "endpoint_url":"https://s3.eu-dkrz-1.dkrz.cloud",
            "key": getenv("S3_KEY"), 
            "secret": getenv("S3_SECRET"),
        }
    }
    
    dst_url="s3://icdc/healpix/era5.zarr"
    if init:
        logging.info("Initializing store")
        ds_hp.to_zarr(dst_url,
                mode = 'w',
                compute=False,
                **opts)
        ds_hp[['lat','lon']].to_zarr(dst_url,
                mode = 'r+',
                **opts)
        
    else:
        region={
                k: slice(v.start, min(v.stop,len(ds.time)), v.step)
                for k,v in region.items()
            }
        
        logging.info("Writting to existing store on region: %s", str(region))
        ds_hp.drop_vars(set(ds.dims)& set(ds.coords)).drop_vars(['crs', 'cell']).isel(region).to_zarr(dst_url,
            mode = 'r+',
            region = region,
            **opts,
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
        print('slice-size must be a multiple of 48 (time chunk)')
        exit(1)

    region = {'time':slice(args.start * args.slice_size, (args.start+1) * args.slice_size)}
    print(f"init={args.init}, region={region}")
    convert(init=args.init, region=region)

if __name__ == "__main__":
    main()
