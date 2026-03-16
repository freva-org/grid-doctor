import logging

from icdc.base import Config, Pipeline

from grid_doctor import latlon_to_healpix_pyramid

HOAPSSpecPT6H = Config(
    dst_s3url="s3://icdc/healpix/atmosphere/HOAPS/PT6H/",
    paths="/pool/data/ICDC/atmosphere/hoaps/DATA/Precipitation/6hourly/*/PREic*SCPOS01GL.nc",
    engine="netcdf4",
    parallel=False,
    open_kwargs={"decode_timedelta": False},
    chunking={"time": 4},
    regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
    init=True,
    region={"time": slice(0, 48)},
    zarr_format=2,
)

HOAPSSpecP1M = Config(
    dst_s3url="s3://icdc/healpix/atmosphere/HOAPS/P1M/",
    paths="/pool/data/ICDC/atmosphere/hoaps/DATA/Precipitation/monthly/PRE*.nc",
    engine="netcdf4",
    parallel=False,
    open_kwargs={"decode_timedelta": False},
    chunking={"time": 12},
    regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
    init=True,
    region={"time": slice(0, 12)},
    zarr_format=2,
)


def run():
    for spec in (
        HOAPSSpecP1M,
        HOAPSSpecPT6H,
    ):
        try:
            HOAPSPipeline = Pipeline(spec)
            HOAPSPipeline.run()
        except Exception as e:
            logging.error(e)
            continue


if __name__ == "__main__":
    run()
