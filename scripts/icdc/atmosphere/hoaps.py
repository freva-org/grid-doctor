import logging

from icdc.core import Config, Collection

from grid_doctor import latlon_to_healpix_pyramid


class HOAPS(Collection):
    PT6H = Config(
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

    P1M = Config(
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


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    HOAPS.run_pipelines()
