from icdc.core import Config, Collection

from grid_doctor import latlon_to_healpix_pyramid

class BNSC(Collection):
    P1M = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/BNSC/P1M/",
        paths="/pool/data/ICDC/atmosphere/bnsc/DATA/v2/air_temperature/monthly/BNSC_Air_Temperature___UHAM_ICDC__v2__1deg__*.nc",
        engine="netcdf4",
        parallel=False,
        chunking={"time": 12},
        regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
        init=True,
        region={"time": slice(0, 12)},
        zarr_format=2,
    )

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    BNSC.run_pipelines()

