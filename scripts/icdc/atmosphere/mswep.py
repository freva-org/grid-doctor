from icdc.core import Config, Collection

from grid_doctor import latlon_to_healpix_pyramid

class MSWEP(Collection):
    PT3H = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/MSWEP/PT3H/",
        paths="/pool/data/ICDC/atmosphere/mswep_precipitation/DATA/*/*.nc",
        engine="netcdf4",
        parallel=False,
        chunking={"time": 8},
        regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
        init=True,
        region={"time": slice(0, 16)},
        zarr_format=2,
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    MSWEP.run_pipelines()

