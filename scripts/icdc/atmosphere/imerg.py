from icdc.core import Config, Collection

from grid_doctor import latlon_to_healpix_pyramid

class IMERG(Collection):
    PT30M = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/IMERG/PT30M/",
        paths="/pool/data/ICDC/atmosphere/imerg/DATA/2025/IMERG_precipitationrate__V07B__halfhourly__0.1degree__*.nc",
        engine="netcdf-4",
        parallel=False,
        chunking={"time": 48},
        regrid_function=latlon_to_healpix_pyramid,
        init=True,
        region={"time": slice(0, 96)},
        zarr_format=2,
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    IMERG.run_pipelines()
