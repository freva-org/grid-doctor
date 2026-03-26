from icdc.core import Config, Collection

from grid_doctor import latlon_to_healpix_pyramid

class CRUTEM(Collection):
    P1M = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/CRUTEM/P1M/",
        paths="/pool/data/ICDC/atmosphere/crutem/DATA/CRUTEM.5.0.2.0.anomalies.nc",
        engine="netcdf4",
        parallel=False,
        chunking={"time": -1, "latitude": -1, "longitude": -1},
        regrid_function=latlon_to_healpix_pyramid,
        init=True,
        region={"time": slice(0, 2102)},
        zarr_format=2,
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    CRUTEM.run_pipelines()
