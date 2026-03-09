from icdc.base import Config, Pipeline

from grid_doctor import latlon_to_healpix_pyramid

CRUTEMSpec = Config(
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


CRUTEMPipeline = Pipeline(CRUTEMSpec)


def run():
    CRUTEMPipeline.run()


if __name__ == "__main__":
    run()  # main()
