from icdc.base import Config, Pipeline

from grid_doctor import latlon_to_healpix_pyramid

NCEPSpecPT6H = Config(
    dst_s3url="s3://icdc/healpix/reanalysis/NCEP/PT6H/",
    paths="/pool/data/ICDC/reanalyses/ncep_reanalysis1/DATA/2m_airtemp/*.nc",
    engine="netcdf4",
    parallel=False,
    chunking={"time": 8},
    regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
    init=True,
    region={"time": slice(0, 16)},
    zarr_format=2,
)

NCEPSpecP1M = Config(
    dst_s3url="s3://icdc/healpix/reanalysis/NCEP/P1M/",
    paths="/pool/data/ICDC/reanalyses/ncep_reanalysis1/DATA/2m_airtemp_monthly/air2m.mon.mean.nc",
    engine="netcdf4",
    parallel=False,
    chunking={"time": -1},
    regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
    init=True,
    region={"time": slice(0, 938)},
    zarr_format=2,
)


def run():
    for spec in (
        NCEPSpecPT6H,
        NCEPSpecP1M,
    ):
        NCEPPipeline = Pipeline(spec)
        NCEPPipeline.run()


if __name__ == "__main__":
    run()  # main()
