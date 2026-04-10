from icdc.core import Config, Collection
from grid_doctor import latlon_to_healpix_pyramid

class NCEP(Collection):
    PT6H = Config(
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

    P1M = Config(
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

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    NCEP.run_pipelines()
