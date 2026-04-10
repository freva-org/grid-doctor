from icdc.core import Config, Pipeline, Collection

from grid_doctor import latlon_to_healpix_pyramid


class ECAD(Collection):
    MEAN    = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/ECAD/P1D/mean/",
        paths="/pool/data/ICDC/atmosphere/ecad_eobs/DATA/t*ens_mean_0.1deg_reg_v31.0e.nc",
        engine="netcdf4",
        parallel=False,
        chunking={"time": 48},
        regrid_function=lambda x: latlon_to_healpix_pyramid(x, method='nearest', max_level=8, missing_policy='propagate'),
        init=True,
        region={"time": slice(0, 96)},
        zarr_format=2,
    )

    SPREAD   = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/ECAD/P1D/spread/",
        paths="/pool/data/ICDC/atmosphere/ecad_eobs/DATA/t*ens_spread_0.1deg_reg_v31.0e.nc",
        engine="netcdf4",
        parallel=False,
        chunking={"time": 48},
        regrid_function=lambda x: latlon_to_healpix_pyramid(x, method='nearest', max_level=8, missing_policy='propagate'),
        init=True,
        region={"time": slice(0, 96)},
        zarr_format=2,
    )




if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    ECAD.run_pipelines()
