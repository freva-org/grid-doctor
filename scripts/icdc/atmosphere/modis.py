from icdc.core import Config, Pipeline, Collection

from grid_doctor import latlon_to_healpix_pyramid

class MODIS(Collection):
    AQUA = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/MODIS/aqua/P1D/",
        paths="/pool/data/ICDC/atmosphere/modis_aqua_watervapor_pwc_temperature/DATA/{year}/MODIS-C6.1__MYD08__daily__watervapor-parameters__[0-9]*__UHAM-ICDC__fv0.1.nc",
        engine="netcdf4",
        parallel=False,
        chunking={"time": 8},
        regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
        init=True,
        region={"time": slice(0, 16)},
        zarr_format=2,
    )

    TERRA = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/MODIS/terra/P1D/",
        paths="/pool/data/ICDC/atmosphere/modis_terra_watervapor_pwc_temperature/DATA/{year}/MODIS-C6.1__MOD08__daily__watervapor-parameters__[0-9]*__UHAM-ICDC__fv0.1.nc",
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
    MODIS.run_pipelines()
