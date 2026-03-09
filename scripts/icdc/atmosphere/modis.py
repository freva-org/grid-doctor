from icdc.base import Config, Pipeline

from grid_doctor import latlon_to_healpix_pyramid

MODISSpecAqua = Config(
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

MODISSpecTerra = Config(
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


def run():
    for spec in (
        MODISSpecAqua,
        MODISSpecTerra,
    ):
        MODISPipeline = Pipeline(spec)
        MODISPipeline.run()


if __name__ == "__main__":
    run()
