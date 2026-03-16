from icdc.base import Config, Pipeline

from grid_doctor import latlon_to_healpix_pyramid

ECADSpecMeanP1D = Config(
    dst_s3url="s3://icdc/healpix/atmosphere/ECAD/P1D/mean/",
    paths="/pool/data/ICDC/atmosphere/ecad_eobs/DATA/t*ens_mean_0.1deg_reg_v31.0e.nc",
    engine="netcdf-4",
    parallel=False,
    chunking={"time": 48},
    regrid_function=lambda x: latlon_to_healpix_pyramid(x, keep_nans=True),
    init=True,
    region={"time": slice(0, 96)},
    zarr_format=2,
)

ECADSpecSpreadP1D = Config(
    dst_s3url="s3://icdc/healpix/atmosphere/ECAD/P1D/spread/",
    paths="/pool/data/ICDC/atmosphere/ecad_eobs/DATA/t*ens_spread_0.1deg_reg_v31.0e.nc",
    engine="netcdf-4",
    parallel=False,
    chunking={"time": 48},
    regrid_function=latlon_to_healpix_pyramid,
    init=True,
    region={"time": slice(0, 96)},
    zarr_format=2,
)


def run():
    for spec in (ECADSpecMeanP1D, ECADSpecSpreadP1D):
        ECADPipeline = Pipeline(spec)
        ECADPipeline.run()


if __name__ == "__main__":
    run()
