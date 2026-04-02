from icdc.core import Config, Pipeline, Collection
from grid_doctor import latlon_to_healpix_pyramid
from xarray import Dataset


def preprocess(ds: Dataset) -> Dataset:
    '''must fix CF convention issues'''
    import xarray as xr
    import pandas as pd
    converted = ds.assign(time = xr.decode_cf(ds[['time']]).time)
    for v in ds.variables:
        # Patch differences between missing and fill value leading to NaNs
        fill = converted[v].attrs.get('_FillValue')
        miss = converted[v].attrs.get('missing_value')
        if fill is not None and miss is not None and fill != miss:
            converted[v].attrs['missing_value'] = fill

        # These are relative to the time value in each file
        if v in ('utctime_desc','utctime_asc',):
            converted[v].attrs["units"] = f"hours since {pd.Timestamp(converted['time'][0].values).strftime('%Y-%m-%d %H:%M:%S')} UTC"
    return xr.decode_cf(converted)

class ASCAT(Collection):
    METOP_A    = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/ASCAT/METEOP_A/P1D/",
        paths="/pool/data/ICDC/atmosphere/ascat_metop_a_surfacewind/DATA/2021/EUMETSAT_MetOp-ASCAT__OceanSurfaceWind__REMSS_UHAM-ICDC__0.25deg__202101*v02.1.nc",
        open_kwargs={'decode_cf':False, 'preprocess':preprocess},
        engine="netcdf4",
        parallel=False,
        chunking={"time": 48},
        regrid_function=latlon_to_healpix_pyramid,
        init=True,
        region={"time": slice(0, 96)},
        zarr_format=2,
    )


    METOP_B  = Config(
        dst_s3url="s3://icdc/healpix/atmosphere/ASCAT/METOP_B/P1D/",
        paths="/pool/data/ICDC/atmosphere/ascat_metop_b_surfacewind/DATA/2021/*v02.1.nc",
        open_kwargs={'decode_cf':False, 'preprocess':preprocess},
        engine="netcdf4",
        parallel=False,
        chunking={"time": 48},
        regrid_function=latlon_to_healpix_pyramid, #lambda x: latlon_to_healpix_pyramid(x, missing_policy='propagate'),
        init=True,
        region={"time": slice(0, 96)},
        zarr_format=2,
    )
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    ASCAT.run_pipelines()
