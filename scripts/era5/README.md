# ERA5 

## Sources
The source of the data is the outcome of Etor CMORization and postprocessing

There is an alternative based on kerchunked data: https://gitlab.dkrz.de/data-infrastructure-services/era5-kerchunks/-/raw/main/main.yaml
However 2023 was the last year for that source (to be confirmed)

## Current state

As a prove of concept only hourly temperature and precipitation were converted and there's no precipitation data for the first 6 hours

https://gridlook.pages.dev/#https://s3.eu-dkrz-1.dkrz.cloud/icdc/healpix/era5.zarr

As of now:
 - `xr.interp` was used because the input grid is regular, but the `regrid` helper function should be used
 - only max healpix level was generated
 - `to_zarr` was used instead of `save_pyramid_to_s3`

## Approach

Open all the files that comprise the dataset via `open_mfdataset` and cache the resulting `xarray.Dataset`

Regrid the dataset `lazily` using `xr.apply_ufunc`
 - Might consider pickle at this stage as well

If first time running: 
 - Write metadata only first

Else:
 - Launch jobs that write slices of the dataset (potencially in parallel)


### Improvements/Alternatives

For this dataset it is possible to open and process the input files in a pipeline.
The first file being processed creates the dataset and the subsequent ones append to it on the `time` dimension

This approach might be benefic for cases were the ammount of input files is very large.




