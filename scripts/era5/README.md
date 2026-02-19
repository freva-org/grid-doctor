# ERA5 

## Sources
The source of the data is the outcome of Etor CMORization and postprocessing

There is an alternative based on kerchunked data: https://gitlab.dkrz.de/data-infrastructure-services/era5-kerchunks/-/raw/main/main.yaml
However 2023 was the last year for that source (to be confirmed)

## Current state

As a prove of concept only hourly temperature and precipitation were converted and there's no precipitation data for the first 6 hours

https://eu-dkrz-1.dkrz.cloud/browser/icdc/healpix/era5/

https://gridlook.pages.dev/#https://s3.eu-dkrz-1.dkrz.cloud/icdc/healpix/era5/P1M/level_7

### Approach

For each time frequency:

 - Open all the files that comprise the dataset via `open_mfdataset` and cache the resulting `xarray.Dataset`

 - Regrid the dataset `lazily` using `xr.apply_ufunc`
  - `scipy.scipy.griddata(method=nearest)` is used
  - Might consider pickle at this stage as well

 - If first time running: 
   - Write metadata only first

 - Else:
   - Launch jobs that write slices of the dataset (potencially in parallel)


`scripts/era5/slum.sh` prints the commands that launch the 2 jobs. The first to initialize the dataset, the second to schedule all the tasks that will populate it. For this readon the **first should succeed before the second is triggered**

It is taking roughly 3 hours and 30 min (max 4:40) for a time slice of 1440 values to be written accross **all** time frequencies for each zoom level (0-7)


## Discuss

  - `lat_bnds` and `lon_bnds` aren't taken into account during the regridding, should they?
  - Only variables with spacial dimensions are kept, should it apply to all?

### Improvements/Alternatives

For this dataset it is possible to open and process the input files in a pipeline.
The first file being processed creates the dataset and the subsequent ones append to it on the `time` dimension

This approach might be benefic for cases were the ammount of input files is very large.
