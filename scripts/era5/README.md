# ERA5 

## Sources
The source of the data is the outcome of Etor CMORization and postprocessing

There is an alternative based on kerchunked data: https://gitlab.dkrz.de/data-infrastructure-services/era5-kerchunks/-/raw/main/main.yaml
However 2023 was the last year for that source (to be confirmed)

## Current state

### Approach

For each time frequency:

 - Open all the files that comprise the dataset via `open_mfdataset` and cache the resulting `xarray.Dataset`

 - Regrid the dataset `lazily` to the max zoom level

 - If first time running
   - `--init` initialized only the coordinates and dimentions in the zarr store

 - Otherwise:
   - Launch jobs that write slices of the dataset in parallel.

 - Lastly, use the coarsen tool for the lower levels

### Array job submission:

```
# Initialize hourly store
python3 scripts/era5/convert.py --init hourly /work/ks1387/era5-redone
# Write full hourly data using 1000 job arrays (max) it will split the time axis over total number of jobs
sbatch -p compute -Ak20200 --mem 16G --array=0-999 --time 00:10:00  <(echo -e '#!/bin/sh\n~/micromamba/envs/grid-doctor/bin/python3 scripts/era5/convert.py hourly /work/ks1387/era5-redonev')
```
