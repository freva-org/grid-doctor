# TODO

## create mapping

### Era5:
we will need to check how the whole grib timestamping goes

## apply zarr/healpix part to mapping

## two interval mode selections:

1. time range: remaps and adds whichever explicit time selection is given. if interval is < 5 months of execution time then global attribute stable batch  within that range, if not global attribute stable batch will be until the latest day of the interval
2. update: fill data to the latest, if exsisting data, replace it. 

## create update logic for the zarr pyramid

both in era5 and era5land there is an update of the data from the last ~ 4 months,
that is every day, the data of the time-4 months will be replaced to incorporate new assimilation data, bias etc. this needs to be replaced in-place, 

the better way would be to chirurgically locate the updated files and replace them in the zarr pyramid instead of rerunning the mapping in bulk for the last 4 months etc.
we should maybe also add a global metadata attribute that tells last permanent file to use it as indicattive that any data after that timestamp is "temporary".

the rough idea is that we have a flag that says --update or so, this:

1. check the current execution time, the latest zarr pyramid datetime, and the timestamps of the files between execution day + latest zarr datetime
2. we select all the files whose timestamps range between latest zarr datetime and execution time:
    - we check that we have the complete datetime range to extend to the "present" (this will be a combo between likely temp and stable data)
    - we will need to replace data in the zarr pyramid of already remapped values to update temporary batch with the stable values
3. we will need to update the global attribute of "stable batch" until latest datetime - 4 months (the latest timestamped file with the earliest data) 

