# TODO



- [x] remove batch-mode, leave it subprocess
- [x] remove the option of record-thread from the code (is already false)
- [x] remove the lazy pyramid strategy, default is already stepwise
---
- [x] vertical level correct adding
- [] areacella calculation
- [] update to the latest dates with quirugicall record replacement
- [] the batchfolders in era5
- [] reflow to be able to send multiple jobs and then collect them.
- [] cf checking?

---
if I need some extra info (`cache_key`) to create pickeld filenames in [grib.py](helpers/grib.py)
```python
        if use_cache:
            ds_raw = gd.cached_open_dataset(
                files_for_var,
                **open_kwargs,
                cache_key={
                    "engine": "cfgrib",
                    "shortName": short_name,
                    "paramId": int(param_id),
                    "typeOfLevel": type_of_level,
                    "level": int(level),
                    "time_normalizer": "valid_time_v1",
                },
            )
        else:
            ds_raw = xr.open_mfdataset(
                files_for_var,
                **open_kwargs,
                parallel=True,
                chunks="auto",
            )
```            


I will modify back in [grid_doctor/utils.py](../../src/grid_doctor/utils.py)

```python
def cached_open_dataset(files: Collection[str], **kwargs: Any) -> xr.Dataset:
    """Open multiple files and cache the merged dataset as a pickle.

    Parameters
    ----------
    files:
        Input file paths or glob-expanded file names.
    **kwargs:
        Extra keyword arguments for `xarray.open_mfdataset`.

    Returns
    -------
    xarray.Dataset
        The opened dataset.
    """
    cache_key = kwargs.pop("cache_key", None) #<---

    digest = hashlib.sha256()
    normalised = sorted({str(path) for path in files})
    digest.update("\0".join(normalised).encode())
    if cache_key is not None: #<--->
        digest.update(repr(cache_key).encode()) #<--->
    pickle_file = cache_dir() / f"{digest.hexdigest()}.pickle"

    if pickle_file.exists():
        try:
            with pickle_file.open("rb") as handle:
                return cast(xr.Dataset, pickle.load(handle))  # nosec B301  # noqa: S301
        except Exception as exc:  # pragma: no cover - defensive cache cleanup
            logger.warning("Could not read cached dataset %s: %s", pickle_file, exc)
            pickle_file.unlink(missing_ok=True)

    from dask.diagnostics.progress import ProgressBar

    merged_kwargs: dict[str, Any] = {"parallel": True, "chunks": "auto"} | kwargs
    with ProgressBar():
        dataset = xr.open_mfdataset(normalised, **merged_kwargs)

    with pickle_file.open("wb") as handle:
        pickle.dump(dataset, handle)
    return dataset
```

---

## create mapping


apparenlty is better to create the highes zzom level and then coarsen, the coarsen can be done directly calling the coarsen_healpix() methiod from src/grid_ctor/helpers.py

this is mapper.map_grib_to_healpix()


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
