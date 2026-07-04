# ICON-DREAM

## What is icon-dream?

Icon dream is a reanalysis product from the German Weather Service.

## Variables, CMOR names and units

By default the pipeline processes **all** variables available on the DWD
open data server for the chosen frequency (the per-variable
sub-directories of e.g.
`https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-Global/hourly/`
are discovered at plan time). To restrict the run, pass an explicit list,
e.g. `--variables t_2m tot_prec`; `--variables all` is the explicit form
of the default.

Variables are renamed to their CMOR/CF counterparts and units are
converted where necessary (e.g. `t_2m -> tas`, `pmsl -> psl`,
`tot_prec [kg m-2 per interval] -> pr [kg m-2 s-1]`). The mapping lives in
one declarative table,
`icon_dream_reflow_helpers/cmor.py::CMOR_TABLE`; add or adjust entries
there. Pass `--cmor false` (slurm mode) or `--no-cmor` (simple mode) to
keep the original ICON names and units. Two caveats worth knowing:

* `tot_prec` is currently treated as the amount accumulated over each
  output interval. If the product turns out to be accumulated since model
  start, flip the entry's `conversion` to `"deaccumulate_rate"`.
* `qv_s` (surface/skin specific humidity) is mapped to `huss` as the
  closest CMOR analogue; the `long_name` records the difference.

Note that the update-only bookkeeping compares against the names stored
in the target, so a store written with `--cmor` must also be updated with
`--cmor` (and vice versa).

## Writing to local disk (serving via versitygw)

The target does not have to be S3. With `--fs-type posix` the `path`
argument is interpreted as a local directory and the Zarr pyramids are
written straight to disk, which is the mode to use when the data are
served by an S3 gateway such as versity afterwards:

```console
REFLOW_ACCOUNT=foo python convert.py submit --run-dir \
    /scratch/k/$USER/grid-doctor --fs-type posix \
    --path /work/ks1387/waterpark-data/icon-dream --freq hourly
```

The final stores end up under
`<path>/healpix/reanalysis/icon-dream-global/icon/<freq>/level_<n>.zarr`
and can be exposed by pointing a versitygw bucket at `<path>`. S3
credentials/endpoint arguments are ignored in this mode.

## Adding variables to an existing store

Re-running `submit` with additional (or `all`) variables against an
existing target is supported: variables already present are only extended
in time (`--update-only`, default), while variables missing from the
store are backfilled across the store's full time axis and then kept in
sync with the shared time axis on subsequent appends.



## How to run:

There are two main modes:

1. the "normal" slurm mode that uses [`reflow`](https://www.reflow.docs.org)
 to split the remap jobs into smaller chunks that can be handled within the
 maximum wall time of 8 hours. The file `convert.py` hosts the "slurm" mode.
1. the "simple" mode doesn't split the tasks into smaller chunks. It calculates
   the HEALPix pyramid using the `cupy` backend and pushes the results directly
   to S3. This mode is not slurm "friendly" but can be directly and easily
   applied on servers without resource restrictions such as the DRKZ
   Grace-Hopper nodes. The file `convert_gh.py` hosts the "simple" mode.


Follow these steps to apply the *slurm mode*:

1. Create and download your s3-secrets file from [https://eu-dkrz-3.dkrz.cloud/access-keys](https://eu-dkrz-3.dkrz.cloud/access-keys)
2. Put the secrets files somewhere into your home on levante.
3. Install the requirements
```console
mamba install -c conda-forge -y -f requirements.yml
```
4. The `convert.py` uses [`reflow`](https://www.reflow.docs.org) to define a
   workflow. Reflow wraps the script in a simple flat cli that define slurm
   jobs:
```console
python convert.py submit --help
Usage: icon_dream_healpix submit [-h] --run-dir RUN_DIR [--store-path STORE_PATH] [--access-pattern {map,time_series}] [--cell-chunk CELL_CHUNK]
                                 [--compression-level COMPRESSION_LEVEL] [--download-chunk-size DOWNLOAD_CHUNK_SIZE] [--download-timeout DOWNLOAD_TIMEOUT] [--end END]
                                 [--freq {hourly,daily,monthly,fx}] [--local-dask-workers LOCAL_DASK_WORKERS] [--max-level MAX_LEVEL]
                                 [--overwrite-downloads OVERWRITE_DOWNLOADS] [--overwrite-static OVERWRITE_STATIC] [--replace-existing-times REPLACE_EXISTING_TIMES]
                                 --s3-bucket S3_BUCKET [--s3-credentials-file S3_CREDENTIALS_FILE] [--s3-endpoint S3_ENDPOINT]
                                 [--source-backend-kwargs-json SOURCE_BACKEND_KWARGS_JSON] [--source-engine SOURCE_ENGINE] [--source-root SOURCE_ROOT] [--start START]
                                 [--strict-access-pattern STRICT_ACCESS_PATTERN] [--time-chunk TIME_CHUNK] [--update-only UPDATE_ONLY]
                                 [--variables VARIABLES [VARIABLES ...]] [--zarr-format {2,3}]

Options:
  -h, --help            show this help message and exit
  --run-dir RUN_DIR     Shared working directory. (default: None)
  --store-path STORE_PATH
                        Explicit path to SQLite manifest. (default: None)
  --access-pattern {map,time_series}
                        Chunking optimisation pattern (default: map)
  --cell-chunk CELL_CHUNK
                        Cell chunk size for temporary Zarr stores (default: 262144)
  --compression-level COMPRESSION_LEVEL
                        Compression level for final Zarr encoding (default: 4)
  --download-chunk-size DOWNLOAD_CHUNK_SIZE
                        HTTP stream chunk size in bytes (default: 1048576)
  --download-timeout DOWNLOAD_TIMEOUT
                        HTTP timeout in seconds (default: 60)
  --end END             Requested UTC end time (default: now)
  --freq {hourly,daily,monthly,fx}
                        ICON-DREAM data frequency (default: hourly)
  --local-dask-workers LOCAL_DASK_WORKERS
                        Optional local distributed workers inside one process (default: 0)
  --max-level MAX_LEVEL
                        Override the automatically chosen HEALPix level (default: None)
  --overwrite-downloads OVERWRITE_DOWNLOADS
                        Re-download the grid file even if it exists (default: False)
  --overwrite-static OVERWRITE_STATIC
                        Overwrite an existing static target store (default: False)
  --replace-existing-times REPLACE_EXISTING_TIMES
                        Rewrite overlapping time slices for already-present variables (default: False)
  --s3-bucket S3_BUCKET
                        Target S3 bucket (default: None)
  --s3-credentials-file S3_CREDENTIALS_FILE
                        Path to S3 credentials JSON (default: /home/k/k204230/.s3-credentials.json)
  --s3-endpoint S3_ENDPOINT
                        S3 endpoint URL (default: https://s3.eu-dkrz-3.dkrz.cloud)
  --source-backend-kwargs-json SOURCE_BACKEND_KWARGS_JSON
                        JSON backend_kwargs for xarray (default: {})
  --source-engine SOURCE_ENGINE
                        Xarray backend engine for source files (default: cfgrib)
  --source-root SOURCE_ROOT
                        Source dataset root URL (default: https://opendata.dwd.de/climate_environment/REA/ICON-DREAM-Global)
  --start START         Requested UTC start time (default: 2010-01-01T00:00)
  --strict-access-pattern STRICT_ACCESS_PATTERN
                        Enforce strict chunking for the chosen access pattern (default: True)
  --time-chunk TIME_CHUNK
                        Time chunk size for temporary Zarr stores (default: 168)
  --update-only UPDATE_ONLY
                        Skip source chunks already covered by existing variables (default: True)
  --variables VARIABLES [VARIABLES ...]
                        Variables to process (default: ['t_2m', 'tot_prec'])
  --zarr-format {2,3}   Target Zarr format version (default: 2)

```
To submit the job choose your slurm partition and add any arguments for the
`convert.py` script for example:

```console
REFLOW_ACCOUNT=foo python convert.py submit --run-dir \
    /scratch/k/$USER/grid-doctor --s3-bucket icon-dream
```
This command will submit a chain of slurm jobs. You can either use `squeue`
to check the job status or

```console
python convert.py runs
python convert.py status <run-id>
```

To apply the *simple mode* use the `convert_gh.py` script:

```console
python convert_gh.py --help
Usage: convert-icon-dream [-h] --s3-bucket S3_BUCKET [--s3-endpoint S3_ENDPOINT] [--s3-credentials-file S3_CREDENTIALS_FILE] [-v] [--variables VARIABLES [VARIABLES ...]]
                          [--freq FREQ] [--run-dir RUN_DIR] [--override]

Convert ICON-DREAM

Options:
  -h, --help            show this help message and exit
  --s3-bucket S3_BUCKET
                        S3 target bucket. (default: None)
  --s3-endpoint S3_ENDPOINT
                        S3 endpoint URL. (default: https://s3.eu-dkrz-3.dkrz.cloud)
  --s3-credentials-file S3_CREDENTIALS_FILE
                        Path to a JSON file with accessKey/secretKey. (default: /home/wilfred/.s3-credentials.json)
  -v, --verbose         Increase verbosity with repeated flags such as -v or -vv. (default: 0)
  --variables VARIABLES [VARIABLES ...]
                        Variables to process (default: ['t_2m', 'tot_prec'])
  --freq, -f FREQ       ICON-DREAM data frequency (default: hourly)
  --run-dir RUN_DIR     The run directory (default: /scratch/w/wilfred/grid-doctor/icon-dream/)
  --override, -o        Override existing files. (default: False)

```

> [!IMPORTANT]
> Once you've downloaded the s3 secrets file apply `chmod 600` to it:
> `chmod 600 ~/.s3-credentials.json`
