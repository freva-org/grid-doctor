# ICON-DREAM

## What is icon-dream?

Icon dream is a reanalysis product from the German Weather Service.


## How to run:

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
pthon convert.py submit --help
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

> [!IMPORTANT]
> Once you've downloaded the s3 secrets file apply `chmod 600` to it:
> `chmod 600 ~/.s3-credentials.json`
