# CMIP6

## How to run:

1. Create and download your s3-secrets file from [https://eu-dkrz-3.dkrz.cloud/access-keys](https://eu-dkrz-3.dkrz.cloud/access-keys)
2. Put the secrets files somewhere into your home on levante.
3. Install the requirements
```console
python -m pip install -r requirements.txt
```

4. The `convert.py` uses [`reflow`](https://www.reflow.docs.org) to submit a
pre-defined workflow

```console
python convert.py --help
Usage: cmip6_healpix [-h] [--version] {submit,status,cancel,retry,runs,dag,describe} ...

reflow workflow: cmip6_healpix

Positional Arguments:
  {submit,status,cancel,retry,runs,dag,describe}
    submit              Create a new run and submit the workflow coordinator.
    status              Show the status of a run.
    cancel              Cancel active tasks in a run.
    retry               Retry failed or cancelled tasks.
    runs                List all runs.
    dag                 Print the task DAG.
    describe            Print the workflow manifest as JSON.

Options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit


```

```console
python convert.py submit --help
Usage: cmip6_healpix submit [-h] --run-dir RUN_DIR [--store-path STORE_PATH] [--force] [--force-tasks TASK [TASK ...]] [--freq FREQ] [--s3-bucket S3_BUCKET]
                            [--s3-credentials-file S3_CREDENTIALS_FILE] [--s3-endpoint S3_ENDPOINT] [--variable VARIABLE [VARIABLE ...]] [--weights-dir WEIGHTS_DIR]

Options:
  -h, --help            show this help message and exit
  --run-dir RUN_DIR     Shared working directory. (default: None)
  --store-path STORE_PATH
                        Explicit path to SQLite manifest. (default: None)
  --force               Skip the Merkle cache entirely and re-run all tasks. (default: False)
  --force-tasks TASK [TASK ...]
                        Skip the cache for specific tasks only. (default: None)
  --freq, -f FREQ       Target time frequency (default: 6hr)
  --s3-bucket S3_BUCKET
                        Target S3 bucket (default: cmip6)
  --s3-credentials-file S3_CREDENTIALS_FILE
                        Path to S3 credentials JSON (default: /home/wilfred/.s3-credentials.json)
  --s3-endpoint S3_ENDPOINT
                        S3 endpoint URL (default: https://s3.eu-dkrz-3.dkrz.cloud)
  --variable VARIABLE [VARIABLE ...]
                        Select the variables (default: ['pr', 'tas'])
  --weights-dir WEIGHTS_DIR
                        Path to the grid weight directory (default:
                        /work/ks1387/healpix-weights)
```

To submit the job choose your slurm partition and add any arguments for the
`convert.py` script for example:

```console
REFLOW_ACCOUNT=foo python convert.py submit --run-dir \
    /scratch/k/$USER/grid-doctor/cmip6 --s3-bucket cmip6
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
