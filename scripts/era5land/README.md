# heal_era5

This repository is a script-driven workflow that:

- resolves local ERA5 or ERA5-Land GRIB source files from the CMOR tables
- normalises the reduced-Gaussian source layout where needed
- remaps the selected variables to HEALPix with `grid_doctor`
- writes one Zarr store per frequency and zoom level

The main entry point is the installed `heal-era5` command.
For the command and helper relationships, see the
[function call graph](FUNCTION_CALL_GRAPH.md).

The installed remapper version is shown by:

```console
heal-era5 --version
```

## Installation

The workflow uses the normal `grid_doctor` package, a small native Conda
dependency overlay, and a local checkout of the ERA5 CMOR tables.

1. Create and activate a conda or mamba environment:

   ```console
   mamba create -n grid_doctor -c conda-forge python=3.11 pip
   mamba activate grid_doctor
   ```

2. From the repository root, install `grid_doctor` itself into that environment:

   ```console
   python3 -m pip install -e .
   ```

3. Install the native GRIB and regridding libraries:

   ```console
   mamba env update -p "$CONDA_PREFIX" -f scripts/era5land/environment.yml
   ```

4. Install the normal remapper workflow in editable mode from the repository root.
   This installs Python dependencies including `cfgrib` and `rich-argparse`; it
   uses the editable local `grid_doctor` package installed in the previous step:

   ```console
   python3 -m pip install -e scripts/era5land
   ```

   This provides the `heal-era5` command.

5. Download or refresh the external ERA5 CMOR tables:

   ```console
   make -C scripts/era5land download
   ```

   This stores tables in `scripts/era5land/tables/era5-cmor-tables`, outside
   the Python package. For an installed wheel or a differently located table
   checkout, set `HEAL_ERA5_TABLES_DIR` to the `era5-cmor-tables` directory:

   ```console
   export HEAL_ERA5_TABLES_DIR=/path/to/era5-cmor-tables
   ```

### Reflow

Reflow is optional because direct `fetch`, `remap`, `update`, `clean`, and
`merge` commands do not use it. Install it when using `remap-reflow` or
`reflow-queue`:

```console
python3 -m pip install -e "scripts/era5land[reflow]"
```

### Development

Install development tools and run the package-local checks:

```console
python3 -m pip install -e "scripts/era5land[dev]"
```

From `scripts/era5land`, run all checks with `tox`, or select one check with
`tox -e lint`, `tox -e types`, or `tox -e test`. The test environment prints
a terminal coverage report and writes an HTML report to `htmlcov/index.html`.

Ruff also sorts imports using its isort-compatible rules. To apply safe lint
and import fixes, format the code, then verify linting and types:

```console
cd scripts/era5land
python3 -m ruff check --fix src tests
python3 -m ruff format src tests
tox -e lint
tox -e types
```

`tox -e types` only reports type issues; fix those annotations manually before
rerunning it.


## Layout

```text
scripts/era5land/
├── pyproject.toml
├── src/heal_era5/
│   ├── main.py
│   ├── cli/
│   ├── helpers/
│   └── assets/
├── tables/era5-cmor-tables/  # downloaded separately; not packaged
└── tests/
```


Sanity-check the installed command:

```console
heal-era5 --help
heal-era5 --version
```

## Environment Notes

`environment.yml` intentionally contains only native Conda packages. Python
dependencies belong in `pyproject.toml` and are installed by `pip`:

- `eccodes` supplies the GRIB runtime library required by `cfgrib`.
- `esmf` and `esmpy` supply the regridding stack used by `grid_doctor`.
- `cfgrib` and `rich-argparse` are normal `heal-era5` Python dependencies.
- `reflow-hpc` is installed with `heal-era5[reflow]` only.

If `heal-era5 ...` fails with
`ModuleNotFoundError: No module named 'grid_doctor'`, the editable install step
above has not been applied in the active environment yet.

## Fetching Source Files

Before converting, you can inspect which GRIB files the CMOR mapping resolves:

```console
heal-era5 fetch --var tas,pr --freq 1hr,day --interval 202603,202603
```

Useful variants:

- `--show-patterns`: print the source glob patterns instead of matching files
- `--json`: emit resolved, missing, and unresolved records as JSON
- `--strict`: exit non-zero if a resolved source pattern matches no files

## Remapping

```console
heal-era5 remap --help
```

Typical full conversion:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr,day,mon \
  --interval 202603,202603 \
  --clean
```

For pressure-level variables, `remap` retains the configured default selection
of `1000,850,700,500,300,200,50,20,10,5` hPa. Surface and fixed-level
variables are unaffected.

Override the selection with `-pl` (or `--pressure-levels`):

```console
heal-era5 remap --dataset era5 --var ta -pl 1000,850,500
```

Use `--pressure-levels all` to retain every available pressure level. Values
are expressed in hPa, so use `1000` rather than the equivalent `100000` Pa
coordinate value.

Two batching modes are available for larger remaps:

- `--batch-months N`: split a bounded `--interval` into sequential calendar windows
- `--batch-files N`: split each resolved source record into sequential groups of `N` files

`--batch-files` and `--batch-months` are mutually exclusive. The file-based mode
is useful when pressure-level or long-running ERA5 workloads need finer-grained
subprocess isolation than calendar batching provides.

### Reflow Operations

Use Reflow to submit a remap as independent scheduler jobs:

```console
heal-era5 remap-reflow submit \
  --run-dir /scratch/$USER/era5land-reflow \
  --dataset era5land \
  --var tas,pr \
  --freq 1hr,day,mon \
  --interval 202603,202603 \
  --output-path /scratch/$USER/era5land-final \
  --clean true
```

`remap-reflow submit` accepts the same `-pl` / `--pressure-levels` selection
as direct `remap`: omit it for the configured hPa default, provide a
comma-separated override, or use `--pressure-levels all` to retain every
available level.

Reflow coordinates the following stages:

```mermaid
flowchart LR
    dispatch[Reflow dispatch] --> plan[gather_plan]
    plan --> workers[remap_variable_frequency*]
    workers --> finalize[finalize_outputs]
```

- `gather_plan` resolves source files, pressure-level groups, and batches once.
- `remap_variable_frequency` runs one `variable x frequency x interval-batch`
  worker item and writes below `<run-dir>/worker-output/`.
- `finalize_outputs` merges temporary stores, publishes special variables, and
  removes the temporary worker directory.

Put `--run-dir` and the weight cache on a filesystem visible to every worker
node. `dispatch` is Reflow's coordinator rather than a remapping task.

For status, retry, and recovery, use the same SQLite manifest (`--store-path`)
that was used for submission. `--run-dir` contains per-run logs and temporary
outputs; `--store-path` tracks workflow state across commands:

```console
heal-era5 remap-reflow runs \
  --store-path $HOME/.cache/reflow/manifest.db

heal-era5 remap-reflow status \
  era5land_healpix-20260731-c66e \
  --store-path $HOME/.cache/reflow/manifest.db
```

Typical status output summarizes workflow state, then each task and any failed
array item:

```text
Status   RUNNING

  gather_plan               SUCCESS=1
  remap_variable_frequency  FAILED=1, SUCCESS=456
    [167]   FAILED       job=26577980
```

Use `--task remap_variable_frequency` for one task, `--json` for scripts, and
`--errors` for captured tracebacks. If a retry unexpectedly marks no work,
first confirm that `--store-path` names the manifest used for submission.

```console
heal-era5 remap-reflow retry \
  era5land_healpix-20260731-c66e \
  --task remap_variable_frequency \
  --store-path $HOME/.cache/reflow/manifest.db

heal-era5 remap-reflow cancel \
  era5land_healpix-20260731-c66e \
  --store-path $HOME/.cache/reflow/manifest.db
```

`retry` marks only failed or cancelled task instances; it does not resubmit
successful shards. Use `cancel` only for active work; it does not change a
terminal task's final state.

Reflow supports the main remap path, including `--clean`, `--from-scratch`,
`--highest-level-only`, and input-cache options. Use direct `heal-era5 remap`
for specialized maintenance modes such as `--coarsen-only`, `--attrs-only`,
and `--rechunk-only`.

### Queueing Long Reflow Campaigns

`heal-era5 reflow-queue` is a lightweight controller for campaigns that must be split
into several time-interval scoped Reflow runs. It submits one interval, polls the
run until its worker and `finalize_outputs` tasks finish, and then submits the
next interval. This keeps only one interval run active while allowing a long
campaign to continue without manual resubmission. The controller stores its
progress in a JSON state file and can be restarted with the same arguments.

Create a plan file with one interval per line. For example, this plan covers
1943 through 2026:

```console
1943,1952
1953,1962
1963,1972
1973,1982
1983,1992
1993,2002
2003,2012
2013,2022
2023,2026
```

Save it as `/shared/era5_from_grib_reflow/zg_1hr_intervals.txt` (or use
another path in `--plan`). Then submit the controller as a long-running,
low-resource Slurm job:

```console
sbatch \
  --account=ch1187 \
  --partition=shared \
  --job-name=era5-reflow-queue \
  --time=7-00:00:00 \
  --cpus-per-task=1 \
  --mem=2G \
  --output=/shared/era5_from_grib_reflow/controller-%j.out \
  --wrap '/home/etor/miniconda3/envs/grid_doctor/bin/heal-era5 reflow-queue --plan /shared/era5_from_grib_reflow/zg_1hr_intervals.txt --run-dir-root /shared/era5_from_grib_reflow/queue_runs --poll-seconds 300 --max-active-runs 1 --command-template "/home/etor/miniconda3/envs/grid_doctor/bin/heal-era5 remap-reflow submit --dataset era5 --freq 1hr --var zg --interval {interval} --batch-files 8 --run-dir {run_dir} --output-path /shared/era5_from_grib_reflow/merged"'
```

The `{interval}` and `{run_dir}` values in `--command-template` are
placeholders. The controller replaces them automatically for each plan entry;
they do not need to be filled in manually. It creates separate directories
such as `queue_runs/001-1943_1952/` for each interval and merges all intervals
into the configured output path in sequence.

If you prefer a reusable file over a long `sbatch --wrap ...` command, generate
an sbatch wrapper script directly from the controller configuration:

```console
/home/etor/miniconda3/envs/grid_doctor/bin/heal-era5 reflow-queue \
  --plan /shared/era5_from_grib_reflow/zg_1hr_intervals.txt \
  --run-dir-root /shared/era5_from_grib_reflow/queue_runs \
  --poll-seconds 300 \
  --max-active-runs 1 \
  --store-path $HOME/.cache/reflow/manifest.db \
  --command-template "/home/etor/miniconda3/envs/grid_doctor/bin/heal-era5 remap-reflow submit --dataset era5 --freq 1hr --var zg --interval {interval} --batch-files 8 --run-dir {run_dir} --output-path /shared/era5_from_grib_reflow/merged" \
  --write-sbatch /shared/era5_from_grib_reflow/run_reflow_queue.sh \
  --sbatch-account ch1187 \
  --sbatch-partition shared \
  --sbatch-job-name era5-reflow-queue \
  --sbatch-time 7-00:00:00 \
  --sbatch-cpus-per-task 1 \
  --sbatch-mem 2G \
  --sbatch-output /shared/era5_from_grib_reflow/controller-%j.out
```

Then submit that generated script with:

```console
sbatch /shared/era5_from_grib_reflow/run_reflow_queue.sh
```

Two queue-controller options are worth calling out explicitly:

- `--state-path` overrides the controller's own JSON checkpoint file. By
  default, the controller stores its progress at
  `<run-dir-root>/reflow-queue-state.json`. This file is separate from the
  Reflow manifest database and records queue-level information such as which
  intervals were submitted, which `run_id` belongs to each interval, and
  whether the controller considers each interval complete.
- `--continue-on-failure` tells the controller to keep submitting later
  intervals even after one interval run reaches `FAILED` or `CANCELLED`.
  Without this flag, the controller stops at the first failed interval so that
  the campaign can be inspected and repaired before more work is launched.

Each Reflow run still submits one Slurm array containing its generated worker
items. `--max-active-runs 1` does not reduce the size of that array, and Slurm
array syntax such as `--array=0-799%20` still counts 800 submitted jobs. Choose
intervals so each generated array remains safely below the user's remaining
job limit, leaving room for Reflow's coordinator jobs and other workloads.
Do not use `--from-scratch` for every queued interval because it can delete
output published by earlier intervals. Use separate run directories, but one
shared output path, as in the example above.

The Reflow workflow reads the default weight-cache directory from
`assets/source_mapper.json` (`weights_path`), and all workers reuse the same
shared cache. The configured directory must already exist and be writable on
the shared filesystem before submission. Override it only when the configured
path is unavailable on the worker nodes by adding
`--weights-dir /path/to/shared/weights` to the Reflow submission template.

## Merging Temporary Outputs

Use `merge` when you already have one or more temporary HEALPix output roots
and want to publish or combine them into a final target tree without running a
new remap:

```console
heal-era5 merge \
  --source '/scratch/$USER/worker-*/era5land/1H' \
  --output-path /scratch/$USER/era5land-final/era5land/1H
```

This command reuses the same incremental Zarr merge behavior as the normal
publication path. Add `--clean` if the first merge into each touched
destination store should recreate that store instead of updating it in place.
Use `--from-scratch` to delete the complete target directory before merging;
this is broader than `--clean`.

**Direct-store mode:** when neither `--dataset` nor `--freq` is supplied, each
source directory must directly contain `level_<n>.zarr` stores. Each matching
store is merged into `<output-path>/level_<n>.zarr`; no dataset or frequency
directory is appended. `--from-scratch` deletes exactly `<output-path>`.
`--source` accepts glob patterns, multiple source values, and comma-separated
values. Quote a glob so the command receives it as a single pattern, for
example `--source '/scratch/$USER/worker-output/*-1hr-zg/era5/PT1H'`.

For output organized below a shared root, the nested dataset and frequency
paths can be discovered with `--dataset` and `--freq`:

```console
heal-era5 merge \
  --source /scratch/$USER/era5land-reflow/merged \
  --dataset era5 \
  --freq 1hr \
  --var zg \
  --output-path /scratch/$USER/era5land-final/era5
```

`--dataset` can be used alone to merge all discovered frequencies and
variables. Add `--freq` to restrict frequencies, and optionally add `--var` to
restrict variables. The output path is a dataset root in selector mode. The
source may be either the dataset directory itself (for example,
`.../merged/era5`) or its parent root (for example, `.../merged`).

Use `--levels` to merge only selected HEALPix levels. It accepts comma-separated
levels and descending ranges such as `7`, `7,5,3`, or `6-0`:

```console
heal-era5 merge \
  --source /scratch/$USER/era5land-reflow/merged \
  --dataset era5 \
  --freq 1hr \
  --levels 6-0 \
  --output-path /scratch/$USER/era5land-final/era5
```

Use `--interval START,END` to restrict time-dependent data to an inclusive
date interval. Dates may be specified as `YYYY`, `YYYYMM`, or `YYYYMMDD`;
static `fx` stores are unaffected:

```console
heal-era5 merge \
  --source /scratch/$USER/era5land-reflow/merged \
  --dataset era5 \
  --freq 1hr \
  --interval 200001,202012 \
  --output-path /scratch/$USER/era5land-final/era5
```

`--levels` and `--interval` can be combined. With `--clean`, only the selected
levels and interval are written to the touched stores; without `--clean`, the
interval is merged incrementally while data outside it is retained.

For example, merge two frequencies and all variables:

```console
heal-era5 merge \
  --source /scratch/$USER/era5land-reflow/worker-output \
  --dataset era5 \
  --freq day,mon \
  --output-path /scratch/$USER/era5land-final/era5
```

Write test output to a different publication root:

```console
heal-era5 remap \
  --var tas,pr \
  --freq mon \
  --interval 202603,202603 \
  --output-path /tmp/era5land-test-output
```

**NOTE:**
`--clean` rewrites only the target `level_*.zarr` stores touched by the run.
Without it, existing stores are updated incrementally by merging overlapping
time slices, adding missing variables, and appending new times where possible.
`--from-scratch` is broader: it deletes the **whole dataset output root **before
the run starts. For `--dataset era5land`, that means the entire
`.../healpix/era5land` subtree across all frequencies and levels.

**NOTE:**
special variables such as `areacella` (that do not depend on GRIB files) are
published through the `fx` output path while still being requestable via `--var`


Example:

```console
heal-era5 remap \
  --var tas,pr \
  --freq mon \
  --interval 202603,202603 \
  --from-scratch
```

## Cleaning Existing Outputs

Use `clean` when you want to remove already published content without
starting a new conversion. The cleanup scope is inferred from the selectors you
pass:

- no `--var`, no `--freq`, no `--levels`: delete the dataset root
- `--var`: remove variables from matching stores
- `--levels` without `--var`: delete matching whole level stores
- `--freq` without `--var` or `--levels`: delete matching whole frequency directories

Preview a cleanup without changing anything:

```console
heal-era5 clean \
  --freq mon \
  --var tas,pr \
  --dry-run
```

Remove variables from all existing levels of one frequency:

```console
heal-era5 clean \
  --freq mon \
  --var tas,pr
```

Remove variables only from selected levels:

```console
heal-era5 clean \
  --freq 1hr \
  --var tas \
  --levels 8-6
```

Delete whole level stores:

```console
heal-era5 clean \
  --freq 1hr,day \
  --levels 3,2,1
```

Delete whole output-frequency directories:

```console
heal-era5 clean \
  --freq fx,mon
```

Delete the whole dataset publication root:

```console
heal-era5 clean \
  --dataset era5
```

Truncate existing time-based stores without deleting or remapping anything:

```console
heal-era5 clean \
  --dataset era5 \
  --freq 1hr,day,mon \
  --truncate-after 1942
```

`--truncate-after` removes timestamps strictly after the cutoff across all
existing levels. It cannot be combined with `--var`, `--levels`, or `--dry-run`.


## Updating Published Data

Use `update` to refresh an existing publication with newly available data. The
command processes only variables and frequencies that already have a published
time series:

```console
heal-era5 update \
  --dataset era5 \
  --freq 1hr,day \
  --output-path /work/ks1387/era5
```

An update has two phases:

- The permanent phase refreshes [source files that have become final](https://docs.dkrz.de/doc/dataservices/finding_and_accessing_data/era_data/index.html#era5-data-via-pool-data-era5).
  Final files lag between 2-3 months behind real time. If a
  variable has a `last_permanent_update` attribute, that watermark determines
  the permanent search boundary. If it does not, the remapper bootstraps from
  approximately three months before the latest stored timestamp and uses source
  file modification times to avoid reprocessing older files.
- The forward phase searches from the latest stored timestamp through today,
  replacing provisional values and appending newer timestamps.

Use `--preview` to resolve and count both phases without modifying Zarr data or
metadata:

```console
heal-era5 update \
  --dataset era5 \
  --freq 1hr,day \
  --output-path /work/ks1387/era5 \
  --preview
```

Update processing is direct by default. Optional batching is available when
the interval or variables are large:

- `--batch-months N`: process each phase in sequential calendar-month batches
- `--batch-files N`: process each phase in groups of `N` source files

The two batching options are mutually exclusive. If neither is supplied, the
resolved records are processed in one direct mapping operation. Successful
data writes update `last_data_update`; successful permanent refreshes update
`last_permanent_update` on the published variables.



### Cache And Parallelism

The remapper separates the two input-side caches:

- GRIB inventory cache: enabled by default
- reduced-Gaussian geometry cache in `grid_doctor.utils.cache_dir()`: enabled by default
- pickled multi-file input-dataset cache: disabled by default

The geometry cache stores the expensive reduced-Gaussian cell-vertex arrays so
later runs can load them instead of rebuilding them. If the cache file
disappears or becomes unreadable, the remapper regenerates it automatically.

Disable the inventory cache:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --no-inventory-cache
```

`--no-cache` is kept as an alias for `--no-inventory-cache`. That same flag
also disables the reduced-Gaussian geometry cache for the run.

Enable the pickled multi-file input-dataset cache explicitly:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --cache-input-datasets
```

### Chunk Layout

By default, new or fully rewritten Zarr stores target about `16` MB per chunk.
You can override that budget explicitly:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --chunk-size 256
```

The chunk-size target is applied when a store is created from scratch, rewritten
with `--clean`, or otherwise rebuilt during an incremental merge.

To rechunk already existing matching Zarr stores and then exit without
continuing into remapping, add `--rechunk-only`:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --chunk-size 256 \
  --rechunk-only
```

When combined with `--highest-level-only`, only the finest existing level for
each selected frequency is rechunked in that standalone pass.

### Pyramid Modes

Write only the finest HEALPix level for each selected frequency:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --clean \
  --highest-level-only
```

`--highest-level-only` keeps only the highest HEALPix level and skips the
coarsening pass for lower zoom levels.

Build lower zoom levels from an already existing highest-level Zarr store:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only \
  --clean
```

In `--coarsen-only` mode, the remapper:

- opens the highest existing `level_*.zarr` store for each selected frequency
- derives lower levels by `coarsen_healpix`
- writes only the coarser levels

Here `--clean` only affects the lower levels being rewritten. The highest-level
source store is read, not replaced.

Restrict coarsening to one time interval in an already existing Zarr store:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only \
  --interval 200301,202112 \
  --clean
```

Target only specific zoom levels instead of rebuilding every lower level:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only 8,0 \
  --interval 200301,202112 \
  --clean
```

You can also use descending ranges:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only 8-0 \
  --interval 200301,202112 \
  --clean
```

When you pass explicit target levels, each requested level assumes its
immediate parent level already exists. For example, `--coarsen-only 8,0`
requires both `level_9.zarr` and `level_1.zarr` to already be present.

The remapper always remaps the highest level first, materialises it, then
coarsens level by level when lower zoom levels are requested.

### Batched Execution

Split a long interval into sequential month-sized batches:

```console
heal-era5 remap \
  --var tas \
  --freq 1hr,day,mon \
  --interval 1950,1962 \
    --batch-months 2 \
  --highest-level-only \
  --from-scratch
```

Batched runs use isolated child processes. This keeps all batches inside the
same job allocation, node, and
environment, but releases the batch-local memory floor when each child exits.

While a batched subprocess run is active, the remapper writes the current batch
process state to `.current_batch_pid.json` in the launch directory when
possible, or falls back to `scripts/era5land/.current_batch_pid.json`.

The state file contains:

- `parent_pid`: PID of the top-level batch controller process
- `batch_pid`: PID of the currently running child batch
- `batch_pgid`: process-group ID of the active child batch
- `batch_index`: 1-based index of the current batch
- `batch_count`: total number of batches in the run
- `batch_interval`: date interval currently being processed

This is mainly intended for manual intervention during a stuck batch. Typical
examples:

```console
cat .current_batch_pid.json
kill 12345
kill -- -12345
```

In practice, `kill <batch_pid>` stops the current child batch, while
`kill -- -<batch_pgid>` targets the whole active batch process group.

### Metadata-Only Maintenance

Refresh metadata on already-published Zarr stores without remapping data:

```console
heal-era5 remap \
  --var tas,pr \
  --freq 1hr,day,mon \
  --attrs-only
```

This updates global and variable attrs in existing stores only.

## Logging

The remapper prints structured progress logs to the terminal. The output is
deliberately milestone-based rather than per-file noisy. Typical stages include:

- `convert_start`
- `frequency_start`
- `grib_merge_done`
- `weight_calculation`
- `remap_start`
- `remap_materialize_done`
- `zarr_write_start`
- `frequency_done`

If stderr is attached to an interactive terminal, these stages are colorised.

## Development Checks

Run the full package suite with terminal and HTML coverage reports:

```console
tox -e test
```

Open `htmlcov/index.html` in a browser to inspect line and branch coverage.
Run the source-resolution tests alone with:

```console
python3 -m pytest tests/test_heal_era5_file_fetcher.py
```

Lint the current ERA5-Land script set with:

```console
python3 -m ruff check src tests
```

Apply auto-fixable Ruff changes with:

```console
python3 -m ruff check --fix src tests
```

Format the ERA5-Land files with Ruff format:

```console
python3 -m ruff format scripts/era5land tests/test_era5land_file_fetcher.py
```

You can also use `python3 -m py_compile` for a fast syntax-only check on the
script files when you do not want to run the workflow itself.

## Notes on Forecast GRIBs

Forecast products such as `tas` and `pr` come from `EL/sf/fc` sources and need
extra handling compared with simpler analysis-style inputs because they mix:

- time plus forecast-step semantics that must be flattened to valid timestamps
- reduced-Gaussian spatial coordinates that are not directly HEALPix-ready

That is why the workflow normalises the source dataset before remapping instead
of assuming a simple regular lat/lon grid straight from `cfgrib`.
