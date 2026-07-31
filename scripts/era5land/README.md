# ERA5 & ERA5-Land

ERA5-Land in this repository is a script-driven workflow that:

- resolves local ERA5 or ERA5-Land GRIB source files from the CMOR tables
- normalises the reduced-Gaussian source layout where needed
- remaps the selected variables to HEALPix with `grid_doctor`
- writes one Zarr store per frequency and zoom level

The main entry point is `scripts/era5land/converter.py`.

## Installation

The workflow uses the normal `grid_doctor` package plus a small ERA5-Land-specific
dependency overlay and a local checkout of the ERA5 CMOR tables.

Create and activate a conda or mamba environment:

```console
mamba create -n grid_doctor -c conda-forge python=3.11 pip
mamba activate grid_doctor
```

From the repository root, install `grid_doctor` itself into that environment:

```console
python3 -m pip install -e .
```

Then layer the extra ERA5-Land dependencies from `scripts/era5land/requirements.yml`:

```console
mamba env update -p "$CONDA_PREFIX" -f scripts/era5land/requirements.yml
```

Download or refresh the local ERA5 CMOR tables:

```console
make -C scripts/era5land download
```

Sanity-check the script entry point:

```console
python3 scripts/era5land/converter.py --help
python3 scripts/era5land/converter.py --version
```

## Environment Notes

The extra environment file is intentionally local to `scripts/era5land/`. It
only adds packages that the generic `grid_doctor` install does not cover for
this workflow:

- `cfgrib` and `eccodes` for GRIB access
- `esmf` and `esmpy` for regridding support
- `kerchunk` and `pyarrow` for supporting data-access utilities
- `python-dotenv` and `reflow-hpc` for local workflow tooling

If `python3 scripts/era5land/converter.py ...` fails with
`ModuleNotFoundError: No module named 'grid_doctor'`, the editable install step
above has not been applied in the active environment yet.

## Fetching Source Files

Before converting, you can inspect which GRIB files the CMOR mapping resolves:

```console
cd scripts/era5land
python3 converter.py fetch --var tas,pr --freq 1hr,day --interval 202603,202603
```

Useful variants:

- `--show-patterns`: print the source glob patterns instead of matching files
- `--json`: emit resolved, missing, and unresolved records as JSON
- `--strict`: exit non-zero if a resolved source pattern matches no files

## Running Conversions

From the `scripts/era5land` directory:

```console
python3 converter.py remap --help
```

Typical full conversion:

```console
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr,day,mon \
  --interval 202603,202603 \
  --clean
```

Two batching modes are available for larger remaps:

- `--batch-months N`: split a bounded `--interval` into sequential calendar windows
- `--batch-files N`: split each resolved source record into sequential groups of `N` files

`--batch-files` and `--batch-months` are mutually exclusive. The file-based mode
is useful when pressure-level or long-running ERA5 workloads need finer-grained
subprocess isolation than calendar batching provides.

Submit the same kind of work as independent scheduler jobs with Reflow:

```console
python3 converter.py remap-reflow submit \
  --run-dir /scratch/$USER/era5land-reflow \
  --dataset era5land \
  --variables tas pr \
  --freq 1hr,day,mon \
  --interval 202603,202603 \
  --output-path /scratch/$USER/era5land-final \
  --clean
```

This workflow follows the standard Reflow pattern described in the
[Reflow user guide](https://reflow-docs.org/latest/guide/):

- `dispatch`: Reflow's own coordinator process. It reads the workflow
  manifest, submits ready jobs, tracks dependencies, and triggers downstream
  tasks. It is not one of the ERA5-Land remap tasks below.
- `gather_plan`: resolves the requested variable and frequency work once and
  returns self-contained array payloads
- `remap_variable_frequency`: runs one array job per item and writes into an
  isolated temporary output root below `--run-dir`
- `finalize_outputs`: merges the temporary Zarr stores into the final
  publication root and consolidates metadata through the existing publisher

ASCII DAG for quick lookup:

```text
dispatch (Reflow coordinator)
    |
    +--> gather_plan
                    |
                    +--> remap_variable_frequency[*]
                               |
            +------------------+
                               |
                               +--> finalize_outputs
```

Where:

- `gather_plan` resolves source records, discovers pressure-level groups,
  computes batching choices, writes the resolved records once to the shared
  run directory, and returns one payload per work item. Each payload contains
  a record key and shared run settings.
- `remap_variable_frequency[*]` is the parallel worker stage. Each array item
  handles one `variable x frequency x interval-batch` unit, optionally slices
  pressure levels for pressure-level variables, and writes temporary Zarr output
  below `<run-dir>/worker-output/`.
- `finalize_outputs` gathers all temporary worker outputs, merges them into the
  final publication tree, publishes special variables such as `areacella` if
  needed, and removes the temporary worker directory.

The shared `--run-dir` should point to a filesystem visible from every worker
node, for example scratch. Temporary worker outputs are written below
`<run-dir>/worker-output/`.

Check the submitted workflow with the generated Reflow CLI:

```console
python3 converter.py remap-reflow runs
python3 converter.py remap-reflow status <run-id>
```

### Reflow Operations

For operational work on an existing run, it is usually best to pass both the
interval run directory and the shared manifest database explicitly. The run
directory holds the per-run files and logs, while the SQLite manifest store
tracks run and task state across commands:

```console
python3 converter.py remap-reflow status \
  era5land_healpix-20260731-c66e \
  --run-dir /shared/era5_from_grib_reflow/queue_runs/002-1953_1962 \
  --store-path $HOME/.cache/reflow/manifest.db
```

Typical status output shows the top-level run state plus one line per task:

```text
Run      era5land_healpix-20260731-c66e
Workflow era5land_healpix
Status   RUNNING

  gather_plan               SUCCESS=1
  remap_variable_frequency  FAILED=1, SUCCESS=456
    [167]   FAILED       job=26577980
```

Read this as:

- `Status` is the overall workflow state.
- `gather_plan`, `remap_variable_frequency`, and `finalize_outputs` are the
  workflow tasks.
- Array tasks show one summary line plus per-shard lines for failed items.
- `job=<slurm-id>` is the Slurm job that handled that task instance.

Useful variants:

- `--task remap_variable_frequency`: only show one task
- `--json`: emit machine-readable state for scripting
- `--errors`: include captured traceback text for failed instances

Retry only the failed or cancelled instances of a task with:

```console
python3 converter.py remap-reflow retry \
  era5land_healpix-20260731-c66e \
  --task remap_variable_frequency \
  --run-dir /shared/era5_from_grib_reflow/queue_runs/002-1953_1962 \
  --store-path $HOME/.cache/reflow/manifest.db
```

If only shard `167` failed, the retry command should mark only that failed
instance. It does not resubmit every successful shard of the same array task.
For example, after:

```text
  remap_variable_frequency  FAILED=1, SUCCESS=456
    [167]   FAILED       job=26577980
```

the retry command should report one task instance marked for retry, and the
workflow should later advance to `finalize_outputs`.

Cancelling a run or one active task uses the same run lookup arguments:

```console
python3 converter.py remap-reflow cancel \
  era5land_healpix-20260731-c66e \
  --task finalize_outputs \
  --run-dir /shared/era5_from_grib_reflow/queue_runs/002-1953_1962 \
  --store-path $HOME/.cache/reflow/manifest.db
```

This is intended for active work. If a task is already terminal, `cancel`
should not change its final state.

The Reflow manifest store is important during debugging and recovery:

- `--run-dir` points to the per-run working directory where task logs, worker
  outputs, and other run-local files live.
- `--store-path` points to the shared SQLite manifest database. This is where
  Reflow remembers runs, task states, retries, and cancellations.
- If commands unexpectedly report `Marked 0 task instance(s) for retry`, first
  confirm that `--store-path` matches the manifest that was used when the run
  was submitted.

Basic manifest-oriented commands for day-to-day operations:

```console
# List recent runs known to the manifest DB.
python3 converter.py remap-reflow runs \
  --store-path $HOME/.cache/reflow/manifest.db

# Show one run from the manifest DB, without relying on the current directory.
python3 converter.py remap-reflow status \
  era5land_healpix-20260731-c66e \
  --store-path $HOME/.cache/reflow/manifest.db

# Show one task only, which is useful for large array jobs.
python3 converter.py remap-reflow status \
  era5land_healpix-20260731-c66e \
  --task remap_variable_frequency \
  --store-path $HOME/.cache/reflow/manifest.db

# Show the same status as JSON for scripting or jq.
python3 converter.py remap-reflow status \
  era5land_healpix-20260731-c66e \
  --json \
  --store-path $HOME/.cache/reflow/manifest.db

# Include captured Python tracebacks for failed task instances.
python3 converter.py remap-reflow status \
  era5land_healpix-20260731-c66e \
  --errors \
  --store-path $HOME/.cache/reflow/manifest.db

# Retry failed or cancelled instances recorded in the manifest DB.
python3 converter.py remap-reflow retry \
  era5land_healpix-20260731-c66e \
  --task remap_variable_frequency \
  --store-path $HOME/.cache/reflow/manifest.db

# Cancel active work for one run recorded in the manifest DB.
python3 converter.py remap-reflow cancel \
  era5land_healpix-20260731-c66e \
  --store-path $HOME/.cache/reflow/manifest.db
```

These commands all read or update the same SQLite manifest, so using the same
`--store-path` consistently matters more than the shell you run them from.

When a queue-controlled campaign is restarted after manual recovery, the
controller state file `reflow-queue-state.json` must also reflect reality.
`reflow_queue.py` stores campaign progress separately from the Reflow manifest.
As of this repository version, the controller refreshes submitted entries
against the actual Reflow run state on startup, so a restarted controller can
reconcile stale `FAILED` entries after a successful manual retry.

At the moment the Reflow workflow is intentionally focused on the main remap
path. It supports the fan-out/gather publication flow plus `--clean`,
`--from-scratch`, `--highest-level-only`, and the input-cache flags. More
specialized maintenance modes such as `--coarsen-only`, `--attrs-only`, and
`--rechunk-only` still live in `converter.py`.

Internally, `converter.py remap-reflow ...` forwards the command to
`scripts/era5land/helpers/reflow_workflow.py`, so there is now one public CLI
for both direct and scheduler-backed operation while the Reflow implementation
stays private.

### Queueing Long Reflow Campaigns

`reflow_queue.py` is a lightweight controller for campaigns that must be split
into several interval-scoped Reflow runs. It submits one interval, polls the
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
  --wrap 'cd /path/to/grid-doctor/scripts/era5land && /path/to/python reflow_queue.py --plan /shared/era5_from_grib_reflow/zg_1hr_intervals.txt --run-dir-root /shared/era5_from_grib_reflow/queue_runs --poll-seconds 300 --max-active-runs 1 --command-template "/path/to/python converter.py remap-reflow submit --dataset era5 --freq 1hr --var zg --interval {interval} --batch-files 8 --run-dir {run_dir} --output-path /shared/era5_from_grib_reflow/merged"'
```

The `{interval}` and `{run_dir}` values in `--command-template` are
placeholders. The controller replaces them automatically for each plan entry;
they do not need to be filled in manually. It creates separate directories
such as `queue_runs/001-1943_1952/` for each interval and merges all intervals
into the configured output path in sequence.

If you prefer a reusable file over a long `sbatch --wrap ...` command, generate
an sbatch wrapper script directly from the controller configuration:

```console
/path/to/python reflow_queue.py \
  --plan /shared/era5_from_grib_reflow/zg_1hr_intervals.txt \
  --run-dir-root /shared/era5_from_grib_reflow/queue_runs \
  --poll-seconds 300 \
  --max-active-runs 1 \
  --store-path $HOME/.cache/reflow/manifest.db \
  --command-template "/path/to/python converter.py remap-reflow submit --dataset era5 --freq 1hr --var zg --interval {interval} --batch-files 8 --run-dir {run_dir} --output-path /shared/era5_from_grib_reflow/merged" \
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
python3 converter.py merge \
  --source '/scratch/$USER/worker-*/era5land/1H' \
  --output-path /scratch/$USER/era5land-final/era5land/1H
```

This command reuses the same incremental Zarr merge behavior as the normal
publication path. Add `--clean` if the first merge into each touched
destination store should recreate that store instead of updating it in place.
Use `--from-scratch` to delete the complete target directory before merging;
this is broader than `--clean`.

**NOTE:**
for `merge`, the source directories and target directory should point directly
at one output-frequency directory, for example `.../era5land/1H` or
`.../era5/day`. The command merges matching `level_<n>.zarr` stores from the
source directories into the target directory in the order they are listed.
`--source` accepts glob patterns, multiple source values, and comma-separated
values. Quote a glob so the command receives it as a single pattern, for
example `--source '/scratch/$USER/worker-output/*-1hr-zg/era5/PT1H'`.

For Reflow worker output, the nested dataset and frequency paths can be
derived from the worker directory names:

```console
python3 converter.py merge \
  --source /scratch/$USER/era5land-reflow/worker-output \
  --dataset era5 \
  --freq 1hr \
  --var zg \
  --output-path /scratch/$USER/era5land-final/era5
```

`--dataset` can be used alone to merge all discovered frequencies and
variables. Add `--freq` to restrict frequencies, and optionally add `--var` to
restrict variables. The output path is a dataset root in selector mode.

For example, merge two frequencies and all variables:

```console
python3 converter.py merge \
  --source /scratch/$USER/era5land-reflow/worker-output \
  --dataset era5 \
  --freq day,mon \
  --output-path /scratch/$USER/era5land-final/era5
```

Write test output to a different publication root:

```console
python3 converter.py remap \
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
special variables such as `areacella` (that DO not depend on GRIB files) are
published through the `fx` output path while still being requestable via `--var`


Example:

```console
python3 converter.py remap \
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
python3 converter.py clean \
  --freq mon \
  --var tas,pr \
  --dry-run
```

Remove variables from all existing levels of one frequency:

```console
python3 converter.py clean \
  --freq mon \
  --var tas,pr
```

Remove variables only from selected levels:

```console
python3 converter.py clean \
  --freq 1hr \
  --var tas \
  --levels 8-6
```

Delete whole level stores:

```console
python3 converter.py clean \
  --freq 1hr,day \
  --levels 3,2,1
```

Delete whole output-frequency directories:

```console
python3 converter.py clean \
  --freq fx,mon
```

Delete the whole dataset publication root:

```console
python3 converter.py clean \
  --dataset era5
```

Truncate existing time-based stores without deleting or remapping anything:

```console
python3 converter.py clean \
  --dataset era5 \
  --freq 1hr,day,mon \
  --truncate-after 1942
```

`--truncate-after` removes timestamps strictly after the cutoff across all
existing levels. It cannot be combined with `--var`, `--levels`, or `--dry-run`.

### Cache And Parallelism

The converter separates the two input-side caches:

- GRIB inventory cache: enabled by default
- reduced-Gaussian geometry cache in `grid_doctor.utils.cache_dir()`: enabled by default
- pickled multi-file input-dataset cache: disabled by default

The geometry cache stores the expensive reduced-Gaussian cell-vertex arrays so
later runs can load them instead of rebuilding them. If the cache file
disappears or becomes unreadable, the converter regenerates it automatically.

Disable the inventory cache:

```console
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --no-inventory-cache
```

`--no-cache` is kept as an alias for `--no-inventory-cache`. That same flag
also disables the reduced-Gaussian geometry cache for the run.

Enable the pickled multi-file input-dataset cache explicitly:

```console
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --cache-input-datasets
```

### Chunk Layout

By default, new or fully rewritten Zarr stores target about `16` MB per chunk.
You can override that budget explicitly:

```console
python3 converter.py remap \
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
python3 converter.py remap \
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
python3 converter.py remap \
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
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only \
  --clean
```

In `--coarsen-only` mode, the converter:

- opens the highest existing `level_*.zarr` store for each selected frequency
- derives lower levels by `coarsen_healpix`
- writes only the coarser levels

Here `--clean` only affects the lower levels being rewritten. The highest-level
source store is read, not replaced.

Restrict coarsening to one time interval in an already existing Zarr store:

```console
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only \
  --interval 200301,202112 \
  --clean
```

Target only specific zoom levels instead of rebuilding every lower level:

```console
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only 8,0 \
  --interval 200301,202112 \
  --clean
```

You can also use descending ranges:

```console
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only 8-0 \
  --interval 200301,202112 \
  --clean
```

When you pass explicit target levels, each requested level assumes its
immediate parent level already exists. For example, `--coarsen-only 8,0`
requires both `level_9.zarr` and `level_1.zarr` to already be present.

The converter always remaps the highest level first, materialises it, then
coarsens level by level when lower zoom levels are requested.

### Batched Execution

Split a long interval into sequential month-sized batches:

```console
python3 converter.py remap \
  --var tas \
  --freq 1hr,day,mon \
  --interval 1950,1962 \
  --batches 2 \
  --highest-level-only \
  --from-scratch
```

Batched runs use isolated child processes. This keeps all batches inside the
same job allocation, node, and
environment, but releases the batch-local memory floor when each child exits.

While a batched subprocess run is active, the converter writes the current batch
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
python3 converter.py remap \
  --var tas,pr \
  --freq 1hr,day,mon \
  --attrs-only
```

This updates global and variable attrs in existing stores only.

## Logging

The converter prints structured progress logs to the terminal. The output is
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

The ERA5-Land-specific automated coverage in this repository is currently
centered on source resolution, especially `tests/test_era5land_file_fetcher.py`.

Run that focused test module with:

```console
python3 -m pytest tests/test_era5land_file_fetcher.py
```

Lint the current ERA5-Land script set with:

```console
python3 -m ruff check scripts/era5land tests/test_era5land_file_fetcher.py
```

Apply auto-fixable Ruff changes with:

```console
python3 -m ruff check --fix scripts/era5land tests/test_era5land_file_fetcher.py
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
