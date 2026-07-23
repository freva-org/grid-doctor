# ERA5-Land

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
python3 converter.py fetch-files --var tas,pr --freq 1hr,day --interval 202603,202603
```

Useful variants:

- `--show-patterns`: print the source glob patterns instead of matching files
- `--json`: emit resolved, missing, and unresolved records as JSON
- `--strict`: exit non-zero if a resolved source pattern matches no files

## Running Conversions

From the `scripts/era5land` directory:

```console
python3 converter.py convert-healpix --help
```

Typical full conversion:

```console
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr,day,mon \
  --interval 202603,202603 \
  --clean
```

Write test output to a different publication root:

```console
python3 converter.py convert-healpix \
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
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq mon \
  --interval 202603,202603 \
  --from-scratch
```

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
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --no-inventory-cache
```

`--no-cache` is kept as an alias for `--no-inventory-cache`. That same flag
also disables the reduced-Gaussian geometry cache for the run.

Enable the pickled multi-file input-dataset cache explicitly:

```console
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --cache-input-datasets
```

Open source records in parallel within each frequency merge:

```console
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --record-threads
```

`--record-threads` is disabled by default. When enabled, the per-record open
stage in `merge_frequency_dataset(...)` uses a thread pool.

### Chunk Layout

By default, new or fully rewritten Zarr stores target about `100` MB per chunk.
You can override that budget explicitly:

```console
python3 converter.py convert-healpix \
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
python3 converter.py convert-healpix \
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
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --clean \
  --highest-level-only
```

`--highest-level-only` automatically forces the stepwise highest-first path. You
do not need to also pass `--pyramid-strategy stepwise`.

Build lower zoom levels from an already existing highest-level Zarr store:

```console
python3 converter.py convert-healpix \
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
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only \
  --interval 200301,202112 \
  --clean
```

Target only specific zoom levels instead of rebuilding every lower level:

```console
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only 8,0 \
  --interval 200301,202112 \
  --clean
```

You can also use descending ranges:

```console
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --coarsen-only 8-0 \
  --interval 200301,202112 \
  --clean
```

When you pass explicit target levels, each requested level assumes its
immediate parent level already exists. For example, `--coarsen-only 8,0`
requires both `level_9.zarr` and `level_1.zarr` to already be present.

Select the pyramid construction strategy explicitly:

```console
python3 converter.py convert-healpix \
  --var tas,pr \
  --freq 1hr \
  --interval 202603,202603 \
  --pyramid-strategy stepwise
```

Available strategies:

- `lazy`: keep `grid_doctor`'s default lazy pyramid construction
- `stepwise`: remap the highest level first, materialise it, then coarsen level by level

### Batched Execution

Split a long interval into sequential month-sized batches:

```console
python3 converter.py convert-healpix \
  --var tas \
  --freq 1hr,day,mon \
  --interval 1950,1962 \
  --batches 2 \
  --highest-level-only \
  --from-scratch
```

By default, batched runs use isolated child processes:

- `--batch-mode subprocess`: default; each batch runs in a fresh Python process
- `--batch-mode inprocess`: legacy single-process loop

The subprocess mode keeps all batches inside the same job allocation, node, and
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

If you explicitly want the legacy behavior for debugging or profiling:

```console
python3 converter.py convert-healpix \
  --var tas \
  --freq 1hr \
  --interval 1950,1951 \
  --batches 2 \
  --batch-mode inprocess
```

### Metadata-Only Maintenance

Refresh metadata on already-published Zarr stores without remapping data:

```console
python3 converter.py convert-healpix \
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
- `grib_read_parallel`
- `grib_merge_done`
- `weight_calculation`
- `remap_start`
- `remap_materialize_done`
- `zarr_write_start`
- `frequency_done`

If stderr is attached to an interactive terminal, these stages are colorised.
`grib_read_parallel` appears only when `--record-threads` is enabled and there
is more than one resolved record for the current frequency.

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
