# CORDEX regional domains → HEALPix pyramid

Converts a limited-area (rotated-pole) dataset to a standard dense
global HEALPix pyramid that is NaN outside the domain. Companion to the
[Regional Datasets recipe](../../docs/technical/recipes/regional.md).

```console
python convert.py my-bucket \
    --input "/pool/data/CORDEX/EUR-11/.../tas_*.nc" \
    --variables tas pr --categorical sftlf \
    --store-name cordex-eur11 -vv
```

Key points:

- **Coverage masking is not optional.** `renormalize` extrapolates
  boundary cells that barely overlap the domain to full-cell values;
  the ones-field coverage pass masks cells below `--coverage-threshold`
  (default 0.5, matching the coarsening `min_valid_fraction` rule) and
  publishes `coverage_fraction` for auditability.
- For categorical variables the mask prevents ESMF nearest from
  smearing the domain over the entire globe.
- Continuous and categorical variables are coarsened separately
  (mean vs. mode) and merged per level — `coarsen_mode="auto"` cannot
  express a mixed dataset.
- Encoding uses `4**6`-cell chunks (each chunk = one level-(L−6) parent
  cell) with NaN fill, so all-empty chunks are elided and storage is
  proportional to the domain (verified: 3.0% domain → 3.1% of chunks).
- `geospatial_*` bounding-box attributes are attached so viewers can
  zoom to the domain.

## Known problems

- The ESMF path (`regrid_to_healpix`, weight generation for rotated-pole
  grids, the ones-field coverage pass) has not been executed in this
  environment (no ESMF available); the orchestration around it is
  covered by a mocked-regrid test. Validate the first real run and note
  findings here.
- Different simulations of one CORDEX domain (e.g. several RCMs on
  EUR-11) share the source grid — pass a common `--weights-path` so the
  weight file is generated once.

---

# NUKLEUS / CORDEX-protocol workflow (`convert_cordex.py`)

Production pipeline for large regional archives, modelled on
`scripts/cmip6/convert.py` (reflow, Slurm array jobs, per-file staging
NetCDFs on Lustre — divide and conquer). Requires the sibling
`scripts/cmip6/` folder (the Freva databrowser client is shared from
there) and `reflow-hpc`.

```console
REFLOW_ACCOUNT=<acct> python convert_cordex.py submit \
    --run-dir /scratch/k/$USER/grid-doctor/nukleus \
    --project nukleus --uri s3://regional -vv
```

Defaults are deliberately open: `--project nukleus`, `--product` empty,
experiments/RCMs/driving models/variables all **discovered** from the
databrowser. `--variable` empty means every dataset carries all
variables available for its combination (`--exclude-variable` to prune).

Differences to the CMIP6 pipeline:

- **Matrix has a driving-model axis.** One RCM driven by two GCMs is
  two output datasets; mixing them would corrupt the time axis.
- **Coverage masking at the finest level.** `create_weights` also
  remaps a ones-field per source grid (shared weight file, so nearly
  free) to obtain per-cell domain coverage; `regrid_file` masks cells
  below `--coverage-threshold` (default 0.5) *before* coarsening, so
  boundary extrapolation from the renormalize policy never contaminates
  parent cells. `coverage_fraction` is published with the data
  (written by file_index 0 only, so the combine sees it exactly once).
- **Regional chunking.** `4**8`-cell spatial chunks (each = one coarse
  parent cell) + NaN `_FillValue`, so all-empty chunks outside the
  domain are elided; the time chunk is sized to ~16 MiB.
- **Wrap-aware bounding-box attributes** from the coverage field, so a
  Greenwich-crossing domain gets `(-12, 35)` and not `(0, 360)`.

## Troubleshooting: run "succeeds" after plan_regrid

If `status` shows only gather_sources/create_weights/plan_regrid, the
plan was empty: an array fan-out over zero items creates no tasks, so
downstream steps never materialize and the coordinator reports SUCCESS.
All three planning stages now raise instead — `gather_sources` with a
dump of the facet values the databrowser actually offers for the
project (the usual culprit: frequency spelled `1hr`/`daily` instead of
`day`, or a differently-cased project value), `create_weights` when
every dataset was skipped (typically ESMF missing in the partition
environment), `plan_regrid` as the last line of defence. Check the
failing task's stdout in the run dir; re-`submit` after fixing (the
Merkle cache reuses the completed steps).

## Known problems

- The ESMF stages (`cached_weights` on rotated-pole grids, the
  ones-field remap, `regrid_to_healpix`) have not run in the review
  environment (no ESMF); the matrix search (against a mocked
  databrowser), grid grouping, masking, chunk plan, combine incl. the
  time-less coverage variable, duplicate-time guard, bbox logic, and
  the reflow DAG/CLI wiring are all verified. Validate the first real
  run on Levante and note findings here.
- NUKLEUS facet names are assumed to follow the standard Freva scheme
  (`project`, `experiment`, `time_frequency`, `model`, `driving_model`,
  `ensemble`, `variable`). If `driving_model` is unused by the project,
  the matrix degrades gracefully to one dataset per RCM. Check
  `--help`/first `gather_sources` output against the actual facets.
- Static (`fx`) variables and categorical fields are out of scope here;
  use the simple `convert.py` in this folder with `method="nearest"` +
  the coverage mask for those.
