# Weights store

Remapping weights depend only on the source grid, the HEALPix target and
the method, so grid-doctor computes them once and reuses them. Every
weight file has one name, derived from what it describes, and lives in a
**weights store**: a local (POSIX) directory. Callers only choose the
store's root; everything below it is grid-doctor's layout.

## Getting weights

[`cached_weights`][grid_doctor.cached_weights] is the usual entry point.
It returns a local weight file, looking it up in this order:

1. the store,
2. the local cache (weights computed while the store was read-only),
3. otherwise the weights are computed into the store, or into the local
   cache when the store is read-only.

```python
import grid_doctor as gd

weights_file = gd.cached_weights(ds, level=9)                    # default store
weights_file = gd.cached_weights(ds, level=9, store="/work/ab1234/weights")
pyramid = gd.create_healpix_pyramid(ds, max_level=9, weights_path=weights_file)
```

The default store is the `GRID_DOCTOR_WEIGHTS_STORE` environment
variable, or `<cache_dir()>/weights` when it is unset. To compute into an
explicit file instead, use
[`compute_healpix_weights(weights_path=...)`][grid_doctor.compute_healpix_weights].

## Sharing weights

Point several users or jobs at the same directory to share weights, e.g.
`GRID_DOCTOR_WEIGHTS_STORE=/work/ab1234/weights`. Files are written
atomically, so readers never see a partial file. A directory you cannot
write to is used read-only: missing weights are computed into your local
cache instead.

Only local directories and in-memory filesystems (`memory://`, for tests)
are supported for now.

### Setting up a shared directory

On POSIX systems, let everyone add weights but only remove their own,
with the sticky bit (and setgid, so files keep the project group):

```bash
mkdir /work/ab1234/weights
chgrp ab1234 /work/ab1234/weights
chmod 3775 /work/ab1234/weights   # group may add; 3777 lets anyone add
```

grid-doctor copies the root's mode onto every directory it creates
below it, and published files follow each user's umask (use `umask 002`
for group-writable files). Files can only be removed by their owner or
the owner of the directory, so let a group account own the root to
clean up. When two users compute the same weights at once, the first
file stays and the second user uses it.

The same grid is recognised whatever its conventions: coordinates in
radians or degrees, longitudes in `0..360` or `[-180, 180)` and differing
variable names all lead to the same weight file.

## Layout

Weight files are stored under Hive-style directories, followed by the
digest of the source grid:

```
<root>/version=1/grid=healpix/ellipsoid=sphere/indexing_scheme=nested/level=9/method=conservative/<digest>.nc
```

Tools such as pyarrow, DuckDB or Polars read the directory names as
columns. One grid has the same digest at every level and method, so
[`WeightsStore.find`][grid_doctor.weights.WeightsStore.find] lists all
targets of a grid, or all grids of a target:

```python
from grid_doctor.weights import WeightsStore

store = WeightsStore("/work/ab1234/weights")
store.find(level=9, method="conservative")
```

Each file also records its identity (source digest, level, indexing
scheme, ellipsoid, method, grid-doctor and layout version) as global
attributes, so a store can be checked from the files alone.

## API

::: grid_doctor.weights.WeightsStore
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.weights.WeightsDescription
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.weights.describe_source
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.weights.SourceDescription
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.weights.WeightsRef
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.weights.local_weights_dir
    options:
      show_root_heading: true
      show_root_full_path: false
