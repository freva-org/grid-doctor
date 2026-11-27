# Working with the data

A HEALPix pyramid written by [grid-doctor](https://grid-doctor.org) is a
set of ordinary Zarr stores, one per level.
Global levels up to the coordinate threshold (level 10 by default) carry
`latitude`/`longitude` arrays and work with any xarray or HEALPix-aware
tool. This page covers what is different about the rest: regions,
very high levels and stores without coordinate arrays.

## Opening a level

```python
import xarray as xr

ds = xr.open_zarr(
    "s3://my-bucket/my-dataset/level_9.zarr",
    storage_options={"anon": True, "endpoint_url": "https://s3.example.org"},
    chunks=None,
)
```

`chunks=None` gives plain lazy Zarr-backed arrays that are sliced in
proportion to what you select. With the default dask chunking, xarray
builds a description of the *global* chunk grid when opening: harmless
at low levels, millions of entries at level 16 and nearly a billion at
level 20, where opening hangs before any data moves. Open high levels
with `chunks=None` and select a region before loading anything.

The cells are equal-area, so a global or domain mean is a plain mean
over cells, with no latitude weighting:

```python
ds["tas"].mean("cell")
```

!!! note "Fields with missing values"
    For variables with missing values (ocean-only fields, observation
    gaps), cells on coarser levels can be only partly valid, and a plain
    mean over cells is biased. If the store has a `<name>_valid_fraction`
    variable, use it as the weight, see
    [Coarsening and averaging masked fields](https://grid-doctor.org/latest/recipes/averaging/):

    ```python
    ds.sst.weighted(ds.sst_valid_fraction.fillna(0)).mean("cell")
    ```

## Selecting a region

grid-doctor's [`select_bbox`](https://grid-doctor.org/latest/api/select/#grid_doctor.select_bbox) and
[`select_cone`](https://grid-doctor.org/latest/api/select/#grid_doctor.select_cone) read only the chunks a region
touches and return a small, loaded dataset:

```python
import grid_doctor as gd

berlin = gd.select_bbox(ds, lon=(13.1, 13.8), lat=(52.3, 52.7))
station = gd.select_cone(ds, lon=13.41, lat=52.52, radius=0.05)  # degrees
```

The result is a compact subset: its `cell` coordinate holds the global
HEALPix indices, and `latitude`/`longitude` are computed for the
selected cells. It *covers* the region, so cells straddling the edge are
included; cut the exact box with the coordinates if you need to:

```python
inside = (
    (berlin.longitude >= 13.1) & (berlin.longitude <= 13.8)
    & (berlin.latitude >= 52.3) & (berlin.latitude <= 52.7)
)
berlin = berlin.isel(cell=inside.values.nonzero()[0])
```

[`select_cells`](https://grid-doctor.org/latest/api/select/#grid_doctor.select_cells) extracts a list of cell
indices, for example the same cells from another dataset.

## Stores without coordinate arrays

At high levels a global coordinate array is useless: level 16 has about
5.2 × 10¹⁰ cells, and `latitude`/`longitude` would take hundreds of
gigabytes while carrying no information, because HEALPix coordinates are
a function of the cell index. Levels above the threshold are therefore
written without them (`save_pyramid(write_coords=...)` controls this).
Such stores carry only the `crs` grid mapping and the attribute
`grid_doctor_implicit_coords = 1`.

Nothing changes for the selection functions: they compute the
coordinates of exactly the cells they return. For cells you sliced
yourself, [`attach_cell_coords`](https://grid-doctor.org/latest/api/select/#grid_doctor.attach_cell_coords) adds
them:

```python
import numpy as np

part = ds.isel(cell=slice(start, stop))
part = gd.attach_cell_coords(part, np.arange(start, stop), level=16)
```

## Comparing levels and datasets

In nested ordering the parent of a cell is a bit shift away: the
level-9 parent of a level-16 cell is `cell >> 14`, two bits per level.
To put a high-resolution selection next to a coarser dataset, select the
parents there, no interpolation needed:

```python
parents = np.unique(berlin["cell"].values >> (2 * (16 - 9)))
era5_berlin = era5_level9.gd.select_cells(parents)
```

Datasets at the same level and with the same definition (sphere or
ellipsoid, see the
[technical decisions](technical-decisions.md#sphere-or-ellipsoid)) can
be compared cell by cell directly.

## Plotting

The selection is a normal dataset with per-cell coordinates:

```python
import matplotlib.pyplot as plt

plt.scatter(berlin["longitude"], berlin["latitude"], c=berlin["t2m"], s=1)
plt.colorbar(label="t2m")
```

At high levels a scatter of cell centres looks like a raster at typical
figure sizes. For a quick overview, open a coarse level of the same
pyramid, which carries coordinates and works in any HEALPix viewer.

## How the selection works

Every cell at a coarse level corresponds to one *contiguous* range of
cells at a fine level: parent `p` at level `L - Δ` is the range
`[p · 4^Δ, (p + 1) · 4^Δ)` at level `L`. The selection functions find the
coarse cells that cover the region (at `level - query_delta`, eight
levels coarser by default), merge neighbouring ranges, and read each as
one slice. When the store is chunked at a power of four, each range is
whole chunks: a city-sized box at level 16 is a handful of contiguous
reads. Setting `query_delta` to the store's chunk exponent reads no
data outside the covering cells.

## Common mistakes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Opening hangs or uses a lot of memory | default dask chunking at a high level | `chunks=None` |
| `MemoryError` on `.load()` or `.values` | loading the global cell dimension | select a region first |
| No `latitude`/`longitude` | store without coordinate arrays | use the selection functions or `attach_cell_coords` |
| Slow reads for a small region | `query_delta` not aligned with the store chunks | set `query_delta` to the chunk exponent |
| Fields shifted by up to ~14 km | comparing sphere and ellipsoid data | use data on the same definition |
