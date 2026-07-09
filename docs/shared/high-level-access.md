# Accessing High-Resolution Regional Data

Some datasets in the hub cover a small area — a city, an island, a
catchment — at very high HEALPix levels (16 ≈ 100 m, 20 ≈ 6 m cells).
These stores look different from the global pyramids in two ways, and
both change how you load them:

1. **No coordinate arrays.**  The global cell dimension at level 16 has
   ~5.2 × 10¹⁰ entries; materialised `latitude`/`longitude` arrays would
   cost hundreds of gigabytes while carrying no information — HEALPix
   coordinates are a pure function of the cell index.  These stores
   carry only the `crs` variable and the `healpix_*` attributes
   (`grid_doctor_implicit_coords = 1` marks them), and coordinates are
   computed on the fly for whatever cells you actually read.
2. **All access is region-driven.**  Nobody loads the full cell
   dimension, the same way nobody downloads a complete XYZ tile set.
   You select a region; the selection is small by construction.

## Opening: always `chunks=None`

```python
import xarray as xr

ds = xr.open_zarr(
    "s3://data/city-example.zarr/level_16.zarr",
    storage_options={"anon": True, "endpoint_url": "https://s3.waterpark.dkrz.de"},
    chunks=None,          # <- essential at high levels
)
```

`chunks=None` gives you plain lazy Zarr-backed arrays that slice in
O(selection).  With the default dask chunking, xarray eagerly builds a
chunk-grid description proportional to the **global** chunk count —
millions of entries at level 16, nearly a billion at level 20.  Opening
would hang long before any data moves.

The opened dataset has an enormous nominal shape and no coordinates.
That is fine — do not call `.load()`, do not print `.values`.  Select
first.

## Selecting a region

```python
import grid_doctor as gd

berlin = gd.select_bbox(ds, lon=(13.1, 13.8), lat=(52.3, 52.7))
```

That one call is fully loaded and ready:

```python
berlin["t2m"].mean()                       # data is in memory
berlin["latitude"], berlin["longitude"]    # computed for these cells only
berlin["cell"]                             # the global HEALPix indices
```

A circular region works the same way:

```python
station = gd.select_cone(ds, lon=13.41, lat=52.52, radius=0.05)
```

and if you already know the cell IDs (e.g. from a previous selection or
another dataset):

```python
subset = gd.select_cells(ds, my_cell_ids)
```

### Why this is fast

In nested ordering, every cell at a coarse level corresponds to one
*contiguous* index range at the fine level.  `select_bbox` rasterises
your box at a coarse level (a cheap geometric query), and each coarse
parent becomes a single contiguous slice — which is exactly one Zarr
chunk when the store is chunked at a power of four.  A city-sized box
resolves to a handful of contiguous byte-range reads from S3, with
nothing wasted.  The `query_delta` parameter controls the granularity:
the selection arrives in whole blocks of `4**query_delta` cells, so
matching it to the store's chunk exponent gives zero-waste reads.

## Plotting

The selection is a normal xarray dataset with per-cell coordinates:

```python
import matplotlib.pyplot as plt

plt.scatter(
    berlin["longitude"], berlin["latitude"],
    c=berlin["t2m"], s=1, cmap="viridis",
)
plt.colorbar(label="t2m")
```

For publication-quality regional maps, bin the cells into a local
raster or use any HEALPix-aware plotting path — at level 16 a scatter
of cell centres is visually indistinguishable from a raster at typical
figure sizes.

## Comparing across levels and datasets

Nested cell IDs make cross-level alignment a bit shift: the level-9
parent of a level-16 cell is `cell_id >> 14` (two bits per level).  To
place your high-resolution selection onto, say, an ERA5 pyramid level:

```python
parents_l9 = np.unique(berlin["cell"].values >> (2 * (16 - 9)))
era5_region = gd.select_cells(era5_level9, parents_l9)
```

No interpolation, no index join — the hierarchies coincide by
construction, which is one of the reasons the hub uses nested ordering
everywhere.

## Coarser overview levels

High-level stores are still pyramids.  For a quick overview, open a
coarse level of the *same* store — levels at or below the coordinate
threshold are written with materialised coordinates and behave exactly
like every other dataset in the hub, including in the browser viewer:

```python
overview = xr.open_zarr(".../city-example.zarr/level_8.zarr", chunks=None)
```

## Common mistakes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Opening hangs or eats memory | default dask chunking at high level | `chunks=None` |
| `MemoryError` on `.load()` / `.values` | loading the global dimension | select a region first |
| "no latitude coordinate" | store has implicit coordinates | use the selectors, or `gd.select_cells` + the returned coords |
| Slow S3 reads for a small box | `query_delta` misaligned with store chunks | match `query_delta` to the chunk exponent |
