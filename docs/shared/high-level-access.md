# Accessing High-Resolution Regional Data

Some datasets in the hub cover a small area (a city, an island, a
catchment) at very high HEALPix levels (16 is roughly 100 m, 20 roughly
6 m cells).  These stores look different from the global pyramids in
two ways, and both change how you load them:

1. **No coordinate arrays.**  The global cell dimension at level 16 has
   about 5.2 × 10¹⁰ entries; materialised `latitude`/`longitude` arrays
   would cost hundreds of gigabytes while carrying no information,
   because HEALPix coordinates are a pure function of the cell index.
   These stores carry only the `crs` variable and the `healpix_*`
   attributes (`grid_doctor_implicit_coords = 1` marks them), and
   coordinates are computed on the fly for whatever cells you actually
   read.
2. **All access is region-driven.**  Nobody loads the full cell
   dimension, the same way nobody downloads a complete XYZ tile set.
   You select a region; the selection is small by construction.

!!! note "Status"
    A dedicated client package with one-call region selection is in
    preparation and will be announced here.  Until then, this page
    shows how to do everything with tools that exist today: `xarray`,
    `numpy`, and [healpix-geo](https://github.com/GRID4EARTH/healpix-geo)
    (`pip install healpix-geo`).  The workflow is a few lines longer
    but uses the exact mechanics the client package will wrap.

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
chunk-grid description proportional to the **global** chunk count:
millions of entries at level 16, nearly a billion at level 20.  Opening
would hang long before any data moves.

The opened dataset has an enormous nominal shape and no coordinates.
That is fine.  Do not call `.load()`, do not print `.values`.  Select
first.

## Selecting a region by hand

The key property is the nested ordering: every cell at a coarse level
corresponds to one *contiguous* index range at the fine level.  A
region selection therefore has three steps: find the coarse cells that
cover your region, expand each to its contiguous fine-level slice, and
compute coordinates for exactly those cells.

### 1. Cover the region at a coarse level

Rasterise the bounding box by sampling points and indexing them at a
coarse level (level 10 cells are about 6 km, a good default for
city-sized boxes):

```python
import numpy as np
from healpix_geo import nested

level, coarse = 16, 10
lon_min, lon_max, lat_min, lat_max = 13.1, 13.8, 52.3, 52.7

lon_s, lat_s = np.meshgrid(
    np.linspace(lon_min, lon_max, 256),
    np.linspace(lat_min, lat_max, 256),
)
parents = np.unique(
    nested.lonlat_to_healpix(lon_s.ravel(), lat_s.ravel(), coarse)
)
```

Make the sampling comfortably denser than the coarse cell size so no
cell inside the box is missed.

### 2. Expand parents to contiguous fine-level slices

Each coarse parent covers `4**(level - coarse)` consecutive fine cells,
so a parent `p` becomes the slice `[p << shift, (p + 1) << shift)` with
`shift = 2 * (level - coarse)`.  Merging runs of adjacent parents keeps
the number of reads minimal:

```python
shift = 2 * (level - coarse)

def merge_ranges(parents, shift):
    starts, stops = [], []
    for p in np.sort(parents):
        lo, hi = int(p) << shift, (int(p) + 1) << shift
        if stops and lo == stops[-1]:
            stops[-1] = hi          # extend the previous run
        else:
            starts.append(lo)
            stops.append(hi)
    return list(zip(starts, stops))

ranges = merge_ranges(parents, shift)
region = xr.concat(
    [ds.isel(cell=slice(a, b)) for a, b in ranges], dim="cell"
)
```

Because the store's cell dimension is the plain nested index,
positional `isel` slices and cell IDs coincide, and each slice maps to
a handful of contiguous byte-range reads from S3.

### 3. Attach coordinates and cut the exact region

```python
cell_ids = np.concatenate(
    [np.arange(a, b, dtype="uint64") for a, b in ranges]
)
lon_c, lat_c = nested.healpix_to_lonlat(cell_ids, level)

region = region.assign_coords(
    cell=("cell", cell_ids),
    longitude=("cell", lon_c),
    latitude=("cell", lat_c),
)

inside = (
    (lon_c >= lon_min) & (lon_c <= lon_max)
    & (lat_c >= lat_min) & (lat_c <= lat_max)
)
berlin = region.isel(cell=np.flatnonzero(inside)).load()
```

`berlin` is now a small, fully loaded dataset with per-cell
coordinates and the global HEALPix indices in `berlin["cell"]`:

```python
berlin["t2m"].mean()
```

### Circular regions

Select a bounding box around the centre as above, then mask by
great-circle distance instead of the box test:

```python
def gc_distance_deg(lon, lat, lon0, lat0):
    lam, phi = np.radians(lon), np.radians(lat)
    lam0, phi0 = np.radians(lon0), np.radians(lat0)
    s = (
        np.sin((phi - phi0) / 2) ** 2
        + np.cos(phi) * np.cos(phi0) * np.sin((lam - lam0) / 2) ** 2
    )
    return np.degrees(2 * np.arcsin(np.sqrt(s)))

inside = gc_distance_deg(lon_c, lat_c, 13.41, 52.52) <= 0.05
station = region.isel(cell=np.flatnonzero(inside)).load()
```

### Why this is fast

In nested ordering, a coarse parent is exactly one contiguous slice at
the fine level, which is exactly one Zarr chunk when the store is
chunked at a power of four.  A city-sized box resolves to a handful of
contiguous byte-range reads from S3, with nothing wasted.  Choosing the
coarse cover level so that `level - coarse` matches the store's chunk
exponent gives zero-waste reads; a coarser cover reads whole aligned
blocks, a finer one reads partial chunks.

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
raster or use any HEALPix-aware plotting path.  At level 16 a scatter
of cell centres is visually indistinguishable from a raster at typical
figure sizes.

## Comparing across levels and datasets

Nested cell IDs make cross-level alignment a bit shift: the level-9
parent of a level-16 cell is `cell_id >> 14` (two bits per level).  To
place your high-resolution selection onto, say, an ERA5 pyramid level,
reuse the slice expansion from above at the coarser level:

```python
parents_l9 = np.unique(berlin["cell"].values >> (2 * (16 - 9)))
era5_region = era5_level9.isel(
    cell=np.concatenate([np.arange(p, p + 1) for p in parents_l9])
)
```

No interpolation, no index join.  The hierarchies coincide by
construction, which is one of the reasons the hub uses nested ordering
everywhere.

## Coarser overview levels

High-level stores are still pyramids.  For a quick overview, open a
coarse level of the *same* store.  Levels at or below the coordinate
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
| "no latitude coordinate" | store has implicit coordinates | compute coordinates from cell IDs (step 3 above) |
| Slow S3 reads for a small box | cover level misaligned with store chunks | match `level - coarse` to the chunk exponent |
| Cells missing at the box edge | sampling grid too sparse in step 1 | sample denser than the coarse cell size |
