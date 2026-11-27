# Regional grids (CORDEX)

This recipe converts regional model output to a HEALPix pyramid, with
CORDEX rotated-pole domains as the canonical case, and shows how to
access the result.  The output is a standard **dense global** store
that is simply NaN outside the domain; with aligned chunks and NaN fill,
storage and reads stay proportional to the domain, not the globe.

Nothing here needs special machinery.  Rotated-pole grids are detected
automatically (`rlat`/`rlon` come first among grid-doctor's dimension
candidates, and the 2-D `lat(rlat, rlon)` coordinates classify the grid
as curvilinear).  Conservative remapping to a global HEALPix mesh leaves
every cell without domain overlap unmapped, that is NaN.

## Creation

```python
import numpy as np
import xarray as xr

import grid_doctor as gd

ds = xr.open_mfdataset("CORDEX/EUR-11/.../tas_*.nc")

# EUR-11: 0.11 deg -> level 9 (~0.115 deg cell spacing)
level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))

hpx = ds.gd.regrid_to_healpix(level, method="conservative")
```

### Domain-edge cells: mask by coverage

Conservative weights are normalised by the **full** destination-cell
area, so a HEALPix cell only partly inside the domain receives weights
summing to its coverage fraction.  The default
`missing_policy="renormalize"` then scales that partial sum back up to
a full-cell value: a cell 5% inside EUR-11 gets a confident-looking
value that represents 5% of its area, and the domain silently grows by
up to one cell.  `"propagate"` does not help here, because cells
outside the domain are *absent* from the weight matrix rather than NaN
sources, so edge cells come out biased low by the coverage factor
instead.

The weight application has no coverage threshold yet, so mask by
coverage explicitly.  The coverage fraction is a field of ones remapped
with the same conservative weights:

```python
ones = xr.Dataset(
    {"coverage_fraction": xr.ones_like(ds["tas"].isel(time=0, drop=True))},
    coords=ds.coords,
)
coverage = ones.gd.regrid_to_healpix(level, method="conservative")[
    "coverage_fraction"
]

hpx = hpx.where(coverage >= 0.5)          # representativeness, cf. coarsening rule
hpx["coverage_fraction"] = coverage.fillna(0.0).astype("float32")
```

The 0.5 threshold is the same representativeness rule that
`min_valid_fraction` applies during coarsening: a cell's value must
cover at least half of its area.  Publishing `coverage_fraction` makes
the boundary auditable and lets strict users apply their own threshold;
it is the gridded analogue of the `<var>_count` companions of binned
point data.  The weight file is reused, so remapping the ones costs
almost nothing.

### Categorical variables: the mask is mandatory

For categorical fields, `method="nearest"` uses ESMF's nearest
source-to-destination method, which assigns **every** global cell its
nearest source value: a EUR-11 land-use field would paint European
classes over the Pacific.  The coverage mask from above is therefore
not a refinement but a requirement:

```python
landuse = ds_categorical.gd.regrid_to_healpix(level, method="nearest")
landuse = landuse.where(coverage >= 0.5)
```

### Pyramid, storage, and upload

Coarsen every level from the finest one and store the valid fractions,
so that means over coarse levels can be weighted (see
[Access](#access) below):

```python
pyramid = {level: hpx}
for lvl in range(level - 1, -1, -1):
    pyramid[lvl] = gd.coarsen_healpix(hpx, lvl, valid_fraction={"tas": True})

encoding = {
    lvl: {
        var: {"chunks": (4**6,), "_FillValue": np.nan}
        for var in pyramid[lvl].data_vars
    }
    for lvl in pyramid
}
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/cordex-eur11-tas.zarr",
    encoding=encoding,
    s3_options=gd.get_s3_options("https://s3.example.org", "~/.s3-credentials.json"),
)
```

Valid fractions are relative to the level they are coarsened from,
which is why every level starts from `hpx` instead of the next finer
level.  Two encoding choices make the mostly-NaN globe cheap:

- **Power-of-four chunks.**  A chunk of length `4**k` is exactly one
  level-`(L-k)` parent cell and therefore spatially compact, so only
  chunks whose parent intersects the domain contain data.
- **`_FillValue = NaN`.**  With NaN as the Zarr fill value, all-NaN
  chunks are not written at all (`write_empty_chunks=False` is the
  zarr-python default).  Verified: a domain covering 3.0% of the sphere
  writes 3.1% of the chunks.

The 50% rule erodes the domain boundary by at most one parent cell per
coarsening step.  That is the representativeness policy working as
documented, not data loss.  Finally, add bounding-box attributes so
that viewers can zoom to the domain instead of opening on an empty
globe:

```python
for lvl, dataset in pyramid.items():
    dataset.attrs.update(
        geospatial_lat_min=float(ds["lat"].min()),
        geospatial_lat_max=float(ds["lat"].max()),
        geospatial_lon_min=float(ds["lon"].min()),
        geospatial_lon_max=float(ds["lon"].max()),
    )
```

## Access

Regional stores at typical CORDEX levels (≤ 10) are ordinary HEALPix
datasets: the coordinates are stored, viewers render them, and any
HEALPix-aware tool works unchanged.  NaN outside the domain is the
expected state, not an error.

On the equal-area grid, a **domain mean is just a NaN-aware mean**, with
no latitude weighting and no domain mask file:

```python
import xarray as xr

ds9 = xr.open_zarr("s3://my-bucket/cordex-eur11-tas.zarr/level_9.zarr", chunks=None)
domain_mean = ds9["tas"].mean("cell", skipna=True)   # area-weighted by construction
```

This holds at the finest level of the pyramid, where every cell is
either inside or outside the domain.  On coarser levels, cells along the
domain edge are only partly covered, and a plain mean gives them full
weight.  Weight by the stored valid fractions there, as shown in
[Coarsening and averaging masked fields](averaging.md):

```python
ds5 = xr.open_zarr("s3://my-bucket/cordex-eur11-tas.zarr/level_5.zarr", chunks=None)
domain_mean = ds5["tas"].weighted(ds5["tas_valid_fraction"].fillna(0)).mean("cell")
```

Sub-domains are extracted with the same selectors as everything else,
which read only the chunks the box touches:

```python
import grid_doctor as gd

alps = gd.select_bbox(ds9, lon=(5.0, 16.0), lat=(43.0, 48.5))
alps["tas"].mean("cell", skipna=True)                # Alpine-box mean
```

And because nested cell IDs align across levels by construction,
comparing the CORDEX domain against a global dataset needs no
interpolation: select the same cells from both.

```python
era5 = xr.open_zarr("s3://my-bucket/era5.zarr/level_9.zarr", chunks=None)
era5_alps = era5.gd.select_cells(alps["cell"].values)
bias = alps["tas"] - era5_alps["t2m"]                # cell by cell, exact
```

For evaluation workflows this is the payoff of one shared indexing
geometry: differences between model and reanalysis are positional
subtractions, never regridding.
