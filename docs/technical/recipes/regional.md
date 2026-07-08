# Regional Datasets (CORDEX)

This recipe converts regional model output — CORDEX rotated-pole domains
being the canonical case — to a HEALPix pyramid, and shows how to access
the result.  The output is a standard **dense global** store that is
simply NaN outside the domain; with aligned chunks and NaN fill, storage
and reads stay proportional to the domain, not the globe.

Nothing here requires new machinery: rotated-pole grids are
auto-detected (`rlat`/`rlon` are first in grid-doctor's dimension
candidates, and the 2-D `lat(rlat, rlon)` coordinates classify the grid
as curvilinear), and conservative remapping to a global HEALPix mesh
leaves every cell without domain overlap unmapped, i.e. NaN.

## Creation

```python
import numpy as np
import xarray as xr
import grid_doctor as gd

ds = gd.cached_open_dataset(["/pool/data/CORDEX/EUR-11/.../tas_*.nc"])

# EUR-11: 0.11 deg -> level 9 (~0.115 deg cell spacing)
level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(ds))

hpx = gd.regrid_to_healpix(ds, level, method="conservative")
```

### Domain-edge cells: mask by coverage

Conservative weights are normalised by the **full** destination-cell
area, so a HEALPix cell only partially inside the domain receives
weights summing to its coverage fraction — and the default
`missing_policy="renormalize"` then scales that partial sum back up to a
full-cell value.  A cell 5% inside EUR-11 gets a confident-looking
value representing 5% of its area: the domain silently bleeds outward
by up to one cell.  (`"propagate"` does not help here: out-of-domain
cells are *absent* from the weight matrix, not NaN sources, so edge
cells come out biased low by the coverage factor instead.)

Until a `min_weight_fraction` floor lands in the apply kernels, mask by
coverage explicitly.  The coverage fraction is just a ones-field pushed
through the same conservative weights:

```python
ones = xr.Dataset(
    {"coverage_fraction": xr.ones_like(ds["tas"].isel(time=0, drop=True))},
    coords=ds.coords,
)
coverage = gd.regrid_to_healpix(ones, level, method="conservative")[
    "coverage_fraction"
]

hpx = hpx.where(coverage >= 0.5)          # representativeness, cf. coarsening rule
hpx["coverage_fraction"] = coverage.fillna(0.0).astype("float32")
```

The 0.5 threshold is the same representativeness rule the pyramid's
`min_valid_fraction` applies during coarsening: a cell's value must
cover at least half its area.  Publishing `coverage_fraction` makes the
boundary auditable and lets strict users apply their own threshold —
it is the gridded analogue of the `<var>_count` companions for binned
point data.  The weight file is cached, so the ones-field pass is
nearly free.

### Categorical variables: the mask is mandatory

For categorical fields, `method="nearest"` uses ESMF's nearest
source-to-destination, which assigns **every** global cell its nearest
source value — a EUR-11 land-use field would paint European classes
over the Pacific.  The coverage mask from above is therefore not a
refinement but a requirement:

```python
landuse = gd.regrid_to_healpix(ds_categorical, level, method="nearest")
landuse = landuse.where(coverage >= 0.5)
```

### Pyramid, storage, and upload

```python
pyramid = {level: hpx}
for lvl in range(level - 1, -1, -1):
    pyramid[lvl] = gd.coarsen_healpix(pyramid[lvl + 1], lvl)

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
    s3_options=gd.get_s3_options(...),
)
```

Two encoding choices make the mostly-NaN globe cheap:

- **Power-of-four chunks.**  A chunk of length `4**k` is exactly one
  level-`(L-k)` parent cell — spatially compact — so only chunks whose
  parent intersects the domain contain data.
- **`_FillValue = NaN`.**  With NaN as the Zarr fill value, all-NaN
  chunks are elided entirely (`write_empty_chunks=False` is the
  zarr-python default).  Verified: a domain covering 3.0% of the sphere
  writes 3.1% of the chunks.

The 50% rule erodes the domain boundary by at most one parent cell per
coarsening step — that is the representativeness policy working as
documented, not data loss.  Finally, add bounding-box attributes so
viewers can zoom to the domain instead of opening on an empty globe:

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

Regional stores at typical CORDEX levels (≤ 10) are completely ordinary
hub datasets: coordinates are materialised, the viewer renders them,
and any HEALPix-aware tool works unchanged.  NaN outside the domain is
the expected state, not an error.

The pleasant property of the equal-area grid: a **domain mean is just a
NaN-aware mean** — no latitude weighting, no domain mask file:

```python
import xarray as xr

ds9 = xr.open_zarr("s3://.../cordex-eur11-tas.zarr/level_9.zarr", chunks=None)
domain_mean = ds9["tas"].mean("cell", skipna=True)   # area-weighted by construction
```

Sub-domain extraction uses the same selectors as everything else —
they read only the chunks the box touches:

```python
import grid_doctor as gd

alps = gd.select_bbox(ds9, lon=(5.0, 16.0), lat=(43.0, 48.5))
alps["tas"].mean("cell", skipna=True)                # Alpine-box mean
```

And because nested cell IDs align across levels by construction,
comparing the CORDEX domain against a global dataset needs no
interpolation — select the same cells from both:

```python
era5 = xr.open_zarr("s3://.../era5.zarr/level_9.zarr", chunks=None)
era5_alps = gd.select_cells(era5, alps["cell"].values)
bias = alps["tas"] - era5_alps["t2m"]                # cell-by-cell, exact
```

For evaluation workflows this is the payoff of one shared indexing
geometry: model-vs-reanalysis differences are positional subtractions,
never regridding.

!!! note "Follow-up: kernel-level boundary handling"
    A `min_weight_fraction` floor in the weight-application kernels
    (masking cells whose valid-weight sum falls below a threshold,
    with an optional built-in `coverage_fraction` output) would replace
    the ones-field workaround above.  Until then, the explicit mask is
    the supported pattern.
