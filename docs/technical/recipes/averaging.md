# Averaging masked fields

Every HEALPix cell has the same area, so for a **complete** field a
global mean is the plain mean over all cells.  Fields with missing
values (ocean-only variables, sea ice, observation gaps, regional
domains) are different on coarser pyramid levels: a coarse cell can be
only partly valid, and its value is the mean over that valid part.  A
plain mean then gives a coastal cell that is 10 % ocean the same weight
as an open-ocean cell.

The fix is to weight every cell by its **valid fraction**, the share of
valid finest-level cells beneath it.  With these weights, means agree
across all levels of a pyramid.  The background is explained in
[Averaging over coarse levels](../technical-decisions.md#averaging-over-coarse-levels-valid-fractions).

## Storing the valid fractions

Fractions are written when the pyramid is created:

```python
import grid_doctor as gd

# ds, level and weights_file as in the structured-grid recipe
pyramid = gd.create_healpix_pyramid(
    ds,
    max_level=level,
    weights_path=weights_file,
    # land-sea masked SST: one cell-only fraction is enough;
    # sea ice changes over time and needs the full shape
    valid_fraction={"sst": "static", "siconc": True},
)
gd.save_pyramid(pyramid, store, mode="w")
```

Every level then contains `sst_valid_fraction` and
`siconc_valid_fraction` (float32, between 0 and 1), linked from the data
variables through the CF `ancillary_variables` attribute.  A static
fraction has only the `cell` dimension and broadcasts against all time
steps.

## Finding the weights of a variable

The `ancillary_variables` attribute names the fraction, so a small
helper works for any variable, with or without fractions:

```python
import xarray as xr

def valid_fraction(ds: xr.Dataset, name: str) -> xr.DataArray | None:
    """Return the valid-fraction weights of ``ds[name]``, if stored."""
    for candidate in ds[name].attrs.get("ancillary_variables", "").split():
        if candidate.endswith("_valid_fraction") and candidate in ds:
            return ds[candidate].fillna(0)
    return None


def weighted_mean(ds: xr.Dataset, name: str, dim: str = "cell") -> xr.DataArray:
    """Area mean over the valid part of ``ds[name]``."""
    weights = valid_fraction(ds, name)
    if weights is None:  # complete field: every cell counts fully
        return ds[name].mean(dim)
    return ds[name].weighted(weights).mean(dim)
```

## Global mean and time series

```python
ds = xr.open_zarr(f"{store}/level_5.zarr")

global_sst = weighted_mean(ds, "sst")      # one value per time step
global_sst.plot()
```

The same call on any other level gives the same result, see
[checking consistency](#checking-consistency-across-levels) below.

## Regional mean

The region selectors keep all variables with a `cell` dimension, so the
fractions are selected together with the data:

```python
north_sea = gd.select_bbox(ds, lon=(-4.0, 9.0), lat=(51.0, 61.0))
regional_sst = weighted_mean(north_sea, "sst")
```

## Zonal mean

xarray's `weighted` does not combine with `groupby`, so form the
weighted sums explicitly.  Weights only count where the data is valid:

```python
import numpy as np

bands = np.arange(-90, 91, 15)
latitude = ds["latitude"].compute()          # grouping needs loaded labels
weights = valid_fraction(ds, "sst").where(ds["sst"].notnull(), 0)
numerator = (ds["sst"].fillna(0) * weights).groupby_bins(latitude, bands).sum()
denominator = weights.groupby_bins(latitude, bands).sum()
zonal_sst = numerator / denominator          # NaN for bands without data
```

Stores above level 10 have no materialised `latitude` coordinate; use a
level at or below 10 for zonal statistics, or the region selectors,
which reconstruct coordinates for the cells they return.

## Area integrals

All cells of a level have the area $4 \pi R^2 / n_\text{cells}$, so the
valid area and area integrals are sums over the fractions:

```python
R = 6371.0                                    # km, the sphere of the grid
cell_area = 4 * np.pi * R**2 / ds.sizes["cell"]

ocean_area = valid_fraction(ds, "sst").sum("cell") * cell_area
sst_integral = (ds["sst"] * valid_fraction(ds, "sst")).sum("cell") * cell_area
```

The valid area is identical on every level: a coarse cell has $4^k$
times the area of a finest-level cell, and its fraction is the number of
its valid finest-level cells divided by $4^k$.  The integral is too, as
long as no cells were masked by `min_valid_fraction` (masked cells drop
out of the sum, see [pitfalls](#pitfalls)).

## Checking consistency across levels

```python
for level in range(5, -1, -1):
    lvl = xr.open_zarr(f"{store}/level_{level}.zarr")
    print(level, weighted_mean(lvl, "sst").isel(time=0).values)
```

With `min_valid_fraction=0` the printed means agree to rounding
precision on every level.

## Pitfalls

- **Plain means over coarse levels.**  `ds["sst"].mean("cell")` gives
  partly valid cells full weight and drifts from level to level.  Use
  the weights whenever a field has missing values.
- **Cells masked by `min_valid_fraction`.**  A cell with less than
  `min_valid_fraction` (default 0.5) of its finest-level cells valid is
  NaN even though its fraction is positive.  Weighted means skip it, so
  with the default threshold coarse-level means describe a slightly
  smaller area than the finest level.  Build the pyramid with
  `min_valid_fraction=0` if exact agreement across levels matters more
  than hiding sparsely covered cells.  The threshold a level was built
  with is stored in its `grid_doctor_min_valid_fraction` attribute.
- **Static fractions for changing masks.**  `"static"` stores the mask
  of the first time step only.  Use the full shape (`True`) for masks
  that change over time or height, such as sea ice or clouds.
- **Unweighted regional means.**  Domain-edge cells of regional datasets
  are partly valid on coarse levels too; weight them in the same way.
