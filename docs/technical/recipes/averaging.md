# Coarsening and averaging masked fields

Every HEALPix cell has the same area, so a **complete** field needs no
area weights: its global mean is the plain mean over cells. This recipe
is for fields that represent only part of a cell—because data is
missing, or because the field is ocean- or land-only.

It explains how to:

- store valid fractions while building a pyramid;
- retain physical ocean or land coverage separately from data validity;
- find the appropriate support for a variable; and
- use that support for global, regional, and zonal means, and area
  integrals.

The background is explained in [Averaging over coarse
levels](../technical-decisions.md#averaging-over-coarse-levels-valid-fractions).

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

## Ocean and land fields: retaining physical coverage

`*_valid_fraction` answers “where was this variable available?” It does
not answer “how much of this cell is ocean or land?” Those are separate
quantities at coastlines and where observations are missing. For an
ocean-only or land-only variable, retain a continuous physical coverage
variable as well:

| Field | Store | Weight a spatial mean by |
|:--|:--|:--|
| Complete global field | value | no weight: `value.mean("cell")` |
| Field with data gaps | value, `value_valid_fraction` | valid fraction |
| Ocean-only field | value, ocean fraction, valid-ocean fraction | valid-ocean fraction |
| Land-only field | value, land fraction or land-sea mask, valid-land fraction | valid-land fraction |

!!! note "Ocean fractions, land fractions, and land-sea masks are continuous fields"
    An ocean fraction, land fraction, or land-sea mask describes
    coverage, not a categorical land/ocean label. Coarsen any of them with
    `coarsen_mode="mean"`; using `mode` would discard partial coverage
    of coastal cells. If a value is valid throughout the ocean (or
    land), the corresponding physical fraction is also the correct
    weight.

Do not derive a production land or ocean fraction from whether a data
variable is non-null. A validity mask cannot distinguish physical land
or ocean from missing observations.

### Building a pyramid for an ocean-only field

`create_healpix_pyramid` can create valid fractions for ordinary masked
fields, but it does not infer a physical ocean or land fraction. Build
and coarsen a numerator and its coverage explicitly when both concepts
matter. The [EasyGEMS ocean-averaging example](https://easy.gems.dkrz.de/Processing/healpix/ocean_averaging.html)
uses the same distinction between an ocean-only value and its physical
coverage. For a surface-ocean velocity `ice_u` $v$:

Let $v_i$ be the value in a finest-level child cell, $f_i$ its physical
ocean fraction, and $w_i$ its valid-ocean fraction:

$$
w_i =
\begin{cases}
f_i, & \text{when } v_i \text{ is valid}, \\
0, & \text{when } v_i \text{ is missing}.
\end{cases}
$$

For a parent $p$ with $k$ children ($k = 4$ for one nested HEALPix
step, or $4^d$ for a direct reduction over $d$ levels), coarsen the
numeric numerator and coverage separately:

$$
N_p = \frac{1}{k}\sum_{i \in p} v_i w_i,
\qquad
C_p = \frac{1}{k}\sum_{i \in p} w_i,
\qquad
v_p = \frac{N_p}{C_p} \quad \text{where } C_p > 0.
$$

The common $1/k$ factor makes each result a fraction of the parent-cell
area and cancels in the reconstructed value. A missing child contributes
zero to both quantities. Thus $v_p$ remains the mean over its valid
ocean area, while $C_p$ retains exactly how much of that area it
represents. A spatial ocean mean then combines parent cells as

$$
\mu_\mathrm{ocean} =
\frac{\sum_p v_p C_p}{\sum_p C_p}.
$$

```python
import xarray as xr
import grid_doctor as gd

ice = finest.ice_u
ocean_fraction = finest.ocean_fraction_surface.fillna(0)
valid_ocean_fraction = ocean_fraction.where(ice.notnull(), 0).fillna(0)

state = xr.Dataset(
    {
        # Zero is safe here because coverage is zero for land and missing data.
        "ice_u_numerator": (ice * valid_ocean_fraction).fillna(0),
        "ice_u_coverage": valid_ocean_fraction,
        "ocean_fraction_surface": ocean_fraction,
    },
    attrs=finest.attrs.copy(),
)

def public_level(state: xr.Dataset) -> xr.Dataset:
    coverage = state.ice_u_coverage
    return xr.Dataset(
        {
            "ice_u": state.ice_u_numerator / coverage.where(coverage > 0),
            "ice_u_valid_fraction": coverage,
            "ocean_fraction_surface": state.ocean_fraction_surface,
        },
        attrs=state.attrs.copy(),
    )

levels = {max_level: public_level(state)}
for level in range(max_level - 1, min_level - 1, -1):
    state = gd.coarsen_healpix(
        state, level, coarsen_mode="mean", min_valid_fraction=0.0
    )
    levels[level] = public_level(state)
```

The stored `ice_u` remains the mean velocity over the valid ocean part
of a cell: a cell that is 20% ocean and has velocity 0.8 m/s stores 0.8
m/s, not 0.16 m/s. The separate `ice_u_valid_fraction` supplies its
weight when cells are combined:

```python
ocean_mean = ds.ice_u.weighted(ds.ice_u_valid_fraction.fillna(0)).mean("cell")
```

If the field is valid everywhere that is physically ocean, weight by
`ocean_fraction_surface` instead.

### Building a pyramid for a land-only field

The same construction applies to a global land-only variable. Use a continuous
`land_fraction_surface` (or a
land-sea mask normalised to 0–1), and coarsen it with `mean`, never
`mode`:

```python
v = finest["land_variable"]  # replace with the ERA5-Land variable name
land_fraction = finest.land_fraction_surface.fillna(0)
valid_land_fraction = land_fraction.where(v.notnull(), 0).fillna(0)

state = xr.Dataset(
    {
        "v_numerator": (v * valid_land_fraction).fillna(0),
        "v_coverage": valid_land_fraction,
        "land_fraction_surface": land_fraction,
    },
    attrs=finest.attrs.copy(),
)
```

Coarsen and reconstruct it exactly as for `ice_u`. Store the public
value, `land_fraction_surface`, and `v_valid_fraction`; weight a global
mean by the latter, or by `land_fraction_surface` when data is valid
everywhere on land. Do not infer physical land coverage from non-null
values—they also reflect missing data.

### Vertical dimensions

Use a depth-specific physical fraction only when coverage really varies
with depth. This is common for a three-dimensional ocean field, where
the fraction must be on the same vertical grid and selected at the same
depth before averaging:

```python
to = ds.to.isel(time=0, depth_full=0)
weights = ds.ocean_fraction_depth_full.isel(depth_full=0)
ocean_mean = to.weighted(weights).mean("cell")
```

Leaving the ocean weights three-dimensional while selecting one data
depth can broadcast arrays and yield a plausible but incorrect result.

For land only fields, soil variables on the other hand have a `depth`
coordinate, but the physical land fraction normally remains cell-only: use the
same `land_fraction_surface` for every soil layer. Only the valid fraction
needs a `depth` dimension when availability differs between layers.

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
