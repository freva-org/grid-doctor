# Point data (satellite swaths, stations, trajectories)

This recipe converts *point-sampled* data to a HEALPix pyramid: anything
where every sample carries its own latitude and longitude instead of
sitting on a fixed grid. Typical sources are satellite Level-2 swath
products (imager or radar granules), in-situ station records, and ship
or aircraft trajectories.

## Why not `regrid_to_healpix`?

The ESMF weight path assumes a *fixed* source geometry:

- **No weight reuse.**  Every satellite granule has unique geometry, so
  the expensive weight-generation step could never be amortised.
- **Nearest smears globally.**  ESMF's nearest source-to-destination
  method assigns a value to *every* destination cell.  With a global
  HEALPix target and a narrow swath as source, the whole sphere,
  antipodal ocean included, would receive swath values.

[`bin_to_healpix`][grid_doctor.bin_to_healpix] instead assigns each
sample to the HEALPix cell containing it and reduces all samples per
cell.  In the oversampled limit (sample spacing much finer than the
target cell spacing) the per-cell **mean** converges to the
area-weighted mean, making it the binning analogue of conservative
remapping; the per-cell **mode** is the analogue of nearest-neighbour
for categorical fields.

!!! tip "Choose the level from the sample spacing"
    Pick the target level so that the HEALPix cell spacing
    (``58.6° / 2**level``) is *coarser* than the sample spacing, the
    same rule
    [`resolution_to_healpix_level`][grid_doctor.resolution_to_healpix_level]
    applies to gridded data.  A 1 km imager swath maps comfortably to
    level 12 (~1.6 km); binning it to level 14 (~0.4 km) would leave
    most cells empty and fabricate detail the samples cannot support.

## Minimal example

```python
import xarray as xr

import grid_doctor as gd

# 1. Open: geolocation may be a data variable rather than a coordinate
ds = xr.open_dataset("granule.h5", group="ScienceData")

# 2. Bin to the finest level
finest = gd.bin_to_healpix(
    ds,
    level=11,
    agg={"radiance": "mean", "cloud_type": "mode"},
    lat_name="latitude",          # explicit names for non-standard products
    lon_name="longitude",
    fill_values={"cloud_type": 255},
    with_counts=True,             # coverage becomes auditable downstream
)

# 3. Coarsen: continuous variables by mean, categorical ones by mode
categorical = ["cloud_type"]
pyramid = {11: finest}
for level in range(10, -1, -1):
    pyramid[level] = xr.merge(
        [
            gd.coarsen_healpix(finest.drop_vars(categorical), level, coarsen_mode="mean"),
            gd.coarsen_healpix(finest[categorical], level, coarsen_mode="mode"),
        ],
        combine_attrs="override",
    )

# 4. Upload
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/my-l2-product.zarr",
    s3_options=gd.get_s3_options("https://s3.example.org", "~/.s3-credentials.json"),
)
```

Mixed continuous and categorical products are binned in one call: pass
a mapping for ``agg``.  Each variable records its own
``grid_doctor_method`` (``binned-mean`` or ``binned-mode``).
[`coarsen_healpix`][grid_doctor.coarsen_healpix], however, chooses one
mode for the whole dataset: with ``coarsen_mode="auto"`` it would
average ``cloud_type`` as well.  Coarsen the two groups separately, as
above, and merge them.  For a product with a single method, one
``coarsen_healpix`` call per level is enough.

## Fill values

Declared ``_FillValue`` and ``missing_value`` attributes are honoured
automatically, and floating-point non-finite values are always treated
as invalid.  For products that do not declare their fills, pass them
explicitly:

```python
gd.bin_to_healpix(ds, 11, fill_values={"quality_flag": 0})
```

``0`` is accepted as a fill value: the check is ``is None``, never
truthiness.  Do **not** rely on dtype conventions ("int16 means
-32768"): products that break the convention will silently pass fill
values into your means.

## Accumulating many granules

At level 12 the dense global array has about 200 million cells, while a
single granule touches only a sliver of it.  Bin each granule to the
compact (sparse) form, which holds only the cells the granule covers:

```python
# granule_paths: your list of files; open each like ds above
sparse_granules = [
    gd.bin_to_healpix(xr.open_dataset(path), level=12, agg="mean", dense=False, with_counts=True)
    for path in granule_paths
]
```

Where orbits overlap, merge the granules with count-weighted means, so
that a cell seen by three orbits is not biased toward the last one
written.  The result equals binning all samples at once:

```python
stacked = xr.concat(sparse_granules, dim="granule", join="outer")
counts = stacked["radiance_count"].fillna(0)
composite = xr.Dataset(
    {
        "radiance": (stacked["radiance"] * counts).sum("granule") / counts.sum("granule"),
        "radiance_count": counts.sum("granule"),
    },
    attrs=sparse_granules[0].attrs,
)
```

Expand to the dense representation once, right before coarsening and
publishing, with [`sparse_to_dense`][grid_doctor.sparse_to_dense].

## Worked example: EarthCARE MSI

EarthCARE MSI Level-2 frames are one concrete instance of this recipe:
2-D swath geolocation (``along_track`` × ``across_track``), ~500 m pixel
spacing (level 13, ~0.8 km, at most; level 12 in practice), continuous
radiances binned with ``mean`` and classification products (cloud mask,
cloud type) binned with ``mode``.  Nothing in the workflow above is
specific to EarthCARE: the same calls handle any L2 swath product.

!!! warning "Sphere and ellipsoid are different grids"
    Satellite geolocation is geodetic (WGS84). Binning assigns each
    position to the HEALPix cell that contains it, on the sphere by
    default or on the WGS84 ellipsoid with ``ellipsoid="WGS84"`` (or
    ``gd.config.ellipsoid``). Both are equally accurate, but they are
    different grids: cells on the ellipsoid are shifted by up to ~0.13°
    (~14 km) relative to cells on the sphere, several to dozens of
    pixels at swath resolutions. Only compare data that uses the same
    definition; the output records which one it uses. See
    [Sphere or ellipsoid](../technical-decisions.md#sphere-or-ellipsoid)
    in the technical decisions.
