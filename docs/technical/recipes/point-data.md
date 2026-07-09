# Point Data (Satellite Swaths, Stations, Trajectories)

This recipe converts *point-sampled* data — anything where every sample
carries its own latitude/longitude rather than living on a fixed grid —
to a HEALPix pyramid.  Typical sources are satellite Level-2 swath
products (imager or radar granules), in-situ station records, and ship
or aircraft trajectories.

## Why not `regrid_to_healpix`?

The ESMF weight path assumes a *fixed* source geometry:

- **No weight reuse.**  Every satellite granule has unique geometry, so
  the expensive weight-generation step could never be amortised.
- **Nearest smears globally.**  ESMF's nearest source-to-destination
  method assigns a value to *every* destination cell.  With a global
  HEALPix target and a narrow swath as source, the whole sphere —
  antipodal ocean included — would receive swath values.

[`bin_to_healpix`][grid_doctor.bin_to_healpix] instead assigns each
sample to the HEALPix cell containing it and reduces all samples per
cell.  In the oversampled limit (sample spacing much finer than the
target cell spacing) the per-cell **mean** converges to the
area-weighted mean, making it the binning analogue of conservative
remapping; the per-cell **mode** is the analogue of nearest-neighbour
for categorical fields.

!!! important "Choose the level from the sample spacing"
    Pick the target level so the HEALPix cell spacing
    (``58.6° / 2**level``) is *coarser* than the sample spacing —
    the same rule
    [`resolution_to_healpix_level`][grid_doctor.resolution_to_healpix_level]
    applies to gridded data.  A 1 km-resolution imager swath maps
    comfortably to level 12 (~0.9 km); binning it to level 14 would
    leave most cells empty and fabricate detail the samples cannot
    support.

## Minimal Example

```python
import grid_doctor as gd
import xarray as xr

# 1. Open — geolocation may be a data variable rather than a coordinate
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

# 3. Coarsen — the standard pyramid machinery applies unchanged
pyramid = {11: finest}
for level in range(10, -1, -1):
    pyramid[level] = gd.coarsen_healpix(pyramid[level + 1], level)

# 4. Upload
gd.save_pyramid(
    pyramid,
    "s3://my-bucket/my-l2-product.zarr",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

Mixed continuous/categorical products work in one call: pass a mapping
for ``agg``.  Each variable records its own ``grid_doctor_method``
(``binned-mean`` / ``binned-mode``), and
[`coarsen_healpix`][grid_doctor.coarsen_healpix] infers mean vs. mode
coarsening automatically when *all* variables share one method — for
mixed datasets, coarsen the categorical variables separately or pass an
explicit ``coarsen_mode``.

## Fill values

Declared ``_FillValue`` / ``missing_value`` attributes are honoured
automatically, and floating-point non-finite values are always treated
as invalid.  For products that do not declare their fills, pass them
explicitly:

```python
gd.bin_to_healpix(ds, 11, fill_values={"quality_flag": 0})
```

``0`` is accepted as a fill value — the check is ``is None``, never
truthiness.  Do **not** rely on dtype conventions ("int16 means
-32768"): products that break the convention will silently pass fill
values into your means.

## Accumulating many granules

At level 12+ the dense global array is large (~50 M cells and beyond)
while a single granule touches only a sliver of it.  Use the sparse
intermediate to keep per-granule memory proportional to the swath:

```python
sparse_granules = [
    gd.bin_to_healpix(open_granule(path), level=12, dense=False)
    for path in granule_paths
]
```

For composites over overlapping orbits, bin with ``agg="mean"`` and
``with_counts=True`` and merge with count-weighted averages, so that a
cell seen by three orbits is not biased toward the last one written.
Expand to the dense representation once, right before coarsening and
publishing, with
[`sparse_to_dense`][grid_doctor.sparse_to_dense].

## Worked example: EarthCARE MSI

EarthCARE MSI Level-2 frames are one concrete instance of this recipe:
2-D swath geolocation (``along_track`` × ``across_track``), ~500 m pixel
spacing (→ level 13 at most, level 12 in practice), continuous radiances
binned with ``mean`` and classification products (cloud mask, cloud
type) binned with ``mode``.  Nothing in the workflow above is
EarthCARE-specific — the same calls handle any L2 swath product.

!!! warning "Keep the sphere"
    Satellite geolocation is geodetic (WGS84), and ``healpix_geo``
    *could* index on the ellipsoid.  grid-doctor deliberately does not:
    all cell geometry uses a perfect sphere so that every dataset in a
    hub shares one indexing geometry.  Indexing one dataset on the
    ellipsoid would shift it by up to ~0.19° (~21 km) relative to all
    others — dozens of pixels at swath resolutions.  See the
    [technical decisions](../technical-decisions.md) document.
