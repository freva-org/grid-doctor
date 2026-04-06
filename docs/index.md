![logo](assets/apple-touch-icon.png)
#grid-doctor

> Your lat/lon data goes to rehab and comes out HEALed.

**grid-doctor** converts climate and Earth-system datasets from regular,
curvilinear, or unstructured grids to multi-resolution
[HEALPix](https://healpix.jpl.nasa.gov/) pyramids stored as
[Zarr](https://zarr.dev/) on S3.


## Features

- **Automatic grid detection** — regular (ERA5, CMIP), curvilinear (NEMO,
  ROMS, WRF), and unstructured (ICON) grids are identified and handled
  transparently.
- **HEALPix pyramid creation** — from the finest level down to level 0 in
  a single call.
- **Weight caching** — expensive triangulation is computed once
  and cached as NetCDF for subsequent runs.
- **S3 upload** — pyramids are written directly to S3 compatible stores
  as Zarr.
- **Dask integration** — regridding and coarsening are lazy and
  parallelised via `xarray.apply_ufunc`.

## Quick Example

```python
import grid_doctor as gd

ds = gd.cached_open_dataset(["data/*.nc"])
weights_file = gd.cached_weights("/path/to/weights/", nproc=4)
pyramid = gd.create_healpix_pyramid(ds, weights_path=weights_file)
gd.save_pyramid_to_s3(
    pyramid,
    "s3://my-bucket/era5",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

See  [Technical Decisions](technical-decisions.md) section for a comprehensive
description on the remapping procedures, the [Getting Started](getting-started.md)
for installation instructions and [Recipes](recipes/index.md)
for full worked examples.
