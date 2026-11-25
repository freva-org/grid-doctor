---
hide:
  - navigation
---

# grid-doctor

> Your lat/lon data goes to rehab and comes out HEALed.

**grid-doctor** converts climate and Earth-system datasets from regular,
curvilinear, or unstructured grids, and from point samples such as
satellite swaths, to multi-resolution [HEALPix](https://healpix.jpl.nasa.gov/)
pyramids stored as [Zarr](https://zarr.dev/) on local disk or S3.

## Features

- **Automatic grid detection.** Regular (ERA5, CMIP), curvilinear and
  rotated-pole (NEMO, CORDEX, WRF) and unstructured (ICON) grids are
  recognised from the dataset itself.
- **Conservative and nearest-neighbour remapping** with
  [ESMF](https://earthsystemmodeling.org/regrid/) weights, on the sphere
  or the WGS84 ellipsoid.
- **A shared weights store.** Weights are computed once per source grid,
  HEALPix level and method, and reused by everyone pointing at the same
  directory.
- **Point binning** for swaths, stations and trajectories, where every
  sample has its own position.
- **Pyramids** from the finest level down to level 0, with mean or mode
  coarsening and optional valid fractions for masked fields.
- **Writing Zarr stores** in one call, or split into independent tasks
  for HPC array jobs.
- **Region selection** by bounding box, cone or cell list, also for
  stores without coordinate arrays.
- **CF metadata.** Outputs carry the CF `healpix` grid mapping and a
  record of how every level was built.
- **Lazy and parallel.** Regridding and coarsening run on dask arrays.

## Quick example

```python
import xarray as xr

import grid_doctor as gd

ds = xr.open_mfdataset("data/*.nc")
pyramid = gd.create_healpix_pyramid(ds)  # finest level from the grid resolution
gd.save_pyramid(pyramid, "/path/to/output", mode="w")
```

`create_healpix_pyramid` computes the remapping weights on the first run
and takes them from the [weights store](api/weights.md) afterwards.

## Where to go next

- [Getting started](getting-started.md): installation, writing to S3 and
  logging.
- [Recipes](recipes/index.md): worked examples for each kind of source
  data.
- [Technical decisions](technical-decisions.md): why HEALPix, and how
  remapping, binning, coarsening and the stored metadata work.
- [API reference](api/index.md): every public function and class.
