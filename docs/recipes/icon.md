# Unstructured Grids (ICON)

This recipe converts ICON model output on a triangular mesh to a
HEALPix pyramid.  ICON data is detected automatically and a weights file
for conservative remapping can be generated.

## Key Difference: Use a grid file

Remapping from unstructured grids is expensive.  For ICON
you should **compute and cache weights once**, then reuse them across
all variables and time steps:

```python
from getpass import getuser
import grid_doctor as gd

# The grid file contains clat/clon but no data variables.
grid_ds = gd.cached_open_dataset(["ICON-DREAM-Global_grid.nc"])

# Derive the HEALPix level from the grid resolution.
level =  gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
weights_file = gd.cached_weights(
    grids_ds,
    level=level
    weights_path="/scratch/{user[0]}/{user}/healpix-weights".format(user=getuser())
)
```

The weights are cached as a NetCDF file in `$SCRATCH` (or `/tmp`).
On subsequent runs with the same grid the cached file is loaded
instantly.

## Conversion

Once you have the weights, conversion works just like for structured
grids:

```python
ds = gd.cached_open_dataset(["icon_dream_hourly_*.grb"])

# ICON GRIB files may use "values" instead of "cell" — rename:
ds = ds.rename_dims({"values": "cell"}).chunk({"cell": -1})

pyramid = gd.create_healpix_pyramid(
    ds, max_level=max_level, weights_path=weights_file,
)

gd.save_pyramid_to_s3(
    pyramid,
    "s3://my-bucket/healpix/icon-dream/hourly",
    s3_options=gd.get_s3_options(
        "https://s3.eu-dkrz-3.dkrz.cloud",
        "~/.s3-credentials.json",
    ),
)
```

## Running on Levante with Dask + SLURM

The ICON-DREAM script under `scripts/icon-dream/convert.py`
demonstrates a full production workflow using
[`reflow`](https://reflow-docs.org):

1. **Download** GRIB files from DWD's open-data server in parallel
   using `dask.distributed` with a concurrency-limited semaphore.
2. **Compute** Delaunay weights once from the grid file.
3. **Regrid** to a HEALPix pyramid.
4. **Rechunk** with `ChunkOptimizer` for optimal Zarr layout.
5. **Upload** to S3.

The script auto-detects whether it is running inside a SLURM allocation
(`$SLURM_JOB_ID` is set) or falls back to a local Dask client for
debugging:

```console
# Submit via reflow
python scripts/icon-dream/convert.py submit --s3-bucket my-bucket \
    --variables t_2m tot_prec \
    --freq hourly \
    --time 2020-01 2020-12 \
    -vv

# Or run locally for testing
python scripts/icon-dream/convert.py --s3-bucket my-bucket \
    --variables t_2m --time 2024-01 2024-01 -vvv
```

## How Grid Detection Works

grid-doctor recognises unstructured grids when any of these conditions
are met:

- A dimension named `cell`, `ncells`, `ncell`, or `nCells` exists.
- A variable has the attribute `CDI_grid_type = "unstructured"`.

The coordinate arrays `clat` / `clon` (or `lat` / `lon`) on the cell
dimension are used directly as the point cloud — no meshgrid or 2-D reshaping
is needed.
