# Unstructured Grids (ICON)

This recipe converts ICON model output on a triangular mesh to a
HEALPix pyramid.  ICON data is detected automatically — the
`easygems.remap` Delaunay interpolation path is used instead of
`scipy.griddata`.

## Key Difference: Pre-compute Weights

Delaunay triangulation on millions of cells is expensive.  For ICON
you should **compute and cache weights once**, then reuse them across
all variables and time steps:

```python
import grid_doctor as gd

# The grid file contains clat/clon but no data variables.
grid_ds = gd.cached_open_dataset(["ICON-DREAM-Global_grid.nc"])

# Derive the HEALPix level from the grid resolution.
max_level = gd.resolution_to_healpix_level(
    gd.get_latlon_resolution(grid_ds)
)

# Compute (or load from cache) the Delaunay weights.
weights = gd.cached_weights(grid_ds, level=max_level)
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

pyramid = gd.latlon_to_healpix_pyramid(
    ds, max_level=max_level, weights=weights
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
demonstrates a full production workflow:

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
# Submit via SLURM
srun python scripts/icon-dream/convert.py my-bucket \
    --variables t_2m tot_prec \
    --freq hourly \
    --time 2020-01 2020-12 \
    -vv

# Or run locally for testing
python scripts/icon-dream/convert.py my-bucket \
    --variables t_2m --time 2024-01 2024-01 -vvv
```

## How Grid Detection Works

grid-doctor recognises unstructured grids when any of these conditions
are met:

- A dimension named `cell`, `ncells`, `ncell`, or `nCells` exists.
- A variable has the attribute `CDI_grid_type = "unstructured"`.

The coordinate arrays `clat` / `clon` (or `lat` / `lon`) on the cell
dimension are used directly as the point cloud for the Delaunay
triangulation — no meshgrid or 2-D reshaping is needed.
