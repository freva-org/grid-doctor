# Unstructured grids (ICON)

This recipe converts ICON model output on a triangular mesh to a HEALPix
pyramid. Unstructured grids are detected automatically; the weights are
computed from the grid file and reused for every variable and time step.

## Weights from the grid file

Remapping from an unstructured grid needs the vertices of every cell,
which ICON keeps in its grid file (`clon_vertices`, `clat_vertices`) and
usually not in the data files. Compute the weights from the grid file:

```python
import xarray as xr

import grid_doctor as gd

# The grid file has the cell centres and vertices, but no data variables.
grid_ds = xr.open_dataset("icon_grid_R02B07.nc")

level = gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
weights_file = gd.cached_weights(grid_ds, level=level, store="/path/to/weights")
```

The weight file is stored in the [weights store](../api/weights.md). Later
runs with the same grid, by you or by anyone using the same store, take
it from there instead of computing it again.

## Conversion

With the weights, the data files are converted like structured grids:

```python
ds = xr.open_mfdataset("icon_output_*.nc")

pyramid = gd.create_healpix_pyramid(ds, max_level=level, weights_path=weights_file)

gd.save_pyramid(
    pyramid,
    "s3://my-bucket/icon/hourly",
    s3_options=gd.get_s3_options("https://s3.example.org", "~/.s3-credentials.json"),
)
```

The cells of the data must be in the order of the grid file. ICON GRIB
files read with cfgrib name the cell dimension `values`; rename it so the
dimension is recognised:

```python
ds = ds.rename_dims({"values": "cell"})
```

## How grid detection works

A dataset is treated as unstructured when

- it has a dimension named `cell`, `ncells`, `ncell` or `nCells`, or
- a data variable has the attribute `CDI_grid_type = "unstructured"`.

The cell vertices are taken from `clon_vertices`/`clat_vertices`,
`lon_vertices`/`lat_vertices`, `lon_bnds`/`lat_bnds`, or the CF `bounds`
variables of the cell-centre coordinates.

## Production runs

For a whole reanalysis or simulation, split the conversion into
independent tasks for an HPC array job, see
[Parallel writes](parallel-writes.md). The repository's
`scripts/icon-dream/` folder holds a complete conversion of the DWD
ICON-DREAM reanalysis with [reflow](https://reflow-hpc.org).
