# Appendix: Remapping methodology and benchmark

This page collects the more technical details behind the Waterpark conversion
pipeline. It is intended as background material for people who want to understand
how the HEALPix/Zarr pyramids are produced.

## Remapping methodology

All datasets are remapped using **pre-computed, reusable ESMF weight files**
applied through sparse matrix multiplication.

Weight generation is the expensive part of the process. Once the weights exist
for a given source grid and HEALPix target level, they can be reused for other
variables or datasets on the same grid.

```text
source grid + target HEALPix level
        ↓
ESMF weight generation
        ↓
sparse weight matrix
        ↓
batched sparse matrix multiplication
        ↓
HEALPix data
```

## Variable-dependent remapping

Waterpark uses different methods for continuous and categorical fields.

| Variable type | Method | Rationale |
|---|---|---|
| Continuous fields, such as SST, temperature, wind, radiation, or precipitation | **Conservative** | Area-weighted averaging preserves integrals and handles sub-pixel variability correctly. |
| Discrete or categorical fields, such as land-sea masks, land cover, or soil type | **Nearest neighbour** | Preserves class labels without creating non-physical intermediate values. |

The target HEALPix level is chosen from the source resolution. Lower pyramid
levels are derived by hierarchical coarsening, not by repeated remapping.

## Technical details

- Weight generation uses ESMF. Large grids can use `ESMF_RegridWeightGen` under
  MPI, while moderate grids can use ESMPy in memory.
- HEALPix geometry is computed on a perfect sphere, consistent with ESMF's
  spherical overlap calculations and the geocentric coordinates used by many
  climate model grids.
- Weight application supports several backends:
  - **cuSPARSE / GPU** via CuPy,
  - **Numba** fused CSR kernels,
  - **batched SciPy** sparse matrix multiplication.
- The tooling is provided by the
  [`grid-doctor`](https://github.com/freva-org/grid-doctor) Python package.

## Why cache weight files?

The same source grid often appears many times:

- multiple variables,
- multiple years,
- multiple ensemble members,
- multiple experiments using the same model grid.

Caching weights by a grid signature avoids repeating the expensive ESMF step.
The first dataset on a grid pays the weight-generation cost; later datasets can
reuse the existing weight file.

## Pipeline shape

For large archives, Waterpark needs more than a single script. A typical archive
conversion is a workflow:

```text
gather_sources
    ↓
create_weights
    ↓
plan_regrid
    ↓
regrid_file
    ↓
group_for_upload
    ↓
combine_and_upload
```

In practice, the regridding step can fan out to many array jobs, while weight
creation and upload planning are singleton tasks. This makes the conversion
pipeline restartable and scalable on HPC systems.

## Benchmark: conservative remapping of HEALPix level 10 on NVIDIA GH200

The example below outlines the full pipeline for conservative remapping on a
4x NVIDIA Grace Hopper 200 Superchip node with 288 CPUs, 856 GB RAM, and
4 GPUs with 382 GB GPU memory.

The dataset used for this benchmark was one year of MODIS-AQUA at roughly
2 km source resolution, remapped to HEALPix level 10.

```python
from dask.diagnostics.progress import ProgressBar
from getpass import getuser
from pathlib import Path

import grid_doctor as gd

# --- 1. Open source data ---
dset = gd.cached_open_dataset(
    Path("/pool/data/ICDC/ocean/modis_aqua_sst/DATA/daily/2025").rglob("*.nc"),
    chunks={"lat": -1},
)

# --- 2. Generate reusable weights (one-time) ---
weights_dir = Path(
    "/scratch/{u[0]}/{u}/healpix-weights".format(u=getuser())
)
resolved_level = gd.resolution_to_healpix_level(
    gd.get_latlon_resolution(dset)
)

%time weights_file = gd.cached_weights(
    dset,
    level=resolved_level,
    cache_path=weights_dir,
    nproc=64,
    prefer_offline=True,
)
# Wall time: ~2 min 30 s (Grace CPUs, 64 MPI ranks)

# --- 3. Build multi-resolution pyramid ---
pyramid = gd.create_healpix_pyramid(
    dset,
    max_level=resolved_level,
    weights_path=weights_file,
    backend="cupy",
)

# --- 4. Verify ---
with ProgressBar():
    hp = pyramid[resolved_level].isel(time=slice(0, 10)).load()
    # Wall time: ~13 s (Hopper GPU)

# --- 5. Write to S3 ---
s3_options = gd.get_s3_options(
    "https://s3-example.org",
    Path("~/.s3-credentials.json").expanduser(),
)

%time gd.save_pyramid(
    pyramid,
    "/icon-dream/healpix/icdc/modis/aqua",
    s3_options,
    mode="w",
)
# Wall time: ~1 h 25 min (Hopper GPU)
```

## Visual comparison

![MODIS SST on HEALPix](assets/gridlook-healpix.png){ width="600" }
/// caption
HEALPix: remapped to HEALPix level 10.
///

![MODIS SST on original grid](assets/gridlook-original.png){ width="600" }
/// caption
   Original: regular latitude/longitude grid at about 2.6 km resolution.
///

## Interpreting the benchmark

The benchmark separates three different costs:

| Step | What it measures |
|---|---|
| Weight generation | One-time geometric overlap calculation for a source grid and target level. |
| Pyramid creation / verification | Sparse matrix application and lazy array execution. |
| S3 upload | End-to-end writing of the resulting Zarr pyramid to object storage. |

The main operational lesson is that weight generation should be cached and
shared, while pyramid generation and upload should be driven by a restartable
workflow system.
