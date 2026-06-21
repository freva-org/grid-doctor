# Waterpark — HEALPix Data Hub

![MODIS SST on HEALPix level 10](assets/modis-aqua.png){ width="600" .img-center }

/// caption
Sea surface temperature seen by MODIS AQUA, remapped to HEALPix level 10.
///

Waterpark is an effort by the German Climate Computing Centre (DKRZ)
to make climate and Earth observation data easier
to discover, compare, and use. The basic idea is simple:

> Remap many datasets onto one common equal-area grid once, store them as
> multi-resolution Zarr pyramids, and serve them through an S3-compatible
> interface.

This removes a common pain point in climate data analysis: every dataset usually
comes on its own grid. ERA5, CMIP6, ICON, DYAMOND, campaign data, and satellite
products all have different geometries, resolutions, chunking strategies, and
access patterns. Waterpark turns them into a more consistent, analysis-ready
data layer. See the [Available datasets](#currently-available-datasets) section
to get an overview over all datasets that have been added.

## Why Waterpark exists

Cross-dataset analysis often starts with the same repeated chores:

- find the right files,
- understand the source grid,
- regrid the data,
- decide on chunking,
- move the result somewhere usable,
- repeat the same work for the next dataset.

Waterpark tries to move those steps out of individual notebooks and into a
shared data preparation pipeline. Users should be able to pick a dataset,
choose a resolution, and stream only the data they need.

## The three core choices

Waterpark combines three technical decisions:

| Choice | What it gives us | Read more |
|---|---|---|
| **HEALPix** | A global equal-area grid with a natural multi-resolution hierarchy. | [Why HEALPix?](why-healpix.md) |
| **Zarr** | Chunked, compressed, lazy arrays that work well with xarray and dask. | [Why Zarr and S3?](zarr-and-s3.md) |
| **S3-compatible object storage** | HTTP-based access without requiring a mounted filesystem. | [Why Zarr and S3?](zarr-and-s3.md) |

The combination matters more than any single component. HEALPix gives Waterpark
a common geometry, Zarr gives partial array access, and S3 makes those chunks
available from laptops, HPC systems, services, and cloud-like workflows.

## Current status

Waterpark is currently a prototype data hub. The first datasets have already
been converted to HEALPix/Zarr pyramids and are served through an S3-compatible
endpoint.

At the moment, the storage backend is still transitional: data is served from
DKRZ file systems through an S3-compatible gateway. The goal is that the public
interface remains stable while the backend can later move to dedicated object
storage.

## Currently available datasets

<div class="wp" data-waterpark
     data-mode="live"
     data-endpoint="https://s3.waterpark.dkrz.de"
     data-buckets="cmip6,cordex,dyamond,earthcare,eerie,era5,icdc,icon-dream,nextgems,orchestra,palmod"
     data-meta="https://raw.githubusercontent.com/freva-org/grid-doctor/main/docs/assets/waterpark-datasets.json"></div>

## What makes the data useful

Waterpark datasets are intended to be:

- **analysis-ready** — no per-user regridding before analysis,
- **cloud-optimized** — read only the chunks needed for a task,
- **multi-resolution** — use coarse levels for averages and fine levels for
  high fidelity statistics,
- **cross-dataset friendly** — datasets share a common grid and can be compared
  more directly,
- **workflow-ready** — suitable for batch processing, interactive analysis, and
  machine-learning pipelines.

## How the data is produced

The conversion pipeline is built around the
[`grid-doctor`](https://github.com/freva-org/grid-doctor) Python package.
For each source grid, Waterpark generates reusable ESMF weight files and applies
them to create HEALPix levels. The finest level is chosen from the source
resolution, and lower levels are derived by hierarchical coarsening rather than
by repeated remapping.

For implementation details see the [Technical Decisions](technical-decisions.md)
and a benchmark on MODIS-AQUA data
[remapping and benchmark appendix](appendix-remapping-benchmark.md).
