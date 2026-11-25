# Point binning

Map *point-sampled* data (satellite Level-2 swaths, station records,
trajectories) onto HEALPix (`grid_doctor.swath`).  Point data cannot use
the ESMF weight path: every granule has its own geometry, so weights
could never be reused, and nearest-neighbour remapping would spread a
narrow swath over the entire globe.  Binning assigns each sample to the
HEALPix cell that contains it and reduces the samples per cell instead.

See the [point data recipe](../recipes/point-data.md) for an end-to-end
workflow and the
[technical decisions](../technical-decisions.md#point-binning-swath-and-station-data)
for the design rationale.

## Binning

::: grid_doctor.bin_to_healpix

## Sparse intermediates

At high HEALPix levels a single granule touches only a tiny fraction of
the global grid.  ``bin_to_healpix(..., dense=False)`` returns a compact
per-granule dataset that can be accumulated cheaply and expanded once
before publishing.

::: grid_doctor.sparse_to_dense
