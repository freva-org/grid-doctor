# Point Binning API

This section contains the functionality for mapping *point-sampled* data
— satellite Level-2 swaths, station records, trajectories — onto the
standard grid-doctor HEALPix representation.  Point data cannot use the
ESMF weight path: every granule has unique geometry (no weight reuse) and
nearest source-to-destination would smear a narrow swath over the entire
globe.  Binning assigns each sample to its containing HEALPix cell and
reduces per cell instead.

See the [point data recipe](../recipes/point-data.md) for an end-to-end
workflow and the shared
[technical decisions](../technical-decisions.md) document for the design
rationale.

## Binning

::: grid_doctor.bin_to_healpix
    options:
      show_root_heading: true
      show_root_full_path: false

## Sparse intermediates

At high HEALPix levels a single granule touches only a tiny fraction of
the global grid.  ``bin_to_healpix(..., dense=False)`` returns a compact
per-granule dataset that can be accumulated cheaply and expanded once
before publishing.

::: grid_doctor.sparse_to_dense
    options:
      show_root_heading: true
      show_root_full_path: false
