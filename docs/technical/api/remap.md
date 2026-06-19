# Regridding

This section covers the core remapping functionality of `grid_doctor`.
The functions below are the main entry points for generating reusable
weight files and transforming data onto HEALPix grids.

## Weight file generation

These functions prepare reusable remapping weights so that later
transformations can be applied efficiently without recomputing the full
mapping each time.

::: grid_doctor.utils.cached_weights
    options:
      show_root_heading: true

::: grid_doctor.compute_healpix_weights
    options:
      show_root_heading: true
      show_root_full_path: false

## Applying remapping

The following functions use either direct remapping logic or precomputed
weights to move data onto the target HEALPix representation.

::: grid_doctor.regrid_to_healpix
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.apply_weight_file
    options:
      show_root_heading: true
      show_root_full_path: false

## Pyramid post-processing

After remapping, these helpers can be used to derive coarser
representations for building multiresolution HEALPix pyramids.

::: grid_doctor.coarsen_healpix
    options:
      show_root_heading: true
      show_root_full_path: false
