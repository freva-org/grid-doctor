# High Level HEALPix API

This section contains the main functionality for creating and working with
HEALPix-based dataset representations, including pyramid construction and
resolution handling.

## Pyramid creation

These functions build HEALPix pyramids from gridded source data and
provide the main higher-level workflow for preparing multi-resolution
outputs.

::: grid_doctor.create_healpix_pyramid
    options:
      show_root_heading: true
      show_root_full_path: false

## Resolution helpers

These utilities help translate between source grid resolution and the
corresponding HEALPix level used for remapping and pyramid generation.

::: grid_doctor.get_latlon_resolution
    options:
      show_root_heading: true
      show_root_full_path: false

::: grid_doctor.resolution_to_healpix_level
    options:
      show_root_heading: true
      show_root_full_path: false
