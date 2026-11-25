# Pyramids

Build multi-resolution HEALPix pyramids (`grid_doctor.pyramid`).
[`create_healpix_pyramid`][grid_doctor.create_healpix_pyramid] remaps the
finest level and coarsens it down to the coarsest;
[`coarsen_healpix`][grid_doctor.coarsen_healpix] coarsens any HEALPix
dataset, for example binned point data.  How coarsening treats missing
values is described in the
[technical decisions](../technical-decisions.md#pyramid-construction-and-coarsening).

## Pyramid creation

::: grid_doctor.create_healpix_pyramid

## Coarsening

::: grid_doctor.coarsen_healpix

## Resolution helpers

Choose the HEALPix level that matches the resolution of a source grid.

::: grid_doctor.get_latlon_resolution

::: grid_doctor.resolution_to_healpix_level
