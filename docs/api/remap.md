# Remapping

Remap gridded data to HEALPix with ESMF weights (`grid_doctor.remap`).
[`regrid_to_healpix`][grid_doctor.regrid_to_healpix] does everything in
one call: it takes the weights from the [weights store](weights.md), or
computes them, and applies them.  The other two functions split these
steps.

## Remapping

::: grid_doctor.regrid_to_healpix

## Weight files

[`cached_weights`][grid_doctor.cached_weights] returns the weights for a
grid from the [weights store](weights.md).
[`compute_healpix_weights`][grid_doctor.compute_healpix_weights] writes
them to a file of your choice.

::: grid_doctor.compute_healpix_weights

## Applying weights

::: grid_doctor.apply_weight_file
