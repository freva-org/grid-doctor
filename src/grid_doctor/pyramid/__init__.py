"""Build HEALPix pyramids from a finest-level HEALPix field.

- ``_build``: [`create_healpix_pyramid`][grid_doctor.create_healpix_pyramid]
  and [`coarsen_healpix`][grid_doctor.coarsen_healpix].
- ``_kernels``: pure NumPy reductions over nested child cells
  (no xarray, no I/O).
- ``_blocks``: the same reductions applied block-wise along ``cell`` for
  NumPy or dask arrays, including exact chained means via
  [`MeanPartials`][grid_doctor.pyramid._blocks.MeanPartials].
- ``_levels``: the xarray layer that turns those arrays into coarse
  HEALPix datasets and chains whole pyramids.

The public names are re-exported from ``grid_doctor``.
"""

from ._build import coarsen_healpix, create_healpix_pyramid

__all__ = ["coarsen_healpix", "create_healpix_pyramid"]
