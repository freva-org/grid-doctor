"""HEALPix pyramid construction (internal, work in progress).

This subpackage is the intended home for everything that turns a
finest-level HEALPix field into a multi-resolution pyramid:

- ``_kernels``: pure NumPy reductions over nested child cells
  (no xarray, no I/O).
- ``_blocks``: the same reductions applied block-wise along ``cell`` for
  NumPy or dask arrays, including exact chained means via
  [`MeanPartials`][grid_doctor.pyramid.MeanPartials].

The xarray-level functions (``coarsen_healpix``,
``create_healpix_pyramid``) still live in
[`grid_doctor.helpers`][grid_doctor.helpers] and will move here in a
later restructuring. Public names stay importable from
``grid_doctor`` throughout.
"""

from ._blocks import (
    MeanPartials,
    coarsen_mean_steps,
    coarsen_mode_steps,
    map_cells,
    rechunk_cells,
)
from ._kernels import coarsen_mean, min_valid_count

__all__ = [
    "MeanPartials",
    "coarsen_mean",
    "coarsen_mean_steps",
    "coarsen_mode_steps",
    "map_cells",
    "min_valid_count",
    "rechunk_cells",
]
