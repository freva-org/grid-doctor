"""HEALPix pyramid construction (internal, work in progress).

This subpackage is the intended home for everything that turns a
finest-level HEALPix field into a multi-resolution pyramid:

- ``_kernels``: pure NumPy reductions over nested child cells
  (no xarray, no I/O).
- ``_blocks``: the same reductions applied block-wise along ``cell`` for
  NumPy or dask arrays, including exact chained means via
  [`MeanPartials`][grid_doctor.pyramid.MeanPartials].
- ``_levels``: the xarray layer that turns those arrays into coarse
  HEALPix datasets and chains whole pyramids.

The public entry points (``coarsen_healpix``,
``create_healpix_pyramid``) still live in
[`grid_doctor.helpers`][grid_doctor.helpers] and delegate here.  Public
names stay importable from ``grid_doctor`` throughout.
"""

from ._blocks import (
    MeanPartials,
    coarsen_counts,
    coarsen_mean_steps,
    coarsen_mode_steps,
    count_valid_steps,
    map_cells,
    rechunk_cells,
)
from ._kernels import coarsen_mean, min_valid_count
from ._levels import (
    assemble_coarse_level,
    coarse_levels,
    coarsen_dataset,
    resolve_coarsen_mode,
    resolve_valid_fraction,
    with_finest_fractions,
)

__all__ = [
    "assemble_coarse_level",
    "coarse_levels",
    "coarsen_counts",
    "coarsen_dataset",
    "coarsen_mean",
    "coarsen_mean_steps",
    "coarsen_mode_steps",
    "count_valid_steps",
    "map_cells",
    "MeanPartials",
    "min_valid_count",
    "rechunk_cells",
    "resolve_coarsen_mode",
    "resolve_valid_fraction",
    "with_finest_fractions",
]
