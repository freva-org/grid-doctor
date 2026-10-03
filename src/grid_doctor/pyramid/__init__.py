"""HEALPix pyramid construction (internal, work in progress).

This subpackage is the intended home for everything that turns a
finest-level HEALPix field into a multi-resolution pyramid:

- ``_kernels``: pure NumPy reductions over nested child cells
  (no xarray, no I/O).

The xarray-level functions (``coarsen_healpix``,
``create_healpix_pyramid``) still live in
[`grid_doctor.helpers`][grid_doctor.helpers] and will move here in a
later restructuring. Public names stay importable from
``grid_doctor`` throughout.
"""

from ._kernels import coarsen_mean, min_valid_count

__all__ = ["coarsen_mean", "min_valid_count"]
