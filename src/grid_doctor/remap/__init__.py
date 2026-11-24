"""Remap source grids to HEALPix through reusable ESMF weight files.

[`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights]
generates a weight file (in memory with ESMPy, or offline with
``ESMF_RegridWeightGen`` under MPI);
[`apply_weight_file`][grid_doctor.remap.apply_weight_file] applies one to a
dataset; [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix] does
both, reusing weights through the weights store.

Supported sources are regular and curvilinear lon/lat grids, unstructured
polygon meshes such as ICON, and spectral data via an external transform
command.
"""

from __future__ import annotations

from ._apply import apply_weight_file
from ._regrid import (
    compute_healpix_weights,
    regrid_to_healpix,
    regrid_unstructured_to_healpix,
)

__all__ = [
    "apply_weight_file",
    "compute_healpix_weights",
    "regrid_to_healpix",
    "regrid_unstructured_to_healpix",
]
