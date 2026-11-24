"""High-level remapping entry points: compute weights, regrid a dataset.

[`compute_healpix_weights`][grid_doctor.remap.compute_healpix_weights]
writes a reusable ESMF weight file;
[`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix] computes or
reuses one and applies it.  Two methods are supported:

- ``"nearest"`` — source-to-destination nearest-neighbour mapping.
- ``"conservative"`` — first-order area-weighted conservative remapping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import xarray as xr

if TYPE_CHECKING:
    from ..weights import SourceDescription

from .._geometry.coords import get_latlon_resolution
from .._geometry.healpix import (
    _ellipsoid_or_config,
    resolution_to_healpix_level,
)
from ..types import (
    ApplyBackend,
    Ellipsoid,
    MissingPolicy,
    RemapMethod,
    SourceKind,
    SourceUnits,
    WeightOptions,
)
from ..weights import cached_weights
from ._apply import apply_weight_file
from ._esmf import (
    OfflineWeightConfig,
    compute_healpix_weights_backend,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Weight generation facade
# ===================================================================


def compute_healpix_weights(
    ds: xr.Dataset | str | Path,
    level: int,
    *,
    method: RemapMethod = "conservative",
    nest: bool = True,
    source_units: SourceUnits = "auto",
    weights_path: str | Path | None = None,
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    ignore_unmapped: bool | None = None,
    large_file: bool = True,
    prefer_offline: bool | None = None,
    nproc: int = 1,
    esmf_regrid_weightgen: str = "ESMF_RegridWeightGen",
    keep_intermediates: bool = False,
    workdir: str | Path | None = None,
    spectral_transform_command: list[str] | tuple[str, ...] | None = None,
    ellipsoid: Ellipsoid | None = None,
    source_description: SourceDescription | None = None,
) -> Path:
    """Generate a reusable NetCDF weight file for HEALPix remapping.

    Parameters
    ----------
    ds:
        Source dataset or a source path.
    level:
        HEALPix refinement level.  ``nside = 2**level``.
    method:
        ``"nearest"`` or ``"conservative"``.
    nest:
        Use nested HEALPix ordering when `True`.
    source_units:
        Unit convention of the source coordinates.
    weights_path:
        Output NetCDF file. Defaults to the local weights directory in the
        [`WeightsRef`][grid_doctor.weights.WeightsRef] layout (see
        [`local_weights_dir`][grid_doctor.weights.local_weights_dir]), so
        the same weights always land at the same path.
    grid:
        Optional external geometry dataset.
    source_kind:
        Explicit source representation or ``"auto"``.
    ignore_unmapped:
        Ignore unmapped destination cells.
    large_file:
        Forwarded to the in-memory ESMPy workflow.
    prefer_offline:
        Force or disable the offline ESMF path.
    nproc:
        Number of MPI ranks for the offline path.
    esmf_regrid_weightgen:
        Name or path of the offline ESMF executable.
    keep_intermediates:
        Keep intermediate mesh files.
    workdir:
        Working directory for offline intermediate files.
    spectral_transform_command:
        External command for ``source_kind="spectral"``.
    ellipsoid:
        Ellipsoid of the HEALPix geometry, ``"sphere"`` or
        ``"WGS84"``.  Defaults to ``grid_doctor.config.ellipsoid``.
    source_description:
        Description of *ds* from
        [`describe_source`][grid_doctor.weights.describe_source],
        e.g. when the caller already built one to look the weights up;
        skips describing the source again.

    Returns
    -------
    pathlib.Path
        Path to the generated NetCDF weight file.

    See Also
    --------
    [`apply_weight_file`][grid_doctor.remap.apply_weight_file]:
        Apply a previously generated weight file.
    """
    offline = OfflineWeightConfig(
        enabled=bool(prefer_offline) if prefer_offline is not None else False,
        nproc=nproc,
        esmf_regrid_weightgen=esmf_regrid_weightgen,
        keep_intermediates=keep_intermediates,
        workdir=Path(workdir) if workdir is not None else None,
    )
    return compute_healpix_weights_backend(
        ds,
        level,
        method=method,
        nest=nest,
        source_units=source_units,
        weights_path=weights_path,
        grid=grid,
        source_kind=source_kind,
        ignore_unmapped=ignore_unmapped,
        large_file=large_file,
        offline=offline,
        spectral_transform_command=spectral_transform_command,
        ellipsoid=_ellipsoid_or_config(ellipsoid),
        source_description=source_description,
    )


# ===================================================================
# High-level remapping entry points
# ===================================================================


def regrid_to_healpix(
    ds: xr.Dataset,
    level: int | None = None,
    *,
    nest: bool = True,
    method: RemapMethod = "conservative",
    source_units: SourceUnits = "auto",
    weights_path: str | Path | None = None,
    missing_policy: MissingPolicy = "renormalize",
    backend: ApplyBackend = "auto",
    grid: xr.Dataset | None = None,
    source_kind: SourceKind = "auto",
    ignore_unmapped: bool | None = None,
    large_file: bool = True,
    prefer_offline: bool | None = None,
    nproc: int = 1,
    esmf_regrid_weightgen: str = "ESMF_RegridWeightGen",
    keep_intermediates: bool = False,
    workdir: str | Path | None = None,
    spectral_transform_command: list[str] | tuple[str, ...] | None = None,
    cell_chunks: int | Literal["auto"] | None = "auto",
    ellipsoid: Ellipsoid | None = None,
) -> xr.Dataset:
    """Regrid *ds* to a HEALPix target grid.

    Parameters
    ----------
    ds:
        Source dataset on a regular, curvilinear, or unstructured grid.
    level:
        HEALPix refinement level. Derived from the grid resolution when
        omitted.
    nest:
        Use nested HEALPix ordering when ``True``.
    method:
        ``"nearest"`` or ``"conservative"``.
    source_units:
        Coordinate units for the source grid.
    weights_path:
        Optional existing or target weight file. When omitted, the
        weights come from the default
        [`WeightsStore`][grid_doctor.weights.WeightsStore] (see
        [`cached_weights`][grid_doctor.weights.cached_weights]): computed
        once, then reused.
    missing_policy:
        How NaN values in the source grid are treated.
        ``"renormalize"`` (default) excludes NaN source cells and
        rescales the remaining weights so that a target cell is valid
        whenever at least one source cell contributes — appropriate
        for most continuous fields.  ``"propagate"`` sets a target
        cell to NaN if any contributing source cell is NaN — useful
        for strict budget-closure checks but very aggressive along
        coastlines and data edges.  See
        [`apply_weight_file`][grid_doctor.remap.apply_weight_file]
        for a detailed description.
    backend:
        Application backend (``"auto"``, ``"scipy"``, ``"numba"``).
    grid:
        Optional external geometry dataset.
    source_kind:
        Explicit source representation or ``"auto"``.
    ignore_unmapped:
        Ignore unmapped destination cells.
    large_file:
        Forwarded to the in-memory ESMPy workflow.
    prefer_offline:
        Force or disable the offline ESMF path.
    nproc:
        Number of MPI ranks for the offline path.
    esmf_regrid_weightgen:
        Name or path of the offline ESMF executable.
    keep_intermediates:
        Keep intermediate mesh files.
    workdir:
        Working directory for offline intermediate files.
    spectral_transform_command:
        External command for ``source_kind="spectral"``.
    cell_chunks:
        Output chunk size along ``cell`` for dask-backed input, see
        [`apply_weight_file`][grid_doctor.remap.apply_weight_file].
        Coarser pyramid levels keep the same chunk size.
    ellipsoid:
        Ellipsoid of the HEALPix geometry, ``"sphere"`` or ``"WGS84"``.
        New weights are generated on it, by default on
        ``grid_doctor.config.ellipsoid``.  An existing
        *weights_path* keeps the ellipsoid it was generated on; a
        different explicit value raises ``ValueError``.

    Returns
    -------
    xarray.Dataset
        Regridded dataset on the HEALPix grid.
    """
    options: WeightOptions = {
        "grid": grid,
        "source_kind": source_kind,
        "prefer_offline": prefer_offline,
        "nproc": nproc,
        "esmf_regrid_weightgen": esmf_regrid_weightgen,
        "keep_intermediates": keep_intermediates,
        "workdir": workdir,
        "spectral_transform_command": spectral_transform_command,
        "ignore_unmapped": ignore_unmapped,
        "large_file": large_file,
    }
    if weights_path is None:
        weight_file = cached_weights(
            ds, level, method=method, source_units=source_units, ellipsoid=ellipsoid, **options
        )
    else:
        weight_file = Path(weights_path)
        if not weight_file.exists():
            if level is None:
                level = resolution_to_healpix_level(get_latlon_resolution(ds))
            weight_file = compute_healpix_weights(
                ds,
                level,
                method=method,
                nest=nest,
                source_units=source_units,
                weights_path=weight_file,
                ellipsoid=ellipsoid,
                **options,
            )
    return apply_weight_file(
        ds,
        weight_file,
        missing_policy=missing_policy,
        backend=backend,
        cell_chunks=cell_chunks,
        ellipsoid=ellipsoid,
        nest=nest,
    )


__all__ = [
    "compute_healpix_weights",
    "regrid_to_healpix",
]
