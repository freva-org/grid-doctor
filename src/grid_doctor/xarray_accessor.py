"""xarray accessors: ``ds.gd`` on datasets and ``dt.gd`` on pyramids.

Importing ``grid_doctor`` registers both, so that grid-doctor's functions
can be called as methods, e.g. ``ds.gd.regrid_to_healpix(level)`` or
``dt.gd.save_pyramid(path)``.  The methods are not there yet: this module
only registers the namespaces.

[`regrid_to_healpix`][grid_doctor.xarray_accessor.DatasetAccessor.regrid_to_healpix] computes or
reuses one and applies it.  Two methods are supported:
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import xarray as xr

from .remap import regrid_to_healpix
from .types import (
    ApplyBackend,
    Ellipsoid,
    MissingPolicy,
    RemapMethod,
    SourceKind,
    SourceUnits,
)

ACCESSOR_NAME = "gd"
"""Attribute under which the accessors are registered."""


@xr.register_dataset_accessor(ACCESSOR_NAME)
class DatasetAccessor:
    """grid-doctor methods on an ``xarray.Dataset`` (``ds.gd``)."""

    def __init__(self, ds: xr.Dataset) -> None:  # noqa: D107
        self._ds = ds

    def regrid_to_healpix(
        self,
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
        """Regrid this dataset to a HEALPix target grid.

        Parameters
        ----------
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
            whenever at least one source cell contributes, which suits
            most continuous fields.  ``"propagate"`` sets a target
            cell to NaN if any contributing source cell is NaN, which is
            useful for strict budget-closure checks but very aggressive along
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
        pathlib.Path
            Path to the generated NetCDF weight file.
        """
        self._ds = regrid_to_healpix(
            self._ds,
            level,
            nest=nest,
            method=method,
            source_units=source_units,
            weights_path=weights_path,
            missing_policy=missing_policy,
            backend=backend,
            grid=grid,
            source_kind=source_kind,
            ignore_unmapped=ignore_unmapped,
            large_file=large_file,
            prefer_offline=prefer_offline,
            nproc=nproc,
            esmf_regrid_weightgen=esmf_regrid_weightgen,
            keep_intermediates=keep_intermediates,
            workdir=workdir,
            spectral_transform_command=spectral_transform_command,
            cell_chunks=cell_chunks,
            ellipsoid=ellipsoid,
        )

        return self._ds


@xr.register_datatree_accessor(ACCESSOR_NAME)
class DataTreeAccessor:
    """grid-doctor methods on an ``xarray.DataTree`` pyramid (``dt.gd``)."""

    def __init__(self, dt: xr.DataTree) -> None:  # noqa: D107
        self._dt = dt


__all__ = ["ACCESSOR_NAME", "DataTreeAccessor", "DatasetAccessor"]
