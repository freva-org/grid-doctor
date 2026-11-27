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
from typing import TYPE_CHECKING, Literal, Unpack

import xarray as xr

if TYPE_CHECKING:
    from .weights import SourceDescription

from .pyramid import create_healpix_pyramid
from .remap import apply_weight_file, compute_healpix_weights, regrid_to_healpix
from .types import (
    ApplyBackend,
    CoarsenMode,
    Ellipsoid,
    MissingPolicy,
    RegridOptions,
    RemapMethod,
    SourceKind,
    SourceUnits,
    ValidFraction,
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
        return regrid_to_healpix(
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

    def create_healpix_pyramid(
        self,
        max_level: int | None = None,
        min_level: int = 0,
        *,
        coarsen_mode: CoarsenMode = "auto",
        min_valid_fraction: float = 0.5,
        valid_fraction: ValidFraction = False,
        **kwargs: Unpack[RegridOptions],
    ) -> dict[int, xr.Dataset]:
        """Create a multi-resolution HEALPix pyramid.

        The dataset is first remapped to *max_level* with
            [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].
        For nested output ordering, lower levels are derived efficiently
        with
            [`coarsen_healpix`][grid_doctor.coarsen_healpix].
        For ring ordering, each lower level is regenerated directly from
            the source dataset.

        For dask-backed input the result is lazy and chunked along ``cell``
            (``cell_chunks``, forwarded to
            [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix]); coarser
            levels keep that chunk size.  Every level builds on the finest one,
            so write the pyramid with
            [`save_pyramid`][grid_doctor.save_pyramid], which computes
            all levels in one pass.  Calling ``.compute()`` per level instead
            regrids the source once per level.

        Parameters
            ----------
        max_level:
            Finest HEALPix level.
        min_level:
            Coarsest HEALPix level to keep.
        coarsen_mode:
            Coarsening strategy for building lower pyramid levels.
            ``"mean"`` for continuous data, ``"mode"`` for categorical,
            ``"auto"`` to infer from the remapping method.
        min_valid_fraction:
            Minimum fraction of valid children for a parent cell to be
            valid.  Default ``0.5``.
        valid_fraction:
            Store ``<name>_valid_fraction`` next to the selected variables on
        every level: the fraction of valid finest-level cells under each
            cell, as float32, linked via the CF ``ancillary_variables``
            attribute.  Cell values are means over their valid area, so
            global or regional means over a coarse level need these weights,
            e.g. ``ds.sst.weighted(ds.sst_valid_fraction.fillna(0)).mean("cell")``.

            ``False`` (default) stores nothing.  ``True`` gives every cell
            variable a fraction of its full shape (correct for masks that
            change over time or height).  ``"static"`` stores one ``cell``-only
            fraction from the first slice along all other dimensions -- small,
            but only correct when the mask never changes (e.g. land/sea).
            A list of names selects variables (full shape); a mapping such as
            ``{"sst": "static", "ice": True}`` sets the shape per variable.
            Requires nested ordering.
        **kwargs:
            Forwarded to
            [`regrid_to_healpix`][grid_doctor.remap.regrid_to_healpix].

        Returns
            -------
            dict[int, xarray.Dataset]
            Pyramid keyed by level.
        """
        return create_healpix_pyramid(
            self._ds,
            max_level=max_level,
            min_level=min_level,
            coarsen_mode=coarsen_mode,
            min_valid_fraction=min_valid_fraction,
            valid_fraction=valid_fraction,
            **kwargs,
        )

    def compute_healpix_weights(
        self,
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
        return compute_healpix_weights(
            self._ds,
            level,
            method=method,
            nest=nest,
            source_units=source_units,
            weights_path=weights_path,
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
            ellipsoid=ellipsoid,
            source_description=source_description,
        )

    def apply_weight_file(
        self,
        weights_path: str | Path,
        *,
        missing_policy: MissingPolicy = "renormalize",
        grid: xr.Dataset | None = None,
        source_dims: tuple[str, ...] | None = None,
        source_units: SourceUnits = "auto",
        backend: ApplyBackend = "auto",
        cell_chunks: int | Literal["auto"] | None = "auto",
        ellipsoid: Ellipsoid | None = None,
        nest: bool | None = None,
    ) -> xr.Dataset:
        """Apply a previously generated ESMF weight file.

            Uses the batched engine in
        `grid_doctor.remap._sparse` which replaces per-slice
        ``vectorize=True`` with a single batched sparse matmul for a
        typical 10–50× speedup.

        Parameters
        ----------
        weights_path:
            Path to the NetCDF weight file.
        missing_policy:
            Strategy for handling NaN (missing) values in the source
            grid during weight application.

            ``"renormalize"`` (default)
                Source cells that are NaN are excluded from the
                weighted sum and the weights of the remaining valid
                source cells are rescaled so they sum to 1.  A target
                cell receives a valid value as long as **at least one**
                contributing source cell is valid.  This is the right
                choice for most climate fields: for instance, a
                HEALPix cell on a coastline that partly overlaps land
                (NaN) and partly ocean still receives a valid SST from
                the ocean fraction.

            ``"propagate"``
                If **any** contributing source cell is NaN, the target
                cell is set to NaN: no partial averages.  Use this
                when completeness of the source coverage matters more
                than spatial coverage of the output, for example when
                computing area-integrated flux budgets where a partial
                average would bias the integral.  Note that this is
                very aggressive: a single NaN source cell at the edge
                of a target cell's footprint is enough to discard it.
                For conservative remapping where a HEALPix cell can
                overlap dozens of source cells, large portions of the
                output along coastlines, swath edges, and orbit gaps
                will be NaN.
        grid:
            Optional grid dataset with source geometry.
        source_dims:
            Optional explicit source dimensions.
        source_units:
            Unit convention for geometry-based inference.
        backend:
            Application backend (``"auto"``, ``"scipy"``, ``"numba"``).

        cell_chunks:
            Chunk size of the output along ``cell`` for dask-backed input.
            Each output chunk is computed from its own slice of weight-matrix
            rows, so no task holds the whole HEALPix field.  ``"auto"`` uses
            ``DEFAULT_CELL_CHUNK`` cells (rounded to a multiple of 4), ``None``
            produces a single chunk.  NumPy-backed input is always computed
            in one piece.
        ellipsoid:
            Ellipsoid the weights must have been generated on.  ``None``
            (default) takes the one recorded in the weight file (a sphere
            for files that record none); the HEALPix cell centres are
            computed on it.
        nest:
            HEALPix order of the output: nested when ``True``, ring when
            ``False``. ``None`` (default) keeps the order of the weight file.
            Stored weight files are nested; their rows are renumbered for
            ring output, so one file serves both orders.

        Returns
        -------
        xarray.Dataset
            Dataset on the HEALPix target grid with a ``cell`` dimension.

        Examples
        --------
        ```python
        ds_hp = ds_icon.gd.apply_weight_file("icon_to_healpix_d8.nc")
        ```
        """
        return apply_weight_file(
            self._ds,
            weights_path=weights_path,
            missing_policy=missing_policy,
            grid=grid,
            source_dims=source_dims,
            source_units=source_units,
            backend=backend,
            cell_chunks=cell_chunks,
            ellipsoid=ellipsoid,
            nest=nest,
        )


@xr.register_datatree_accessor(ACCESSOR_NAME)
class DataTreeAccessor:
    """grid-doctor methods on an ``xarray.DataTree`` pyramid (``dt.gd``)."""

    def __init__(self, dt: xr.DataTree) -> None:  # noqa: D107
        self._dt = dt


__all__ = ["ACCESSOR_NAME", "DataTreeAccessor", "DatasetAccessor"]
