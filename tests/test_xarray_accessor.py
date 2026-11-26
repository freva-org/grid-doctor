"""Tests for the ``gd`` xarray accessors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import xarray as xr


import grid_doctor as gd
from grid_doctor import xarray_accessor
from grid_doctor.xarray_accessor import DatasetAccessor, DataTreeAccessor


def test_dataset_accessor_is_registered() -> None:
    ds = xr.Dataset({"t": ("cell", [1.0, 2.0])})
    assert isinstance(ds.gd, DatasetAccessor)
    assert ds.gd._ds is ds


def test_datatree_accessor_is_registered() -> None:
    dt = xr.DataTree.from_dict({"/level_0": xr.Dataset({"t": ("cell", [1.0])})})
    assert isinstance(dt.gd, DataTreeAccessor)
    assert dt.gd._dt is dt


def test_import_grid_doctor_registers_accessors() -> None:
    """``import grid_doctor`` alone is enough; no submodule needs importing."""
    src = str(Path(gd.__file__).parents[1])
    code = "import xarray as xr, grid_doctor; print(type(xr.Dataset().gd).__name__)"
    env = {"PYTHONPATH": src, "PATH": ""}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "DatasetAccessor"


def test_regrid_to_healpix_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    level = 1
    ds = xr.Dataset()
    fake_regrid = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "regrid_to_healpix", fake_regrid)

    ds.gd.regrid_to_healpix(level)

    # Confirm all the default values are passed through.
    fake_regrid.assert_called_with(
        ds,
        1,
        nest=True,
        method="conservative",
        source_units="auto",
        weights_path=None,
        missing_policy="renormalize",
        backend="auto",
        grid=None,
        source_kind="auto",
        ignore_unmapped=None,
        large_file=True,
        prefer_offline=None,
        nproc=1,
        esmf_regrid_weightgen="ESMF_RegridWeightGen",
        keep_intermediates=False,
        workdir=None,
        spectral_transform_command=None,
        cell_chunks="auto",
        ellipsoid=None,
    )


def create_healpix_pyramid_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    min_level = 1
    kwargs = {"arg1": "val1", "arg2": "val2"}
    ds = xr.Dataset()
    fake_create = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "regrid_to_healpix", fake_create)

    ds.gd.create_healpix_pyramid(
        max_level=None,
        min_level=min_level,
    )

    # Confirm all the default values are passed through.
    fake_create.assert_called_with(
        ds,
        None,
        min_level,
        coarsen_mode="auto",
        min_valid_fraction=0.5,
        valid_fraction=False,
        kwargs=kwargs,
    )


def test_compute_healpix_weights_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    level = 1
    ds = xr.Dataset()
    fake_compute = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "compute_healpix_weights", fake_compute)

    ds.gd.compute_healpix_weights(level)

    # Confirm all the default values are passed through.
    fake_compute.assert_called_with(
        ds,
        1,
        method="conservative",
        nest=True,
        source_units="auto",
        weights_path=None,
        grid=None,
        source_kind="auto",
        ignore_unmapped=None,
        large_file=True,
        prefer_offline=None,
        nproc=1,
        esmf_regrid_weightgen="ESMF_RegridWeightGen",
        keep_intermediates=False,
        workdir=None,
        spectral_transform_command=None,
        ellipsoid=None,
        source_description=None,
    )
