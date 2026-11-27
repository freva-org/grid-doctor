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
from grid_doctor.select import _DEFAULT_QUERY_DELTA
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


def test_create_healpix_pyramid_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    min_level = 1
    ds = xr.Dataset()
    fake_create = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "create_healpix_pyramid", fake_create)

    ds.gd.create_healpix_pyramid(
        max_level=None,
        min_level=min_level,
    )

    # Confirm all the default values are passed through.
    fake_create.assert_called_with(
        ds,
        max_level=None,
        min_level=min_level,
        coarsen_mode="auto",
        min_valid_fraction=0.5,
        valid_fraction=False,
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


def test_apply_weight_file_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    weights_path = "no/where"
    ds = xr.Dataset()
    fake_apply = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "apply_weight_file", fake_apply)

    ds.gd.apply_weight_file(weights_path)

    # Confirm all the default values are passed through.
    fake_apply.assert_called_with(
        ds,
        weights_path=weights_path,
        missing_policy="renormalize",
        grid=None,
        source_dims=None,
        source_units="auto",
        backend="auto",
        cell_chunks="auto",
        ellipsoid=None,
        nest=None,
    )


def test_select_cells_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset()
    fake_select = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "select_cells", fake_select)
    cells = [
        42,
        137,
    ]

    ds.gd.select_cells(cells)

    # Confirm all the default values are passed through.
    fake_select.assert_called_with(ds, cells, level=None, load=True)


def test_select_bbox_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset()
    fake_select = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "select_bbox", fake_select)
    lon = (8.5, 11.3)
    lat = (52.8, 54.2)

    ds.gd.select_bbox(lon=lon, lat=lat)

    # Confirm all the default values are passed through.
    fake_select.assert_called_with(ds, lon=lon, lat=lat, level=None, query_delta=_DEFAULT_QUERY_DELTA, load=True)


def test_select_cone_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset()
    fake_select = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "select_cone", fake_select)
    lon = 10.0
    lat = 53.5
    radius = 0.3

    ds.gd.select_cone(lon=lon, lat=lat, radius=radius)

    # Confirm all the default values are passed through.
    fake_select.assert_called_with(
        ds, lon=lon, lat=lat, radius=radius, level=None, query_delta=_DEFAULT_QUERY_DELTA, load=True
    )


def test_attach_cell_coords_passes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = xr.Dataset()
    fake_attach = mock.MagicMock()
    monkeypatch.setattr(xarray_accessor, "attach_cell_coords", fake_attach)
    cells = [
        42,
        137,
    ]
    level = 7

    ds.gd.attach_cell_coords(cells, level=level)

    # Confirm all the default values are passed through.
    fake_attach.assert_called_with(ds, cells, level=level, attrs=None)
