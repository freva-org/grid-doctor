"""Tests for the ``gd`` xarray accessors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import xarray as xr

import grid_doctor as gd
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
