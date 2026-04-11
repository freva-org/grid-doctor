"""Tests for the weight-generation + application workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap, remap_backend
from grid_doctor.remap import (
    apply_weight_file,
    compute_healpix_weights,
)

from .helpers import _FakeHealpixModule

# ===================================================================
# Fake ESMPy objects for in-memory weight generation
# ===================================================================


class _FakeMesh:
    def __init__(self, **_: Any) -> None:
        self.element_count = 0
        self.node_count = 0

    def add_nodes(
        self,
        *,
        node_count: int,
        node_ids: np.ndarray,
        node_coords: np.ndarray,
        node_owners: np.ndarray,
    ) -> None:
        self.node_count = node_count
        assert node_ids.shape[0] == node_count
        assert node_coords.shape[0] == 2 * node_count
        assert node_owners.shape[0] == node_count

    def add_elements(
        self,
        *,
        element_count: int,
        element_ids: np.ndarray,
        element_types: np.ndarray,
        element_conn: np.ndarray,
        element_coords: np.ndarray,
    ) -> None:
        self.element_count = element_count
        assert element_ids.shape[0] == element_count
        assert element_types.shape[0] == element_count
        assert element_coords.shape[0] == 2 * element_count
        assert element_conn.size >= 3 * element_count


class _FakeField:
    def __init__(
        self, mesh: _FakeMesh, name: str, meshloc: object
    ) -> None:
        self.mesh = mesh
        self.name = name
        self.meshloc = meshloc
        self.data = np.zeros(mesh.element_count, dtype=np.float64)


class _FakeRegrid:
    last_kwargs: dict[str, Any] | None = None

    def __init__(
        self,
        src_field: _FakeField,
        dst_field: _FakeField,
        **kwargs: Any,
    ) -> None:
        self.src_field = src_field
        self.dst_field = dst_field
        self.kwargs = kwargs
        _FakeRegrid.last_kwargs = kwargs
        filename = Path(kwargs["filename"])
        n_target = dst_field.data.size
        ds = xr.Dataset(
            {
                "row": (
                    "nnz",
                    np.arange(1, n_target + 1, dtype=np.int32),
                ),
                "col": ("nnz", np.ones(n_target, dtype=np.int32)),
                "S": ("nnz", np.ones(n_target, dtype=np.float64)),
            }
        )
        ds.to_netcdf(filename)

    def destroy(self) -> None:
        return None


class _FakeESMPy:
    Mesh = _FakeMesh
    Field = _FakeField
    Regrid = _FakeRegrid
    MeshElemType = SimpleNamespace(TRI=3, QUAD=4)
    CoordSys = SimpleNamespace(SPH_DEG="SPH_DEG")
    MeshLoc = SimpleNamespace(ELEMENT="ELEMENT")
    RegridMethod = SimpleNamespace(
        NEAREST_STOD="nearest", CONSERVE="conserve"
    )
    UnmappedAction = SimpleNamespace(IGNORE="ignore", ERROR="error")
    NormType = SimpleNamespace(DSTAREA="dstarea")
    LineType = SimpleNamespace(GREAT_CIRCLE="great_circle")


# ===================================================================
# Weight generation tests
# ===================================================================


class TestWeightGeneration:
    def test_compute_healpix_weights_nearest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            remap_backend, "_require_esmpy", lambda: _FakeESMPy()
        )
        monkeypatch.setattr(
            remap_backend,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        monkeypatch.setattr(
            remap,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        path = tmp_path / "weights.nc"
        result = compute_healpix_weights(
            regular_ds, 1, method="nearest", weights_path=path
        )
        assert result == path
        with xr.open_dataset(path) as ds:
            assert ds.attrs["grid_doctor_method"] == "nearest"
            assert ds.attrs["grid_doctor_level"] == 1
        assert _FakeRegrid.last_kwargs is not None
        assert (
            _FakeRegrid.last_kwargs["regrid_method"]
            == _FakeESMPy.RegridMethod.NEAREST_STOD
        )

    def test_compute_healpix_weights_conservative(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            remap_backend, "_require_esmpy", lambda: _FakeESMPy()
        )
        monkeypatch.setattr(
            remap_backend,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        monkeypatch.setattr(
            remap,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        path = tmp_path / "weights.nc"
        compute_healpix_weights(
            regular_ds, 1, method="conservative", weights_path=path
        )
        assert _FakeRegrid.last_kwargs is not None
        assert (
            _FakeRegrid.last_kwargs["regrid_method"]
            == _FakeESMPy.RegridMethod.CONSERVE
        )
        assert (
            _FakeRegrid.last_kwargs["line_type"]
            == _FakeESMPy.LineType.GREAT_CIRCLE
        )

    def test_limited_area_ignores_unmapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        limited_area_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            remap_backend, "_require_esmpy", lambda: _FakeESMPy()
        )
        monkeypatch.setattr(
            remap_backend,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        monkeypatch.setattr(
            remap,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        path = tmp_path / "weights.nc"
        compute_healpix_weights(
            limited_area_ds,
            1,
            method="conservative",
            weights_path=path,
        )
        assert _FakeRegrid.last_kwargs is not None
        assert (
            _FakeRegrid.last_kwargs["unmapped_action"]
            == _FakeESMPy.UnmappedAction.IGNORE
        )


# ===================================================================
# End-to-end weight application with grid argument
# ===================================================================


class TestApplyWeightFileWithGrid:
    def test_with_grid_argument(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[1.0, 2.0, 3.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1, 2]},
                )
            }
        )
        grid = xr.Dataset(coords={"cell": [0, 1, 2]})

        weights = xr.Dataset(
            {
                "row": (
                    "nnz",
                    np.array([1, 2, 2], dtype=np.int32),
                ),
                "col": (
                    "nnz",
                    np.array([1, 2, 3], dtype=np.int32),
                ),
                "S": (
                    "nnz",
                    np.array([1.0, 0.25, 0.75], dtype=np.float64),
                ),
            },
            attrs={
                "grid_doctor_level": 1,
                "grid_doctor_order": "nested",
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        monkeypatch.setattr(
            remap,
            "_source_centre_arrays",
            lambda obj, source_units="auto": (
                np.array([0.0, 1.0, 2.0], dtype=np.float64),
                np.array([10.0, 20.0, 30.0], dtype=np.float64),
                ("cell",),
            ),
        )
        monkeypatch.setattr(
            remap,
            "_healpix_centres",
            lambda level, nest: (
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
        )

        result = apply_weight_file(ds, path, grid=grid)
        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, [1.0, 2.75])
