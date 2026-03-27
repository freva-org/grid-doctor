from __future__ import annotations
from types import SimpleNamespace
from typing import Any
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor import remap_backend
from grid_doctor.remap import (
    _flattened_size,
    _guess_source_dims_from_size,
    _parse_source_dims_attr,
    _resolve_source_dims_for_weight_application,
    apply_weight_file,
    compute_healpix_weights,
)
from .helpers import _FakeHealpixModule


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
    def __init__(self, mesh: _FakeMesh, name: str, meshloc: object) -> None:
        self.mesh = mesh
        self.name = name
        self.meshloc = meshloc
        self.data = np.zeros(mesh.element_count, dtype=np.float64)


class _FakeRegrid:
    last_kwargs: dict[str, Any] | None = None

    def __init__(
        self, src_field: _FakeField, dst_field: _FakeField, **kwargs: Any
    ) -> None:
        self.src_field = src_field
        self.dst_field = dst_field
        self.kwargs = kwargs
        _FakeRegrid.last_kwargs = kwargs
        filename = Path(kwargs["filename"])
        n_target = dst_field.data.size
        ds = xr.Dataset(
            {
                "row": ("nnz", np.arange(1, n_target + 1, dtype=np.int32)),
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
    RegridMethod = SimpleNamespace(NEAREST_STOD="nearest", CONSERVE="conserve")
    UnmappedAction = SimpleNamespace(IGNORE="ignore", ERROR="error")
    NormType = SimpleNamespace(DSTAREA="dstarea")
    LineType = SimpleNamespace(GREAT_CIRCLE="great_circle")


class TestFlattenedSize:
    def test_single_dim(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        assert _flattened_size(ds, ("cell",)) == 5

    def test_multiple_dims(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        assert _flattened_size(ds, ("y", "x")) == 12

    def test_missing_dim_raises(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        with pytest.raises(ValueError, match="Missing dimensions"):
            _flattened_size(ds, ("cell", "time"))


class TestParseSourceDimsAttr:
    def test_none_returns_none(self) -> None:
        assert _parse_source_dims_attr(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_source_dims_attr("") is None

    def test_json_string(self) -> None:
        assert _parse_source_dims_attr('["y", "x"]') == ("y", "x")

    def test_plain_string(self) -> None:
        assert _parse_source_dims_attr("cell") == ("cell",)

    def test_comma_separated_string(self) -> None:
        assert _parse_source_dims_attr("y, x") == ("y", "x")

    def test_tuple(self) -> None:
        assert _parse_source_dims_attr(("y", "x")) == ("y", "x")

    def test_list(self) -> None:
        assert _parse_source_dims_attr(["y", "x"]) == ("y", "x")

    def test_invalid_value_returns_none(self) -> None:
        assert _parse_source_dims_attr(123) is None
        assert _parse_source_dims_attr(["y", 1]) is None


class TestGuessSourceDimsFromSize:
    def test_prefers_unstructured_dim(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(10), "member": np.arange(10)})
        assert _guess_source_dims_from_size(ds, 10) == ("cell",)

    def test_unique_non_unstructured_match(self) -> None:
        ds = xr.Dataset(coords={"points": np.arange(7), "time": np.arange(3)})
        assert _guess_source_dims_from_size(ds, 7) == ("points",)

    def test_ambiguous_match_returns_none(self) -> None:
        ds = xr.Dataset(coords={"a": np.arange(8), "b": np.arange(8)})
        assert _guess_source_dims_from_size(ds, 8) is None

    def test_no_match_returns_none(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        assert _guess_source_dims_from_size(ds, 99) is None


class TestResolveSourceDimsForWeightApplication:
    def test_uses_explicit_source_dims(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=5,
            grid=None,
            source_dims=("cell",),
            source_units="auto",
            stored_source_dims=None,
        )
        assert result == ("cell",)

    def test_explicit_source_dims_size_mismatch_raises(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        with pytest.raises(ValueError, match="do not match the weight file geometry"):
            _resolve_source_dims_for_weight_application(
                ds,
                n_source=4,
                grid=None,
                source_dims=("cell",),
                source_units="auto",
                stored_source_dims=None,
            )

    def test_uses_stored_source_dims_when_valid(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=12,
            grid=None,
            source_dims=None,
            source_units="auto",
            stored_source_dims=("y", "x"),
        )
        assert result == ("y", "x")

    def test_uses_grid_when_dataset_lacks_geometry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5), dtype=np.float64))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )
        grid = xr.Dataset(coords={"cell": np.arange(5)})

        monkeypatch.setattr(
            remap,
            "_source_centre_arrays",
            lambda obj, source_units="auto": (
                np.arange(5, dtype=np.float64),
                np.arange(5, dtype=np.float64),
                ("cell",),
            ),
        )

        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=5,
            grid=grid,
            source_dims=None,
            source_units="auto",
            stored_source_dims=None,
        )
        assert result == ("cell",)

    def test_falls_back_to_size_guess_when_geometry_inference_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5), dtype=np.float64))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )

        def raise_no_geometry(
            obj: xr.Dataset,
            source_units: str = "auto",
        ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
            raise ValueError("no geometry")

        monkeypatch.setattr(remap, "_source_centre_arrays", raise_no_geometry)

        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=5,
            grid=None,
            source_dims=None,
            source_units="auto",
            stored_source_dims=None,
        )
        assert result == ("cell",)

    def test_geometry_mismatch_raises_helpful_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5), dtype=np.float64))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )

        monkeypatch.setattr(
            remap,
            "_source_centre_arrays",
            lambda obj, source_units="auto": (
                np.arange(8, dtype=np.float64),
                np.arange(8, dtype=np.float64),
                ("cell",),
            ),
        )

        with pytest.raises(ValueError, match="Pass `source_dims=...` or `grid=...`"):
            _resolve_source_dims_for_weight_application(
                ds,
                n_source=6,
                grid=None,
                source_dims=None,
                source_units="auto",
                stored_source_dims=None,
            )


class TestApplyWeightFileWithExternalGeometry:
    def test_apply_weight_file_with_explicit_source_dims(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[1.0, np.nan, 3.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1, 2]},
                ),
                "static": xr.DataArray(
                    [10.0, 20.0],
                    dims=("level",),
                    coords={"level": [0, 1]},
                ),
            }
        )

        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 3], dtype=np.int32)),
                "S": ("nnz", np.array([0.5, 0.5, 1.0], dtype=np.float64)),
            },
            attrs={
                "grid_doctor_level": 1,
                "grid_doctor_order": "nested",
                "grid_doctor_source_dims": json.dumps(["cell"]),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        monkeypatch.setattr(
            remap,
            "_healpix_centres",
            lambda level, nest, ellipsoid="sphere": (
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
        )

        result = apply_weight_file(
            ds,
            path,
            missing_policy="renormalize",
            source_dims=("cell",),
        )

        assert "cell" in result.dims
        assert result.sizes["cell"] == 2

        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, np.array([1.0, 3.0], dtype=np.float64))

        np.testing.assert_allclose(result["static"].values, ds["static"].values)

    def test_apply_weight_file_with_grid_argument(
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
                "row": ("nnz", np.array([1, 2, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 3], dtype=np.int32)),
                "S": ("nnz", np.array([1.0, 0.25, 0.75], dtype=np.float64)),
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
            lambda level, nest, ellipsoid="sphere": (
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
        )

        result = apply_weight_file(ds, path, grid=grid)

        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, np.array([1.0, 2.75], dtype=np.float64))

    def test_apply_weight_file_uses_stored_source_dims_without_geometry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.arange(6, dtype=np.float64).reshape(1, 2, 3),
                    dims=("time", "y", "x"),
                    coords={"time": [0], "y": [0, 1], "x": [0, 1, 2]},
                )
            }
        )

        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 6], dtype=np.int32)),
                "S": ("nnz", np.array([0.5, 0.5, 1.0], dtype=np.float64)),
            },
            attrs={
                "grid_doctor_source_dims": json.dumps(["y", "x"]),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        def no_geometry(
            obj: xr.Dataset,
            source_units: str = "auto",
        ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
            raise ValueError("geometry not embedded")

        monkeypatch.setattr(remap, "_source_centre_arrays", no_geometry)

        result = apply_weight_file(ds, path)

        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, np.array([0.5, 5.0], dtype=np.float64))

    def test_apply_weight_file_geometry_mismatch_still_raises(
        self,
        tmp_path: Path,
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

        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1], dtype=np.int32)),
                "col": ("nnz", np.array([4], dtype=np.int32)),
                "S": ("nnz", np.array([1.0], dtype=np.float64)),
            }
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        with pytest.raises(ValueError, match="source_dims"):
            apply_weight_file(ds, path, source_dims=("cell",))


class TestWeightFileWorkflow:
    def test_compute_healpix_weights_rejects_linear(
        self, regular_ds: xr.Dataset
    ) -> None:
        with pytest.raises(ValueError, match="Only 'nearest' and 'conservative'"):
            compute_healpix_weights(regular_ds, 1, method="linear")

    def test_compute_healpix_weights_nearest(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(remap_backend, "_require_esmpy", lambda: _FakeESMPy())
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

    def test_compute_healpix_weights_limited_area_ignores_unmapped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        limited_area_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(remap_backend, "_require_esmpy", lambda: _FakeESMPy())
        monkeypatch.setattr(
            remap,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        path = tmp_path / "weights.nc"
        compute_healpix_weights(
            limited_area_ds, 1, method="conservative", weights_path=path
        )
        assert _FakeRegrid.last_kwargs is not None
        assert (
            _FakeRegrid.last_kwargs["unmapped_action"]
            == _FakeESMPy.UnmappedAction.IGNORE
        )
        assert _FakeRegrid.last_kwargs["line_type"] == _FakeESMPy.LineType.GREAT_CIRCLE

    def test_apply_weight_file_renormalizes_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[np.array([1.0, np.nan, 3.0], dtype=np.float32)]],
                    dims=("time", "level", "cell"),
                    coords={
                        "time": [0],
                        "level": [0],
                        "cell": [0, 1, 2],
                    },
                ),
                "static": xr.DataArray(
                    [10.0, 20.0],
                    dims=("level",),
                    coords={"level": [0, 1]},
                ),
            },
            attrs={"source": "synthetic"},
        )

        weights = xr.Dataset(
            {
                # ESMF-style 1-based indexing
                "row": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 3], dtype=np.int32)),
                "S": ("nnz", np.array([0.5, 0.5, 1.0], dtype=np.float64)),
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
            lambda ds, source_units="auto": (
                np.array([0.0, 1.0, 2.0], dtype=np.float64),
                np.array([10.0, 20.0, 30.0], dtype=np.float64),
                ("cell",),
            ),
        )
        monkeypatch.setattr(
            remap,
            "_healpix_centres",
            lambda level, nest, ellipsoid="sphere": (
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
        )

        result = apply_weight_file(ds, path, missing_policy="renormalize")

        assert "cell" in result.dims
        assert result.sizes["cell"] == 2

        out = result["temperature"].isel(time=0, level=0).values
        np.testing.assert_allclose(out, np.array([1.0, 3.0], dtype=np.float64))

        # Non-spatial variables should pass through unchanged
        np.testing.assert_allclose(result["static"].values, ds["static"].values)

    def test_apply_weight_file_rejects_mismatched_geometry(
        self, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        path = tmp_path / "weights.nc"
        xr.Dataset(
            {
                "row": ("nnz", np.array([1], dtype=np.int32)),
                "col": ("nnz", np.array([100], dtype=np.int32)),
                "S": ("nnz", np.array([1.0], dtype=np.float64)),
            }
        ).to_netcdf(path)
        with pytest.raises(ValueError, match="does not match"):
            apply_weight_file(regular_ds, path)
