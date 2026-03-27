"""Tests for `grid_doctor.remap`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

from grid_doctor import remap
from grid_doctor.remap import (
    _apply_sparse_array,
    _attach_healpix_coords,
    _canonical_lon,
    _extract_sparse_weights,
    _get_spatial_dims,
    _healpix_centres,
    _infer_bounds_1d,
    _infer_curvilinear_corners,
    _interpolate_linear_array,
    _looks_global,
    _normalise_angle_units,
    _source_polygons,
    apply_weight_file,
    compute_healpix_weights,
    regrid_to_healpix,
    regrid_unstructured_to_healpix,
)


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


class _FakeHealpixModule:
    @staticmethod
    def vertices(
        ipix: np.ndarray, level: int, ellipsoid: str = "sphere"
    ) -> tuple[np.ndarray, np.ndarray]:
        del level, ellipsoid
        n = ipix.size
        lon = np.stack(
            [
                np.asarray(ipix, dtype=np.float64),
                np.asarray(ipix, dtype=np.float64) + 0.5,
                np.asarray(ipix, dtype=np.float64) + 0.5,
                np.asarray(ipix, dtype=np.float64),
            ],
            axis=1,
        )
        lat = np.tile(np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float64), (n, 1))
        return lon, lat

    @staticmethod
    def healpix_to_lonlat(
        ipix: np.ndarray, level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del level
        n = ipix.size
        lon = np.linspace(-180.0, 180.0, n, endpoint=False)
        lat = np.linspace(-90.0, 90.0, n)
        return lon, lat


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


class TestPrimitiveHelpers:
    def test_canonical_lon_wraps(self) -> None:
        lon = np.array([-190.0, -180.0, 0.0, 180.0, 190.0])
        wrapped = _canonical_lon(lon)
        np.testing.assert_allclose(
            wrapped, np.array([170.0, -180.0, 0.0, -180.0, -170.0])
        )

    def test_normalise_angle_units_auto_detects_radians(self) -> None:
        values = np.array([0.0, np.pi / 2.0, np.pi])
        result = _normalise_angle_units(values, "auto")
        np.testing.assert_allclose(result, np.array([0.0, 90.0, 180.0]))

    def test_infer_bounds_1d(self) -> None:
        values = np.array([0.0, 2.0, 4.0])
        bounds = _infer_bounds_1d(values)
        np.testing.assert_allclose(bounds, np.array([-1.0, 1.0, 3.0, 5.0]))

    def test_infer_curvilinear_corners_shape(self, curvilinear_ds: xr.Dataset) -> None:
        lat = np.asarray(curvilinear_ds["lat"].values, dtype=np.float64)
        lon = np.asarray(curvilinear_ds["lon"].values, dtype=np.float64)
        lat_corner, lon_corner = _infer_curvilinear_corners(lat, lon)
        assert lat_corner.shape == (lat.shape[0] + 1, lat.shape[1] + 1)
        assert lon_corner.shape == (lon.shape[0] + 1, lon.shape[1] + 1)

    def test_get_spatial_dims_regular(self, regular_ds: xr.Dataset) -> None:
        assert _get_spatial_dims(regular_ds) == ("y", "x")

    def test_get_spatial_dims_era5(self, era5_ds: xr.Dataset) -> None:
        assert _get_spatial_dims(era5_ds) == ("lat", "lon")

    def test_get_spatial_dims_rejects_unstructured(
        self, unstructured_ds: xr.Dataset
    ) -> None:
        with pytest.raises(ValueError, match="Could not determine"):
            _get_spatial_dims(unstructured_ds)


class TestSourcePolygons:
    def test_regular_grid_polygons(self, regular_ds: xr.Dataset) -> None:
        polygons, dims = _source_polygons(regular_ds, source_units="auto")
        assert dims == ("y", "x")
        assert len(polygons) == regular_ds.sizes["y"] * regular_ds.sizes["x"]
        assert len(polygons[0][0]) == 4

    def test_curvilinear_grid_polygons(self, curvilinear_ds: xr.Dataset) -> None:
        polygons, dims = _source_polygons(curvilinear_ds, source_units="auto")
        assert dims == ("y", "x")
        assert len(polygons) == curvilinear_ds.sizes["y"] * curvilinear_ds.sizes["x"]

    def test_unstructured_grid_polygons_convert_radians(
        self, unstructured_rad_ds: xr.Dataset
    ) -> None:
        polygons, dims = _source_polygons(unstructured_rad_ds, source_units="auto")
        assert dims == ("cell",)
        first_lon, first_lat = polygons[0]
        assert np.nanmax(np.abs(first_lon)) > 0.1
        assert np.nanmax(np.abs(first_lat)) > 0.1

    def test_unstructured_requires_vertices(self, unstructured_ds: xr.Dataset) -> None:
        ds = unstructured_ds.drop_vars(["clon_vertices", "clat_vertices"])
        with pytest.raises(ValueError, match="require per-cell vertex"):
            _source_polygons(ds, source_units="auto")


class TestGlobalCoverage:
    def test_global_regular_grid(self, regular_ds: xr.Dataset) -> None:
        assert _looks_global(regular_ds, source_units="auto") is True

    def test_limited_area_grid(self, limited_area_ds: xr.Dataset) -> None:
        assert _looks_global(limited_area_ds, source_units="auto") is False


class TestHealpixCentres:
    def test_uses_healpix_geo_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            remap,
            "_require_healpix_geo_module",
            lambda nest: (_FakeHealpixModule(), {}),
        )
        lat, lon = _healpix_centres(1, nest=True)
        assert lat.shape == lon.shape
        assert lat.size == 48


class TestSparseWeights:
    def test_extracts_row_col_s(self, tmp_path: Path) -> None:
        path = tmp_path / "weights.nc"
        xr.Dataset(
            {
                "row": ("nnz", np.array([1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2], dtype=np.int32)),
                "S": ("nnz", np.array([0.25, 0.75], dtype=np.float64)),
            }
        ).to_netcdf(path)
        with xr.open_dataset(path) as ds:
            matrix, n_target, n_source = _extract_sparse_weights(ds)
        assert n_target == 2
        assert n_source == 2
        np.testing.assert_allclose(matrix.toarray(), np.diag([0.25, 0.75]))

    def test_extracts_scrip_style_columns(self) -> None:
        ds = xr.Dataset(
            {
                "dst_address": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "src_address": ("nnz", np.array([1, 2, 2], dtype=np.int32)),
                "remap_matrix": ("nnz", np.array([0.25, 0.75, 1.0], dtype=np.float64)),
            }
        )
        matrix, n_target, n_source = _extract_sparse_weights(ds)
        assert (n_target, n_source) == (2, 2)
        np.testing.assert_allclose(
            matrix.toarray(), np.array([[0.25, 0.75], [0.0, 1.0]])
        )

    def test_apply_sparse_array_renormalize(self) -> None:
        matrix = csr_matrix(np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float64))
        values = np.array([10.0, np.nan], dtype=np.float64)
        result = _apply_sparse_array(
            values, matrix=matrix, missing_policy="renormalize"
        )
        np.testing.assert_allclose(result, np.array([10.0, np.nan]), equal_nan=True)

    def test_apply_sparse_array_propagate(self) -> None:
        matrix = csr_matrix(np.array([[0.5, 0.5]], dtype=np.float64))
        values = np.array([10.0, np.nan], dtype=np.float64)
        result = _apply_sparse_array(values, matrix=matrix, missing_policy="propagate")
        assert np.isnan(result[0])


class TestInterpolation:
    def test_interpolate_linear_all_nan(self) -> None:
        points_xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        target_xyz = np.array([[1.0, 0.0, 0.0]])
        result = _interpolate_linear_array(
            np.array([np.nan, np.nan]), points_xyz=points_xyz, target_xyz=target_xyz
        )
        assert np.isnan(result[0])

    def test_interpolate_linear_falls_back_to_nearest(self) -> None:
        points_xyz = np.eye(3, dtype=np.float64)
        target_xyz = np.array([[0.9, 0.05, 0.05]], dtype=np.float64)
        values = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        result = _interpolate_linear_array(
            values, points_xyz=points_xyz, target_xyz=target_xyz
        )
        assert result.shape == (1,)
        assert np.isfinite(result[0])


class TestWeightFileWorkflow:
    def test_compute_healpix_weights_rejects_linear(
        self, regular_ds: xr.Dataset
    ) -> None:
        with pytest.raises(ValueError, match="Only 'nearest' and 'conservative'"):
            compute_healpix_weights(regular_ds, 1, method="linear")

    def test_compute_healpix_weights_nearest(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(remap, "_require_esmpy", lambda: _FakeESMPy())
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
        monkeypatch.setattr(remap, "_require_esmpy", lambda: _FakeESMPy())
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


class TestPublicRegridding:
    def test_regrid_to_healpix_linear(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset
    ) -> None:
        monkeypatch.setattr(
            remap,
            "_healpix_centres",
            lambda level, nest: (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])),
        )
        result = regrid_to_healpix(regular_ds, 1, method="linear")
        assert "cell" in result.dims
        assert result.sizes["cell"] == 3
        assert result.attrs["healpix_order"] == "nested"

    def test_regrid_to_healpix_nearest_uses_weight_file(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        path = tmp_path / "weights.nc"
        called: dict[str, Any] = {}

        def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
            del ds, level, kwargs
            called["computed"] = True
            path.write_text("placeholder")
            return path

        def fake_apply(
            ds: xr.Dataset,
            weights_path: str | Path,
            *,
            missing_policy: str = "renormalize",
        ) -> xr.Dataset:
            del weights_path, missing_policy
            called["applied"] = True
            return xr.Dataset(
                {"t": ("cell", np.arange(3, dtype=np.float64))}, attrs=ds.attrs.copy()
            )

        monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)
        monkeypatch.setattr(remap, "apply_weight_file", fake_apply)
        result = regrid_to_healpix(regular_ds, 2, method="nearest", weights_path=path)
        assert called == {"computed": True, "applied": True}
        assert result.sizes["cell"] == 3

    def test_regrid_unstructured_rejects_structured(
        self, regular_ds: xr.Dataset
    ) -> None:
        with pytest.raises(ValueError, match="expects an unstructured"):
            regrid_unstructured_to_healpix(regular_ds, 1)

    def test_regrid_unstructured_delegates(
        self, monkeypatch: pytest.MonkeyPatch, unstructured_ds: xr.Dataset
    ) -> None:
        monkeypatch.setattr(
            remap,
            "regrid_to_healpix",
            lambda ds, level, **kwargs: xr.Dataset(
                {"t": ("cell", np.arange(2))}, attrs=ds.attrs.copy()
            ),
        )
        result = regrid_unstructured_to_healpix(unstructured_ds, 1)
        assert result.sizes["cell"] == 2


class TestAttachHealpixCoords:
    def test_attach_coords(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            remap,
            "_healpix_centres",
            lambda level, nest: (np.array([10.0, 20.0]), np.array([30.0, 40.0])),
        )
        ds = xr.Dataset({"t": ("cell", np.array([1.0, 2.0]))})
        result = _attach_healpix_coords(ds, level=2, nest=False)
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        assert result.attrs["healpix_nside"] == 4
        assert result.attrs["healpix_order"] == "ring"
