"""Tests for `grid_doctor.remap`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor.remap import (
    _attach_healpix_coords,
    _flattened_size,
    _guess_source_dims_from_size,
    _healpix_centres,
    _parse_source_dims_attr,
    _resolve_source_dims_for_weight_application,
    apply_weight_file,
    regrid_to_healpix,
    regrid_unstructured_to_healpix,
)
from grid_doctor.remap_backend import (
    _canonical_lon,
    _get_spatial_dims,
    _infer_bounds_1d,
    _infer_curvilinear_corners,
    _looks_global,
    _normalise_angle_units,
    _regular_grid_mesh,
)
from .helpers import _FakeHealpixModule


# ===================================================================
# Primitive helpers (live in remap_backend, tested here for coverage)
# ===================================================================


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


# ===================================================================
# Regular grid mesh (antimeridian & periodicity)
# ===================================================================


class TestRegularGridMesh:
    """Tests for the longitude unwrapping and periodic node wrapping."""

    def test_global_grid_no_giant_polygon(self) -> None:
        """All source polygons must be ≈ dlon wide, not ~180°."""
        lat = np.linspace(-90.0, 90.0, 8)
        lon = np.linspace(0.125, 359.875, 16)
        mesh = _regular_grid_mesh(lat, lon)
        # Gather corner longitudes for every face: shape (n_face, 4).
        corner_lons = mesh.node_lon[mesh.face_nodes]
        # Unwrap per-face to handle the ±180 wraparound.
        deltas = np.diff(corner_lons, axis=1)
        deltas = (deltas + 180.0) % 360.0 - 180.0
        unwrapped = np.concatenate(
            [corner_lons[:, :1], corner_lons[:, :1] + np.cumsum(deltas, axis=1)],
            axis=1,
        )
        spans = unwrapped.max(axis=1) - unwrapped.min(axis=1)
        dlon = 360.0 / lon.size
        worst = int(np.argmax(spans))
        assert spans.max() < 2.0 * dlon, (
            f"Face {worst} spans {spans[worst]:.1f}° in longitude"
        )

    def test_global_grid_node_count_excludes_duplicate_column(self) -> None:
        """Periodic grids should have (ny+1)*nx nodes, not (ny+1)*(nx+1)."""
        lat = np.linspace(-90.0, 90.0, 12)
        lon = np.linspace(0.0, 360.0, 24, endpoint=False)
        mesh = _regular_grid_mesh(lat, lon)
        ny, nx = lat.size, lon.size
        assert mesh.node_lon.size == (ny + 1) * nx

    def test_global_grid_last_face_wraps(self) -> None:
        """The last face in each lat row should reference column-0 nodes."""
        lat = np.linspace(-90.0, 90.0, 4)
        lon = np.linspace(0.0, 360.0, 8, endpoint=False)
        mesh = _regular_grid_mesh(lat, lon)
        nx = lon.size
        # Last face of the first lat row: face index nx-1.
        last_face = mesh.face_nodes[nx - 1]
        # The right-hand nodes (indices 1 and 2) should be in column 0,
        # i.e. node indices 0 and nx (first column, rows 0 and 1).
        assert last_face[1] == 0  # top-right wraps to column 0
        assert last_face[2] == nx  # bottom-right wraps to column 0

    def test_regional_grid_no_wrapping(self) -> None:
        """A regional grid must not have periodic wrapping."""
        lat = np.linspace(20.0, 40.0, 5)
        lon = np.linspace(10.0, 30.0, 6)
        mesh = _regular_grid_mesh(lat, lon)
        ny, nx = lat.size, lon.size
        # Non-periodic: (ny+1)*(nx+1) nodes.
        assert mesh.node_lon.size == (ny + 1) * (nx + 1)

    def test_antimeridian_bounds_are_monotonic(self) -> None:
        """Node longitudes across the antimeridian must stay monotonic."""
        lat = np.linspace(-90.0, 90.0, 4)
        lon = np.linspace(0.125, 359.875, 8)
        mesh = _regular_grid_mesh(lat, lon)
        # Extract the first row of node longitudes (ny+1 rows of nx nodes).
        nx = lon.size
        first_row_lons = mesh.node_lon[:nx]
        # The sequence must be monotonically increasing.
        assert np.all(np.diff(first_row_lons) > 0), (
            f"Node longitudes are not monotonic: {first_row_lons}"
        )


# ===================================================================
# Global coverage
# ===================================================================


class TestGlobalCoverage:
    def test_global_regular_grid(self, regular_ds: xr.Dataset) -> None:
        assert _looks_global(regular_ds, source_units="auto") is True

    def test_limited_area_grid(self, limited_area_ds: xr.Dataset) -> None:
        assert _looks_global(limited_area_ds, source_units="auto") is False


# ===================================================================
# HEALPix centres
# ===================================================================


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


# ===================================================================
# Source-dimension resolution
# ===================================================================


class TestFlattenedSize:
    def test_single_dim(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        assert _flattened_size(ds, ("cell",)) == 5

    def test_multiple_dims(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        assert _flattened_size(ds, ("y", "x")) == 12

    def test_missing_dim_raises(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        with pytest.raises(ValueError, match="Missing"):
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
        ds = xr.Dataset(
            coords={
                "cell": np.arange(10),
                "member": np.arange(10),
            }
        )
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


class TestResolveSourceDims:
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
        with pytest.raises(ValueError, match="mismatch"):
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
            {"temperature": (("time", "cell"), np.ones((2, 5)))},
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

    def test_falls_back_to_size_guess(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5)))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )

        def raise_no_geometry(
            obj: xr.Dataset, source_units: str = "auto"
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

    def test_geometry_mismatch_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5)))},
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
        with pytest.raises(ValueError, match="source_dims"):
            _resolve_source_dims_for_weight_application(
                ds,
                n_source=6,
                grid=None,
                source_dims=None,
                source_units="auto",
                stored_source_dims=None,
            )


# ===================================================================
# Weight-file application (end-to-end through apply_weight_file)
# ===================================================================


class TestApplyWeightFile:
    def test_renormalizes_missing(
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
            lambda level, nest: (
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
        )

        result = apply_weight_file(ds, path, missing_policy="renormalize")

        assert "cell" in result.dims
        assert result.sizes["cell"] == 2
        out = result["temperature"].isel(time=0, level=0).values
        np.testing.assert_allclose(out, [1.0, 3.0])
        np.testing.assert_allclose(result["static"].values, ds["static"].values)

    def test_with_explicit_source_dims(
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
            },
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
            lambda level, nest: (
                np.array([0.0, 1.0], dtype=np.float64),
                np.array([2.0, 3.0], dtype=np.float64),
            ),
        )

        result = apply_weight_file(
            ds,
            path,
            source_dims=("cell",),
        )
        assert result.sizes["cell"] == 2
        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, [1.0, 3.0])

    def test_stored_source_dims_without_geometry(
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
            },
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
            obj: xr.Dataset, source_units: str = "auto"
        ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
            raise ValueError("geometry not embedded")

        monkeypatch.setattr(remap, "_source_centre_arrays", no_geometry)

        result = apply_weight_file(ds, path)
        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, [0.5, 5.0])

    def test_rejects_mismatched_geometry(
        self,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "weights.nc"
        xr.Dataset(
            {
                "row": ("nnz", np.array([1], dtype=np.int32)),
                "col": ("nnz", np.array([100], dtype=np.int32)),
                "S": ("nnz", np.array([1.0], dtype=np.float64)),
            }
        ).to_netcdf(path)
        with pytest.raises(ValueError, match="does not match|source_dims"):
            apply_weight_file(regular_ds, path)

    def test_geometry_mismatch_with_explicit_dims_raises(
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
        with pytest.raises(ValueError, match="mismatch"):
            apply_weight_file(ds, path, source_dims=("cell",))

    def test_backend_parameter_forwarded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the backend kwarg reaches apply_weights_nd."""
        ds = xr.Dataset(
            {
                "t": xr.DataArray(
                    [[1.0, 2.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1]},
                )
            },
        )
        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2], dtype=np.int32)),
                "S": ("nnz", np.array([1.0, 1.0], dtype=np.float64)),
            },
            attrs={
                "grid_doctor_source_dims": json.dumps(["cell"]),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        result = apply_weight_file(ds, path, backend="scipy")
        np.testing.assert_allclose(result["t"].isel(time=0).values, [1.0, 2.0])


# ===================================================================
# Public regridding entry points
# ===================================================================


class TestPublicRegridding:
    def test_regrid_to_healpix_nearest(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "weights.nc"
        called: dict[str, Any] = {}

        def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
            called["computed"] = True
            path.write_text("placeholder")
            return path

        def fake_apply(
            ds: xr.Dataset,
            weights_path: str | Path,
            **kwargs: Any,
        ) -> xr.Dataset:
            called["applied"] = True
            return xr.Dataset(
                {"t": ("cell", np.arange(3, dtype=np.float64))},
                attrs=ds.attrs.copy(),
            )

        monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)
        monkeypatch.setattr(remap, "apply_weight_file", fake_apply)

        result = regrid_to_healpix(regular_ds, 2, method="nearest", weights_path=path)
        assert called == {"computed": True, "applied": True}
        assert result.sizes["cell"] == 3

    def test_regrid_unstructured_rejects_structured(
        self,
        regular_ds: xr.Dataset,
    ) -> None:
        with pytest.raises(ValueError, match="unstructured"):
            regrid_unstructured_to_healpix(regular_ds, 1)

    def test_regrid_unstructured_delegates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        unstructured_ds: xr.Dataset,
    ) -> None:
        monkeypatch.setattr(
            remap,
            "regrid_to_healpix",
            lambda ds, level, **kw: xr.Dataset(
                {"t": ("cell", np.arange(2))}, attrs=ds.attrs.copy()
            ),
        )
        result = regrid_unstructured_to_healpix(unstructured_ds, 1)
        assert result.sizes["cell"] == 2


# ===================================================================
# Attach HEALPix coords
# ===================================================================


class TestAttachHealpixCoords:
    def test_attach_coords(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            remap,
            "_healpix_centres",
            lambda level, nest: (
                np.array([10.0, 20.0]),
                np.array([30.0, 40.0]),
            ),
        )
        ds = xr.Dataset({"t": ("cell", np.array([1.0, 2.0]))})
        result = _attach_healpix_coords(ds, level=2, nest=False)
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        assert result.attrs["healpix_nside"] == 4
        assert result.attrs["healpix_order"] == "ring"
