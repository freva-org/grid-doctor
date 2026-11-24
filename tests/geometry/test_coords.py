"""Coordinate detection and resolution helpers (``grid_doctor._geometry.coords``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor._geometry.coords import (
    _get_latlon_arrays,
    _get_spatial_dims,
    _get_unstructured_dim,
    _get_unstructured_vertices,
    _is_unstructured,
    get_latlon_resolution,
)
from grid_doctor._geometry.healpix import resolution_to_healpix_level
from grid_doctor._geometry.mesh import (
    _infer_bounds_1d,
    _infer_curvilinear_corners,
    _source_mesh,
)
from grid_doctor._geometry.normalise import (
    _canonical_lon,
    _looks_like_radians,
    _normalise_angle_units,
)


class TestGridDetection:
    def test_is_unstructured_from_dimension(self, unstructured_ds: xr.Dataset) -> None:
        assert _is_unstructured(unstructured_ds) is True

    def test_is_unstructured_from_attribute(self) -> None:
        ds = xr.Dataset(
            {"t": (("points",), np.zeros(3), {"CDI_grid_type": "unstructured"})},
            coords={"lat": ("points", np.zeros(3)), "lon": ("points", np.zeros(3))},
        )
        assert _is_unstructured(ds) is True

    def test_get_unstructured_dim_prefers_known_name(
        self, unstructured_ds: xr.Dataset
    ) -> None:
        assert _get_unstructured_dim(unstructured_ds) == "cell"

    def test_get_unstructured_dim_falls_back_to_lat_dim(self) -> None:
        ds = xr.Dataset(
            {"t": (("points",), np.zeros(4))},
            coords={"clat": ("points", np.zeros(4)), "clon": ("points", np.zeros(4))},
        )
        assert _get_unstructured_dim(ds) == "points"

    def test_get_latlon_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not locate"):
            _get_latlon_arrays(xr.Dataset({"t": (("x",), np.zeros(3))}))


class TestResolutionHelpers:
    def test_regular_resolution(self, regular_ds: xr.Dataset) -> None:
        resolution = get_latlon_resolution(regular_ds)
        assert 0.0 < resolution < 30.0

    def test_curvilinear_resolution(self, curvilinear_ds: xr.Dataset) -> None:
        resolution = get_latlon_resolution(curvilinear_ds)
        assert 0.0 < resolution < 30.0

    def test_unstructured_resolution(self, unstructured_ds: xr.Dataset) -> None:
        resolution = get_latlon_resolution(unstructured_ds)
        assert resolution > 0.0

    def test_resolution_to_healpix_level(self) -> None:
        assert resolution_to_healpix_level(1.0) >= 0
        with pytest.raises(ValueError, match="positive"):
            resolution_to_healpix_level(0.0)


# ===================================================================
# Primitive helpers (live in _geometry and remap._esmf, tested here for coverage)
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

    @pytest.mark.parametrize(
        ("test_ds", "is_rad"),
        [("unstructured_ds", False), ("unstructured_rad_ds", True)],
        indirect=["test_ds"],
    )
    def test_unstructured_looks_like_radians(self, test_ds: xr.Dataset, is_rad: bool) -> None:
        latv, lonv = _get_latlon_arrays(test_ds)
        assert is_rad == _looks_like_radians(latv)
        assert is_rad == _looks_like_radians(lonv)

    def test_unstructured_vertices_resolution(self, unstructured_ds: xr.Dataset) -> None:
        assert _get_unstructured_vertices(unstructured_ds) == (
            "clon_vertices",
            "clat_vertices",
        )

        ds = unstructured_ds.rename(
            {
                "clon": "lon",
                "clat": "lat",
                "clon_vertices": "cell_longitude_bounds",
                "clat_vertices": "cell_latitude_bounds",
            }
        )
        ds["lon"].attrs["bounds"] = "cell_longitude_bounds"
        ds["lat"].attrs["bounds"] = "cell_latitude_bounds"

        assert _get_unstructured_vertices(ds) == (
            "cell_longitude_bounds",
            "cell_latitude_bounds",
        )

        ds = xr.Dataset({"temperature": ("cell", np.zeros(2))})

        assert _get_unstructured_vertices(ds) == (None, None)

    def test_source_mesh_geometry_dispatch(
        self, unstructured_ds: xr.Dataset, curvilinear_ds: xr.Dataset
    ) -> None:
        with pytest.raises(ValueError, match="require per-cell vertex coordinates"):
            _source_mesh(
                unstructured_ds.drop_vars(["clon_vertices", "clat_vertices"]),
                source_units="deg",
            )

        mesh, source_dims = _source_mesh(unstructured_ds, source_units="deg")

        assert source_dims == ("cell",)
        assert mesh.face_nodes.shape == (4, 3)

        mesh, source_dims = _source_mesh(curvilinear_ds, source_units="deg")

        assert source_dims == ("y", "x")
        assert mesh.face_nodes.shape == (12 * 24, 4)

        invalid_ds = xr.Dataset(
            coords={
                "lat": (("y", "x", "z"), np.zeros((2, 2, 2))),
                "lon": (("y", "x", "z"), np.zeros((2, 2, 2))),
            }
        )

        with pytest.raises(ValueError, match="must be 1-D or 2-D"):
            _source_mesh(invalid_ds, source_units="deg")
