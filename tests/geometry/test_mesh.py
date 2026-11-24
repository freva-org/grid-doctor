"""Source meshes, polygons and coverage (``grid_doctor._geometry.mesh``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor._geometry import sphere as sphere_module
from grid_doctor._geometry.coords import (
    _get_spatial_dims,
    _get_unstructured_dim,
)
from grid_doctor._geometry.mesh import (
    _classify_source_kind,
    _collapse_repeated_corners,
    _corner_mesh_from_arrays,
    _infer_bounds_1d,
    _infer_curvilinear_corners,
    _lat_coverage_from_centres,
    _lon_coverage_from_centres,
    _looks_global,
    _median_positive_step,
    _mesh_to_polygons,
    _polygons_to_corner_arrays,
    _regular_grid_mesh,
)
from grid_doctor._geometry.sphere import _vectorized_polygon_centres

# ===================================================================
# Backend source classification and coverage helpers
# ===================================================================


class TestBackendClassificationAndCoverage:
    def test_dimension_and_coordinate_failures(self) -> None:
        unstructured_without_cell_dim = xr.Dataset(
            {
                "data": (("a", "b"), np.zeros((2, 2)), {"CDI_grid_type": "unstructured"}),
            },
            coords={
                "lat": (("a", "b"), np.zeros((2, 2))),
                "lon": (("a", "b"), np.zeros((2, 2))),
            },
        )
        with pytest.raises(ValueError, match="source cell dimension"):
            _get_unstructured_dim(unstructured_without_cell_dim)

        curvilinear = xr.Dataset(
            coords={
                "lat": (("row", "column"), np.zeros((2, 3))),
                "lon": (("row", "column"), np.zeros((2, 3))),
            }
        )
        assert _get_spatial_dims(curvilinear) == ("row", "column")

        with pytest.raises(ValueError, match="one-dimensional"):
            _infer_bounds_1d(np.zeros((2, 2)))
        with pytest.raises(ValueError, match="At least two"):
            _infer_bounds_1d(np.array([0.0]))
        with pytest.raises(ValueError, match="expects 2-D"):
            _infer_curvilinear_corners(np.zeros(2), np.zeros(2))

    def test_coverage_helpers_handle_empty_and_singleton_values(self) -> None:
        assert _median_positive_step(np.array([1.0])) == 0.0
        assert _median_positive_step(np.array([1.0, 1.0])) == 0.0
        assert _lon_coverage_from_centres(np.array([np.nan])) == 0.0
        assert _lon_coverage_from_centres(np.array([0.0])) == 0.0
        assert _lat_coverage_from_centres(np.array([np.nan])) == 0.0
        assert _lat_coverage_from_centres(np.array([0.0])) == 0.0

    def test_classify_source_kind_all_paths(
        self, regular_ds: xr.Dataset, curvilinear_ds: xr.Dataset, unstructured_ds: xr.Dataset
    ) -> None:
        assert _classify_source_kind(regular_ds, explicit_kind="regular") == "regular"
        assert _classify_source_kind(unstructured_ds, explicit_kind="auto") == "unstructured"
        assert _classify_source_kind(regular_ds, explicit_kind="auto") == "regular"
        assert _classify_source_kind(curvilinear_ds, explicit_kind="auto") == "curvilinear"
        invalid = xr.Dataset(coords={"lat": ("z", np.zeros(2)), "lon": ("z", np.zeros(2))})
        invalid["lat"] = (("a", "b", "c"), np.zeros((1, 1, 1)))
        invalid["lon"] = (("a", "b", "c"), np.zeros((1, 1, 1)))
        with pytest.raises(ValueError, match="Could not classify"):
            _classify_source_kind(invalid, explicit_kind="auto")
# ===================================================================
# Backend polygon mesh helpers
# ===================================================================


class TestBackendPolygonMeshes:
    def test_polygon_centres_reject_invalid_faces(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="at least one valid corner"):
            _vectorized_polygon_centres(
                np.array([[np.nan]]), np.array([[np.nan]])
            )
        monkeypatch.setattr(
            sphere_module.np, "sqrt", lambda values: np.zeros_like(values)
        )
        with pytest.raises(ValueError, match="zero-length"):
            _vectorized_polygon_centres(
                np.array([[0.0, 180.0]]), np.array([[0.0, 0.0]])
            )

    def test_corner_mesh_validates_and_collapses_repeated_nodes(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            _corner_mesh_from_arrays(np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError, match="Source cell 0"):
            _corner_mesh_from_arrays(
                np.array([[0.0, 1.0, np.nan]]), np.array([[0.0, 0.0, np.nan]])
            )
        with pytest.raises(ValueError, match="fewer than three distinct"):
            _corner_mesh_from_arrays(
                np.array([[0.0, 1.0, 0.0, 0.0]]),
                np.array([[0.0, 0.0, 0.0, 0.0]]),
            )

        collapsed = _collapse_repeated_corners(
            np.array([[-1, -1, -1], [1, 2, 1]], dtype=np.int32)
        )
        np.testing.assert_array_equal(collapsed, np.array([[-1, -1, -1], [1, 2, -1]]))

    def test_polygon_corner_array_round_trip(self) -> None:
        polygons = [
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([1.0, 2.0, 2.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])),
        ]
        cell_lon, cell_lat = _polygons_to_corner_arrays(polygons)
        assert cell_lon.shape == (2, 4)
        assert np.isnan(cell_lon[0, -1])
        mesh = _corner_mesh_from_arrays(cell_lon, cell_lat)
        round_tripped = _mesh_to_polygons(mesh)
        assert [len(lon) for lon, _ in round_tripped] == [3, 4]
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
