"""Tests for `grid_doctor.remap`."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor import remap_backend as remap_backend_module
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
    _get_latlon_names,
    _canonical_lon,
    _get_latlon_arrays,
    _looks_like_radians,
    _classify_source_kind,
    _collapse_repeated_corners,
    _corner_mesh_from_arrays,
    _default_offline_enabled,
    _ensure_ccw,
    _get_unstructured_vertices,
    _get_unstructured_dim,
    _get_spatial_dims,
    _infer_bounds_1d,
    _infer_curvilinear_corners,
    _lat_coverage_from_centres,
    _looks_global,
    _lon_coverage_from_centres,
    _materialise_spectral_source,
    _median_positive_step,
    _mesh_to_polygons,
    _norm_lon,
    _normalise_angle_units,
    _polygons_to_corner_arrays,
    _regular_grid_mesh,
    _require_esmpy,
    _require_healpix_geo_module,
    _run_offline_esmf,
    _run_subprocess,
    _source_mesh,
    _spherical_centroid,
    _vectorized_polygon_centres,
    _xyz_to_lonlat,
    OfflineWeightConfig,
    SpectralTransformError,
    compute_healpix_weights_backend,
    describe_source,
    run_esmf_regrid_weightgen,
    write_ugrid_mesh_file,
    normalise_dataset,
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


# ===================================================================
# Backend coordinate and configuration helpers
# ===================================================================


class TestBackendCoordinateGuards:
    def test_offline_config_selects_slurm_launcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SLURM_JOB_ID", "123")
        assert OfflineWeightConfig().mpirun == "srun"
        monkeypatch.delenv("SLURM_JOB_ID")
        assert OfflineWeightConfig().mpirun == "mpirun"

    def test_polygon_mesh_max_face_nodes(self) -> None:
        mesh = _corner_mesh_from_arrays(
            np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
        )
        assert mesh.max_face_nodes == 3

    def test_optional_module_imports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        esmpy = ModuleType("esmpy")
        monkeypatch.setitem(__import__("sys").modules, "esmpy", esmpy)
        assert _require_esmpy() is esmpy

        healpix = ModuleType("healpix_geo")
        ring = ModuleType("healpix_geo.ring")
        healpix.ring = ring  # type: ignore[attr-defined]
        monkeypatch.setitem(__import__("sys").modules, "healpix_geo", healpix)
        monkeypatch.setitem(__import__("sys").modules, "healpix_geo.ring", ring)
        module, kwargs = _require_healpix_geo_module(nest=False)
        assert module is ring
        assert kwargs == {"ellipsoid": "sphere"}

    def test_xyz_to_lonlat_and_spherical_helpers(self) -> None:
        with pytest.raises(ValueError, match="zero-length"):
            _xyz_to_lonlat(np.zeros(3))
        with pytest.raises(ValueError, match="zero-length"):
            _xyz_to_lonlat(np.zeros((1, 3)), batch=True)

        lon, lat = _xyz_to_lonlat(np.array([0.0, 1.0, 0.0]))
        assert lon == 90.0
        assert lat == 0.0
        centroid_lon, centroid_lat = _spherical_centroid(
            np.array([0.0, 2.0]), np.array([0.0, 0.0])
        )
        np.testing.assert_allclose((centroid_lon, centroid_lat), (1.0, 0.0))

        lon, lat = _ensure_ccw(
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
        )
        np.testing.assert_array_equal(lon, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_array_equal(lat, np.array([0.0, 1.0, 0.0]))
        lon, lat = _ensure_ccw(
            np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        )
        np.testing.assert_array_equal(lon, np.array([0.0, 1.0, 0.0]))
        np.testing.assert_array_equal(lat, np.array([0.0, 0.0, 1.0]))


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
            remap_backend_module.np, "sqrt", lambda values: np.zeros_like(values)
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
# Backend external workflow helpers
# ===================================================================


class TestBackendExternalWorkflows:
    def test_write_ugrid_mesh_file(self, tmp_path: Path) -> None:
        path = write_ugrid_mesh_file(
            [(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))],
            tmp_path / "nested" / "mesh.nc",
            mesh_name="source_mesh",
        )
        with xr.open_dataset(path) as ds:
            assert ds.attrs["Conventions"] == "UGRID-1.0"
            assert ds["source_mesh"].attrs["cf_role"] == "mesh_topology"
            assert ds["source_mesh_face_nodes"].attrs["start_index"] == 0

        mesh = _corner_mesh_from_arrays(
            np.array([[0.0, 1.0, 0.0]]), np.array([[0.0, 0.0, 1.0]])
        )
        assert write_ugrid_mesh_file(
            mesh, tmp_path / "mesh-object.nc", mesh_name="mesh"
        ).exists()

    def test_subprocess_and_weightgen_command(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            remap_backend_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="out", stderr="err"),
        )
        with pytest.raises(RuntimeError, match="STDOUT"):
            _run_subprocess(["bad-command"])

        captured: list[str] = []
        monkeypatch.setattr(
            remap_backend_module, "_run_subprocess", lambda cmd: captured.extend(cmd)
        )
        config = OfflineWeightConfig(
            nproc=2, weight_only=True, netcdf4=True
        )
        result = run_esmf_regrid_weightgen(
            src_mesh="source.nc",
            dst_mesh="target.nc",
            weights_path=tmp_path / "weights.nc",
            method="nearest",
            config=config,
            ignore_unmapped=True,
            src_regional=True,
            dst_regional=True,
        )
        assert result == tmp_path / "weights.nc"
        assert captured[:4] == ["mpirun", "-np", "2", "ESMF_RegridWeightGen"]
        assert {"--weight_only", "--netcdf4", "--ignore_unmapped", "--src_regional", "--dst_regional"} <= set(captured)

    def test_offline_and_spectral_helpers(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        assert _default_offline_enabled(
            method="nearest",
            source_cell_count=1,
            target_cell_count=1,
            config=OfflineWeightConfig(enabled=True),
        )
        with pytest.raises(ValueError, match="requires an external transform"):
            _materialise_spectral_source(
                regular_ds, transform_command=None, workdir=tmp_path
            )

        def write_transformed(cmd: list[str]) -> None:
            xr.Dataset(coords={"lat": ("y", [0.0, 1.0]), "lon": ("x", [0.0, 1.0])}).to_netcdf(cmd[-1])

        monkeypatch.setattr(remap_backend_module, "_run_subprocess", write_transformed)
        transformed = _materialise_spectral_source(
            regular_ds,
            transform_command=["transform", "{input}", "{output}"],
            workdir=tmp_path,
        )
        assert set(transformed.coords) == {"lat", "lon"}

        source_path = tmp_path / "spectral-input.nc"
        regular_ds.to_netcdf(source_path)
        transformed_from_path = _materialise_spectral_source(
            source_path,
            transform_command=["transform", "{input}", "{output}"],
            workdir=tmp_path,
        )
        assert set(transformed_from_path.coords) == {"lat", "lon"}

        monkeypatch.setattr(
            remap_backend_module,
            "_run_subprocess",
            lambda cmd: (_ for _ in ()).throw(RuntimeError("bad transform")),
        )
        with pytest.raises(SpectralTransformError, match="bad transform"):
            _materialise_spectral_source(
                regular_ds, transform_command=["transform"], workdir=tmp_path
            )

    def test_describe_source_from_path_and_spectral(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        source_path = tmp_path / "source.nc"
        regular_ds.to_netcdf(source_path)
        description = describe_source(source_path, workdir=tmp_path)
        assert description.kind == "regular"
        assert description.dataset.identical(regular_ds)

        monkeypatch.setattr(
            remap_backend_module,
            "_materialise_spectral_source",
            lambda source, **kwargs: regular_ds,
        )
        spectral = describe_source(
            source_path,
            source_kind="spectral",
            spectral_transform_command=["transform"],
            workdir=tmp_path,
        )
        assert spectral.kind == "regular"

    def test_compute_backend_rejects_invalid_method(
        self, regular_ds: xr.Dataset
    ) -> None:
        with pytest.raises(ValueError, match="Only 'nearest'"):
            compute_healpix_weights_backend(
                regular_ds,
                1,
                method="invalid",  # type: ignore[arg-type]
                nest=False,
                source_units="deg",
                weights_path=None,
            )

    def test_compute_backend_uses_offline_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        description = describe_source(regular_ds, workdir=tmp_path)
        target_mesh = _regular_grid_mesh(
            np.array([0.0, 1.0]), np.array([0.0, 1.0])
        )
        monkeypatch.setattr(
            remap_backend_module, "describe_source", lambda *args, **kwargs: description
        )
        monkeypatch.setattr(
            remap_backend_module,
            "_target_healpix_mesh",
            lambda level, nest: (np.array([0], dtype=np.int64), target_mesh),
        )

        def offline(**kwargs: Any) -> Path:
            output = kwargs["target_path"]
            xr.Dataset({"S": ("n_s", np.array([1.0]))}).to_netcdf(output)
            return output

        monkeypatch.setattr(remap_backend_module, "_run_offline_esmf", offline)
        output = tmp_path / "weights.nc"
        result = compute_healpix_weights_backend(
            regular_ds,
            1,
            method="nearest",
            nest=False,
            source_units="deg",
            weights_path=output,
            offline=OfflineWeightConfig(enabled=True),
        )
        assert result == output
        with xr.open_dataset(result) as weights:
            assert weights.attrs["grid_doctor_backend"] == "offline-esmf"

    def test_run_offline_esmf_validates_and_orchestrates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        description = describe_source(regular_ds, workdir=tmp_path)
        target_mesh = _regular_grid_mesh(
            np.array([0.0, 1.0]), np.array([0.0, 1.0])
        )
        config = OfflineWeightConfig(
            esmf_regrid_weightgen="missing-esmf"
        )
        monkeypatch.setattr(remap_backend_module.shutil, "which", lambda executable: None)
        with pytest.raises(FileNotFoundError, match="missing-esmf"):
            _run_offline_esmf(
                source_desc=description,
                target_mesh=target_mesh,
                target_path=tmp_path / "weights.nc",
                level=1,
                nest=False,
                method="nearest",
                config=config,
                ignore_unmapped=True,
            )

        config = OfflineWeightConfig(nproc=2)
        def executable(name: str) -> str | None:
            return "/bin/esmf" if name == config.esmf_regrid_weightgen else None

        monkeypatch.setattr(remap_backend_module.shutil, "which", executable)
        with pytest.raises(FileNotFoundError, match="MPI launcher"):
            _run_offline_esmf(
                source_desc=description,
                target_mesh=target_mesh,
                target_path=tmp_path / "weights.nc",
                level=1,
                nest=False,
                method="nearest",
                config=config,
                ignore_unmapped=True,
            )

        written: list[Path] = []
        monkeypatch.setattr(remap_backend_module.shutil, "which", lambda executable: "/bin/tool")
        monkeypatch.setattr(
            remap_backend_module,
            "write_ugrid_mesh_file",
            lambda mesh, path, **kwargs: written.append(Path(path)) or Path(path),
        )
        monkeypatch.setattr(
            remap_backend_module,
            "run_esmf_regrid_weightgen",
            lambda **kwargs: kwargs["weights_path"],
        )
        output = tmp_path / "offline.nc"
        result = _run_offline_esmf(
            source_desc=description,
            target_mesh=target_mesh,
            target_path=output,
            level=2,
            nest=True,
            method="conservative",
            config=OfflineWeightConfig(),
            ignore_unmapped=False,
        )
        assert result == output
        assert [path.name for path in written] == ["source_mesh.nc", "healpix_level_2_nest.nc"]
        assert not written[0].parent.exists()

        configured_workdir = tmp_path / "configured-workdir"
        _run_offline_esmf(
            source_desc=description,
            target_mesh=target_mesh,
            target_path=output,
            level=1,
            nest=False,
            method="nearest",
            config=OfflineWeightConfig(workdir=configured_workdir),
            ignore_unmapped=True,
        )
        assert configured_workdir.exists()


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
    @staticmethod
    def _unused_trailing_sources(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_source: int, source_size: int | None
    ) -> tuple[xr.Dataset, Path]:
        """Weights where only source cells 1 and 3 are used, like ``nearest`` output."""
        ds = xr.Dataset(
            {"t": ("cell", np.arange(1.0, n_source + 1.0))},
            coords={"cell": np.arange(n_source)},
        )
        # Toy weights with 2 targets are no real HEALPix grid: leave the
        # level unset so no coordinates are attached.
        attrs: dict[str, Any] = {}
        if source_size is not None:
            attrs["grid_doctor_source_size"] = source_size
        path = tmp_path / "weights.nc"
        xr.Dataset(
            {
                "row": ("nnz", np.array([1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 3], dtype=np.int32)),
                "S": ("nnz", np.array([1.0, 1.0])),
            },
            attrs=attrs,
        ).to_netcdf(path)

        monkeypatch.setattr(
            remap,
            "_source_centre_arrays",
            lambda ds, source_units="auto": (
                np.zeros(n_source),
                np.arange(float(n_source)),
                ("cell",),
            ),
        )
        return ds, path

    def test_uses_stored_source_size(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Trailing source cells without weights must not shrink the matrix."""
        ds, path = self._unused_trailing_sources(tmp_path, monkeypatch, n_source=5, source_size=5)
        result = apply_weight_file(ds, path)
        np.testing.assert_array_equal(result["t"].values, [1.0, 3.0])

    def test_infers_source_size_without_attribute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older weight files without grid_doctor_source_size still work."""
        ds, path = self._unused_trailing_sources(tmp_path, monkeypatch, n_source=3, source_size=None)
        result = apply_weight_file(ds, path)
        np.testing.assert_array_equal(result["t"].values, [1.0, 3.0])

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
            attrs={"grid_doctor_source_dims": json.dumps(["cell"])},
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

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
    def test_attach_coords(self) -> None:
        ds = xr.Dataset({"t": ("cell", np.arange(12.0))})
        result = _attach_healpix_coords(ds, level=0, nest=False, method="nearest")
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        np.testing.assert_array_equal(result["cell"].values, np.arange(12))
        assert result["cell"].attrs["standard_name"] == "healpix_index"
        assert result["crs"].attrs == {
            "grid_mapping_name": "healpix",
            "indexing_scheme": "ring",
            "refinement_level": 0,
        }
        assert result["t"].attrs["grid_mapping"] == "crs"
        assert result.attrs["grid_doctor_method"] == "nearest"

    def test_rejects_wrong_cell_count(self) -> None:
        ds = xr.Dataset({"t": ("cell", np.arange(2.0))})
        with pytest.raises(ValueError, match="conflicting sizes|cells"):
            _attach_healpix_coords(ds, level=2, nest=True)


class TestNormaliseDataset:
    """Normalisation of xarray objects (``normalise_dataset`` and friends)."""

    def test_micro_degrees(self):
        ds = xr.Dataset(
            coords={"lat": [1e-7, 1_111_111.012e-6, 12.34567890], "lon": [2e-7, 2_222_222.012e-6, 98.76543210]}
        )
        ds["lat"].attrs["units"] = "degrees_north"
        ds["lon"].attrs["units"] = "degrees_east"
        norm_ds = normalise_dataset(ds)
        np.testing.assert_array_equal(norm_ds["lat"], [0, 1_111_111e-6, 12.345679])
        np.testing.assert_array_equal(norm_ds["lon"], [0, 2_222_222e-6, 98.765432])
    @pytest.mark.parametrize(
        "test_ds",
        ["curvilinear_ds", "era5_ds", "limited_area_ds", "regular_ds", "unstructured_ds"],
        indirect=True,
    )
    def test_normalise_dataset_lonlat_ranges(self, test_ds):
        norm_ds = normalise_dataset(test_ds)
        lat, lon = (norm_ds[name] for name in _get_latlon_names(norm_ds))
        assert np.isfinite(lon).any() and np.isfinite(lat).any()
        assert -180.0 <= lon.min() and lon.max() < 180.0
        assert -90.0 <= lat.min() and lat.max() <= 90.0
        assert lat.attrs["units"] == "degrees_north"
        assert lon.attrs["units"] == "degrees_east"

    @pytest.mark.parametrize("dims", [("rlat", "rlon"), ("y", "x")])
    def test_normalise_curvilinear_latlon(self, dims):
        """Normalise the 2-D lat/lon, not the (rotated) dimension coordinates."""
        ydim, xdim = dims
        lon2d, lat2d = np.meshgrid(np.linspace(190.0, 350.0, 4), np.linspace(-20.0, 20.0, 3))
        coords = {
            "lat": ((ydim, xdim), lat2d, {"units": "degrees_north"}),
            "lon": ((ydim, xdim), lon2d, {"units": "degrees_east"}),
        }
        if ydim == "rlat":
            coords |= {
                "rlat": (ydim, np.linspace(-10.0, 10.0, 3), {"units": "degrees"}),
                "rlon": (xdim, np.linspace(-20.0, 20.0, 4), {"units": "degrees"}),
            }
        ds = xr.Dataset(coords=coords)

        norm_ds = normalise_dataset(ds)

        np.testing.assert_allclose(norm_ds["lon"], lon2d - 360.0)
        np.testing.assert_allclose(norm_ds["lat"], lat2d)
        assert norm_ds["lon"].attrs["units"] == "degrees_east"
        assert norm_ds["lat"].attrs["units"] == "degrees_north"
        for dim in dims:
            if dim in ds.coords:
                xr.testing.assert_identical(norm_ds[dim], ds[dim])
            else:
                assert dim not in norm_ds.coords

    @pytest.mark.parametrize("with_vertices", [True, False])
    def test_normalise_unstructured_centres_and_vertices(self, with_vertices):
        """Centres and (optional) vertices end up in the same lon convention."""
        clon = np.array([10.0, 190.0, 350.0])
        clat = np.array([-30.0, 0.0, 30.0])
        coords = {
            "clat": ("cell", clat, {"units": "degrees_north"}),
            "clon": ("cell", clon, {"units": "degrees_east"}),
        }
        if with_vertices:
            offsets = np.array([-1.0, 0.0, 1.0])
            coords |= {
                "clat_vertices": (("cell", "nv"), clat[:, None] + offsets, {"units": "degrees_north"}),
                "clon_vertices": (("cell", "nv"), clon[:, None] + offsets, {"units": "degrees_east"}),
            }
        norm_ds = normalise_dataset(xr.Dataset(coords=coords))

        np.testing.assert_allclose(norm_ds["clon"], [10.0, -170.0, -10.0])
        np.testing.assert_allclose(norm_ds["clat"], clat)
        if with_vertices:
            np.testing.assert_allclose(norm_ds["clon_vertices"].mean("nv"), norm_ds["clon"])
            np.testing.assert_allclose(norm_ds["clat_vertices"].mean("nv"), norm_ds["clat"])
        else:
            assert "clon_vertices" not in norm_ds

    def test_normalise_unstructured_nan_vertices(self):
        """NaN-padded vertices (cells with fewer corners) stay NaN; real corners are normalised."""
        clat_v = np.array([[-1.0, 0.0, 1.0, np.nan], [9.0, 10.0, 11.0, 10.0]])
        clon_v = np.array([[189.0, 190.0, 191.0, np.nan], [349.0, 350.0, 351.0, 352.0]])
        ds = xr.Dataset(
            coords={
                "clat": ("cell", [0.0, 10.0], {"units": "degrees_north"}),
                "clon": ("cell", [190.0, 350.0], {"units": "degrees_east"}),
                "clat_vertices": (("cell", "nv"), clat_v, {"units": "degrees_north"}),
                "clon_vertices": (("cell", "nv"), clon_v, {"units": "degrees_east"}),
            }
        )
        norm_ds = normalise_dataset(ds)

        lat_v, lon_v = norm_ds["clat_vertices"].values, norm_ds["clon_vertices"].values
        np.testing.assert_array_equal(np.isnan(lon_v), np.isnan(clon_v))
        np.testing.assert_array_equal(np.isnan(lat_v), np.isnan(clat_v))
        np.testing.assert_allclose(lon_v, [[-171.0, -170.0, -169.0, np.nan], [-11.0, -10.0, -9.0, -8.0]])
        np.testing.assert_allclose(lat_v, clat_v)
        np.testing.assert_allclose(norm_ds["clon"], [-170.0, -10.0])

    @pytest.mark.parametrize("set_units", [False, True])
    def test_normalise_unit(self, set_units):
        coords = {"lat": [-90, -45, 0, 45, 90], "lon": [0, 90, 180, 270, 360]}
        attrs = {"test": "This should prevail"}
        deg_ds = xr.Dataset(coords=coords, attrs=attrs)
        rad_ds = xr.Dataset(coords={k: np.radians(v) for k, v in coords.items()}, attrs=attrs)
        if set_units:
            for ds, units in ((deg_ds, "deg"), (rad_ds, "rad")):
                for c in ds.coords:
                    ds[c].attrs["units"] = units

        if set_units:
            n_deg_ds, n_rad_ds = normalise_dataset(deg_ds), normalise_dataset(rad_ds)
        else:
            n_deg_ds = normalise_dataset(deg_ds, source_units="deg")
            n_rad_ds = normalise_dataset(rad_ds, source_units="rad")
        for n_ds in (n_deg_ds, n_rad_ds):
            assert n_ds["lon"].attrs["units"] == "degrees_east"
            assert n_ds["lat"].attrs["units"] == "degrees_north"
        xr.testing.assert_identical(n_deg_ds, n_rad_ds)

        # check lon [-180, 180)
        assert n_deg_ds["lon"].min() == -180
        assert n_deg_ds["lon"].max() < 180

    def test_normalise_missing_units_assumes_degrees(self, caplog):
        ds = xr.Dataset(coords={"lat": [0.0, 0.5, 1.0], "lon": [0.0, 1.0, 2.0]})
        with caplog.at_level(logging.WARNING, logger="grid_doctor.remap_backend"):
            norm_ds = normalise_dataset(ds)
        np.testing.assert_allclose(norm_ds["lat"], ds["lat"])
        np.testing.assert_allclose(norm_ds["lon"], ds["lon"])
        assert "assuming degrees" in caplog.text

    def test_normalise_source_units_overrides_attrs(self, caplog):
        ds = xr.Dataset(coords={"lat": [0.0, 0.5], "lon": [0.0, 1.0]})
        for c in ds.coords:
            ds[c].attrs["units"] = "degrees"
        with caplog.at_level(logging.WARNING, logger="grid_doctor.remap_backend"):
            norm_ds = normalise_dataset(ds, source_units="rad")
        np.testing.assert_allclose(norm_ds["lat"], np.degrees([0.0, 0.5]), atol=1e-6)
        assert not caplog.text

    @pytest.mark.parametrize(
        "coords,errmsg",
        [
            ({}, "Could not locate latitude/longitude"),
            ({"lat": []}, "Could not locate latitude/longitude"),
            ({"lon": []}, "Could not locate latitude/longitude"),
            ({"lat": [360], "lon": [0]}, "Located coordinate `lat` has values outside [-90, 90]."),
            ({"lat": [91], "lon": [0]}, "Located coordinate `lat` has values outside [-90, 90]."),
            ({"lat": [-91], "lon": [0]}, "Located coordinate `lat` has values outside [-90, 90]."),
            # Input checks:
            ({"lat": [], "lon":[]}, "Located coordinate `lat` is empty."),
            ({"lat": [1.0], "lon":[]}, "Located coordinate `lon` is empty."),
            ({"lat": [float("inf")], "lon":[]}, "Located coordinate `lat` contains Inf or NaN"),
            ({"lat": [1.0], "lon":[float("inf")]}, "Located coordinate `lon` contains Inf or NaN"),
            ({"lat": [float("nan")], "lon":[]}, "Located coordinate `lat` contains Inf or NaN"),
            ({"lat": [1.0], "lon":[float("nan")]}, "Located coordinate `lon` contains Inf or NaN"),
            # with vertex: centres valid, vertices invalid (NaN padding is allowed)
            (
                {"lat": [1.0], "lon": [1.0], "lat_bnds": (("cell", "nv"), np.empty((0, 3))), "lon_bnds": (("cell", "nv"), np.empty((0, 3)))},
                "Located coordinate `lat_bnds` is empty.",
            ),
            (
                {"lat": [1.0], "lon": [1.0], "lat_bnds": (("cell", "nv"), [[0.0, float("inf"), 2.0]]), "lon_bnds": (("cell", "nv"), [[0.0, 1.0, 2.0]])},
                "Located coordinate `lat_bnds` contains Inf values.",
            ),
            (
                {"lat": [1.0], "lon": [1.0], "lat_bnds": (("cell", "nv"), [[0.0, 1.0, 2.0]]), "lon_bnds": (("cell", "nv"), [[0.0, -float("inf"), 2.0]])},
                "Located coordinate `lon_bnds` contains Inf values.",
            ),
            (
                {"lat": [1.0], "lon": [1.0], "lat_bnds": (("cell", "nv"), [[0.0, 91.0, 2.0]]), "lon_bnds": (("cell", "nv"), [[0.0, 1.0, 2.0]])},
                "Located coordinate `lat_bnds` has values outside [-90, 90].",
            ),
            # centres are checked before vertices
            (
                {"lat": [float("nan")], "lon": [1.0], "lat_bnds": (("cell", "nv"), [[0.0, float("inf"), 2.0]]), "lon_bnds": (("cell", "nv"), [[0.0, 1.0, 2.0]])},
                "Located coordinate `lat` contains Inf or NaN",
            ),
        ],
    )
    def test_normalise_raises(self, coords, errmsg):
        with pytest.raises(ValueError, match=re.escape(errmsg)):
            normalise_dataset(xr.Dataset(coords=coords))


class TestNormaliseArrays:
    """Normalisation of plain float arrays (``_norm_lon``, ``_norm_lat``)."""

    @pytest.mark.parametrize(
        "lon",
        [
            np.linspace(-180, 180, num=360 * 2),  # includes 180
            np.arange(-180, 180),  # doesn't include 180
            np.arange(-180, 181),  # includes 180
            np.arange(-180, 180, 1 + 1e-6),
            np.arange(-180, 180, 1 + 1e-9),
        ],
        ids=[
            "linspace",
            "arange",
            "arange-wrap",
            "arange-m-err",
            "arange-n-err",
        ],
    )
    @pytest.mark.parametrize("scale", [1_000_000, 1_000_000_000], ids=["micro", "nano"])
    def test_norm_lon_noop(self, lon, scale):
        n_lon = _norm_lon(lon, scale=scale)
        # last element may change because of wrapping
        np.testing.assert_allclose(lon[:-1], n_lon[:-1], atol=1 / scale)
        assert (n_lon < 180.0).all()
        assert (n_lon == -180.0).any()

    @pytest.mark.parametrize(
        "lon",
        [
            np.linspace(-360, 360, num=360 * 4),
            np.arange(-360, 360),
            np.arange(-360, 360, 1 + 1e-6),
            np.arange(-360, 360, 1 + 1e-9),
        ],
        ids=[
            "linspace",
            "arange",
            "arange-m-err",
            "arange-n-err",
        ],
    )
    @pytest.mark.parametrize("scale", [1_000_000, 1_000_000_000], ids=["micro", "nano"])
    def test_norm_lon_duplicates(self, lon, scale):
        n_lon = _norm_lon(lon, scale=scale)
        np.testing.assert_allclose(n_lon, ((lon + 180.0) % 360.0) - 180.0, atol=1 / scale)
