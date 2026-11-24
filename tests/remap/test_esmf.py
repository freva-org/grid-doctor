"""ESMF/ESMPy backend guards and offline workflows (``grid_doctor.remap._esmf``)."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import weights as weights_module
from grid_doctor._geometry import spectral as spectral_module
from grid_doctor._geometry.healpix import _require_healpix_geo_module
from grid_doctor._geometry.mesh import (
    _corner_mesh_from_arrays,
    _regular_grid_mesh,
)
from grid_doctor._geometry.spectral import (
    SpectralTransformError,
    _materialise_spectral_source,
    _run_subprocess,
)
from grid_doctor._geometry.sphere import (
    _ensure_ccw,
    _spherical_centroid,
    _xyz_to_lonlat,
)
from grid_doctor.metadata import HealpixGrid
from grid_doctor.remap import _esmf as esmf_module
from grid_doctor.remap._esmf import (
    OfflineWeightConfig,
    _default_offline_enabled,
    _require_esmpy,
    _run_offline_esmf,
    compute_healpix_weights_backend,
    run_esmf_regrid_weightgen,
    write_ugrid_mesh_file,
)
from grid_doctor.weights import describe_source

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
        module, kwargs = _require_healpix_geo_module(HealpixGrid(level=1, indexing_scheme="ring"))
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
            spectral_module.subprocess,
            "run",
            lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="out", stderr="err"),
        )
        with pytest.raises(RuntimeError, match="STDOUT"):
            _run_subprocess(["bad-command"])

        captured: list[str] = []
        monkeypatch.setattr(
            esmf_module, "_run_subprocess", lambda cmd: captured.extend(cmd)
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

        monkeypatch.setattr(spectral_module, "_run_subprocess", write_transformed)
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
            spectral_module,
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
            weights_module,
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
            weights_module, "describe_source", lambda *args, **kwargs: description
        )
        monkeypatch.setattr(
            esmf_module,
            "_target_healpix_mesh",
            lambda grid: (np.array([0], dtype=np.int64), target_mesh),
        )

        def offline(**kwargs: Any) -> Path:
            output = kwargs["target_path"]
            xr.Dataset({k: ("n_s", np.array([v])) for k, v in (("row", 1), ("col", 1), ("S", 1.0))}).to_netcdf(output)
            return output

        monkeypatch.setattr(esmf_module, "_run_offline_esmf", offline)
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
        monkeypatch.setattr(esmf_module.shutil, "which", lambda executable: None)
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

        monkeypatch.setattr(esmf_module.shutil, "which", executable)
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
        monkeypatch.setattr(esmf_module.shutil, "which", lambda executable: "/bin/tool")
        monkeypatch.setattr(
            esmf_module,
            "write_ugrid_mesh_file",
            lambda mesh, path, **kwargs: written.append(Path(path)) or Path(path),
        )
        monkeypatch.setattr(
            esmf_module,
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
