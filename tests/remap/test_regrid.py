"""Weight generation and ``regrid_to_healpix`` (``grid_doctor.remap._regrid``)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import config, remap
from grid_doctor._geometry import healpix as healpix_module
from grid_doctor._geometry import mesh as mesh_module
from grid_doctor.metadata import ELLIPSOID_ATTR
from grid_doctor.remap import _esmf as esmf_module
from grid_doctor.remap import _regrid as regrid_module
from grid_doctor.remap import (
    compute_healpix_weights,
    regrid_to_healpix,
)
from grid_doctor.types import Ellipsoid

from ..helpers import _FakeHealpixModule


@pytest.mark.parametrize(
    "missing_policy",
    ["renormalize", "propagate"],
)
def test_regrid_to_healpix_weight_methods_delegate(
    regular_ds: xr.Dataset,
    missing_policy: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weight_file = tmp_path / "weights.nc"
    calls: dict[str, Any] = {}

    def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
        del ds, level
        calls["compute"] = kwargs
        weight_file.write_text("ok")
        return weight_file

    def fake_apply(
        ds: xr.Dataset,
        path: str | Path,
        *,
        missing_policy: str = "renormalize",
        **kwargs: Any,
    ) -> xr.Dataset:
        calls["apply"] = {"path": Path(path), "missing_policy": missing_policy}
        return xr.Dataset(
            {"temperature": (("cell",), np.arange(3, dtype=np.float64))},
            attrs=ds.attrs.copy(),
        )

    monkeypatch.setattr(regrid_module, "compute_healpix_weights", fake_compute)
    monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)  # cached_weights
    monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)

    result = regrid_to_healpix(
        regular_ds, level=1, method="nearest", missing_policy=missing_policy
    )
    assert result.sizes["cell"] == 3
    assert calls["apply"]["missing_policy"] == missing_policy


def test_regrid_to_healpix_reuses_default_store(
    regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, _isolated_weights_store: Path
) -> None:
    """Without ``weights_path`` weights come from the default store: computed once, then reused."""
    computed: list[Path] = []

    def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
        path = Path(kwargs["weights_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("weights")
        computed.append(path)
        return path

    applied: list[Path] = []

    def fake_apply(ds: xr.Dataset, path: str | Path, **kwargs: Any) -> xr.Dataset:
        applied.append(Path(path))
        return xr.Dataset({"temperature": (("cell",), np.zeros(3))})

    monkeypatch.setattr(regrid_module, "compute_healpix_weights", fake_compute)
    monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)  # cached_weights
    monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)

    regrid_to_healpix(regular_ds)  # level derived from the grid resolution
    regrid_to_healpix(regular_ds)
    assert len(computed) == 1
    assert applied == computed * 2
    assert computed[0].is_relative_to(_isolated_weights_store / "weights")  # the default store


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

        monkeypatch.setattr(regrid_module, "compute_healpix_weights", fake_compute)
        monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)

        result = regrid_to_healpix(regular_ds, 2, method="nearest", weights_path=path)
        assert called == {"computed": True, "applied": True}
        assert result.sizes["cell"] == 3
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
            esmf_module, "_require_esmpy", lambda: _FakeESMPy()
        )
        monkeypatch.setattr(
            mesh_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
        )
        monkeypatch.setattr(
            healpix_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
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

    def _fake_healpix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(esmf_module, "_require_esmpy", lambda: _FakeESMPy())
        for module in (mesh_module, healpix_module):
            monkeypatch.setattr(module, "_require_healpix_geo_module", lambda grid: (_FakeHealpixModule(), {}))

    def test_write_is_atomic(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """Only the finished, annotated file appears; no scratch files are left behind."""
        self._fake_healpix(monkeypatch)
        out = tmp_path / "out" / "weights.nc"
        assert compute_healpix_weights(regular_ds, 1, method="nearest", weights_path=out) == out
        assert [p.name for p in out.parent.iterdir()] == ["weights.nc"]
        with xr.open_dataset(out) as ds:
            assert ds.attrs["grid_doctor_method"] == "nearest"

    def test_provenance_matches_weights_ref(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        """A weight file names its whole identity, so a store can be verified from the files alone."""
        from grid_doctor.metadata import HealpixGrid
        from grid_doctor.metadata import _keys as keys
        from grid_doctor.weights import (
            WEIGHTS_VERSION,
            WeightsDescription,
            describe_source,
        )

        self._fake_healpix(monkeypatch)
        out = compute_healpix_weights(regular_ds, 1, method="nearest", nest=False, weights_path=tmp_path / "w.nc")
        ref = WeightsDescription(describe_source(regular_ds), HealpixGrid(level=1, indexing_scheme="ring"), "nearest").ref
        with xr.open_dataset(out) as ds:
            attrs = ds.attrs
        assert attrs[keys.WEIGHTS_VERSION_ATTR] == WEIGHTS_VERSION
        assert attrs[keys.SOURCE_HASH_ATTR] == ref.key
        assert attrs[keys.LEVEL_ATTR] == ref.level
        assert attrs[keys.ORDER_ATTR] == "ring"  # stored files are nested; this one was renumbered
        assert attrs[keys.ELLIPSOID_ATTR] == ref.ellipsoid
        assert attrs[keys.METHOD_ATTR] == ref.method
        assert keys.VERSION_ATTR in attrs  # grid-doctor version
        assert attrs[keys.ESMF_VERSION_ATTR]  # installed ESMPy version, or "unknown"
        assert attrs[keys.UNMAPPED_CELLS_ATTR] >= 0

    def test_failed_generation_leaves_nothing(
        self, monkeypatch: pytest.MonkeyPatch, regular_ds: xr.Dataset, tmp_path: Path
    ) -> None:
        self._fake_healpix(monkeypatch)

        def broken(**kwargs: Any) -> Path:
            Path(kwargs["weights_path"]).write_bytes(b"partial")
            raise RuntimeError("ESMF crashed")

        monkeypatch.setattr(esmf_module, "_compute_weights_in_memory", broken)
        out = tmp_path / "out" / "weights.nc"
        with pytest.raises(RuntimeError, match="ESMF crashed"):
            compute_healpix_weights(regular_ds, 1, method="nearest", weights_path=out)
        assert list(out.parent.iterdir()) == []

    def test_default_path_follows_weights_ref(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        """Without ``weights_path`` the same weights always land at the same local path."""
        from grid_doctor.metadata import HealpixGrid
        from grid_doctor.weights import WeightsDescription, describe_source

        monkeypatch.setattr(esmf_module, "_require_esmpy", lambda: _FakeESMPy())
        for module in (mesh_module, healpix_module):
            monkeypatch.setattr(module, "_require_healpix_geo_module", lambda grid: (_FakeHealpixModule(), {}))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        result = compute_healpix_weights(regular_ds, 1, method="nearest", nest=False)

        desc = WeightsDescription(describe_source(regular_ds), HealpixGrid(level=1, indexing_scheme="ring"), "nearest")
        assert result == tmp_path / "grid-doctor" / "weights" / desc.ref.path  # the default layout holds nested weights
        assert result.exists()
        with xr.open_dataset(result) as ds:
            assert ds.attrs["grid_doctor_order"] == "nested"
        assert compute_healpix_weights(regular_ds, 1, method="nearest", nest=False) == result

    @pytest.mark.parametrize(
        ("argument", "expected"), [(None, "WGS84"), ("sphere", "sphere")], ids=["config", "argument"]
    )
    def test_weight_file_records_its_ellipsoid(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
        argument: Ellipsoid | None,
        expected: Ellipsoid,
    ) -> None:
        monkeypatch.setattr(esmf_module, "_require_esmpy", lambda: _FakeESMPy())
        monkeypatch.setattr(
            mesh_module, "_require_healpix_geo_module", lambda grid: (_FakeHealpixModule(), {})
        )
        path = tmp_path / "weights.nc"
        with config.set(ellipsoid="WGS84"):
            compute_healpix_weights(regular_ds, 1, method="nearest", weights_path=path, ellipsoid=argument)
        with xr.open_dataset(path) as ds:
            assert ds.attrs[ELLIPSOID_ATTR] == expected

    def test_compute_healpix_weights_conservative(
        self,
        monkeypatch: pytest.MonkeyPatch,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            esmf_module, "_require_esmpy", lambda: _FakeESMPy()
        )
        monkeypatch.setattr(
            mesh_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
        )
        monkeypatch.setattr(
            healpix_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
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
            esmf_module, "_require_esmpy", lambda: _FakeESMPy()
        )
        monkeypatch.setattr(
            mesh_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
        )
        monkeypatch.setattr(
            healpix_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
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
