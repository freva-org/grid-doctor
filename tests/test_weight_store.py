"""Tests for the WeightsStore."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import fsspec
import numpy as np
import pytest
import xarray as xr

from grid_doctor.metadata import HealpixGrid
from grid_doctor.weights import describe_source
from grid_doctor.weights import (
    WEIGHTS_STORE_ENV,
    WeightsDescription,
    WeightsStore,
)


def _desc(ds: xr.Dataset, level: int = 2, method: str = "conservative") -> WeightsDescription:
    return WeightsDescription(describe_source(ds), HealpixGrid(level=level), method)  # type: ignore[arg-type]


def _weight_file(path: Path, value: float = 1.0) -> Path:
    xr.Dataset({"S": ("nnz", np.array([value]))}).to_netcdf(path)
    return path


@pytest.fixture
def memory_fs() -> Iterator[fsspec.AbstractFileSystem]:
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    yield fs
    fs.store.clear()


class TestLocation:
    def test_local_path(self, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path / "store")
        assert store.is_local
        assert store.url_root.startswith("file://")
        assert not store.read_only

    def test_url(self, memory_fs: fsspec.AbstractFileSystem) -> None:
        store = WeightsStore("memory://weights")
        assert not store.is_local
        assert store.url_root == "memory:///weights"

    def test_fsspec_mapper(self, memory_fs: fsspec.AbstractFileSystem, tmp_path: Path) -> None:
        store = WeightsStore(memory_fs.get_mapper("weights"), cache_dir=tmp_path)
        assert store.fs is memory_fs
        assert store.root == "/weights"

    def test_env_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(WEIGHTS_STORE_ENV, str(tmp_path / "from-env"))
        assert WeightsStore().root == str(tmp_path / "from-env")

    @pytest.mark.parametrize("url", ["s3://bucket/weights", "https://example.com/weights"])
    def test_rejects_other_filesystems(self, url: str) -> None:
        pytest.importorskip("s3fs" if url.startswith("s3") else "aiohttp")
        with pytest.raises(ValueError, match="Unsupported weights store filesystem"):
            WeightsStore(url)

    def test_rejects_other_mappings(self) -> None:
        with pytest.raises(TypeError, match="Unsupported weights store location"):
            WeightsStore({})  # type: ignore[arg-type]


class TestLocalStore:
    def test_publish_fetch(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path / "store")
        desc = _desc(regular_ds)
        assert not store.exists(desc)
        with pytest.raises(FileNotFoundError):
            store.fetch(desc)

        ref = store.publish(desc, _weight_file(tmp_path / "w.nc"))
        assert ref == desc.ref
        assert store.exists(desc) and store.exists(ref) and store.exists(ref.ref)
        assert store.fetch(desc) == tmp_path / "store" / ref.ref
        assert store.url(desc) == f"file://{tmp_path}/store/{ref.ref}"

    def test_publish_leaves_no_partial_files(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path / "store")
        ref = store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))
        siblings = list((tmp_path / "store" / ref.ref).parent.iterdir())
        assert [p.name for p in siblings] == [f"{ref.key}.nc"]

    def test_read_only(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path, read_only=True)
        with pytest.raises(PermissionError, match="read-only"):
            store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))

    def test_find(self, regular_ds: xr.Dataset, unstructured_ds: xr.Dataset, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path / "store")
        w = _weight_file(tmp_path / "w.nc")
        regular_2 = store.publish(_desc(regular_ds, level=2), w)
        regular_3 = store.publish(_desc(regular_ds, level=3), w)
        unstructured = store.publish(_desc(unstructured_ds, level=2, method="nearest"), w)
        (tmp_path / "store" / "not-a-weight-file.nc").touch()  # ignored

        assert set(store.find()) == {regular_2, regular_3, unstructured}
        assert store.find(level=3) == [regular_3]
        assert store.find(method="nearest") == [unstructured]
        assert set(store.find(key=regular_2.key)) == {regular_2, regular_3}  # one grid, all targets
        with pytest.raises(TypeError, match="Unknown filters"):
            store.find(resolution=2)


class TestRemoteStore:
    def test_publish_and_fetch_via_local_cache(
        self, regular_ds: xr.Dataset, tmp_path: Path, memory_fs: fsspec.AbstractFileSystem
    ) -> None:
        store = WeightsStore("memory://weights", cache_dir=tmp_path / "cache")
        ref = store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))
        assert store.url(ref) == f"memory:///weights/{ref.ref}"

        local = store.fetch(ref)
        assert local == tmp_path / "cache" / ref.ref  # cache mirrors the store layout
        with xr.open_dataset(local) as ds:
            assert float(ds["S"][0]) == 1.0

        memory_fs.rm(f"/weights/{ref.ref}")
        assert store.fetch(ref) == local  # served from the local cache, no download
        assert not list((tmp_path / "cache").rglob("*.part"))

    def test_fetch_missing(self, regular_ds: xr.Dataset, tmp_path: Path, memory_fs: fsspec.AbstractFileSystem) -> None:
        store = WeightsStore("memory://weights", cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.fetch(_desc(regular_ds))

    def test_find(self, regular_ds: xr.Dataset, tmp_path: Path, memory_fs: fsspec.AbstractFileSystem) -> None:
        store = WeightsStore("memory://weights", cache_dir=tmp_path)
        ref = store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))
        assert store.find(level=2) == [ref]
