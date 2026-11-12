"""Tests for the WeightsStore."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from itertools import product
from pathlib import Path

import fsspec
import numpy as np
import pytest
import xarray as xr

from grid_doctor.metadata import HealpixGrid
from grid_doctor.weights import (
    WEIGHTS_STORE_ENV,
    WeightsDescription,
    WeightsRef,
    WeightsStore,
    describe_source,
)


def _desc(ds: xr.Dataset, level: int = 2, method: str = "conservative") -> WeightsDescription:
    return WeightsDescription(describe_source(ds), HealpixGrid(level=level), method)  # type: ignore[arg-type]


def _weight_file(path: Path, value: float = 1.0, level: int = 2) -> Path:
    """A minimal nested-order weight file: target cell 1 gets source cell 1."""
    from grid_doctor.metadata import _keys as keys

    data = {"row": ("nnz", np.array([1])), "col": ("nnz", np.array([1])), "S": ("nnz", np.array([value]))}
    xr.Dataset(data, attrs={keys.LEVEL_ATTR: level, keys.ORDER_ATTR: "nested"}).to_netcdf(path)
    return path


@pytest.fixture
def memory_fs() -> Iterator[fsspec.AbstractFileSystem]:
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    yield fs
    fs.store.clear()


class TestWeightsStore:
    """Every spelling of one location is the same store, and instances share its contents."""

    URI = "memory://weights"

    @pytest.fixture(autouse=True)
    def _default_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Keep the default cache in ``tmp_path``, never in the user's cache."""
        monkeypatch.setattr("grid_doctor.utils.cache_dir", lambda: tmp_path)

    @pytest.mark.parametrize(
        ("location", "url_root"),
        [
            pytest.param("/data/weights", "file:///data/weights", id="path-str"),
            pytest.param(Path("/data/weights"), "file:///data/weights", id="path"),
            pytest.param("/data/weights/", "file:///data/weights", id="path-slash"),
            pytest.param("file:///data/weights", "file:///data/weights", id="file-url"),
            pytest.param("memory://weights", "memory:///weights", id="url"),
            pytest.param("memory://weights/", "memory:///weights", id="url-slash"),
            pytest.param(fsspec.get_mapper("memory://weights"), "memory:///weights", id="mapper"),
        ],
    )
    def test_spellings_resolve_to_one_root(self, location: str | Path | fsspec.FSMap, url_root: str) -> None:
        store = WeightsStore(location)
        assert store.url_root == url_root
        assert store.is_local == url_root.startswith("file://")

    def test_different_roots_differ(self, tmp_path: Path) -> None:
        assert WeightsStore(tmp_path / "a").url_root != WeightsStore(tmp_path / "b").url_root

    @pytest.mark.parametrize(
        ("cache_dir", "shared"),
        [
            pytest.param(None, True, id="default-cache"),
            pytest.param("cache", True, id="same-cache"),
            pytest.param("dif-cache", False, id="different-cache"),
        ],
    )
    def test_instances_share_contents(
        self,
        cache_dir: str | None,
        shared: bool,
        regular_ds: xr.Dataset,
        tmp_path: Path,
        memory_fs: fsspec.AbstractFileSystem,
    ) -> None:
        writer_c = reader_c = None
        if cache_dir:
            writer_c = reader_c = tmp_path / cache_dir
            if not shared:
                writer_c = writer_c / 'writer'
                reader_c = reader_c / 'reader'
        writer = WeightsStore(self.URI, cache_dir=writer_c)
        reader = WeightsStore(self.URI, cache_dir=reader_c)

        desc = _desc(regular_ds)
        assert not reader.exists(desc)
        writer.publish(desc, _weight_file(tmp_path / "w.nc"))
        assert reader.exists(desc)

        local = writer.fetch(desc)
        assert local == writer.cache_dir / desc.ref.path

        memory_fs.rm(f"/weights/{desc.ref.path}")
        # the reader can now only use a local copy
        if shared:
            # removed actual the store location but still cached on shared cache
            assert reader.fetch(desc) == local
        else:
            with pytest.raises(FileNotFoundError):
                reader.fetch(desc)

            writer.fetch(desc) # still cached on writer


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
        assert store.exists(desc) and store.exists(ref) and store.exists(ref.path)
        assert store.fetch(desc) == tmp_path / "store" / ref.path
        assert store.url(desc) == f"file://{tmp_path}/store/{ref.path}"

    def test_publish_leaves_no_partial_files(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path / "store")
        ref = store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))
        siblings = list((tmp_path / "store" / ref.path).parent.iterdir())
        assert [p.name for p in siblings] == [f"{ref.key}.nc"]

    @pytest.mark.skipif(os.name != "posix", reason="needs POSIX permissions")
    def test_published_file_follows_umask(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        store = WeightsStore(tmp_path / "store")
        old = os.umask(0o022)
        try:
            ref = store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))
        finally:
            os.umask(old)
        assert (tmp_path / "store" / ref.path).stat().st_mode & 0o777 == 0o644  # not mkstemp's 0600

    @pytest.mark.skipif(os.name != "posix", reason="needs POSIX permissions")
    @pytest.mark.parametrize("via", ["publish", "compute"])
    def test_new_dirs_copy_root_mode(self, via: str, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        """A shared root (sticky, world-writable) stays shared all the way down."""
        root = tmp_path / "store"
        root.mkdir()
        root.chmod(0o1777)
        store, desc = WeightsStore(root, cache_dir=tmp_path / "cache"), _desc(regular_ds)
        if via == "publish":
            store.publish(desc, _weight_file(tmp_path / "w.nc"))
        else:
            store.fetch_or_compute(desc, lambda path: _weight_file(path))
        new_dirs = (root / desc.ref.path).parents[: len(WeightsRef.HIERARCHY)]
        assert {d.stat().st_mode & 0o7777 for d in new_dirs} == {0o1777}

    @staticmethod
    def _sticky(*args: object) -> None:
        raise PermissionError("Operation not permitted")

    def test_publish_keeps_file_of_another_user(
        self, regular_ds: xr.Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In a sticky directory another user's file cannot be replaced; it holds the same weights."""
        store, desc = WeightsStore(tmp_path / "store"), _desc(regular_ds)
        theirs = store.local_path(desc)
        theirs.parent.mkdir(parents=True)
        _weight_file(theirs, value=2.0)
        monkeypatch.setattr("grid_doctor.weights.os.replace", self._sticky)
        assert store.publish(desc, _weight_file(tmp_path / "w.nc")) == desc.ref
        with xr.open_dataset(theirs) as ds:
            assert float(ds["S"][0]) == 2.0  # theirs, untouched
        assert not list(theirs.parent.glob("*.part"))

    def test_compute_keeps_file_of_another_user(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        """Another user finished the same weights first; their file is used."""
        store, desc = WeightsStore(tmp_path / "store", cache_dir=tmp_path / "cache"), _desc(regular_ds)

        def compute(path: Path) -> Path:
            _weight_file(path)  # the other user's file lands first...
            self._sticky()  # ...so ours cannot replace it
            return path

        assert store.fetch_or_compute(desc, compute) == store.local_path(desc)

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
        assert store.find(method="Nearest") == [unstructured]  # like WeightsRef, case-insensitive
        assert set(store.find(key=regular_2.key)) == {regular_2, regular_3}  # one grid, all targets
        with pytest.raises(TypeError, match="Unknown filters"):
            store.find(resolution=2)


    def test_hive_globs(self, regular_ds: xr.Dataset, unstructured_ds: xr.Dataset, tmp_path: Path) -> None:
        """Plain globs, without grid-doctor, find all targets of a grid and all grids of a target.

        The same grid has the same digest at every level and method, so its
        file name is the same in every target directory.
        """
        root = tmp_path / "store"
        store = WeightsStore(root)
        w = _weight_file(tmp_path / "w.nc")
        source_desc = describe_source(regular_ds)
        targets = [
            WeightsDescription(source_desc, HealpixGrid(level=level), method)  # type: ignore[arg-type]
            for level, method in product((2, 3), ("nearest", "conservative"))
        ]
        other = WeightsDescription(describe_source(unstructured_ds), HealpixGrid(level=2), "conservative")
        for desc in (*targets, other):
            store.publish(desc, w)

        assert {d.ref.key for d in targets} == {source_desc.digest}

        # All targets of one grid: any value in every directory level, fixed file name.
        any_target = "/".join(["*"] * len(WeightsRef.HIERARCHY))
        assert sorted(root.glob(f"{any_target}/{source_desc.digest}.nc")) == sorted(root / d.ref.path for d in targets)
       # All grids of one target: fixed directories, any file name.
        target = "version=1/grid=healpix/ellipsoid=sphere/level=2/method=conservative"
        assert {p.stem for p in root.glob(f"{target}/*.nc")} == {source_desc.digest, other.source.digest}


class TestRemoteStore:
    def test_publish_and_fetch_via_local_cache(
        self, regular_ds: xr.Dataset, tmp_path: Path, memory_fs: fsspec.AbstractFileSystem
    ) -> None:
        store = WeightsStore("memory://weights", cache_dir=tmp_path / "cache")
        ref = store.publish(_desc(regular_ds), _weight_file(tmp_path / "w.nc"))
        assert store.url(ref) == f"memory:///weights/{ref.path}"

        local = store.fetch(ref)
        assert local == tmp_path / "cache" / ref.path  # cache mirrors the store layout
        with xr.open_dataset(local) as ds:
            assert float(ds["S"][0]) == 1.0

        memory_fs.rm(f"/weights/{ref.path}")
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


class TestReadOnlyStore:
    """Stores that cannot be written: weights are computed into, and reused from, ``cache_dir``."""

    @staticmethod
    def _compute(calls: list[Path]) -> Callable[[Path], Path]:
        def compute(path: Path) -> Path:
            calls.append(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            return _weight_file(path)

        return compute

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs POSIX permissions, not root")
    def test_unwritable_directory_is_read_only(self, tmp_path: Path) -> None:
        root = tmp_path / "store"
        root.mkdir()
        root.chmod(0o555)
        try:
            assert WeightsStore(root).read_only
            assert WeightsStore(root / "not" / "created" / "yet").read_only  # nearest parent decides
        finally:
            root.chmod(0o755)
        assert not WeightsStore(tmp_path / "new").read_only

    def test_existing_weights_are_used(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        desc = _desc(regular_ds)
        WeightsStore(tmp_path / "store").publish(desc, _weight_file(tmp_path / "w.nc"))
        store = WeightsStore(tmp_path / "store", cache_dir=tmp_path / "cache", read_only=True)
        calls: list[Path] = []
        assert store.fetch_or_compute(desc, self._compute(calls)) == tmp_path / "store" / desc.ref.path
        assert not calls

    def test_missing_weights_are_computed_into_cache(self, regular_ds: xr.Dataset, tmp_path: Path) -> None:
        desc = _desc(regular_ds)
        store = WeightsStore(tmp_path / "store", cache_dir=tmp_path / "cache", read_only=True)
        calls: list[Path] = []
        path = store.fetch_or_compute(desc, self._compute(calls))
        assert path == tmp_path / "cache" / desc.ref.path
        assert not store.exists(desc)  # store untouched
        assert store.fetch_or_compute(desc, self._compute(calls)) == path  # reused from the cache
        assert len(calls) == 1
