"""Shared setup for the performance benchmarks."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from grid_doctor import remap_backend


@pytest.fixture(autouse=True)
def _no_healpix_mesh_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Benchmark real work: no disk cache, empty in-memory caches."""
    monkeypatch.setattr(remap_backend, "_HEALPIX_MESH_DISK_CACHE_MIN_LEVEL", 99)
    remap_backend._nested_healpix_mesh.cache_clear()
    remap_backend._healpix_ring_to_nest.cache_clear()
    yield
    remap_backend._nested_healpix_mesh.cache_clear()
    remap_backend._healpix_ring_to_nest.cache_clear()
