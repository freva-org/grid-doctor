"""Benchmarks for mesh construction and weight-key lookup.

Run with ``tox -e bench`` (see ``benchmarks/README.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap_backend as rb
from grid_doctor.remap_backend import TargetDescription, WeightsDescription, describe_source

LEVELS = [5, 7]


def _healpix_corners(level: int) -> tuple[np.ndarray, np.ndarray]:
    module, kwargs = rb._require_healpix_geo_module(True)
    lon, lat = module.vertices(np.arange(12 * 4**level), level, **kwargs)
    return rb._canonical_lon(np.asarray(lon, dtype=np.float64)), np.asarray(lat, dtype=np.float64)


def _regular_ds(nlat: int = 360, nlon: int = 720) -> xr.Dataset:
    return xr.Dataset(
        coords={
            "lat": ("lat", np.linspace(-89.75, 89.75, nlat), {"units": "degrees_north"}),
            "lon": ("lon", np.linspace(0.25, 359.75, nlon), {"units": "degrees_east"}),
        }
    )


@pytest.mark.parametrize("level", LEVELS)
def test_healpix_mesh_cold_build(benchmark, level: int) -> None:
    """First build of the nested target mesh (vertices + de-duplication)."""
    benchmark.pedantic(
        lambda: rb._target_healpix_mesh(level, nest=True),
        setup=rb._nested_healpix_mesh.cache_clear,
        rounds=3,
    )


@pytest.mark.parametrize("level", LEVELS)
def test_healpix_mesh_ring_from_nested(benchmark, level: int) -> None:
    """Ring mesh derived from an already cached nested mesh."""
    rb._target_healpix_mesh(level, nest=True)
    benchmark.pedantic(
        lambda: rb._target_healpix_mesh(level, nest=False),
        setup=rb._healpix_ring_to_nest.cache_clear,
        rounds=5,
    )


@pytest.mark.parametrize("level", LEVELS)
def test_corner_mesh_from_arrays(benchmark, level: int) -> None:
    """Node de-duplication, the dominant cost of any mesh build."""
    lon, lat = _healpix_corners(level)
    benchmark.pedantic(rb._corner_mesh_from_arrays, args=(lon, lat), rounds=3)


def test_collapse_repeated_corners(benchmark) -> None:
    face_nodes = np.random.default_rng(0).integers(-1, 1000, size=(500_000, 6)).astype(np.int32)
    benchmark(rb._collapse_repeated_corners, face_nodes)


def test_source_mesh_regular_quarter_degree(benchmark) -> None:
    ds = _regular_ds()
    benchmark.pedantic(lambda: describe_source(ds).source_mesh, rounds=3)


def test_weights_key_lookup(benchmark) -> None:
    """What ``cached_weights`` pays on a cache hit: must stay mesh-free."""
    ds = _regular_ds()

    def key() -> str:
        return WeightsDescription(
            source=describe_source(ds),
            target=TargetDescription(level=10, order="nest"),
            method="conservative",
        ).key

    benchmark(key)
