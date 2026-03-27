"""Shared test fixtures for grid-doctor."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _xy_to_latlon(
    x: np.ndarray,
    y: np.ndarray,
    grid: Literal["regular", "curvilinear"] = "regular",
) -> dict[str, object]:
    if grid == "regular":
        return {
            "lat": np.linspace(-90.0, 90.0, len(y)),
            "lon": np.linspace(0.0, 360.0, len(x), endpoint=False),
        }
    lon2d, lat2d = np.meshgrid(x, y)
    return {
        "lat": (("y", "x"), np.clip(lat2d + 2.0 * np.sin(np.deg2rad(lon2d)), -90.0, 90.0)),
        "lon": (("y", "x"), (lon2d + 2.0 * np.cos(np.deg2rad(lat2d))) % 360.0),
    }


def _make_structured_dataset(
    grid: Literal["regular", "curvilinear"] = "regular",
    ny: int = 12,
    nx: int = 24,
) -> xr.Dataset:
    dims = {"time": 2, "level": 2, "y": ny, "x": nx}
    shape = tuple(dims.values())
    y = np.linspace(-90.0, 90.0, ny)
    x = np.linspace(0.0, 360.0, nx, endpoint=False)

    coords: dict[str, object] = {
        "time": np.arange(dims["time"]),
        "level": np.arange(dims["level"]),
    }
    coords |= _xy_to_latlon(x, y, grid=grid)

    return xr.Dataset(
        {
            "temperature": (("time", "level", "y", "x"), np.random.default_rng(0).random(shape).astype("float32")),
            "pressure": (("time", "level", "y", "x"), (1000.0 + 50.0 * np.random.default_rng(1).random(shape)).astype("float64")),
            "static": (("level",), np.array([1.0, 2.0], dtype=np.float64)),
        },
        coords=coords,
        attrs={"source": "synthetic", "grid_type": grid},
    )


def _make_era5_dataset() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=3, freq="6h")
    lon = np.linspace(0.0, 359.0, 16)
    lat = np.linspace(-88.0, 88.0, 8)
    shape = (len(time), len(lat), len(lon))
    return xr.Dataset(
        data_vars={
            "tas": (("time", "lat", "lon"), np.random.default_rng(2).random(shape).astype("float32")),
            "pr": (("time", "lat", "lon"), np.random.default_rng(3).random(shape).astype("float32")),
        },
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"source": "era5-synthetic"},
    )


def _make_unstructured_dataset(ncells: int = 4, *, radians: bool = False) -> xr.Dataset:
    lon_vertices_deg = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0] + np.array([0.0, 0.0, 1.0]),
            [1.0, 1.0, 0.0] + np.array([0.0, 1.0, 1.0]),
        ],
        dtype=np.float64,
    )[:ncells]
    lat_vertices_deg = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 2.0],
        ],
        dtype=np.float64,
    )[:ncells]
    clon_deg = lon_vertices_deg.mean(axis=1)
    clat_deg = lat_vertices_deg.mean(axis=1)
    scale = np.pi / 180.0 if radians else 1.0
    rng = np.random.default_rng(42)
    return xr.Dataset(
        data_vars={
            "temperature": (
                ("time", "cell"),
                rng.standard_normal((2, ncells)).astype("float32"),
                {"CDI_grid_type": "unstructured"},
            ),
            "pressure": (
                ("time", "cell"),
                (1000.0 + rng.standard_normal((2, ncells))).astype("float64"),
                {"CDI_grid_type": "unstructured"},
            ),
            "clon_vertices": (("cell", "nv"), lon_vertices_deg * scale),
            "clat_vertices": (("cell", "nv"), lat_vertices_deg * scale),
        },
        coords={
            "time": np.arange(2),
            "clon": ("cell", clon_deg * scale),
            "clat": ("cell", clat_deg * scale),
        },
        attrs={"source": "icon-synthetic"},
    )


@pytest.fixture
def test_ds(request: pytest.FixtureRequest) -> xr.Dataset:
    if request.param == "era5":
        return _make_era5_dataset()
    return _make_structured_dataset(grid=request.param)


@pytest.fixture
def regular_ds() -> xr.Dataset:
    return _make_structured_dataset("regular")


@pytest.fixture
def curvilinear_ds() -> xr.Dataset:
    return _make_structured_dataset("curvilinear")


@pytest.fixture
def era5_ds() -> xr.Dataset:
    return _make_era5_dataset()


@pytest.fixture
def unstructured_ds() -> xr.Dataset:
    return _make_unstructured_dataset()


@pytest.fixture
def unstructured_rad_ds() -> xr.Dataset:
    return _make_unstructured_dataset(radians=True)


@pytest.fixture
def limited_area_ds() -> xr.Dataset:
    lat = np.linspace(20.0, 40.0, 5)
    lon = np.linspace(10.0, 30.0, 6)
    data = np.random.default_rng(4).random((2, 5, 6)).astype("float32")
    return xr.Dataset(
        {"temperature": (("time", "lat", "lon"), data)},
        coords={"time": np.arange(2), "lat": lat, "lon": lon},
        attrs={"source": "limited-area"},
    )


@pytest.fixture
def healpix_ds() -> xr.Dataset:
    level = 3
    npix = 12 * (4**level)
    lat = np.linspace(-90.0, 90.0, npix)
    lon = np.linspace(-180.0, 180.0, npix, endpoint=False)
    rng = np.random.default_rng(99)
    return xr.Dataset(
        {"temperature": (("time", "cell"), rng.random((2, npix)).astype("float32"))},
        coords={
            "time": np.arange(2),
            "cell": np.arange(npix),
            "latitude": ("cell", lat),
            "longitude": ("cell", lon),
        },
        attrs={"healpix_nside": 2**level, "healpix_level": level, "healpix_order": "nested"},
    )
