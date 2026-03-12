"""Shared test fixtures for grid-doctor."""

from __future__ import annotations

from typing import Literal

import dask.array as da
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
            "lat": np.linspace(-90, 90, len(y)),
            "lon": np.linspace(0, 360, len(x), endpoint=False),
        }
    elif grid == "curvilinear":
        lon2d, lat2d = np.meshgrid(x, y)
        return {
            "lat": (
                ("y", "x"),
                np.clip(lat2d + 2 * np.sin(np.deg2rad(lon2d)), -90, 90),
            ),
            "lon": (("y", "x"), (lon2d + 2 * np.cos(np.deg2rad(lat2d))) % 360),
        }
    raise ValueError(f"Unknown grid type: {grid}")


def _make_structured_dataset(
    grid: Literal["regular", "curvilinear"] = "regular",
    ny: int = 64,
    nx: int = 128,
) -> xr.Dataset:
    dims = {"time": 2, "level": 2, "y": ny, "x": nx}
    shape = tuple(dims.values())
    chunks = (1, 1, -1, -1)

    y = np.linspace(-90, 90, ny)
    x = np.linspace(0, 360, nx, endpoint=False)

    coords: dict[str, object] = {
        "time": np.arange(dims["time"]),
        "level": np.arange(dims["level"]),
    }
    coords |= _xy_to_latlon(x, y, grid=grid)

    lat_vals = np.linspace(-90, 90, ny)
    lon_vals = np.linspace(0, 360, nx, endpoint=False)
    lat_bnds = np.column_stack([lat_vals - 0.5, lat_vals + 0.5])
    lon_bnds = np.column_stack([lon_vals - 0.5, lon_vals + 0.5])

    ds = xr.Dataset(
        {
            "temperature": (
                list(dims.keys()),
                da.random.random(shape, chunks=chunks).astype("float32"),
            ),
            "pressure": (
                list(dims.keys()),
                (1000 + 50 * da.random.random(shape, chunks=chunks)).astype(
                    "float64"
                ),
            ),
            "lat_bnds": (("lat", "bnds"), lat_bnds),
            "lon_bnds": (("lon", "bnds"), lon_bnds),
        },
        coords=coords,
        attrs={"source": "synthetic", "grid_type": grid},
    )
    return ds


def _make_era5_dataset() -> xr.Dataset:
    time = pd.date_range("2020-01-01", periods=3, freq="6h")
    lon = np.linspace(0.0, 359.71875, 128)
    lat = np.linspace(-89.78125, 89.78125, 64)

    shape = (len(time), len(lat), len(lon))
    chunks = (1, 64, 128)

    lat_bnds = np.column_stack([lat - 0.14, lat + 0.14])
    lon_bnds = np.column_stack([lon - 0.14, lon + 0.14])

    ds = xr.Dataset(
        data_vars={
            "tas": (
                ("time", "lat", "lon"),
                da.random.random(shape, chunks=chunks).astype("float32"),
            ),
            "pr": (
                ("time", "lat", "lon"),
                da.random.random(shape, chunks=chunks).astype("float32"),
            ),
            "lat_bnds": (("lat", "bnds"), lat_bnds),
            "lon_bnds": (("lon", "bnds"), lon_bnds),
        },
        coords={"time": time, "lon": lon, "lat": lat, "bnds": np.arange(2)},
        attrs={"source": "era5-synthetic"},
    )
    return ds


def _make_unstructured_dataset(
    ncells: int = 5000,
    dim_name: str = "cell",
) -> xr.Dataset:
    """Create a synthetic ICON-like unstructured dataset."""
    rng = np.random.default_rng(42)
    clat = rng.uniform(-90, 90, ncells).astype(np.float64)
    clon = rng.uniform(0, 360, ncells).astype(np.float64)

    shape = (3, ncells)
    chunks = (1, -1)

    ds = xr.Dataset(
        data_vars={
            "temperature": (
                ("time", dim_name),
                da.from_array(
                    rng.standard_normal(shape).astype("float32"), chunks=chunks
                ),
                {"CDI_grid_type": "unstructured"},
            ),
            "pressure": (
                ("time", dim_name),
                da.from_array(
                    (1000 + 50 * rng.standard_normal(shape)).astype("float64"),
                    chunks=chunks,
                ),
                {"CDI_grid_type": "unstructured"},
            ),
        },
        coords={
            "time": np.arange(3),
            "clat": (dim_name, clat),
            "clon": (dim_name, clon),
        },
        attrs={"source": "icon-synthetic"},
    )
    return ds


@pytest.fixture
def test_ds(request: pytest.FixtureRequest) -> xr.Dataset:
    """Parametrised fixture: 'regular', 'curvilinear', 'era5'."""
    if request.param == "era5":
        return _make_era5_dataset()
    return _make_structured_dataset(grid=request.param)


@pytest.fixture
def regular_ds() -> xr.Dataset:
    return _make_structured_dataset(grid="regular")


@pytest.fixture
def curvilinear_ds() -> xr.Dataset:
    return _make_structured_dataset(grid="curvilinear")


@pytest.fixture
def era5_ds() -> xr.Dataset:
    return _make_era5_dataset()


@pytest.fixture
def unstructured_ds() -> xr.Dataset:
    """ICON-like dataset with 'cell' dimension and clat/clon."""
    return _make_unstructured_dataset(ncells=5000, dim_name="cell")


@pytest.fixture
def unstructured_ncells_ds() -> xr.Dataset:
    """ICON-like dataset with 'ncells' dimension and lat/lon."""
    ds = _make_unstructured_dataset(ncells=3000, dim_name="ncells")
    return ds.rename({"clat": "lat", "clon": "lon"})


@pytest.fixture
def healpix_ds() -> xr.Dataset:
    """A pre-built HEALPix dataset at level 3 for coarsening tests."""
    import healpy as hp

    level = 3
    nside = 2**level
    npix = hp.nside2npix(nside)

    hp_theta, hp_phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    hp_lat = 90.0 - np.degrees(hp_theta)
    hp_lon = np.degrees(hp_phi)

    rng = np.random.default_rng(99)
    ds = xr.Dataset(
        {
            "temperature": (
                ("time", "cell"),
                rng.random((2, npix)).astype("float32"),
            )
        },
        coords={
            "time": np.arange(2),
            "cell": np.arange(npix),
            "latitude": ("cell", hp_lat),
            "longitude": ("cell", hp_lon),
        },
        attrs={
            "healpix_nside": nside,
            "healpix_level": level,
            "healpix_order": "nested",
        },
    )
    return ds
