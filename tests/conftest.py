import numpy as np
import dask.array as da
import xarray as xr
import pytest
import pandas as pd

from typing import Literal, Dict

def _xy_to_latlon(x, y, grid: Literal['regular','curvilinear'] = 'regular') -> Dict | None:
    match grid:
        case 'regular':
            return {
                "lat": np.linspace(-90, 90, len(x)),
                "lon": np.linspace(0, 360, len(y), endpoint=False),
            }
        case 'curvilinear':
            lon2d, lat2d = np.meshgrid(x, y)
            return {
                "lat": (('x','y'), lat2d + 5 * np.sin(np.deg2rad(lon2d))),
                "lon": (('x','y'), lon2d + 2 * np.cos(np.deg2rad(lat2d))),
            }

    return None


def era5(
    start="1940-01-01",
    end="2026-02-05T23:00:00",
    freq="h",
) -> xr.Dataset:
    rand = da.random.random

    time = pd.date_range(start, end, freq=freq)
    lon = np.linspace(0.0, 359.71875, 1280)
    lat = np.linspace(-89.78125, 89.78125, 640)
    bnds = np.arange(2)

    tas = rand(
        (len(time), len(lat), len(lon)),
        chunks=(40, 640, 1280),
    ).astype("float32")

    pr = rand(
        (len(time), len(lat), len(lon)),
        chunks=(39, 640, 1280),
    ).astype("float32")

    lon_bnds = rand(
        (len(time), len(lon), 2),
        chunks=(729, 1280, 2),
    )

    lat_bnds = rand(
        (len(time), len(lat), 2),
        chunks=(729, 640, 2),
    )

    ds = xr.Dataset(
        data_vars={
            "tas": (("time", "lat", "lon"), tas),
            "pr": (("time", "lat", "lon"), pr),
            "lon_bnds": (("time", "lon", "bnds"), lon_bnds),
            "lat_bnds": (("time", "lat", "bnds"), lat_bnds),
        },
        coords={
            "time": time,
            "lon": lon,
            "lat": lat,
            "bnds": bnds,
        },
        attrs={
            "description": "Synthetic dataset matching requested structure"
        },
    )

    return ds

def _make_dataset(grid=Literal['regular','curvilinear']) -> xr.Dataset:
    dims = {
        "time": 4,
        "level": 3,
        "y": 512,
        "x": 512,
    }

    shape = tuple(dims.values())
    chunks = (1, 1, -1,-1)

    y = np.linspace(-90, 90, dims["y"])
    x = np.linspace(0, 360, dims["x"], endpoint=False)
    coords = {
        "time": np.arange(dims["time"]),
        "level": np.arange(dims["level"]),
    }
    coords |= _xy_to_latlon(x,y, grid=grid)

    ds = xr.Dataset(
        {
            "temperature": (
                dims.keys(),
                da.random.random(shape, chunks=chunks).astype("float32"),
            ),
            "pressure": (
                dims.keys(),
                (1000 + 50 * da.random.random(shape, chunks=chunks)).astype("float64"),
            ),
            "quality_flag": (
                dims.keys(),
                da.random.randint(0, 5, shape, chunks=chunks).astype("int16"),
            ),
            "land_mask": (
                dims.keys(),
                (da.random.random(shape, chunks=chunks) > 0.7),
            ),
        },
        coords=coords,
    )

    return ds
    
@pytest.fixture
def test_ds(request):
    if request.param == 'era5':
        return era5()
    else:
        return _make_dataset(grid = request.param)

