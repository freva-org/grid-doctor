import numpy as np
import dask.array as da
import xarray as xr
import pytest

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

#@pytest.fixture
@pytest.fixture
def test_ds(request):
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

    return _make_dataset(grid = request.param)

