"""Tests for grid_doctor.misc.

Datasets should come from the shared fixtures in ``conftest.py``;
"""

import re

import numpy as np
import pytest
import xarray as xr

from grid_doctor.misc import _get_spatial_dims, normalize_dataset
from grid_doctor.misc.dataset import _norm_lon

from .helpers import MISC, STRUCTURED, TEST_DS, UNSTRUCTURED



class TestNormalizeHelpers:
    def test_micro_degrees(self):
        ds = xr.Dataset(
            coords={"lat": [1e-7, 1_111_111.012e-6, 12.34567890], "lon": [2e-7, 2_222_222.012e-6, 98.76543210]}
        )
        ds["lat"].attrs["units"] = "degrees_north"
        ds["lon"].attrs["units"] = "degrees_east"
        norm_ds = normalize_dataset(ds)
        np.testing.assert_array_equal(norm_ds["lat"], [0, 1_111_111e-6, 12.345679])
        np.testing.assert_array_equal(norm_ds["lon"], [0, 2_222_222e-6, 98.765432])

    @TEST_DS(STRUCTURED | MISC)
    def test_normalize_structured_dataset(self, test_ds):
        norm_ds = normalize_dataset(test_ds)
        lat, lon = (norm_ds[name] for name in _get_spatial_dims(norm_ds))
        assert np.isfinite(lon).any() and np.isfinite(lat).any()
        assert -180.0 <= lon.min() and lon.max() < 180.0
        assert -90.0 <= lat.min() and lat.max() <= 90.0

    @pytest.mark.parametrize("set_units", [False, True])
    def test_normalize_unit(self, set_units):
        coords = {"lat": [-90, -45, 0, 45, 90], "lon": [0, 90, 180, 270, 360]}
        attrs = {"test": "This should prevail"}
        deg_ds = xr.Dataset(coords=coords, attrs=attrs)
        rad_ds = xr.Dataset(coords={k: np.radians(v) for k, v in coords.items()}, attrs=attrs)
        if set_units:
            for ds, units in ((deg_ds, "deg"), (rad_ds, "rad")):
                for c in ds.coords:
                    ds[c].attrs["units"] = units

        n_deg_ds, n_rad_ds = normalize_dataset(deg_ds), normalize_dataset(rad_ds)
        for n_ds in (n_deg_ds, n_rad_ds):
            assert n_ds["lon"].attrs["units"] == "degrees_east"
            assert n_ds["lat"].attrs["units"] == "degrees_north"
        xr.testing.assert_identical(n_deg_ds, n_rad_ds)

        # check lon [-180, 180)
        assert n_deg_ds["lon"].min() == -180
        assert n_deg_ds["lon"].max() < 180

    @pytest.mark.parametrize(
        "coords,errmsg",
        [
            ({}, "Could not locate latitude/longitude"),
            ({"lat": [np.radians(360)]}, "Could not locate latitude/longitude"),
            ({"lat": [360], "lon": [0]}, "latitude outside [-90, 90]"),
            ({"lat": [91], "lon": [0]}, "latitude outside [-90, 90]"),
            ({"lat": [-91], "lon": [0]}, "latitude outside [-90, 90]"),
        ],
    )
    def test_normalize_raises(self, coords, errmsg):
        with pytest.raises(ValueError, match=re.escape(errmsg)):
            normalize_dataset(xr.Dataset(coords=coords))

    @pytest.mark.parametrize(
        "lon",
        [
            np.linspace(-180, 180, num=360 * 2),  # includes 180
            np.arange(-180, 180),  # doesn't include 180
            np.arange(-180, 181),  # includes 180
            np.arange(-180, 180, 1 + 1e-6),
            np.arange(-180, 180, 1 + 1e-9),
        ],
        ids=[
            "linspace",
            "arange",
            "arange-wrap",
            "arange-m-err",
            "arange-n-err",
        ],
    )
    @pytest.mark.parametrize("scale", [1_000_000, 1_000_000_000], ids=["micro", "nano"])
    def test_norm_lon_noop(self, lon, scale):
        n_lon = _norm_lon(lon, scale=scale)
        # last element may change because of wrapping
        np.testing.assert_allclose(lon[:-1], n_lon[:-1], atol=1 / scale)
        assert (n_lon < 180.0).all()
        assert (n_lon == -180.0).any()

    @pytest.mark.parametrize(
        "lon",
        [
            np.linspace(-360, 360, num=360 * 4),
            np.arange(-360, 360),
            np.arange(-360, 360, 1 + 1e-6),
            np.arange(-360, 360, 1 + 1e-9),
        ],
        ids=[
            "linspace",
            "arange",
            "arange-m-err",
            "arange-n-err",
        ],
    )
    @pytest.mark.parametrize("scale", [1_000_000, 1_000_000_000], ids=["micro", "nano"])
    def test_norm_lon_duplicates(self, lon, scale):
        n_lon = _norm_lon(lon, scale=scale)
        np.testing.assert_allclose(n_lon, ((lon + 180.0) % 360.0) - 180.0, atol=1 / scale)
