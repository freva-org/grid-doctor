"""Tests for grid_doctor.misc.

Datasets should come from the shared fixtures in ``conftest.py``;
"""

import logging
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

    @TEST_DS(UNSTRUCTURED)
    def test_normalize_unstructured_dataset(self, test_ds):
        n_ds = normalize_dataset(test_ds)
        lat, lon = n_ds["clat_vertices"], n_ds["clon_vertices"]
        assert -180.0 <= lon.min() and lon.max() < 180.0
        assert -90.0 <= lat.min() and lat.max() <= 90.0
        assert lat.attrs["units"] == "degrees_north"

    @TEST_DS(STRUCTURED | MISC)
    def test_normalize_structured_dataset(self, test_ds):
        norm_ds = normalize_dataset(test_ds)
        lat, lon = (norm_ds[name] for name in _get_spatial_dims(norm_ds))
        assert np.isfinite(lon).any() and np.isfinite(lat).any()
        assert -180.0 <= lon.min() and lon.max() < 180.0
        assert -90.0 <= lat.min() and lat.max() <= 90.0

    @pytest.mark.parametrize("dims", [("rlat", "rlon"), ("y", "x")])
    def test_normalize_curvilinear_latlon(self, dims):
        """Normalize the 2-D lat/lon, not the (rotated) dimension coordinates."""
        ydim, xdim = dims
        lon2d, lat2d = np.meshgrid(np.linspace(190.0, 350.0, 4), np.linspace(-20.0, 20.0, 3))
        coords = {
            "lat": ((ydim, xdim), lat2d, {"units": "degrees_north"}),
            "lon": ((ydim, xdim), lon2d, {"units": "degrees_east"}),
        }
        if ydim == "rlat":
            coords |= {
                "rlat": (ydim, np.linspace(-10.0, 10.0, 3), {"units": "degrees"}),
                "rlon": (xdim, np.linspace(-20.0, 20.0, 4), {"units": "degrees"}),
            }
        ds = xr.Dataset(coords=coords)

        norm_ds = normalize_dataset(ds)

        np.testing.assert_allclose(norm_ds["lon"], lon2d - 360.0)
        np.testing.assert_allclose(norm_ds["lat"], lat2d)
        assert norm_ds["lon"].attrs["units"] == "degrees_east"
        assert norm_ds["lat"].attrs["units"] == "degrees_north"
        for dim in dims:
            if dim in ds.coords:
                xr.testing.assert_identical(norm_ds[dim], ds[dim])
            else:
                assert dim not in norm_ds.coords

    @pytest.mark.parametrize("with_vertices", [True, False])
    def test_normalize_unstructured_centres_and_vertices(self, with_vertices):
        """Centres and (optional) vertices end up in the same lon convention."""
        clon = np.array([10.0, 190.0, 350.0])
        clat = np.array([-30.0, 0.0, 30.0])
        coords = {
            "clat": ("cell", clat, {"units": "degrees_north"}),
            "clon": ("cell", clon, {"units": "degrees_east"}),
        }
        if with_vertices:
            offsets = np.array([-1.0, 0.0, 1.0])
            coords |= {
                "clat_vertices": (("cell", "nv"), clat[:, None] + offsets, {"units": "degrees_north"}),
                "clon_vertices": (("cell", "nv"), clon[:, None] + offsets, {"units": "degrees_east"}),
            }
        norm_ds = normalize_dataset(xr.Dataset(coords=coords))

        np.testing.assert_allclose(norm_ds["clon"], [10.0, -170.0, -10.0])
        np.testing.assert_allclose(norm_ds["clat"], clat)
        if with_vertices:
            np.testing.assert_allclose(norm_ds["clon_vertices"].mean("nv"), norm_ds["clon"])
            np.testing.assert_allclose(norm_ds["clat_vertices"].mean("nv"), norm_ds["clat"])
        else:
            assert "clon_vertices" not in norm_ds

    def test_normalize_unstructured_nan_vertices(self):
        """NaN-padded vertices (cells with fewer corners) stay NaN; real corners are normalized."""
        clat_v = np.array([[-1.0, 0.0, 1.0, np.nan], [9.0, 10.0, 11.0, 10.0]])
        clon_v = np.array([[189.0, 190.0, 191.0, np.nan], [349.0, 350.0, 351.0, 352.0]])
        ds = xr.Dataset(
            coords={
                "clat": ("cell", [0.0, 10.0], {"units": "degrees_north"}),
                "clon": ("cell", [190.0, 350.0], {"units": "degrees_east"}),
                "clat_vertices": (("cell", "nv"), clat_v, {"units": "degrees_north"}),
                "clon_vertices": (("cell", "nv"), clon_v, {"units": "degrees_east"}),
            }
        )
        norm_ds = normalize_dataset(ds)

        lat_v, lon_v = norm_ds["clat_vertices"].values, norm_ds["clon_vertices"].values
        np.testing.assert_array_equal(np.isnan(lon_v), np.isnan(clon_v))
        np.testing.assert_array_equal(np.isnan(lat_v), np.isnan(clat_v))
        np.testing.assert_allclose(lon_v, [[-171.0, -170.0, -169.0, np.nan], [-11.0, -10.0, -9.0, -8.0]])
        np.testing.assert_allclose(lat_v, clat_v)
        np.testing.assert_allclose(norm_ds["clon"], [-170.0, -10.0])

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

        if set_units:
            n_deg_ds, n_rad_ds = normalize_dataset(deg_ds), normalize_dataset(rad_ds)
        else:
            n_deg_ds = normalize_dataset(deg_ds, source_units="deg")
            n_rad_ds = normalize_dataset(rad_ds, source_units="rad")
        for n_ds in (n_deg_ds, n_rad_ds):
            assert n_ds["lon"].attrs["units"] == "degrees_east"
            assert n_ds["lat"].attrs["units"] == "degrees_north"
        xr.testing.assert_identical(n_deg_ds, n_rad_ds)

        # check lon [-180, 180)
        assert n_deg_ds["lon"].min() == -180
        assert n_deg_ds["lon"].max() < 180

    def test_normalize_missing_units_assumes_degrees(self, caplog):
        ds = xr.Dataset(coords={"lat": [0.0, 0.5, 1.0], "lon": [0.0, 1.0, 2.0]})
        with caplog.at_level(logging.WARNING, logger="grid_doctor.misc.dataset"):
            norm_ds = normalize_dataset(ds)
        np.testing.assert_allclose(norm_ds["lat"], ds["lat"])
        np.testing.assert_allclose(norm_ds["lon"], ds["lon"])
        assert "assuming degrees" in caplog.text

    def test_normalize_source_units_overrides_attrs(self, caplog):
        ds = xr.Dataset(coords={"lat": [0.0, 0.5], "lon": [0.0, 1.0]})
        for c in ds.coords:
            ds[c].attrs["units"] = "degrees"
        with caplog.at_level(logging.WARNING, logger="grid_doctor.misc.dataset"):
            norm_ds = normalize_dataset(ds, source_units="rad")
        np.testing.assert_allclose(norm_ds["lat"], np.degrees([0.0, 0.5]), atol=1e-6)
        assert not caplog.text

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
