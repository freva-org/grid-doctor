"""Tests for the source description and its digest."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor.remap_backend import describe_source
from grid_doctor.weights import _source_hash


def _wrf_ds(offset: float = 0.0, lat: str = "XLAT", lon: str = "XLONG") -> xr.Dataset:
    xlong, xlat = np.meshgrid(np.linspace(0.0, 10.0, 4) + offset, np.linspace(40.0, 50.0, 3))
    return xr.Dataset(
        coords={lat: (("south_north", "west_east"), xlat), lon: (("south_north", "west_east"), xlong)}
    )


def _rename_vertices(ds: xr.Dataset, scheme: str) -> tuple[xr.Dataset, str, str]:
    """Rename ICON-style vertices to ``lat_bnds`` or CF ``bounds``; return ``(ds, lat, lon)``."""
    lat_v, lon_v = ("lat_bnds", "lon_bnds") if scheme == "bnds" else ("my_lat_b", "my_lon_b")
    ds = ds.rename({"clat_vertices": lat_v, "clon_vertices": lon_v})
    if scheme == "cf-bounds":
        ds["clat"].attrs["bounds"] = lat_v
        ds["clon"].attrs["bounds"] = lon_v
    return ds, lat_v, lon_v


class TestSourceDescription:
    def test_eq_and_hash_use_digest(self, regular_ds: xr.Dataset) -> None:
        a, b = describe_source(regular_ds), describe_source(regular_ds)
        assert a is not b
        assert a == b  # no ValueError from comparing datasets
        assert hash(a) == hash(b) == hash(a.digest)  # no TypeError from hashing datasets
        assert len({a, b}) == 1
        assert a != "not a description"
        assert "dataset" not in repr(a)


class TestSourceHash:
    def test_different_grids_differ(self) -> None:
        assert _source_hash(_wrf_ds()) != _source_hash(_wrf_ds(offset=1.0))

    def test_names_do_not_matter(self) -> None:
        assert _source_hash(_wrf_ds(lat="lat", lon="lon")) == _source_hash(_wrf_ds())

    def test_float32_and_float64_match(self) -> None:
        f64 = _wrf_ds(offset=0.1)
        f32 = f64.assign_coords({k: f64[k].astype(np.float32) for k in ("XLAT", "XLONG")})
        f64 = f32.assign_coords({k: f32[k].astype(np.float64) for k in ("XLAT", "XLONG")})
        assert _source_hash(f32) == _source_hash(f64)

    def test_lat_lon_order_matters(self) -> None:
        ds = _wrf_ds()
        swapped = ds.rename({"XLAT": "tmp"}).rename({"XLONG": "XLAT", "tmp": "XLONG"})
        assert _source_hash(ds) != _source_hash(swapped)

    def test_shape_is_hashed(self) -> None:
        ds = _wrf_ds()
        flat = xr.Dataset(coords={k: ("cell", ds[k].values.ravel()) for k in ("XLAT", "XLONG")})
        assert _source_hash(ds) != _source_hash(flat)

    def test_shared_dims_are_hashed(self) -> None:
        """Whether lat and lon share a dimension changes the grid, not only their values.

        The same four values make a 4x4 regular grid (16 cells) on separate
        ``lat``/``lon`` dims, but 4 unstructured points on a shared ``cell`` dim.
        """
        v = np.array([0.0, 10.0, 20.0, 30.0])
        regular = xr.Dataset(coords={"lat": ("lat", v), "lon": ("lon", v)})
        unstructured = xr.Dataset(coords={"lat": ("cell", v), "lon": ("cell", v)})
        assert _source_hash(regular) != _source_hash(unstructured)

    def test_vertices_are_hashed(self) -> None:
        clat, clon = np.array([0.0, 10.0]), np.array([0.0, 20.0])
        base = {"clat": ("cell", clat), "clon": ("cell", clon)}

        def verts(d: float) -> dict[str, tuple[tuple[str, str], np.ndarray]]:
            return {
                "clat_vertices": (("cell", "nv"), clat[:, None] + [-d, 0.0, d]),
                "clon_vertices": (("cell", "nv"), clon[:, None] + [-d, 0.0, d]),
            }

        hashes = {
            _source_hash(xr.Dataset(coords=base)),
            _source_hash(xr.Dataset(coords=base | verts(1.0))),
            _source_hash(xr.Dataset(coords=base | verts(2.0))),
        }
        assert len(hashes) == 3

    @pytest.mark.parametrize("scheme", ["bnds", "cf-bounds"])
    def test_vertices_are_hashed_for_all_names(self, unstructured_ds: xr.Dataset, scheme: str) -> None:
        ds, lat_v, _ = _rename_vertices(unstructured_ds, scheme)
        shifted = ds.assign({lat_v: ds[lat_v] + 1.0})
        assert _source_hash(ds) != _source_hash(shifted)
        assert _source_hash(ds) == _source_hash(unstructured_ds)

    def test_signed_zero_matches(self) -> None:
        ds = _wrf_ds()
        lon = ds["XLONG"].values.copy()
        lon[lon == 0.0] = -0.0
        signed = ds.assign_coords(XLONG=(ds["XLONG"].dims, lon))
        assert _source_hash(ds) == _source_hash(signed)

    def test_nan_payloads_match(self) -> None:
        """NaNs with different bit patterns must not change the key.

        Many bit patterns are NaN (``np.nan`` is ``0x7ff8000000000000``,
        ``0x7ff8000000000001`` is one too). They compare as NaN but their
        bytes differ, so hashing raw bytes would give one grid two keys.
        """
        ds = _wrf_ds()
        lat = ds["XLAT"].values.copy()
        lat.flat[0] = np.nan
        quiet = ds.assign_coords(XLAT=(ds["XLAT"].dims, lat))
        lat = lat.copy()
        lat.view(np.uint64).flat[0] |= 1  # still a valid NaN
        payload = ds.assign_coords(XLAT=(ds["XLAT"].dims, lat))

        assert np.isnan(payload["XLAT"].values.flat[0])
        assert quiet["XLAT"].values.tobytes() != payload["XLAT"].values.tobytes()
        assert _source_hash(quiet) == _source_hash(payload)

    def test_hash_is_pinned(self) -> None:
        """The same grid gets the same hash on any host byte order.

        Update *expected* only when the hash scheme changes on purpose.
        """
        ds = _wrf_ds()
        big = ds.assign_coords({k: ds[k].astype(">f8") for k in ("XLAT", "XLONG")})
        expected = "5965fbcec230c4066fae8d21eec71431c9bf4913e930b80622f16e76882bb589"
        assert _source_hash(ds) == _source_hash(big) == expected

    def test_raises_without_coordinates(self) -> None:
        with pytest.raises(ValueError, match="Could not locate latitude/longitude"):
            _source_hash(xr.Dataset(coords={"x": [0, 1]}))



class TestCanonicalSourceDigest:
    """The digest is computed from the normalised geometry, so conventions do not matter."""

    LON = np.arange(8) * 45.0  # 0, 45, ..., 315

    @staticmethod
    def _grid(lon: np.ndarray = LON, radians: bool = False) -> xr.Dataset:
        lat = np.linspace(-80.0, 80.0, 5)
        if radians:
            lat, lon = np.radians(lat), np.radians(lon)
        units = "rad" if radians else "degrees"
        return xr.Dataset(coords={"lat": ("lat", lat, {"units": units}), "lon": ("lon", lon, {"units": units})})

    @staticmethod
    def _digest(ds: xr.Dataset) -> str:
        return describe_source(ds).digest

    def test_same_grid_any_convention_shares_digest(self) -> None:
        ref = self._digest(self._grid())
        assert self._digest(self._grid(radians=True)) == ref  # radians vs degrees
        assert self._digest(self._grid(self.LON + 360.0)) == ref  # lon shifted by a full turn

    def test_0_360_and_minus180_180_share_digest(self) -> None:
        """The same cells in the same order share weights, whatever the lon convention."""
        lon_180 = (self.LON + 180.0) % 360.0 - 180.0  # 0, ..., 135, -180, ..., -45
        assert self._digest(self._grid(lon_180)) == self._digest(self._grid())

    def test_reordered_cells_get_own_digest(self) -> None:
        """Weight columns follow the data's cell order, so a sorted grid must not reuse weights."""
        lon_sorted = np.sort((self.LON + 180.0) % 360.0 - 180.0)  # -180, ..., 135
        assert self._digest(self._grid(lon_sorted)) != self._digest(self._grid())
