"""Tests for the weight descriptions and references."""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from grid_doctor.metadata import HealpixGrid
from grid_doctor.remap_backend import _source_mesh
from grid_doctor.weights import (
    WEIGHTS_VERSION,
    WeightsDescription,
    WeightsRef,
    _source_hash,
    describe_source,
    nested_to_ring,
    ring_to_nested,
)


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

    def test_description_is_lazy(self, regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
        """Describing a source and taking its digest must not build the mesh."""
        import grid_doctor.weights as weights_module

        def boom(*args: object, **kwargs: object) -> None:
            raise AssertionError("mesh built eagerly")

        monkeypatch.setattr(weights_module, "_source_mesh", boom)
        desc = describe_source(regular_ds)
        assert desc.digest
        with pytest.raises(AssertionError, match="eagerly"):
            desc.source_mesh  # noqa: B018

    def test_mesh_is_built_once(self, regular_ds: xr.Dataset) -> None:
        desc = describe_source(regular_ds)
        assert desc.source_mesh is desc.source_mesh
        assert set(desc.source_dims) <= set(regular_ds.dims)
        assert desc.metadata["grid_doctor_source_size"] == desc.source_mesh.face_count

    @pytest.mark.parametrize("source_units", ["auto", "rad"])
    def test_geometry_is_described_in_degrees(
        self, unstructured_ds: xr.Dataset, unstructured_rad_ds: xr.Dataset, source_units: str
    ) -> None:
        """The same grid in radians and degrees gives the same mesh, recorded as ``"deg"``."""
        deg = describe_source(unstructured_ds)
        rad = describe_source(unstructured_rad_ds, source_units=source_units)

        for desc in (deg, rad):
            assert desc.metadata["grid_doctor_source_units"] == "deg"
        assert rad.metadata == deg.metadata
        for name in ("node_lon", "node_lat", "face_nodes", "face_lon", "face_lat"):
            np.testing.assert_allclose(getattr(rad.source_mesh, name), getattr(deg.source_mesh, name), atol=1e-6)

    @pytest.mark.parametrize(
        "lon",
        [np.arange(1.0, 360.0, 2.0), np.arange(161.0, 200.0, 2.0)],
        ids=["global-0-360", "regional-antimeridian"],
    )
    def test_regular_mesh_unchanged_by_normalisation(self, lon):
        """``describe_source`` normalises the geometry; wrapping a 1-D lon axis must not change the cells."""
        lat = np.arange(-89.0, 90.0, 2.0)
        ds = xr.Dataset(
            {"t": (("lat", "lon"), np.zeros((lat.size, lon.size)))},
            coords={
                "lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"}),
            },
        )
        raw, _ = _source_mesh(ds, source_units="deg")
        norm = describe_source(ds).source_mesh

        np.testing.assert_allclose(norm.face_lon, raw.face_lon)
        np.testing.assert_allclose(norm.face_lat, raw.face_lat)
        corners = norm.node_lon[norm.face_nodes]
        width = np.ptp(corners, axis=1)
        np.testing.assert_allclose(np.minimum(width, 360.0 - width), 2.0)


class TestWeightsDescription:
    def test_identity_per_target_and_method(self, regular_ds: xr.Dataset) -> None:
        source = describe_source(regular_ds)
        descs = [
            WeightsDescription(source, HealpixGrid(level=level, indexing_scheme=scheme), method)
            for method, level, scheme in product(("nearest", "conservative"), (0, 1), ("nested", "ring"))
        ]
        assert len(set(descs)) == len(descs)
        assert len({d.ref for d in descs}) == len(descs) // 2  # nested and ring share the nested file
        assert {d.ref.key for d in descs} == {source.digest}  # one source digest for all targets

    def test_eq_and_hash(self, regular_ds: xr.Dataset) -> None:
        def make() -> WeightsDescription:
            return WeightsDescription(describe_source(regular_ds), HealpixGrid(level=1), "nearest")

        a, b = make(), make()
        assert a is not b
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1
        assert a != "not a description"


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
        expected = "46bd11acb66b91f664ff104b273ad05b1d84a4be5d29d4107575ef24b0e45dbf"
        assert _source_hash(ds) == _source_hash(big) == expected

    def test_documented_algorithm(self) -> None:
        """Other tools can compute the same digest from the spec in the docs alone."""
        import hashlib

        lat = np.array([[-10.0, -10.0], [10.0, 10.0]])
        lon = np.array([[170.0, -170.0], [170.0, -170.0]])
        ds = xr.Dataset(coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)})

        spec = hashlib.sha256()
        for values in (lat, lon):  # centres: lat then lon; both on dims 0, 1
            spec.update(b"dims=(0, 1);shape=(2, 2);")
            spec.update(np.rint(values * 1_000_000).astype("<i8").tobytes())
        assert _source_hash(ds) == spec.hexdigest()

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

    def test_float_noise_shares_digest(self) -> None:
        """Values within half a micro-degree are the same grid (#86)."""
        noisy = self.LON + 4e-7 * np.cos(self.LON)  # |noise| < 0.5e-6 degrees
        assert self._digest(self._grid(noisy)) == self._digest(self._grid())

    def test_antimeridian_rounding_shares_digest(self) -> None:
        """179.9999999 rounds to 180 and wraps to -180, like an exact -180."""
        lon = np.array([-180.0, 0.0])
        nearly = np.array([179.9999999, 0.0])
        assert self._digest(self._grid(nearly)) == self._digest(self._grid(lon))

    def test_reordered_cells_get_own_digest(self) -> None:
        """Weight columns follow the data's cell order, so a sorted grid must not reuse weights."""
        lon_sorted = np.sort((self.LON + 180.0) % 360.0 - 180.0)  # -180, ..., 135
        assert self._digest(self._grid(lon_sorted)) != self._digest(self._grid())


KEY = "0123456789abcdef" * 4  # a sha256 hex digest


class TestWeightRef:
    PREFIX = "version=1/grid=healpix/ellipsoid=sphere"

    def test_roundtrip(self, regular_ds: xr.Dataset) -> None:
        desc = WeightsDescription(describe_source(regular_ds), HealpixGrid(level=3, indexing_scheme="ring"), "nearest")
        ref = desc.ref
        assert ref.path == f"{self.PREFIX}/level=3/method=nearest/{desc.source.digest}.nc"
        assert WeightsRef.parse(ref.path) == ref
        assert WeightsRef.parse("/" + ref.path + "/") == ref

    def test_ref_is_lowercase(self) -> None:
        """Mixed-case input must not give the same weights a second path."""
        upper = WeightsRef("NEAREST", 3, KEY.upper())  # type: ignore[arg-type]
        assert upper == WeightsRef("nearest", 3, KEY)
        assert upper.path == upper.path.lower()
        assert WeightsRef.parse(upper.path) == upper

    def test_hierarchy_is_versioned(self) -> None:
        """Changing the path scheme must come with a new ``WEIGHTS_VERSION``.

        If this fails, bump ``WEIGHTS_VERSION``, keep the old paths readable
        and give every new segment a default matching the old behaviour.
        """
        assert (WEIGHTS_VERSION, WeightsRef.HIERARCHY) == (
            1,
            ("version", "grid", "ellipsoid", "level", "method"),
        ), "weight path scheme changed: bump WEIGHTS_VERSION"

    def test_known_path_is_stable(self) -> None:
        """A version-1 reference must keep its exact path, or stores lose their files."""
        ref = WeightsRef("conservative", 10, KEY)
        expected = f"version=1/grid=healpix/ellipsoid=sphere/level=10/method=conservative/{KEY}.nc"
        assert ref.path == expected
        assert WeightsRef.parse(expected) == ref

    @pytest.mark.parametrize(
        "bad",
        [
            f"{PREFIX}/level=3/{KEY}.nc",  # method missing
            f"{PREFIX}/method=nearest/level=3/{KEY}.nc",  # wrong order
            f"grid=healpix/ellipsoid=sphere/level=3/method=nearest/{KEY}.nc",  # no version
            f"version=2/grid=healpix/ellipsoid=sphere/level=3/method=nearest/{KEY}.nc",
            f"version=1/grid=other/ellipsoid=sphere/level=3/method=nearest/{KEY}.nc",
            f"version=1/grid=healpix/level=3/method=nearest/{KEY}.nc",  # no ellipsoid
            f"version=1/grid=healpix/ellipsoid=wgs84/level=3/method=nearest/{KEY}.nc",
            f"{PREFIX}/indexing_scheme=nested/level=3/method=nearest/{KEY}.nc",  # indexing scheme is not a level
            f"{PREFIX}/level=x/method=nearest/{KEY}.nc",
            f"{PREFIX}/level=3/method=bilinear/{KEY}.nc",
            f"{PREFIX}/level=3/method=nearest/{KEY}.zarr",
            f"{PREFIX}/level/method=nearest/{KEY}.nc",  # no '='
            f"{PREFIX}/level=3/method=nearest/abc123.nc",  # not a sha256 digest
        ],
    )
    def test_parse_rejects(self, bad: str) -> None:
        with pytest.raises(ValueError, match="Not a weight reference"):
            WeightsRef.parse(bad)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("method", "bilinear"),
            ("level", -1),
            ("level", 30),
            ("level", 3.0),
            ("level", True),
            ("key", "abc123"),  # not a sha256 digest
            ("key", "../" + KEY[3:]),  # would escape the directory
            ("grid", "other"),
            ("ellipsoid", "wgs84"),
            ("version", 2),
        ],
    )
    def test_constructor_validates(self, field: str, value: object) -> None:
        """A constructed reference is as strict as a parsed one."""
        fields: dict[str, object] = {"method": "nearest", "level": 3, "key": KEY}
        with pytest.raises(ValueError, match=f"Invalid weight reference {field}"):
            WeightsRef(**{**fields, field: value})  # type: ignore[arg-type]


class TestNestedToRing:
    """Weight files are nested; ring output renumbers their target rows when they are applied."""

    LEVEL = 1
    N = 12 * 4**LEVEL

    def _nested_file(self, path: Path) -> Path:
        """Weights from N source cells that put source cell j into nested target cell (j * 7) % N."""
        from grid_doctor.metadata import _keys as keys

        col = np.arange(1, self.N + 1, dtype=np.int32)
        row = ((col - 1) * 7 % self.N + 1).astype(np.int32)
        data = {"row": ("n_s", row), "col": ("n_s", col), "S": ("n_s", np.ones(self.N))}
        xr.Dataset(data, attrs={keys.LEVEL_ATTR: self.LEVEL, keys.ORDER_ATTR: "nested"}).to_netcdf(path)
        return path

    @staticmethod
    def _apply(path: Path, data: np.ndarray, nest: bool | None = None) -> np.ndarray:
        from grid_doctor.remap import _read_weight_file

        return np.asarray(_read_weight_file(path, nest=nest).matrix @ data)

    def test_conversions_are_inverse(self) -> None:
        cells = np.arange(self.N)
        ring = nested_to_ring(cells, self.LEVEL)
        assert sorted(ring) == list(cells)  # a permutation...
        assert not np.array_equal(ring, cells)  # ...that is not the identity
        np.testing.assert_array_equal(ring_to_nested(ring, self.LEVEL), cells)

    def test_nested_file_must_be_permuted_for_ring(self, tmp_path: Path) -> None:
        """Read as is, a nested file puts ring output into the wrong cells; renumbering fixes that."""
        from grid_doctor.remap import _read_weight_file

        nested = self._nested_file(tmp_path / "w.nc")
        data = np.arange(self.N, dtype=np.float64) + 1.0
        out_nested, out_ring = self._apply(nested, data), self._apply(nested, data, nest=False)
        ring_of = nested_to_ring(np.arange(self.N), self.LEVEL)

        assert not np.array_equal(out_nested, out_ring)  # the nested rows as is would be wrong for ring
        np.testing.assert_array_equal(out_ring[ring_of], out_nested)  # same values, ring cell numbers
        assert _read_weight_file(nested, nest=False).order == "ring"
        np.testing.assert_array_equal(self._apply(nested, data, nest=True), out_nested)  # nothing to do
