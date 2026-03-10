"""Tests for grid_doctor.helpers."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from dask.array import Array as DaskArray

from grid_doctor.helpers import (
    _attach_healpix_coords,
    _get_latlon_arrays,
    _get_spatial_dims,
    _get_unstructured_dim,
    _get_unstructured_resolution,
    _is_unstructured,
    coarsen_healpix,
    compute_weights_delaunay,
    create_healpix_pyramid,
    get_latlon_resolution,
    latlon_to_healpix_pyramid,
    regrid_to_healpix,
    regrid_unstructured_to_healpix,
    resolution_to_healpix_level,
)


class TestIsUnstructured:
    def test_cell_dimension(self, unstructured_ds: xr.Dataset) -> None:
        assert _is_unstructured(unstructured_ds) is True

    def test_ncells_dimension(self, unstructured_ncells_ds: xr.Dataset) -> None:
        assert _is_unstructured(unstructured_ncells_ds) is True

    def test_regular_grid(self, regular_ds: xr.Dataset) -> None:
        assert _is_unstructured(regular_ds) is False

    def test_curvilinear_grid(self, curvilinear_ds: xr.Dataset) -> None:
        assert _is_unstructured(curvilinear_ds) is False

    def test_cdi_attribute(self) -> None:
        """Detect unstructured via CDI_grid_type even without known dim names."""
        ds = xr.Dataset(
            {"t": (("points",), np.zeros(10), {"CDI_grid_type": "unstructured"})},
            coords={
                "lat": ("points", np.zeros(10)),
                "lon": ("points", np.zeros(10)),
            },
        )
        assert _is_unstructured(ds) is True


class TestGetUnstructuredDim:
    def test_cell_dim(self, unstructured_ds: xr.Dataset) -> None:
        assert _get_unstructured_dim(unstructured_ds) == "cell"

    def test_ncells_dim(self, unstructured_ncells_ds: xr.Dataset) -> None:
        assert _get_unstructured_dim(unstructured_ncells_ds) == "ncells"

    def test_fallback_from_coords(self) -> None:
        """Falls back to the dim that lat is indexed on."""
        ds = xr.Dataset(
            {"t": (("points",), np.zeros(10))},
            coords={
                "clat": ("points", np.zeros(10)),
                "clon": ("points", np.zeros(10)),
            },
        )
        assert _get_unstructured_dim(ds) == "points"

    def test_raises_unknown(self) -> None:
        ds = xr.Dataset({"t": (("x",), np.zeros(10))})
        with pytest.raises(ValueError, match="Cannot determine"):
            _get_unstructured_dim(ds)


class TestGetLatlonArrays:
    def test_regular(self, regular_ds: xr.Dataset) -> None:
        lat, lon = _get_latlon_arrays(regular_ds)
        assert lat.ndim == 1
        assert lon.ndim == 1

    def test_curvilinear(self, curvilinear_ds: xr.Dataset) -> None:
        lat, lon = _get_latlon_arrays(curvilinear_ds)
        assert lat.ndim == 2
        assert lon.ndim == 2

    def test_unstructured(self, unstructured_ds: xr.Dataset) -> None:
        lat, lon = _get_latlon_arrays(unstructured_ds)
        assert lat.ndim == 1
        assert lat.shape == lon.shape

    def test_icon_names(self, unstructured_ds: xr.Dataset) -> None:
        """clat/clon are found before lat/lon."""
        lat, lon = _get_latlon_arrays(unstructured_ds)
        assert lat.shape[0] == unstructured_ds.sizes["cell"]

    def test_raises_missing(self) -> None:
        ds = xr.Dataset({"t": (("x",), np.zeros(5))})
        with pytest.raises(ValueError, match="Could not find"):
            _get_latlon_arrays(ds)

    def test_data_var_fallback(self) -> None:
        """Finds lat/lon in data_vars when not in coords."""
        ds = xr.Dataset(
            {"lat": (("y",), np.arange(5.0)), "lon": (("x",), np.arange(10.0))},
        )
        lat, lon = _get_latlon_arrays(ds)
        assert lat.shape == (5,)
        assert lon.shape == (10,)


class TestGetSpatialDims:
    def test_regular(self, regular_ds: xr.Dataset) -> None:
        y, x = _get_spatial_dims(regular_ds)
        assert y == "y"
        assert x == "x"

    def test_era5(self, era5_ds: xr.Dataset) -> None:
        y, x = _get_spatial_dims(era5_ds)
        assert y == "lat"
        assert x == "lon"

    def test_raises_on_unstructured(self, unstructured_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="Could not identify"):
            _get_spatial_dims(unstructured_ds)


class TestResolution:
    def test_regular_grid(self, regular_ds: xr.Dataset) -> None:
        res = get_latlon_resolution(regular_ds)
        assert 0 < res < 10  # reasonable for a 64×128 grid

    def test_curvilinear_grid(self, curvilinear_ds: xr.Dataset) -> None:
        res = get_latlon_resolution(curvilinear_ds)
        assert 0 < res < 10

    def test_unstructured_grid(self, unstructured_ds: xr.Dataset) -> None:
        res = get_latlon_resolution(unstructured_ds)
        # 5000 cells → ~3.2°
        assert 1 < res < 10

    def test_unstructured_resolution_from_cell_count(
        self, unstructured_ds: xr.Dataset
    ) -> None:
        res = _get_unstructured_resolution(unstructured_ds)
        n = unstructured_ds.sizes["cell"]
        expected = float(np.degrees(np.sqrt(4 * np.pi / n)))
        assert abs(res - expected) < 1e-10


class TestResolutionToHealpixLevel:
    @pytest.mark.parametrize(
        "res, expected_level",
        [
            (1.0, 5),
            (0.1, 9),
            (0.01, 12),
            (10.0, 2),
        ],
    )
    def test_known_values(self, res: float, expected_level: int) -> None:
        assert resolution_to_healpix_level(res) == expected_level

    def test_very_coarse(self) -> None:
        assert resolution_to_healpix_level(100.0) == 0

    def test_monotonic(self) -> None:
        """Finer resolution → higher level."""
        levels = [resolution_to_healpix_level(r) for r in [10, 1, 0.1, 0.01]]
        assert levels == sorted(levels)


class TestRegridToHealpix:
    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_output_has_cell_dim(self, test_ds: xr.Dataset) -> None:
        hp_ds = regrid_to_healpix(test_ds, level=3)
        assert "cell" in hp_ds.dims

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_attrs_preserved(self, test_ds: xr.Dataset) -> None:
        hp_ds = regrid_to_healpix(test_ds, level=3)
        for key, val in test_ds.attrs.items():
            assert hp_ds.attrs[key] == val

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_healpix_metadata(self, test_ds: xr.Dataset) -> None:
        level = 3
        hp_ds = regrid_to_healpix(test_ds, level=level)
        assert hp_ds.attrs["healpix_nside"] == 2**level
        assert hp_ds.attrs["healpix_level"] == level
        assert hp_ds.attrs["healpix_order"] == "nested"

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_crs_coordinate(self, test_ds: xr.Dataset) -> None:
        hp_ds = regrid_to_healpix(test_ds, level=3)
        assert "crs" in hp_ds.coords
        assert hp_ds.coords["crs"].attrs["grid_mapping_name"] == "healpix"

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_dtypes_preserved(self, test_ds: xr.Dataset) -> None:
        hp_ds = regrid_to_healpix(test_ds, level=3)
        for name in hp_ds.data_vars:
            assert hp_ds[name].dtype == test_ds[name].dtype

    def test_ring_ordering(self, regular_ds: xr.Dataset) -> None:
        hp_ds = regrid_to_healpix(regular_ds, level=3, nest=False)
        assert hp_ds.attrs["healpix_order"] == "ring"

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_vars_subset(self, test_ds: xr.Dataset) -> None:
        """Regridded dataset only contains variables with spatial dims."""
        hp_ds = regrid_to_healpix(test_ds, level=3)
        assert set(hp_ds.data_vars).issubset(set(test_ds.data_vars))


class TestRegridUnstructuredToHealpix:
    def test_output_has_cell_dim(self, unstructured_ds: xr.Dataset) -> None:
        hp_ds = regrid_unstructured_to_healpix(unstructured_ds, level=2)
        assert "cell" in hp_ds.dims

    def test_healpix_metadata(self, unstructured_ds: xr.Dataset) -> None:
        level = 2
        hp_ds = regrid_unstructured_to_healpix(unstructured_ds, level=level)
        assert hp_ds.attrs["healpix_nside"] == 2**level
        assert hp_ds.attrs["healpix_level"] == level

    def test_attrs_preserved(self, unstructured_ds: xr.Dataset) -> None:
        hp_ds = regrid_unstructured_to_healpix(unstructured_ds, level=2)
        assert hp_ds.attrs["source"] == "icon-synthetic"

    def test_crs_coordinate(self, unstructured_ds: xr.Dataset) -> None:
        hp_ds = regrid_unstructured_to_healpix(unstructured_ds, level=2)
        assert "crs" in hp_ds.coords
        assert hp_ds.coords["crs"].attrs["grid_mapping_name"] == "healpix"

    def test_lat_lon_coords(self, unstructured_ds: xr.Dataset) -> None:
        hp_ds = regrid_unstructured_to_healpix(unstructured_ds, level=2)
        assert "latitude" in hp_ds.coords
        assert "longitude" in hp_ds.coords

    def test_precomputed_weights(self, unstructured_ds: xr.Dataset) -> None:
        level = 2
        weights = compute_weights_delaunay(unstructured_ds, level=level)
        hp_ds = regrid_unstructured_to_healpix(
            unstructured_ds, level=level, weights=weights
        )
        assert "cell" in hp_ds.dims

    def test_ncells_dim(self, unstructured_ncells_ds: xr.Dataset) -> None:
        """Works when the dimension is 'ncells' instead of 'cell'."""
        hp_ds = regrid_unstructured_to_healpix(unstructured_ncells_ds, level=2)
        assert "cell" in hp_ds.dims

    def test_lazy_output(self, unstructured_ds: xr.Dataset) -> None:
        hp_ds = regrid_unstructured_to_healpix(unstructured_ds, level=2)
        for var in hp_ds.data_vars.values():
            assert isinstance(var.data, DaskArray)


class TestComputeWeightsDelaunay:
    def test_returns_dataset(self, unstructured_ds: xr.Dataset) -> None:
        weights = compute_weights_delaunay(unstructured_ds, level=2)
        assert isinstance(weights, xr.Dataset)

    def test_has_tgt_idx(self, unstructured_ds: xr.Dataset) -> None:
        weights = compute_weights_delaunay(unstructured_ds, level=2)
        assert "tgt_idx" in weights.dims

    def test_ring_ordering(self, unstructured_ds: xr.Dataset) -> None:
        weights = compute_weights_delaunay(unstructured_ds, level=2, nest=False)
        assert isinstance(weights, xr.Dataset)


class TestCoarsenHealpix:
    def test_reduces_npix(self, healpix_ds: xr.Dataset) -> None:
        import healpy as hp

        coarse = coarsen_healpix(healpix_ds, target_level=1)
        expected_npix = hp.nside2npix(2**1)
        assert coarse.sizes["cell"] == expected_npix

    def test_metadata_updated(self, healpix_ds: xr.Dataset) -> None:
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert coarse.attrs["healpix_nside"] == 2**1
        assert coarse.attrs["healpix_level"] == 1

    def test_crs_updated(self, healpix_ds: xr.Dataset) -> None:
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert coarse.coords["crs"].attrs["healpix_nside"] == 2

    def test_coords_present(self, healpix_ds: xr.Dataset) -> None:
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert "latitude" in coarse.coords
        assert "longitude" in coarse.coords

    def test_raises_on_higher_level(self, healpix_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="must result in lower"):
            coarsen_healpix(healpix_ds, target_level=5)

    def test_non_cell_vars_preserved(self, healpix_ds: xr.Dataset) -> None:
        """Variables without 'cell' dim are passed through unchanged."""
        ds = healpix_ds.assign(scalar=xr.DataArray(42.0))
        coarse = coarsen_healpix(ds, target_level=1)
        assert float(coarse["scalar"]) == 42.0


class TestCreateHealpixPyramid:
    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear"],
        ids=["regular", "curvilinear"],
        indirect=True,
    )
    def test_structured(self, test_ds: xr.Dataset) -> None:
        pyramid = create_healpix_pyramid(test_ds, max_level=3, min_level=1)
        assert set(pyramid.keys()) == {1, 2, 3}

    def test_unstructured(self, unstructured_ds: xr.Dataset) -> None:
        pyramid = create_healpix_pyramid(
            unstructured_ds, max_level=2, min_level=0
        )
        assert set(pyramid.keys()) == {0, 1, 2}

    def test_levels_decrease_in_size(self, regular_ds: xr.Dataset) -> None:
        pyramid = create_healpix_pyramid(regular_ds, max_level=3, min_level=0)
        sizes = [pyramid[lvl].sizes["cell"] for lvl in sorted(pyramid)]
        assert sizes == sorted(sizes)


class TestLatlonToHealpixPyramid:
    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_auto_level(self, test_ds: xr.Dataset) -> None:
        pyramid = latlon_to_healpix_pyramid(test_ds)
        assert len(pyramid) > 0

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_explicit_max_level(self, test_ds: xr.Dataset) -> None:
        pyramid = latlon_to_healpix_pyramid(test_ds, max_level=3, min_level=2)
        assert set(pyramid.keys()) == {2, 3}

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    @pytest.mark.parametrize("method", ["nearest", "linear"])
    def test_lazy_output(self, test_ds: xr.Dataset, method: str) -> None:
        pyramid = latlon_to_healpix_pyramid(test_ds, max_level=3, method=method)
        for hp_ds in pyramid.values():
            for var in hp_ds.data_vars.values():
                assert isinstance(var.data, DaskArray)

    @pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear", "era5"],
        ids=["regular", "curvilinear", "era5"],
        indirect=True,
    )
    def test_attrs_preserved(self, test_ds: xr.Dataset) -> None:
        pyramid = latlon_to_healpix_pyramid(test_ds, max_level=3)
        for hp_ds in pyramid.values():
            for key, val in test_ds.attrs.items():
                assert hp_ds.attrs[key] == val

    def test_unstructured_auto(self, unstructured_ds: xr.Dataset) -> None:
        pyramid = latlon_to_healpix_pyramid(unstructured_ds, max_level=2)
        assert 2 in pyramid
        assert "cell" in pyramid[2].dims

    def test_unstructured_with_weights(self, unstructured_ds: xr.Dataset) -> None:
        weights = compute_weights_delaunay(unstructured_ds, level=2)
        pyramid = latlon_to_healpix_pyramid(
            unstructured_ds, max_level=2, weights=weights
        )
        assert 2 in pyramid


class TestAttachHealpixCoords:
    def test_adds_coords(self, regular_ds: xr.Dataset) -> None:
        import healpy as hp

        nside = 4
        npix = hp.nside2npix(nside)
        ds_hp = xr.Dataset(
            {"t": (("cell",), np.zeros(npix))},
        )
        result = _attach_healpix_coords(
            ds_hp, regular_ds, nside=4, level=2, nest=True
        )
        assert "cell" in result.coords
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        assert "crs" in result.coords
        assert result.attrs["healpix_nside"] == 4
        assert result.attrs["healpix_order"] == "nested"

    def test_source_attrs_copied(self) -> None:
        import healpy as hp

        nside = 4
        npix = hp.nside2npix(nside)
        src = xr.Dataset(attrs={"history": "test", "custom": 42})
        ds_hp = xr.Dataset({"t": (("cell",), np.zeros(npix))})
        result = _attach_healpix_coords(ds_hp, src, nside=4, level=2, nest=True)
        assert result.attrs["history"] == "test"
        assert result.attrs["custom"] == 42
