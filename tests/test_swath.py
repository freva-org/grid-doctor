"""Tests for `grid_doctor.swath`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor import coarsen_healpix
from grid_doctor.swath import (
    _bin_mode,
    _declared_fill_value,
    _masked_float64,
    _resolve_point_coords,
    bin_to_healpix,
    sparse_to_dense,
)

LEVEL = 4
NPIX = 12 * 4**LEVEL


def _cell_centres(cells: np.ndarray, level: int = LEVEL) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) centres of *cells* on the sphere."""
    from healpix_geo import nested

    lon, lat = nested.healpix_to_lonlat(
        np.asarray(cells, dtype=np.int64), level, ellipsoid="sphere"
    )
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def _make_swath(
    cells: np.ndarray,
    n_across: int = 5,
    *,
    lat_name: str = "latitude",
    lon_name: str = "longitude",
) -> xr.Dataset:
    """Build a synthetic swath with *n_across* samples per target cell.

    All samples of swath row *i* sit exactly at the centre of
    ``cells[i]``, so the expected binning result is fully known.
    """
    lon_c, lat_c = _cell_centres(cells)
    n_along = len(cells)
    lat = np.repeat(lat_c, n_across).reshape(n_along, n_across)
    lon = np.repeat(lon_c, n_across).reshape(n_along, n_across)
    field = np.arange(n_along, dtype=np.float64)[:, None] * np.ones(n_across)
    return xr.Dataset(
        {"field": (("along_track", "across_track"), field)},
        coords={
            lat_name: (("along_track", "across_track"), lat),
            lon_name: (("along_track", "across_track"), lon),
        },
        attrs={"source": "swath-synthetic"},
    )


@pytest.fixture
def target_cells() -> np.ndarray:
    return np.array([3, 100, 101, 500, 2000, NPIX - 1], dtype=np.int64)


@pytest.fixture
def swath_ds(target_cells: np.ndarray) -> xr.Dataset:
    return _make_swath(target_cells)


# ===================================================================
# Coordinate resolution
# ===================================================================


class TestResolvePointCoords:
    def test_auto_detection(self, swath_ds: xr.Dataset) -> None:
        lat, lon, dims = _resolve_point_coords(
            swath_ds, lat_name=None, lon_name=None, source_units="auto"
        )
        assert dims == ("along_track", "across_track")
        assert lat.size == lon.size == swath_ds["latitude"].size

    def test_explicit_names(self, target_cells: np.ndarray) -> None:
        ds = _make_swath(target_cells, lat_name="lat_swath", lon_name="lon_swath")
        lat, lon, dims = _resolve_point_coords(
            ds, lat_name="lat_swath", lon_name="lon_swath", source_units="auto"
        )
        assert dims == ("along_track", "across_track")
        assert np.isfinite(lat).all() and np.isfinite(lon).all()

    def test_explicit_name_missing_raises(self, swath_ds: xr.Dataset) -> None:
        with pytest.raises(KeyError, match="not found"):
            _resolve_point_coords(
                swath_ds, lat_name="nope", lon_name=None, source_units="auto"
            )

    def test_no_candidate_raises(self) -> None:
        ds = xr.Dataset({"t": (("x",), np.zeros(3))})
        with pytest.raises(ValueError, match="Could not locate"):
            _resolve_point_coords(
                ds, lat_name=None, lon_name=None, source_units="auto"
            )

    def test_mismatched_dims_raise(self, swath_ds: xr.Dataset) -> None:
        ds = swath_ds.assign_coords(
            latitude=("along_track", swath_ds["latitude"].values[:, 0])
        )
        with pytest.raises(ValueError, match="same dimensions"):
            _resolve_point_coords(
                ds, lat_name=None, lon_name=None, source_units="auto"
            )

    def test_radians_and_negative_longitudes(
        self, target_cells: np.ndarray
    ) -> None:
        ds = _make_swath(target_cells)
        lon180 = ((ds["longitude"].values + 180.0) % 360.0) - 180.0
        rad = ds.assign_coords(
            latitude=(ds["latitude"].dims, np.deg2rad(ds["latitude"].values)),
            longitude=(ds["longitude"].dims, np.deg2rad(lon180)),
        )
        result = bin_to_healpix(rad, LEVEL, source_units="rad")
        expected = bin_to_healpix(ds, LEVEL)
        xr.testing.assert_allclose(result["field"], expected["field"])


# ===================================================================
# Fill-value handling
# ===================================================================


class TestFillValues:
    def test_declared_fill_from_attrs(self) -> None:
        da = xr.DataArray(np.zeros(3), attrs={"_FillValue": 255})
        assert _declared_fill_value(da) == 255.0

    def test_declared_fill_from_encoding(self) -> None:
        da = xr.DataArray(np.zeros(3))
        da.encoding["missing_value"] = -32768
        assert _declared_fill_value(da) == -32768.0

    def test_no_declared_fill(self) -> None:
        assert _declared_fill_value(xr.DataArray(np.zeros(3))) is None

    def test_zero_is_a_valid_fill_value(self) -> None:
        """Regression: a fill value of 0 must not be dropped as falsy."""
        da = xr.DataArray(np.array([0, 1, 2], dtype=np.int16))
        masked = _masked_float64(da, fill_value=0)
        assert np.isnan(masked[0])
        assert masked[1:].tolist() == [1.0, 2.0]

    def test_large_positive_cf_fill(self) -> None:
        """The CF default fill 9.9692e36 is finite and must be honoured."""
        fill = np.float32(9.96921e36)
        da = xr.DataArray(
            np.array([1.0, fill], dtype=np.float32), attrs={"_FillValue": fill}
        )
        masked = _masked_float64(da, fill_value=None)
        assert masked[0] == 1.0
        assert np.isnan(masked[1])

    def test_nan_invalid_even_with_declared_fill(self) -> None:
        da = xr.DataArray(
            np.array([np.nan, 1.0, -999.0]), attrs={"_FillValue": -999.0}
        )
        masked = _masked_float64(da, fill_value=None)
        assert np.isnan(masked[0]) and np.isnan(masked[2])
        assert masked[1] == 1.0

    def test_explicit_fill_overrides_attrs(self) -> None:
        da = xr.DataArray(np.array([5.0, 7.0]), attrs={"_FillValue": 7.0})
        masked = _masked_float64(da, fill_value=5.0)
        assert np.isnan(masked[0])
        assert masked[1] == 7.0

    def test_integer_fill_on_unsigned_dtype(self) -> None:
        da = xr.DataArray(np.array([1, 255], dtype=np.uint8))
        masked = _masked_float64(da, fill_value=255)
        assert masked[0] == 1.0 and np.isnan(masked[1])


# ===================================================================
# Mode reduction
# ===================================================================


class TestBinMode:
    def test_majority_wins(self) -> None:
        group_idx = np.zeros(5, dtype=np.int64)
        values = np.array([1.0, 2.0, 2.0, 2.0, 3.0])
        result = _bin_mode(group_idx, values, n_cells=1)
        assert result[0] == 2.0

    def test_tie_breaks_to_lowest_class(self) -> None:
        group_idx = np.zeros(4, dtype=np.int64)
        values = np.array([3.0, 3.0, 1.0, 1.0])
        result = _bin_mode(group_idx, values, n_cells=1)
        assert result[0] == 1.0

    def test_all_invalid_gives_nan(self) -> None:
        result = _bin_mode(
            np.zeros(2, dtype=np.int64), np.array([np.nan, np.nan]), n_cells=1
        )
        assert np.isnan(result[0])

    def test_too_many_classes_raises(self) -> None:
        values = np.arange(10.0)
        with pytest.raises(ValueError, match="distinct classes"):
            _bin_mode(
                np.zeros(10, dtype=np.int64), values, n_cells=1, max_classes=5
            )


# ===================================================================
# bin_to_healpix — dense path
# ===================================================================


class TestBinToHealpixDense:
    def test_known_cells_receive_known_means(
        self, swath_ds: xr.Dataset, target_cells: np.ndarray
    ) -> None:
        result = bin_to_healpix(swath_ds, LEVEL)
        assert result.sizes["cell"] == NPIX
        assert np.allclose(
            result["field"].values[target_cells], np.arange(len(target_cells))
        )
        untouched = np.setdiff1d(np.arange(NPIX), target_cells)
        assert np.isnan(result["field"].values[untouched]).all()

    def test_standard_metadata(self, swath_ds: xr.Dataset) -> None:
        result = bin_to_healpix(swath_ds, LEVEL)
        assert result.attrs["healpix_level"] == LEVEL
        assert result.attrs["healpix_nside"] == 2**LEVEL
        assert result.attrs["healpix_order"] == "nested"
        assert result.attrs["grid_doctor_method"] == "binned-mean"
        assert "grid_doctor_version" in result.attrs
        assert "crs" in result.coords
        assert result["crs"].attrs["grid_mapping_name"] == "healpix"
        assert result["field"].attrs["grid_mapping"] == "crs"
        assert {"latitude", "longitude"} <= set(result.coords)

    def test_batch_dimensions_preserved(self, target_cells: np.ndarray) -> None:
        ds = _make_swath(target_cells)
        stacked = ds["field"].expand_dims(time=[0, 1]).copy()
        stacked.loc[{"time": 1}] = stacked.sel(time=1) + 10.0
        ds = ds.assign(field=stacked)
        result = bin_to_healpix(ds, LEVEL)
        assert result["field"].dims == ("time", "cell")
        assert np.allclose(
            result["field"].values[1, target_cells] -
            result["field"].values[0, target_cells],
            10.0,
        )
        assert "time" in result.coords

    def test_non_spatial_variables_pass_through(
        self, swath_ds: xr.Dataset
    ) -> None:
        ds = swath_ds.assign(meta=((), np.float64(42.0)))
        result = bin_to_healpix(ds, LEVEL)
        assert float(result["meta"]) == 42.0

    def test_partial_dimension_overlap_is_skipped(
        self, swath_ds: xr.Dataset
    ) -> None:
        ds = swath_ds.assign(
            profile=(("along_track",), np.zeros(swath_ds.sizes["along_track"]))
        )
        result = bin_to_healpix(ds, LEVEL)
        assert "profile" not in result

    def test_min_count_masks_sparse_cells(
        self, target_cells: np.ndarray
    ) -> None:
        ds = _make_swath(target_cells, n_across=2)
        result = bin_to_healpix(ds, LEVEL, min_count=3)
        assert np.isnan(result["field"].values[target_cells]).all()

    def test_fill_values_argument(self, target_cells: np.ndarray) -> None:
        ds = _make_swath(target_cells)
        classes = np.full(ds["field"].shape, 2, dtype=np.uint8)
        classes[0] = 255
        ds = ds.assign(klass=(ds["field"].dims, classes))
        result = bin_to_healpix(
            ds,
            LEVEL,
            agg={"field": "mean", "klass": "mode"},
            fill_values={"klass": 255},
        )
        assert np.isnan(result["klass"].values[target_cells[0]])
        assert result["klass"].values[target_cells[1]] == 2.0
        # per-variable method attributes
        assert result["klass"].attrs["grid_doctor_method"] == "binned-mode"
        assert result["field"].attrs["grid_doctor_method"] == "binned-mean"

    def test_count_aggregation(
        self, swath_ds: xr.Dataset, target_cells: np.ndarray
    ) -> None:
        result = bin_to_healpix(swath_ds, LEVEL, agg="count")
        assert (result["field"].values[target_cells] == 5.0).all()

    def test_with_counts_companion(
        self, swath_ds: xr.Dataset, target_cells: np.ndarray
    ) -> None:
        result = bin_to_healpix(swath_ds, LEVEL, with_counts=True)
        assert "field_count" in result
        assert result["field_count"].dtype == np.int32
        assert (result["field_count"].values[target_cells] == 5).all()

    def test_min_max(self, target_cells: np.ndarray) -> None:
        ds = _make_swath(target_cells)
        spread = ds["field"].values.copy()
        spread[:, 0] -= 1.0
        spread[:, -1] += 1.0
        ds = ds.assign(field=(ds["field"].dims, spread))
        lo = bin_to_healpix(ds, LEVEL, agg="min")
        hi = bin_to_healpix(ds, LEVEL, agg="max")
        base = np.arange(len(target_cells), dtype=np.float64)
        assert np.allclose(lo["field"].values[target_cells], base - 1.0)
        assert np.allclose(hi["field"].values[target_cells], base + 1.0)

    def test_unknown_aggregation_raises(self, swath_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="Unknown aggregation"):
            bin_to_healpix(swath_ds, LEVEL, agg="sum")

    def test_invalid_level_raises(self, swath_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="level"):
            bin_to_healpix(swath_ds, 30)

    def test_invalid_min_count_raises(self, swath_ds: xr.Dataset) -> None:
        with pytest.raises(ValueError, match="min_count"):
            bin_to_healpix(swath_ds, LEVEL, min_count=0)

    def test_no_valid_coordinates_raises(self, swath_ds: xr.Dataset) -> None:
        broken = swath_ds.assign_coords(
            latitude=(
                swath_ds["latitude"].dims,
                np.full(swath_ds["latitude"].shape, np.nan),
            )
        )
        with pytest.raises(ValueError, match="valid"):
            bin_to_healpix(broken, LEVEL)

    def test_invalid_coordinate_samples_are_dropped(
        self, target_cells: np.ndarray
    ) -> None:
        ds = _make_swath(target_cells)
        lat = ds["latitude"].values.copy()
        lat[0, :] = np.nan
        ds = ds.assign_coords(latitude=(ds["latitude"].dims, lat))
        result = bin_to_healpix(ds, LEVEL)
        assert np.isnan(result["field"].values[target_cells[0]])
        assert np.allclose(
            result["field"].values[target_cells[1:]],
            np.arange(1, len(target_cells), dtype=np.float64),
        )


# ===================================================================
# bin_to_healpix — sparse path
# ===================================================================


class TestBinToHealpixSparse:
    def test_cell_coordinate_holds_healpix_indices(
        self, swath_ds: xr.Dataset, target_cells: np.ndarray
    ) -> None:
        result = bin_to_healpix(swath_ds, LEVEL, dense=False)
        assert result.sizes["cell"] == len(target_cells)
        assert (np.sort(result["cell"].values) == np.sort(target_cells)).all()
        assert int(result.attrs["grid_doctor_sparse"]) == 1
        assert "crs" in result.coords

    def test_sparse_to_dense_round_trip(self, swath_ds: xr.Dataset) -> None:
        sparse = bin_to_healpix(swath_ds, LEVEL, dense=False)
        dense = sparse_to_dense(sparse)
        expected = bin_to_healpix(swath_ds, LEVEL)
        xr.testing.assert_allclose(dense["field"], expected["field"])
        assert "grid_doctor_sparse" not in dense.attrs
        assert dense.attrs["grid_doctor_method"] == "binned-mean"

    def test_sparse_to_dense_rejects_dense_input(
        self, swath_ds: xr.Dataset
    ) -> None:
        dense = bin_to_healpix(swath_ds, LEVEL)
        with pytest.raises(ValueError, match="sparse"):
            sparse_to_dense(dense)


# ===================================================================
# Pyramid integration
# ===================================================================


class TestCoarsenIntegration:
    def test_binned_mean_output_coarsens(self, swath_ds: xr.Dataset) -> None:
        dense = bin_to_healpix(swath_ds, LEVEL)
        coarse = coarsen_healpix(dense, LEVEL - 1, min_valid_fraction=0.25)
        assert coarse.attrs["healpix_level"] == LEVEL - 1
        assert coarse.attrs["grid_doctor_coarsened_from_level"] == LEVEL
        assert np.isfinite(coarse["field"].values).any()

    def test_binned_mode_selects_mode_coarsening(self) -> None:
        """`coarsen_mode="auto"` must pick the mode for binned-mode data.

        Cells 100 and 101 share the same level-3 parent (25).  With
        classes {7, 7, 9} among the parent's children the mode is 7,
        while a mean would produce a fabricated class of ~7.67.
        """
        cells = np.array([100, 100, 101], dtype=np.int64)
        lon, lat = _cell_centres(cells)
        ds = xr.Dataset(
            {"klass": (("obs",), np.array([7.0, 7.0, 9.0]))},
            coords={"latitude": ("obs", lat), "longitude": ("obs", lon)},
        )
        dense = bin_to_healpix(ds, LEVEL, agg="mode")
        assert dense.attrs["grid_doctor_method"] == "binned-mode"
        coarse = coarsen_healpix(dense, LEVEL - 1, min_valid_fraction=0.25)
        assert coarse["klass"].values[25] == 7.0

    def test_one_dimensional_track(self) -> None:
        """A nadir-only track (1-D sample dimension) also works."""
        cells = np.array([10, 20, 30], dtype=np.int64)
        lon, lat = _cell_centres(cells)
        ds = xr.Dataset(
            {"t": (("time_track",), np.array([1.0, 2.0, 3.0]))},
            coords={
                "latitude": ("time_track", lat),
                "longitude": ("time_track", lon),
            },
        )
        result = bin_to_healpix(ds, LEVEL)
        assert np.allclose(result["t"].values[cells], [1.0, 2.0, 3.0])
