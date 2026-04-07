"""Tests for `grid_doctor.helpers`."""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor import helpers
from grid_doctor.helpers import (
    _coarsen_array,
    _coarsen_array_mode,
    _get_latlon_arrays,
    _get_unstructured_dim,
    _is_unstructured,
    coarsen_healpix,
    create_healpix_pyramid,
    get_latlon_resolution,
    latlon_to_healpix_pyramid,
    resolution_to_healpix_level,
    save_pyramid_to_s3,
)


class TestGridDetection:
    def test_is_unstructured_from_dimension(self, unstructured_ds: xr.Dataset) -> None:
        assert _is_unstructured(unstructured_ds) is True

    def test_is_unstructured_from_attribute(self) -> None:
        ds = xr.Dataset(
            {"t": (("points",), np.zeros(3), {"CDI_grid_type": "unstructured"})},
            coords={"lat": ("points", np.zeros(3)), "lon": ("points", np.zeros(3))},
        )
        assert _is_unstructured(ds) is True

    def test_get_unstructured_dim_prefers_known_name(
        self, unstructured_ds: xr.Dataset
    ) -> None:
        assert _get_unstructured_dim(unstructured_ds) == "cell"

    def test_get_unstructured_dim_falls_back_to_lat_dim(self) -> None:
        ds = xr.Dataset(
            {"t": (("points",), np.zeros(4))},
            coords={"clat": ("points", np.zeros(4)), "clon": ("points", np.zeros(4))},
        )
        assert _get_unstructured_dim(ds) == "points"

    def test_get_latlon_arrays_raises(self) -> None:
        with pytest.raises(ValueError, match="Could not locate"):
            _get_latlon_arrays(xr.Dataset({"t": (("x",), np.zeros(3))}))


class TestResolutionHelpers:
    def test_regular_resolution(self, regular_ds: xr.Dataset) -> None:
        resolution = get_latlon_resolution(regular_ds)
        assert 0.0 < resolution < 30.0

    def test_curvilinear_resolution(self, curvilinear_ds: xr.Dataset) -> None:
        resolution = get_latlon_resolution(curvilinear_ds)
        assert 0.0 < resolution < 30.0

    def test_unstructured_resolution(self, unstructured_ds: xr.Dataset) -> None:
        resolution = get_latlon_resolution(unstructured_ds)
        assert resolution > 0.0

    def test_resolution_to_healpix_level(self) -> None:
        assert resolution_to_healpix_level(1.0) >= 0
        with pytest.raises(ValueError, match="positive"):
            resolution_to_healpix_level(0.0)


class TestCoarsenArray:
    """Tests for the low-level ``_coarsen_array`` mean kernel."""

    def test_all_valid(self) -> None:
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        np.testing.assert_allclose(result, [[2.5, 6.5]])

    def test_below_threshold_becomes_nan(self) -> None:
        """1 of 4 valid children (25%) is below the 50% threshold."""
        data = np.array([[1.0, np.nan, np.nan, np.nan, 5.0, 6.0, 7.0, 8.0]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0]), "1/4 valid should be NaN"
        np.testing.assert_allclose(result[0, 1], 6.5)

    def test_at_threshold_is_valid(self) -> None:
        """2 of 4 valid children (50%) meets the 50% threshold."""
        data = np.array([[1.0, 2.0, np.nan, np.nan]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        np.testing.assert_allclose(result[0, 0], 1.5)

    def test_all_nan_becomes_nan(self) -> None:
        data = np.full((1, 4), np.nan)
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0])

    def test_siberian_lake_killed(self) -> None:
        """A single valid pixel among 3 NaN siblings must not survive."""
        data = np.full((1, 4), np.nan)
        data[0, 2] = 15.0
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0])

    def test_batch_dims_preserved(self) -> None:
        data = np.ones((3, 5, 16))
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.5)
        assert result.shape == (3, 5, 4)

    def test_custom_threshold(self) -> None:
        """With threshold 0.25, 1 of 4 valid should survive."""
        data = np.array([[7.0, np.nan, np.nan, np.nan]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=0.25)
        np.testing.assert_allclose(result[0, 0], 7.0)

    def test_strict_threshold(self) -> None:
        """With threshold 1.0, any NaN child kills the parent."""
        data = np.array([[1.0, 2.0, 3.0, np.nan]])
        result = _coarsen_array(data, factor=4, min_valid_fraction=1.0)
        assert np.isnan(result[0, 0])


class TestCoarsenArrayMode:
    """Tests for the low-level ``_coarsen_array_mode`` kernel."""

    def test_clear_majority(self) -> None:
        data = np.array([[5.0, 5.0, 5.0, 3.0]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] == 5.0

    def test_tie_returns_valid_class(self) -> None:
        """All unique values — any of them is acceptable."""
        data = np.array([[1.0, 2.0, 3.0, 4.0]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] in {1.0, 2.0, 3.0, 4.0}

    def test_nan_excluded_from_vote(self) -> None:
        data = np.array([[3.0, 3.0, 7.0, np.nan]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] == 3.0

    def test_below_threshold_becomes_nan(self) -> None:
        data = np.array([[5.0, np.nan, np.nan, np.nan]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert np.isnan(result[0, 0])

    def test_at_threshold_is_valid(self) -> None:
        data = np.array([[5.0, 5.0, np.nan, np.nan]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] == 5.0

    def test_batch_dims_preserved(self) -> None:
        data = np.ones((2, 3, 8))
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result.shape == (2, 3, 2)

    def test_does_not_average_classes(self) -> None:
        """Mode must return an actual class label, never an average."""
        data = np.array([[2.0, 2.0, 8.0, 8.0]])
        result = _coarsen_array_mode(data, factor=4, min_valid_fraction=0.5)
        assert result[0, 0] in {2.0, 8.0}, "Result must be an existing class"
        assert result[0, 0] != 5.0, "Must not be the average"


class TestCoarsenHealpix:
    def test_coarsens_nested_dataset(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert coarse.sizes["cell"] == 48
        assert coarse.attrs["healpix_level"] == 1

    def test_rejects_ring_order(self, healpix_ds: xr.Dataset) -> None:
        ds = healpix_ds.copy()
        ds.attrs["healpix_order"] = "ring"
        with pytest.raises(ValueError, match="only supports nested"):
            coarsen_healpix(ds, target_level=1)

    def test_preserves_non_cell_variable(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = healpix_ds.assign(scale=xr.DataArray(2.0))
        coarse = coarsen_healpix(ds, target_level=1)
        assert float(coarse["scale"]) == 2.0

    def test_auto_mode_conservative_uses_mean(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto coarsen_mode with conservative method should use mean."""
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = healpix_ds.copy()
        ds.attrs["grid_doctor_method"] = "conservative"
        # Pattern repeats every 4 cells; groups of 16 contain 4 copies → mean = 3
        ds["temperature"].values[:] = np.tile([0.0, 2.0, 4.0, 6.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="auto")
        np.testing.assert_allclose(coarse["temperature"].values, 3.0)

    def test_auto_mode_nearest_uses_mode(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto coarsen_mode with nearest method should use mode."""
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = healpix_ds.copy()
        ds.attrs["grid_doctor_method"] = "nearest"
        # Groups of 16: twelve 7s and four 3s → mode = 7
        ds["temperature"].values[:] = np.tile([7.0, 7.0, 7.0, 3.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="auto")
        np.testing.assert_array_equal(coarse["temperature"].values, 7.0)

    def test_explicit_mean_overrides_nearest(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit coarsen_mode='mean' should average even with nearest method."""
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = healpix_ds.copy()
        ds.attrs["grid_doctor_method"] = "nearest"
        ds["temperature"].values[:] = np.tile([0.0, 2.0, 4.0, 6.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="mean")
        np.testing.assert_allclose(coarse["temperature"].values, 3.0)

    def test_explicit_mode_overrides_conservative(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit coarsen_mode='mode' takes the mode even with conservative method."""
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = healpix_ds.copy()
        ds.attrs["grid_doctor_method"] = "conservative"
        ds["temperature"].values[:] = np.tile([5.0, 5.0, 5.0, 9.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="mode")
        np.testing.assert_array_equal(coarse["temperature"].values, 5.0)

    def test_min_valid_fraction_kills_sparse_cells(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = healpix_ds.copy()
        # All NaN except every 4th cell → 4 of 16 valid per group (25% < 50%).
        ds["temperature"].values[:] = np.nan
        ds["temperature"].values[:, 0::4] = 42.0
        coarse = coarsen_healpix(ds, target_level=1, min_valid_fraction=0.5)
        assert np.all(np.isnan(coarse["temperature"].values)), (
            "Cells with only 4/16 valid children should be NaN"
        )

    def test_has_crs_variable(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert "crs" in coarse.coords
        assert coarse.coords["crs"].attrs["grid_mapping_name"] == "healpix"
        assert coarse.coords["crs"].attrs["healpix_nside"] == 2
        assert coarse.coords["crs"].attrs["healpix_level"] == 1
        assert coarse.coords["crs"].attrs["healpix_order"] == "nested"

    def test_data_vars_have_grid_mapping(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        for name in coarse.data_vars:
            if "cell" in coarse[name].dims:
                assert coarse[name].attrs["grid_mapping"] == "crs"

    def test_coarsened_from_level_attr(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            helpers,
            "_healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert coarse.attrs["grid_doctor_coarsened_from_level"] == 3


class TestPyramidBuilders:
    def test_create_healpix_pyramid_nested(
        self, regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_regrid(ds: xr.Dataset, level: int, **kwargs: Any) -> xr.Dataset:
            del kwargs
            npix = 12 * (4**level)
            return xr.Dataset(
                {"t": (("cell",), np.zeros(npix))},
                attrs=ds.attrs
                | {
                    "healpix_nside": 2**level,
                    "healpix_level": level,
                    "healpix_order": "nested",
                },
            )

        monkeypatch.setattr(helpers, "regrid_to_healpix", fake_regrid)
        monkeypatch.setattr(
            helpers,
            "coarsen_healpix",
            lambda ds, level, **kwargs: xr.Dataset(
                {"t": (("cell",), np.zeros(12 * (4**level)))},
                attrs=ds.attrs
                | {
                    "healpix_level": level,
                    "healpix_nside": 2**level,
                    "healpix_order": "nested",
                },
            ),
        )
        pyramid = create_healpix_pyramid(regular_ds, max_level=3, min_level=1)
        assert set(pyramid) == {1, 2, 3}

    def test_create_healpix_pyramid_ring_regrids_every_level(
        self, regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def fake_regrid(ds: xr.Dataset, level: int, **kwargs: Any) -> xr.Dataset:
            calls.append(level)
            return xr.Dataset(
                {"t": (("cell",), np.zeros(12 * (4**level)))},
                attrs=ds.attrs
                | {
                    "healpix_level": level,
                    "healpix_nside": 2**level,
                    "healpix_order": "ring",
                },
            )

        monkeypatch.setattr(helpers, "regrid_to_healpix", fake_regrid)
        pyramid = create_healpix_pyramid(
            regular_ds, max_level=2, min_level=0, nest=False
        )
        assert set(pyramid) == {0, 1, 2}
        assert calls == [2, 1, 0]

    @pytest.mark.parametrize("method", ["nearest", "linear", "conservative"])
    def test_latlon_to_healpix_pyramid_passes_arguments(
        self, regular_ds: xr.Dataset, method: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_create(
            ds: xr.Dataset, max_level: int, min_level: int = 0, **kwargs: Any
        ) -> dict[int, xr.Dataset]:
            captured.update(kwargs)
            return {max_level: ds}

        monkeypatch.setattr(helpers, "create_healpix_pyramid", fake_create)
        pyramid = latlon_to_healpix_pyramid(
            regular_ds, max_level=2, min_level=1, method=method
        )
        assert set(pyramid) == {2}
        assert captured["method"] == method

    def test_create_healpix_pyramid_forwards_coarsen_params(
        self, regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coarsen_calls: list[dict[str, Any]] = []

        def fake_regrid(ds: xr.Dataset, level: int, **kwargs: Any) -> xr.Dataset:
            npix = 12 * (4**level)
            return xr.Dataset(
                {"t": (("cell",), np.zeros(npix))},
                attrs=ds.attrs
                | {
                    "healpix_nside": 2**level,
                    "healpix_level": level,
                    "healpix_order": "nested",
                },
            )

        def fake_coarsen(
            ds: xr.Dataset, level: int, **kwargs: Any
        ) -> xr.Dataset:
            coarsen_calls.append(kwargs)
            npix = 12 * (4**level)
            return xr.Dataset(
                {"t": (("cell",), np.zeros(npix))},
                attrs=ds.attrs
                | {
                    "healpix_level": level,
                    "healpix_nside": 2**level,
                    "healpix_order": "nested",
                },
            )

        monkeypatch.setattr(helpers, "regrid_to_healpix", fake_regrid)
        monkeypatch.setattr(helpers, "coarsen_healpix", fake_coarsen)
        create_healpix_pyramid(
            regular_ds,
            max_level=2,
            min_level=0,
            coarsen_mode="mode",
            min_valid_fraction=0.75,
        )
        assert len(coarsen_calls) == 2
        for call in coarsen_calls:
            assert call["coarsen_mode"] == "mode"
            assert call["min_valid_fraction"] == 0.75


class TestSavePyramidToS3:
    def _make_pyramid(self) -> dict[int, xr.Dataset]:
        return {
            level: xr.Dataset(
                {"t": (("cell",), np.zeros(12 * (4**level), dtype=np.float32))},
                coords={"cell": np.arange(12 * (4**level), dtype=np.int64)},
                attrs={
                    "healpix_nside": 2**level,
                    "healpix_level": level,
                    "healpix_order": "nested",
                },
            )
            for level in (0, 1)
        }

    @mock.patch("grid_doctor.helpers.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.helpers.s3fs.S3Map")
    def test_calls_to_zarr(self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock) -> None:
        del mock_s3fs
        pyramid = self._make_pyramid()
        mock_s3map.return_value = mock.MagicMock()
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w")
            assert mock_zarr.call_count == 2

    @mock.patch("grid_doctor.helpers.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.helpers.s3fs.S3Map")
    def test_zarr_format_3_omits_consolidated(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        del mock_s3fs
        pyramid = self._make_pyramid()
        mock_s3map.return_value = mock.MagicMock()
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(
                pyramid, "s3://bucket/test", s3_options={}, zarr_format=3
            )
            for call in mock_zarr.call_args_list:
                assert "consolidated" not in call.kwargs


    @mock.patch("grid_doctor.helpers.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.helpers.s3fs.S3Map")
    def test_calls_to_zarr_init_region(self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock) -> None:
        del mock_s3fs
        pyramid = self._make_pyramid()
        mock_s3map.return_value = mock.MagicMock()
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid_to_s3(pyramid, "s3://bucket/test", s3_options={}, mode="w", compute=False, region={'cell':slice(0,-1)})
            # Ensure to_zarr is called twice (per dataset), when region is specified!
            assert mock_zarr.call_count == len(pyramid) * 2

            # Ensure the first call is metadata only and second is 'r+' for coords
            def pairwise(iterable):
                a = iter(iterable)
                return zip(a,a)

            for first, second in pairwise(mock_zarr.call_args_list):
                assert first.kwargs.get('mode') == 'w'
                assert first.kwargs.get('compute') == False
                assert first.kwargs.get('region') is not None
                assert second.kwargs.get('mode') == 'r+'
                assert second.kwargs.get('compute') == True
                assert second.kwargs.get('region') == None
