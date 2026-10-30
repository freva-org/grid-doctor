"""Tests for `grid_doctor.helpers`."""

from __future__ import annotations

import os
from concurrent.futures import (
    ThreadPoolExecutor,
)
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
    init_pyramid_store,
    latlon_to_healpix_pyramid,
    resolution_to_healpix_level,
    save_pyramid,
)
from grid_doctor.metadata import HealpixMetadata, PyramidMetadata
from grid_doctor.pyramid import _levels as pyramid_levels

from .helpers import with_healpix_metadata


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
            pyramid_levels,
            "healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert coarse.sizes["cell"] == 48
        assert HealpixMetadata.from_dataset(coarse).grid.level == 1

    def test_rejects_ring_order(self, healpix_ds: xr.Dataset) -> None:
        ds = with_healpix_metadata(healpix_ds, 3, nest=False)
        with pytest.raises(ValueError, match="only supports nested"):
            coarsen_healpix(ds, target_level=1)

    def test_preserves_non_cell_variable(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
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
            pyramid_levels,
            "healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = with_healpix_metadata(healpix_ds.copy(), 3, method="conservative")
        # Pattern repeats every 4 cells; groups of 16 contain 4 copies → mean = 3
        ds["temperature"].values[:] = np.tile([0.0, 2.0, 4.0, 6.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="auto")
        np.testing.assert_allclose(coarse["temperature"].values, 3.0)

    def test_auto_mode_nearest_uses_mode(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Auto coarsen_mode with nearest method should use mode."""
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = with_healpix_metadata(healpix_ds.copy(), 3, method="nearest")
        # Groups of 16: twelve 7s and four 3s → mode = 7
        ds["temperature"].values[:] = np.tile([7.0, 7.0, 7.0, 3.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="auto")
        np.testing.assert_array_equal(coarse["temperature"].values, 7.0)

    def test_explicit_mean_overrides_nearest(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit coarsen_mode='mean' should average even with nearest method."""
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = with_healpix_metadata(healpix_ds.copy(), 3, method="nearest")
        ds["temperature"].values[:] = np.tile([0.0, 2.0, 4.0, 6.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="mean")
        np.testing.assert_allclose(coarse["temperature"].values, 3.0)

    def test_explicit_mode_overrides_conservative(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit coarsen_mode='mode' takes the mode even with conservative method."""
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        ds = with_healpix_metadata(healpix_ds.copy(), 3, method="conservative")
        ds["temperature"].values[:] = np.tile([5.0, 5.0, 5.0, 9.0], 192)
        coarse = coarsen_healpix(ds, target_level=1, coarsen_mode="mode")
        np.testing.assert_array_equal(coarse["temperature"].values, 5.0)

    def test_min_valid_fraction_kills_sparse_cells(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
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
            pyramid_levels,
            "healpix_coords",
            lambda level, nest: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert "crs" in coarse.coords
        assert coarse.coords["crs"].attrs == {
            "grid_mapping_name": "healpix",
            "indexing_scheme": "nested",
            "refinement_level": 1,
        }
        assert coarse["cell"].attrs["standard_name"] == "healpix_index"

    def test_data_vars_have_grid_mapping(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
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
            pyramid_levels,
            "healpix_coords",
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
            return with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(npix))},
                    attrs=ds.attrs,
                ),
                level,
            )

        monkeypatch.setattr(helpers, "regrid_to_healpix", fake_regrid)
        monkeypatch.setattr(
            helpers,
            "coarsen_healpix",
            lambda ds, level, **kwargs: with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(12 * (4**level)))}, attrs=ds.attrs
                ),
                level,
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
            return with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(12 * (4**level)))},
                    attrs=ds.attrs,
                ),
                level, nest=False,
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
            return with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(npix))},
                    attrs=ds.attrs,
                ),
                level,
            )

        def fake_coarse_levels(
            finest: xr.Dataset, **kwargs: Any
        ) -> dict[int, xr.Dataset]:
            coarsen_calls.append(kwargs)
            return {}

        monkeypatch.setattr(helpers, "regrid_to_healpix", fake_regrid)
        monkeypatch.setattr(helpers, "coarse_levels", fake_coarse_levels)
        create_healpix_pyramid(
            regular_ds,
            max_level=2,
            min_level=0,
            coarsen_mode="mode",
            min_valid_fraction=0.75,
        )
        assert len(coarsen_calls) == 1
        call = coarsen_calls[0]
        assert call["max_level"] == 2
        assert call["min_level"] == 0
        assert call["coarsen_mode"] == "mode"
        assert call["min_valid_fraction"] == 0.75


class TestSavePyramid:
    def _make_pyramid(self, chunked: bool = False) -> dict[int, xr.Dataset]:
        def _create_dataset(level: int, chunked: bool) -> xr.Dataset:
            ds = with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell", "time"), np.random.randn(12 * (4**level), 100))},
                    coords={
                        "cell": np.arange(12 * (4**level), dtype=np.int64),
                        "time": np.arange(100, dtype=np.int64),
                    },
                ),
                level,
            )
            if chunked:
                return ds.chunk({"time": 5})
            return ds

        return {level: _create_dataset(level, chunked) for level in (0, 1)}

    def _make_pyramid_chunked_coord(self) -> dict[int, xr.Dataset]:
        def _create_dataset(level: int) -> xr.Dataset:
            lat = xr.DataArray(
                np.random.randn(12 * (4**level)),
                coords=[
                    np.arange(12 * (4**level)),
                ],
                dims=[
                    "cell",
                ],
            ).chunk({"cell": 12 * (4**level)})
            static = xr.DataArray(
                [1.0, 2.0],
                coords=[
                    [0, 1],
                ],
                dims=["level"],
            ).chunk({"level": 2})

            ds = xr.Dataset(
                data_vars={
                    "temp": (
                        ["time", "level", "cell"],
                        np.random.randn(4, 2, 12 * (4**level)),
                    ),
                    "static": static,
                },
                coords={
                    "time": ("time", range(4)),
                    "level": ("level", range(2)),
                    "cell": ("cell", range(12 * (4**level))),
                    "latitude": lat,
                },
            )
            ds["temp"] = ds.temp.chunk({"time": 2})
            return ds

        return {level: _create_dataset(level) for level in (0, 1)}

    @mock.patch("grid_doctor.helpers.s3fs.S3FileSystem")
    @mock.patch("grid_doctor.helpers.s3fs.S3Map")
    def test_calls_to_zarr(self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock) -> None:
        del mock_s3fs
        pyramid = self._make_pyramid()
        mock_s3map.return_value = mock.MagicMock()
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid(pyramid, "s3://bucket/test", s3_options={}, mode="w")
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
            save_pyramid(pyramid, "s3://bucket/test", s3_options={}, zarr_format=3)
            for call in mock_zarr.call_args_list:
                assert "consolidated" not in call.kwargs

    @pytest.mark.parametrize("chunked", [True, False])
    def test_save_pyramid_local(self, tmp_path, chunked):
        pyramid = self._make_pyramid(chunked)

        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="w",
        )
        level0_path = tmp_path / "level_0.zarr"
        assert os.path.isdir(str(level0_path))
        level1_path = tmp_path / "level_1.zarr"
        assert os.path.isdir(str(level1_path))

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0, pyramid[0])
        xr.testing.assert_equal(ds1, pyramid[1])

    @pytest.mark.parametrize("chunked", [True, False])
    def test_save_pyramid_local_region_init(self, tmp_path, chunked):
        pyramid = self._make_pyramid(chunked)

        init_pyramid_store(pyramid=pyramid, path=str(tmp_path))

        level0_path = tmp_path / "level_0.zarr"
        assert os.path.isdir(str(level0_path))
        level1_path = tmp_path / "level_1.zarr"
        assert os.path.isdir(str(level1_path))

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        # Datasets should have the full coordinates, and data variables.
        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["t"]
        assert list(ds1.data_vars) == ["t"]

        if chunked:
            assert ds0.t.chunksizes == pyramid[0].t.chunksizes

        # Confirm that the data is nan.
        ds0t = ds0.t.data.compute()
        ds1t = ds1.t.data.compute()
        assert len(ds0t[~np.isnan(ds0t)]) == 0
        assert len(ds1t[~np.isnan(ds1t)]) == 0

    @pytest.mark.parametrize("chunked", [True, False])
    def test_save_pyramid_local_region_update(self, tmp_path, chunked):
        pyramid = self._make_pyramid(chunked)
        # Initialise zarr store
        init_pyramid_store(pyramid=pyramid, path=str(tmp_path))

        # Update region
        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="a",
            region={"time": slice(0, 5)},
        )

        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"
        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        # Confirm that the coordinates and data variables are unchanged in the datasets.
        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["t"]
        assert list(ds1.data_vars) == ["t"]

        ds0t = ds0.t.data.compute()
        ds1t = ds1.t.data.compute()

        # Confirm that the selected region has been written correctly.
        np.testing.assert_allclose(ds0.t.data[:, 0:5], pyramid[0].t.data[:, 0:5])
        np.testing.assert_allclose(ds1.t.data[:, 0:5], pyramid[1].t.data[:, 0:5])

        # Confirm that the rest of the data is nan.
        ds0_rest = ds0t[:, 5:]
        ds1_rest = ds1t[:, 5:]
        assert len(ds0_rest[~np.isnan(ds0_rest)]) == 0
        assert len(ds1_rest[~np.isnan(ds1_rest)]) == 0

        # Update second region
        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="a",
            region={"time": slice(5, 10)},
        )

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))
        # Confirm that both regions have been written correctly.
        np.testing.assert_allclose(ds0.t.data[:, 0:10], pyramid[0].t.data[:, 0:10])
        np.testing.assert_allclose(ds1.t.data[:, 0:10], pyramid[1].t.data[:, 0:10])
        # Confirm that the rest of the data is still nan.
        ds0_rest = ds0t[:, 10:]
        ds1_rest = ds1t[:, 10:]
        assert len(ds0_rest[~np.isnan(ds0_rest)]) == 0
        assert len(ds1_rest[~np.isnan(ds1_rest)]) == 0

    def test_save_pyramid_local_region_parallel(self, tmp_path):
        # Data has to be chunked for parallel  writes, and regions have to
        # be split along chunks.
        pyramid = self._make_pyramid(chunked=True)
        # Initialise zarr store
        init_pyramid_store(pyramid=pyramid, path=str(tmp_path))
        region_limits = [(i, i + 10) for i in range(0, 100, 10)]
        regions = [{"time": slice(*lim)} for lim in region_limits]

        def _save_section(region: dict):
            save_pyramid(pyramid, str(tmp_path), region=region)

        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            for future in [executor.submit(_save_section, i) for i in regions]:
                future.result()

        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"
        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        # Confirm that the coordinates and data variables are unchanged in the datasets.
        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["t"]
        assert list(ds1.data_vars) == ["t"]

        # Confirm that all regions have been written correctly.
        np.testing.assert_allclose(ds0.t.data, pyramid[0].t.data)
        np.testing.assert_allclose(ds1.t.data, pyramid[1].t.data)

    def test_save_pyramid_local_chunked_coord(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()

        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="w",
        )
        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0.compute(), pyramid[0].compute())
        xr.testing.assert_equal(ds1.compute(), pyramid[1].compute())

    def test_save_pyramid_local_chunked_coord_regions(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()
        init_pyramid_store(
            pyramid=pyramid,
            path=str(tmp_path),
            materialise=[
                "static",
            ],
        )

        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["static", "temp"]
        assert list(ds1.data_vars) == ["static", "temp"]

        assert ds0.static.chunksizes == pyramid[0].static.chunksizes
        assert ds0.latitude.chunksizes == pyramid[0].latitude.chunksizes
        assert ds1.static.chunksizes == pyramid[1].static.chunksizes
        assert ds1.latitude.chunksizes == pyramid[1].latitude.chunksizes

        # Confirm that the data is nan.
        ds0t = ds0.temp.data.compute()
        ds1t = ds1.temp.data.compute()
        assert len(ds0t[~np.isnan(ds0t)]) == 0
        assert len(ds1t[~np.isnan(ds1t)]) == 0

        ds0stat = ds0.static.compute()
        ds1stat = ds1.static.compute()
        xr.testing.assert_equal(ds0stat, pyramid[0].static.compute())
        xr.testing.assert_equal(ds1stat, pyramid[1].static.compute())

        regions = [
            {"time": slice(i, i + 2)} for i in range(0, pyramid[0].sizes["time"], 2)
        ]

        def _save_section(region: dict):
            save_pyramid(pyramid, str(tmp_path), region=region)

        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            for future in [executor.submit(_save_section, i) for i in regions]:
                future.result()

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0.compute(), pyramid[0].compute())
        xr.testing.assert_equal(ds1.compute(), pyramid[1].compute())


def test_save_pyramid_region_init_huge_finely_chunked(tmp_path):
    """``region='init'`` on ~70 TiB in ~573k chunks writes metadata only, in seconds.

    Its cost must not grow with the number of chunks: a metadata-only
    init takes well under 1 s here, one that builds a write graph about a minute.
    """
    ncell = 12 * 4**10
    ntime = 70 * 2**40 // (ncell * 4)  # float32 -> ~70 TiB
    shape = xr.DataArray(
        np.broadcast_to(np.float32(0), (ncell, ntime)), dims=("cell", "time")
    )
    t = xr.zeros_like(
        shape, chunks=(4**9, 128), chunked_array_type="dask"
    )  # lazy, no memory
    ds = xr.Dataset({"t": t})
    with ThreadPoolExecutor(1) as pool:
        start = os.times().elapsed
        pool.submit(init_pyramid_store, {10: ds}, str(tmp_path)).result()
        duration = os.times().elapsed - start

    assert duration < 10, (
        f"region='init' took {duration:.0f} s: its cost grows with the number of chunks"
    )
    store = tmp_path / "level_10.zarr"
    assert xr.open_zarr(store)["t"].encoding["chunks"] == (4**9, 128), (
        "init must keep the chunk layout"
    )
    assert not [f for f in os.listdir(store / "t") if not f.startswith(".")], (
        "init must not write data"
    )


class TestSavePyramidMultiscales:
    def _make_pyramid_chunked_coord(self) -> dict[int, xr.Dataset]:
        def _create_dataset(level: int) -> xr.Dataset:
            num_cells = 3 * (4 ** int(level / 2))
            lat = xr.DataArray(
                np.random.randn(num_cells),
                coords=[
                    np.arange(num_cells),
                ],
                dims=[
                    "cell",
                ],
            ).chunk({"cell": num_cells})
            static = xr.DataArray(
                [1.0, 2.0],
                coords=[
                    [0, 1],
                ],
                dims=["level"],
            ).chunk({"level": 2})

            ds = xr.Dataset(
                data_vars={
                    "temp": (
                        ["time", "level", "cell"],
                        np.random.randn(4, 2, num_cells),
                    ),
                    "static": static,
                },
                coords={
                    "time": ("time", range(4)),
                    "level": ("level", range(2)),
                    "cell": ("cell", range(num_cells)),
                    "latitude": lat,
                },
            )
            ds["temp"] = ds.temp.chunk({"time": 2})
            return ds

        # Pick levels so one keeps the coordinates and one drops them.
        return {level: _create_dataset(level) for level in (7, 11)}

    def test_save_pyramid_local_multiscales(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()
        zarr_store = tmp_path / "multi.zarr"
        save_pyramid(
            pyramid=pyramid,
            path=str(zarr_store),
            structure="multiscales",
        )

        dt = xr.open_datatree(zarr_store, chunks="auto")
        ds0 = dt["/level_7"].dataset
        ds1 = dt["/level_11"].dataset

        layout = PyramidMetadata.from_attrs(dt.attrs).levels
        assert [level.asset for level in layout] == ["level_11", "level_7"]

        xr.testing.assert_allclose(ds0.compute(), pyramid[7].compute())
        xr.testing.assert_allclose(
            ds1.compute(), pyramid[11].compute().drop_vars(["cell", "latitude"])
        )

    def test_save_pyramid_local_multiscales_regions(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()
        zarr_store = tmp_path / "multi.zarr"
        init_pyramid_store(
            pyramid=pyramid,
            path=str(zarr_store),
            materialise=[
                "static",
            ],
            structure="multiscales",
        )

        dt = xr.open_datatree(zarr_store, chunks="auto")
        ds0 = dt["/level_7"].dataset
        ds1 = dt["/level_11"].dataset

        xr.testing.assert_equal(ds0.coords, pyramid[7].coords)
        xr.testing.assert_equal(
            ds1.coords, pyramid[11].drop_vars(["cell", "latitude"]).coords
        )
        assert list(ds0.data_vars) == ["static", "temp"]
        assert list(ds1.data_vars) == ["static", "temp"]

        assert ds0.static.chunksizes == pyramid[7].static.chunksizes
        assert ds0.latitude.chunksizes == pyramid[7].latitude.chunksizes
        assert ds1.static.chunksizes == pyramid[11].static.chunksizes
        assert "latitude" not in ds1

        # Confirm that the data is nan.
        ds0t = ds0.temp.data.compute()
        ds1t = ds1.temp.data.compute()
        assert len(ds0t[~np.isnan(ds0t)]) == 0
        assert len(ds1t[~np.isnan(ds1t)]) == 0

        ds0stat = ds0.static.compute()
        ds1stat = ds1.static.compute()
        xr.testing.assert_equal(ds0stat, pyramid[7].static.compute())
        xr.testing.assert_equal(ds1stat, pyramid[11].static.compute())

        regions = [
            {"time": slice(i, i + 2)} for i in range(0, pyramid[7].sizes["time"], 2)
        ]

        def _save_section(region: dict):
            save_pyramid(
                pyramid,
                str(zarr_store),
                region=region,
                structure="multiscales",
            )

        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            for future in [executor.submit(_save_section, i) for i in regions]:
                future.result()

        dt = xr.open_datatree(zarr_store, chunks="auto")
        ds0 = dt["/level_7"].dataset
        ds1 = dt["/level_11"].dataset

        xr.testing.assert_allclose(ds0.compute(), pyramid[7].compute())
        xr.testing.assert_allclose(
            ds1.compute(), pyramid[11].compute().drop_vars(["cell", "latitude"])
        )
