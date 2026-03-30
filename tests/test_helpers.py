"""Tests for `grid_doctor.helpers`."""

from __future__ import annotations

from typing import Any
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor import helpers
from grid_doctor.helpers import (
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
            lambda ds, level: xr.Dataset(
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
