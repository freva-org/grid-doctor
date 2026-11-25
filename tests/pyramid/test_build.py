"""``coarsen_healpix`` and ``create_healpix_pyramid`` (``grid_doctor.pyramid._build``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor.io._save import save_pyramid
from grid_doctor.metadata import HealpixMetadata
from grid_doctor.pyramid import _build as build_module
from grid_doctor.pyramid import _levels as pyramid_levels
from grid_doctor.pyramid._build import (
    coarsen_healpix,
    create_healpix_pyramid,
)
from grid_doctor.pyramid._kernels import coarsen_mean
from grid_doctor.remap import _apply as apply_module
from grid_doctor.xarray_accessor import DatasetAccessor

from ..helpers import _weight_file, with_healpix_metadata


class TestCoarsenHealpix:
    def test_coarsens_nested_dataset(
        self, healpix_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pyramid_levels,
            "healpix_coords",
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
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
            lambda grid: (np.array([0.0] * 48), np.array([1.0] * 48)),
        )
        coarse = coarsen_healpix(healpix_ds, target_level=1)
        assert coarse.attrs["grid_doctor_coarsened_from_level"] == 3


class TestPyramidBuilders:
    def test_create_healpix_pyramid_nested(
        self, regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_regrid(self, level: int, **kwargs: Any) -> xr.Dataset:
            del kwargs
            npix = 12 * (4**level)
            return with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(npix))},
                    attrs=self._ds.attrs,
                ),
                level,
            )

        monkeypatch.setattr(DatasetAccessor, "regrid_to_healpix", fake_regrid)
        monkeypatch.setattr(
            build_module,
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

        def fake_regrid(self, level: int, **kwargs: Any) -> xr.Dataset:
            calls.append(level)
            return with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(12 * (4**level)))},
                    attrs=self._ds.attrs,
                ),
                level, nest=False,
            )

        monkeypatch.setattr(DatasetAccessor, "regrid_to_healpix", fake_regrid)
        pyramid = create_healpix_pyramid(
            regular_ds, max_level=2, min_level=0, nest=False
        )
        assert set(pyramid) == {0, 1, 2}
        assert calls == [2, 1, 0]

    def test_create_healpix_pyramid_forwards_coarsen_params(
        self, regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        coarsen_calls: list[dict[str, Any]] = []

        def fake_regrid(self, level: int, **kwargs: Any) -> xr.Dataset:
            npix = 12 * (4**level)
            return with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell",), np.zeros(npix))},
                    attrs=self._ds.attrs,
                ),
                level,
            )

        def fake_coarse_levels(
            finest: xr.Dataset, **kwargs: Any
        ) -> dict[int, xr.Dataset]:
            coarsen_calls.append(kwargs)
            return {}

        monkeypatch.setattr(DatasetAccessor, "regrid_to_healpix", fake_regrid)
        monkeypatch.setattr(build_module, "coarse_levels", fake_coarse_levels)
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


LEVEL = 3
NPIX = 12 * 4**LEVEL
NLAT, NLON = 8, 16


def _values_with_nans(shape: tuple[int, ...], seed: int = 54) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape)
    values[..., : shape[-1] // 3] = np.nan  # a clustered "coastline"
    values[rng.random(shape) < 0.3] = np.nan
    return values


def _source(chunked: bool) -> xr.Dataset:
    values = _values_with_nans((4, NLAT, NLON), seed=1)
    values[:, :2, :] = np.nan
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), values)},
        coords={
            "time": np.arange(4),
            "lat": np.linspace(-80, 80, NLAT),
            "lon": np.linspace(0, 337.5, NLON),
        },
    )
    return ds.chunk(time=1) if chunked else ds


class TestCoarsenHealpixChunked:
    def test_dask_matches_numpy(self) -> None:
        ds = with_healpix_metadata(
            xr.Dataset(
                {"sst": (("time", "cell"), _values_with_nans((2, NPIX)))},
                coords={"time": [0, 1], "cell": np.arange(NPIX)},
            ),
            LEVEL,
        )
        eager = coarsen_healpix(ds, 1, coarsen_mode="mean")
        lazy = coarsen_healpix(ds.chunk(cell=64), 1, coarsen_mode="mean")
        assert lazy["sst"].chunks is not None
        xr.testing.assert_allclose(eager, lazy.compute())


class TestSinglePassPyramid:
    def test_regrids_each_chunk_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        weights = _weight_file(tmp_path / "w.nc")
        calls: list[int] = []
        original = apply_module.apply_weights_nd

        def counting(values: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(values, **kwargs)

        monkeypatch.setattr(apply_module, "apply_weights_nd", counting)
        monkeypatch.setattr(
            "grid_doctor.xarray_accessor.DatasetAccessor.regrid_to_healpix",
            lambda self, level, **kw: remap.apply_weight_file(
                self._ds, weights, cell_chunks=256
            ),
        )
        pyramid = create_healpix_pyramid(_source(True), max_level=LEVEL, min_level=0)
        assert not calls, "building the pyramid must stay lazy"

        save_pyramid(pyramid, str(tmp_path / "out"), mode="w")
        n_time_chunks, n_row_blocks = 4, 3
        assert len(calls) == n_time_chunks * n_row_blocks

        for level, expected in pyramid.items():
            stored = xr.open_zarr(tmp_path / "out" / f"level_{level}.zarr")
            np.testing.assert_allclose(
                stored["sst"].values, expected["sst"].values, equal_nan=True
            )

    def test_levels_match_direct_coarsening(self, tmp_path: Path) -> None:
        weights = _weight_file(tmp_path / "w.nc")
        finest = remap.apply_weight_file(_source(True), weights, cell_chunks=256)
        pyramid = create_healpix_pyramid(
            _source(True),
            max_level=LEVEL,
            min_level=0,
            min_valid_fraction=0.3,
            weights_path=weights,
            cell_chunks=256,
        )
        values = finest["sst"].values
        for level in range(LEVEL):
            expected = coarsen_mean(
                values, factor=4 ** (LEVEL - level), min_valid_fraction=0.3
            )
            np.testing.assert_allclose(
                pyramid[level]["sst"].values, expected, equal_nan=True
            )
            assert pyramid[level].attrs["grid_doctor_coarsened_from_level"] == LEVEL
