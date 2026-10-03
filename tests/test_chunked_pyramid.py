"""Chunked regridding, exact chained coarsening and single-pass pyramid writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor.helpers import (
    coarsen_healpix,
    create_healpix_pyramid,
    save_pyramid,
)
from grid_doctor.pyramid import MeanPartials, coarsen_mean, coarsen_mean_steps

LEVEL = 3
NPIX = 12 * 4**LEVEL
NLAT, NLON = 8, 16


def _values_with_nans(shape: tuple[int, ...], seed: int = 54) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape)
    values[..., : shape[-1] // 3] = np.nan  # a clustered "coastline"
    values[rng.random(shape) < 0.3] = np.nan
    return values


def _weight_file(path: Path, n_target: int = NPIX, seed: int = 7) -> Path:
    """Synthetic conservative-style weights: 3 normalised sources per target."""
    rng = np.random.default_rng(seed)
    n_source = NLAT * NLON
    rows = np.repeat(np.arange(n_target), 3)
    cols = rng.integers(0, n_source, size=rows.size)
    weights = rng.random(rows.size)
    weights /= np.bincount(rows, weights)[rows]
    xr.Dataset(
        {
            "row": ("n_s", rows + 1),
            "col": ("n_s", cols + 1),
            "S": ("n_s", weights),
        },
        attrs={
            "grid_doctor_level": LEVEL,
            "grid_doctor_order": "nested",
            "grid_doctor_method": "conservative",
            "grid_doctor_source_dims": json.dumps(["lat", "lon"]),
        },
    ).to_netcdf(path)
    return path


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


class TestMeanPartials:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_chained_partials_match_direct_mean(self, chunked: bool) -> None:
        values = _values_with_nans((2, NPIX))
        arr = da.from_array(values, chunks=(1, 64)) if chunked else values
        partials = MeanPartials.from_values(arr).rechunk(64)
        for steps in range(1, LEVEL + 1):
            expected = coarsen_mean(values, factor=4**steps, min_valid_fraction=0.4)
            np.testing.assert_allclose(
                np.asarray(partials.mean(0.4)), expected, equal_nan=True
            )
            partials = partials.coarsen().rechunk(64)

    def test_dask_stays_chunked_along_cells(self) -> None:
        arr = da.from_array(_values_with_nans((2, NPIX)), chunks=(1, 64))
        out = coarsen_mean_steps(arr, 1, min_valid_fraction=0.5, cell_chunk=64)
        assert isinstance(out, da.Array)
        assert out.chunks[-1] == (64, 64, 64)


class TestCoarsenHealpixChunked:
    def test_dask_matches_numpy(self) -> None:
        ds = xr.Dataset(
            {"sst": (("time", "cell"), _values_with_nans((2, NPIX)))},
            coords={"time": [0, 1], "cell": np.arange(NPIX)},
            attrs={
                "healpix_nside": 2**LEVEL,
                "healpix_level": LEVEL,
                "healpix_order": "nested",
            },
        )
        eager = coarsen_healpix(ds, 1, coarsen_mode="mean")
        lazy = coarsen_healpix(ds.chunk(cell=64), 1, coarsen_mode="mean")
        assert lazy["sst"].chunks is not None
        xr.testing.assert_allclose(eager, lazy.compute())


class TestRowBlockedRegrid:
    def test_matches_unchunked(self, tmp_path: Path) -> None:
        weights = _weight_file(tmp_path / "w.nc")
        reference = remap.apply_weight_file(_source(False), weights, cell_chunks=None)
        chunked = remap.apply_weight_file(_source(True), weights, cell_chunks=256)
        assert chunked["sst"].chunks == ((1, 1, 1, 1), (256, 256, 256))
        xr.testing.assert_identical(reference, chunked.compute())

    @pytest.mark.parametrize("policy", ["renormalize", "propagate"])
    def test_missing_policies(self, tmp_path: Path, policy: str) -> None:
        weights = _weight_file(tmp_path / "w.nc")
        kwargs: dict[str, Any] = {"missing_policy": policy}
        reference = remap.apply_weight_file(
            _source(False), weights, cell_chunks=None, **kwargs
        )
        chunked = remap.apply_weight_file(
            _source(True), weights, cell_chunks=100, **kwargs
        )
        np.testing.assert_allclose(
            reference["sst"].values, chunked["sst"].values, equal_nan=True
        )

    def test_auto_skips_small_targets(self, tmp_path: Path) -> None:
        weights = _weight_file(tmp_path / "w.nc")
        out = remap.apply_weight_file(_source(True), weights)
        assert out["sst"].chunks[-1] == (NPIX,)

    def test_rejects_tiny_chunks(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cell_chunks"):
            remap.apply_weight_file(
                _source(True), _weight_file(tmp_path / "w.nc"), cell_chunks=2
            )


class TestSinglePassPyramid:
    def test_regrids_each_chunk_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        weights = _weight_file(tmp_path / "w.nc")
        calls: list[int] = []
        original = remap.apply_weights_nd

        def counting(values: Any, **kwargs: Any) -> Any:
            calls.append(1)
            return original(values, **kwargs)

        monkeypatch.setattr(remap, "apply_weights_nd", counting)
        monkeypatch.setattr(
            "grid_doctor.helpers.regrid_to_healpix",
            lambda ds, level, **kw: remap.apply_weight_file(
                ds, weights, cell_chunks=256
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


class TestSavePyramidSingleStore:
    @staticmethod
    def _pyramid() -> dict[int, xr.Dataset]:
        values = _values_with_nans((4, NPIX))
        ds = xr.Dataset(
            {"sst": (("time", "cell"), values)},
            coords={"time": np.arange(4), "cell": np.arange(NPIX)},
        )
        return {LEVEL: ds.chunk(time=1, cell=192)}

    def test_encoding_and_nan_fill(self, tmp_path: Path) -> None:
        pyramid = self._pyramid()
        save_pyramid(
            pyramid,
            str(tmp_path),
            mode="w",
            encoding={LEVEL: {"sst": {"dtype": "float32", "_FillValue": -999.0}}},
        )
        stored = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr")
        raw = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr", mask_and_scale=False)
        assert raw["sst"].dtype == np.float32
        assert (raw["sst"].values == -999.0).any()
        np.testing.assert_allclose(
            stored["sst"].values,
            pyramid[LEVEL]["sst"].values.astype(np.float32),
            equal_nan=True,
        )

    def test_region_write(self, tmp_path: Path) -> None:
        pyramid = self._pyramid()
        save_pyramid(pyramid, str(tmp_path), mode="w", compute=False)
        save_pyramid(
            pyramid,
            str(tmp_path),
            mode="r+",
            region={"time": slice(1, 3)},
        )
        stored = xr.open_zarr(tmp_path / f"level_{LEVEL}.zarr")["sst"].values
        expected = pyramid[LEVEL]["sst"].values
        np.testing.assert_allclose(stored[1:3], expected[1:3], equal_nan=True)
        assert np.isnan(stored[[0, 3]]).all()
