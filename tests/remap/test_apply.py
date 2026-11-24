"""Applying weight files (``grid_doctor.remap._apply``)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor.remap import _apply as apply_module
from grid_doctor.remap import apply_weight_file
from grid_doctor.remap._apply import (
    _flattened_size,
    _guess_source_dims_from_size,
    _parse_source_dims_attr,
    _resolve_source_dims_for_weight_application,
)

from ..helpers import _weight_file

# ===================================================================
# Source-dimension resolution
# ===================================================================


class TestFlattenedSize:
    def test_single_dim(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        assert _flattened_size(ds, ("cell",)) == 5

    def test_multiple_dims(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        assert _flattened_size(ds, ("y", "x")) == 12

    def test_missing_dim_raises(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        with pytest.raises(ValueError, match="Missing"):
            _flattened_size(ds, ("cell", "time"))


class TestParseSourceDimsAttr:
    def test_none_returns_none(self) -> None:
        assert _parse_source_dims_attr(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_source_dims_attr("") is None

    def test_json_string(self) -> None:
        assert _parse_source_dims_attr('["y", "x"]') == ("y", "x")

    def test_plain_string(self) -> None:
        assert _parse_source_dims_attr("cell") == ("cell",)

    def test_comma_separated_string(self) -> None:
        assert _parse_source_dims_attr("y, x") == ("y", "x")

    def test_tuple(self) -> None:
        assert _parse_source_dims_attr(("y", "x")) == ("y", "x")

    def test_list(self) -> None:
        assert _parse_source_dims_attr(["y", "x"]) == ("y", "x")

    def test_invalid_value_returns_none(self) -> None:
        assert _parse_source_dims_attr(123) is None
        assert _parse_source_dims_attr(["y", 1]) is None


class TestGuessSourceDimsFromSize:
    def test_prefers_unstructured_dim(self) -> None:
        ds = xr.Dataset(
            coords={
                "cell": np.arange(10),
                "member": np.arange(10),
            }
        )
        assert _guess_source_dims_from_size(ds, 10) == ("cell",)

    def test_unique_non_unstructured_match(self) -> None:
        ds = xr.Dataset(coords={"points": np.arange(7), "time": np.arange(3)})
        assert _guess_source_dims_from_size(ds, 7) == ("points",)

    def test_ambiguous_match_returns_none(self) -> None:
        ds = xr.Dataset(coords={"a": np.arange(8), "b": np.arange(8)})
        assert _guess_source_dims_from_size(ds, 8) is None

    def test_no_match_returns_none(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        assert _guess_source_dims_from_size(ds, 99) is None


class TestResolveSourceDims:
    def test_uses_explicit_source_dims(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=5,
            grid=None,
            source_dims=("cell",),
            source_units="auto",
            stored_source_dims=None,
        )
        assert result == ("cell",)

    def test_explicit_source_dims_size_mismatch_raises(self) -> None:
        ds = xr.Dataset(coords={"cell": np.arange(5)})
        with pytest.raises(ValueError, match="mismatch"):
            _resolve_source_dims_for_weight_application(
                ds,
                n_source=4,
                grid=None,
                source_dims=("cell",),
                source_units="auto",
                stored_source_dims=None,
            )

    def test_uses_stored_source_dims_when_valid(self) -> None:
        ds = xr.Dataset(coords={"y": np.arange(3), "x": np.arange(4)})
        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=12,
            grid=None,
            source_dims=None,
            source_units="auto",
            stored_source_dims=("y", "x"),
        )
        assert result == ("y", "x")

    def test_uses_grid_when_dataset_lacks_geometry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5)))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )
        grid = xr.Dataset(coords={"cell": np.arange(5)})

        monkeypatch.setattr(
            apply_module, "_source_centre_arrays",
            lambda obj, source_units="auto": (
                np.arange(5, dtype=np.float64),
                np.arange(5, dtype=np.float64),
                ("cell",),
            ),
        )
        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=5,
            grid=grid,
            source_dims=None,
            source_units="auto",
            stored_source_dims=None,
        )
        assert result == ("cell",)

    def test_falls_back_to_size_guess(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5)))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )

        def raise_no_geometry(
            obj: xr.Dataset, source_units: str = "auto"
        ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
            raise ValueError("no geometry")

        monkeypatch.setattr(apply_module, "_source_centre_arrays", raise_no_geometry)
        result = _resolve_source_dims_for_weight_application(
            ds,
            n_source=5,
            grid=None,
            source_dims=None,
            source_units="auto",
            stored_source_dims=None,
        )
        assert result == ("cell",)

    def test_geometry_mismatch_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {"temperature": (("time", "cell"), np.ones((2, 5)))},
            coords={"time": [0, 1], "cell": np.arange(5)},
        )
        monkeypatch.setattr(
            apply_module, "_source_centre_arrays",
            lambda obj, source_units="auto": (
                np.arange(8, dtype=np.float64),
                np.arange(8, dtype=np.float64),
                ("cell",),
            ),
        )
        with pytest.raises(ValueError, match="source_dims"):
            _resolve_source_dims_for_weight_application(
                ds,
                n_source=6,
                grid=None,
                source_dims=None,
                source_units="auto",
                stored_source_dims=None,
            )


# ===================================================================
# Weight-file application (end-to-end through apply_weight_file)
# ===================================================================


class TestApplyWeightFile:
    @staticmethod
    def _unused_trailing_sources(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_source: int, source_size: int | None
    ) -> tuple[xr.Dataset, Path]:
        """Weights where only source cells 1 and 3 are used, like ``nearest`` output."""
        ds = xr.Dataset(
            {"t": ("cell", np.arange(1.0, n_source + 1.0))},
            coords={"cell": np.arange(n_source)},
        )
        # Toy weights with 2 targets are no real HEALPix grid: leave the
        # level unset so no coordinates are attached.
        attrs: dict[str, Any] = {}
        if source_size is not None:
            attrs["grid_doctor_source_size"] = source_size
        path = tmp_path / "weights.nc"
        xr.Dataset(
            {
                "row": ("nnz", np.array([1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 3], dtype=np.int32)),
                "S": ("nnz", np.array([1.0, 1.0])),
            },
            attrs=attrs,
        ).to_netcdf(path)

        monkeypatch.setattr(
            apply_module, "_source_centre_arrays",
            lambda ds, source_units="auto": (
                np.zeros(n_source),
                np.arange(float(n_source)),
                ("cell",),
            ),
        )
        return ds, path

    def test_uses_stored_source_size(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Trailing source cells without weights must not shrink the matrix."""
        ds, path = self._unused_trailing_sources(tmp_path, monkeypatch, n_source=5, source_size=5)
        result = apply_weight_file(ds, path)
        np.testing.assert_array_equal(result["t"].values, [1.0, 3.0])

    def test_infers_source_size_without_attribute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Older weight files without grid_doctor_source_size still work."""
        ds, path = self._unused_trailing_sources(tmp_path, monkeypatch, n_source=3, source_size=None)
        result = apply_weight_file(ds, path)
        np.testing.assert_array_equal(result["t"].values, [1.0, 3.0])

    def test_renormalizes_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[np.array([1.0, np.nan, 3.0], dtype=np.float32)]],
                    dims=("time", "level", "cell"),
                    coords={
                        "time": [0],
                        "level": [0],
                        "cell": [0, 1, 2],
                    },
                ),
                "static": xr.DataArray(
                    [10.0, 20.0],
                    dims=("level",),
                    coords={"level": [0, 1]},
                ),
            },
            attrs={"source": "synthetic"},
        )

        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 3], dtype=np.int32)),
                "S": ("nnz", np.array([0.5, 0.5, 1.0], dtype=np.float64)),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        monkeypatch.setattr(
            apply_module, "_source_centre_arrays",
            lambda ds, source_units="auto": (
                np.array([0.0, 1.0, 2.0], dtype=np.float64),
                np.array([10.0, 20.0, 30.0], dtype=np.float64),
                ("cell",),
            ),
        )
        result = apply_weight_file(ds, path, missing_policy="renormalize")

        assert "cell" in result.dims
        assert result.sizes["cell"] == 2
        out = result["temperature"].isel(time=0, level=0).values
        np.testing.assert_allclose(out, [1.0, 3.0])
        np.testing.assert_allclose(result["static"].values, ds["static"].values)

    def test_with_explicit_source_dims(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[1.0, np.nan, 3.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1, 2]},
                ),
            },
        )
        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 3], dtype=np.int32)),
                "S": ("nnz", np.array([0.5, 0.5, 1.0], dtype=np.float64)),
            },
            attrs={"grid_doctor_source_dims": json.dumps(["cell"])},
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        result = apply_weight_file(
            ds,
            path,
            source_dims=("cell",),
        )
        assert result.sizes["cell"] == 2
        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, [1.0, 3.0])

    def test_stored_source_dims_without_geometry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    np.arange(6, dtype=np.float64).reshape(1, 2, 3),
                    dims=("time", "y", "x"),
                    coords={"time": [0], "y": [0, 1], "x": [0, 1, 2]},
                )
            },
        )
        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2, 6], dtype=np.int32)),
                "S": ("nnz", np.array([0.5, 0.5, 1.0], dtype=np.float64)),
            },
            attrs={
                "grid_doctor_source_dims": json.dumps(["y", "x"]),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        def no_geometry(
            obj: xr.Dataset, source_units: str = "auto"
        ) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
            raise ValueError("geometry not embedded")

        monkeypatch.setattr(apply_module, "_source_centre_arrays", no_geometry)

        result = apply_weight_file(ds, path)
        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, [0.5, 5.0])

    def test_rejects_mismatched_geometry(
        self,
        regular_ds: xr.Dataset,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "weights.nc"
        xr.Dataset(
            {
                "row": ("nnz", np.array([1], dtype=np.int32)),
                "col": ("nnz", np.array([100], dtype=np.int32)),
                "S": ("nnz", np.array([1.0], dtype=np.float64)),
            }
        ).to_netcdf(path)
        with pytest.raises(ValueError, match="does not match|source_dims"):
            apply_weight_file(regular_ds, path)

    def test_geometry_mismatch_with_explicit_dims_raises(
        self,
        tmp_path: Path,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[1.0, 2.0, 3.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1, 2]},
                )
            }
        )
        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1], dtype=np.int32)),
                "col": ("nnz", np.array([4], dtype=np.int32)),
                "S": ("nnz", np.array([1.0], dtype=np.float64)),
            }
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)
        with pytest.raises(ValueError, match="mismatch"):
            apply_weight_file(ds, path, source_dims=("cell",))

    def test_backend_parameter_forwarded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verify the backend kwarg reaches apply_weights_nd."""
        ds = xr.Dataset(
            {
                "t": xr.DataArray(
                    [[1.0, 2.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1]},
                )
            },
        )
        weights = xr.Dataset(
            {
                "row": ("nnz", np.array([1, 2], dtype=np.int32)),
                "col": ("nnz", np.array([1, 2], dtype=np.int32)),
                "S": ("nnz", np.array([1.0, 1.0], dtype=np.float64)),
            },
            attrs={
                "grid_doctor_source_dims": json.dumps(["cell"]),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        result = apply_weight_file(ds, path, backend="scipy")
        np.testing.assert_allclose(result["t"].isel(time=0).values, [1.0, 2.0])
# ===================================================================
# End-to-end weight application with grid argument
# ===================================================================


class TestApplyWeightFileWithGrid:
    def test_with_grid_argument(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(
                    [[1.0, 2.0, 3.0]],
                    dims=("time", "cell"),
                    coords={"time": [0], "cell": [0, 1, 2]},
                )
            }
        )
        grid = xr.Dataset(coords={"cell": [0, 1, 2]})

        weights = xr.Dataset(
            {
                "row": (
                    "nnz",
                    np.array([1, 2, 2], dtype=np.int32),
                ),
                "col": (
                    "nnz",
                    np.array([1, 2, 3], dtype=np.int32),
                ),
                "S": (
                    "nnz",
                    np.array([1.0, 0.25, 0.75], dtype=np.float64),
                ),
            },
        )
        path = tmp_path / "weights.nc"
        weights.to_netcdf(path)

        monkeypatch.setattr(
            apply_module, "_source_centre_arrays",
            lambda obj, source_units="auto": (
                np.array([0.0, 1.0, 2.0], dtype=np.float64),
                np.array([10.0, 20.0, 30.0], dtype=np.float64),
                ("cell",),
            ),
        )
        result = apply_weight_file(ds, path, grid=grid)
        out = result["temperature"].isel(time=0).values
        np.testing.assert_allclose(out, [1.0, 2.75])
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


def test_row_blocks_share_weight_buffers() -> None:
    from scipy.sparse import random as sparse_random

    matrix = sparse_random(100, 40, density=0.2, format="csr", random_state=0)
    blocks = apply_module._row_blocks(matrix, 32)
    assert [b.shape[0] for b in blocks] == [32, 32, 32, 4]
    assert all(np.shares_memory(b.data, matrix.data) for b in blocks)
    from scipy.sparse import vstack

    assert (vstack(blocks) != matrix).nnz == 0


@pytest.mark.parametrize(
    ("chunked", "cell_chunks", "expected"),
    [(True, 256, False), (True, None, False), (False, None, True)],
)
def test_dask_tasks_use_serial_kernels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chunked: bool,
    cell_chunks: int | None,
    expected: bool,
) -> None:
    """Dask threads must not start their own thread teams (thread limits)."""
    seen: list[bool] = []
    original = apply_module.apply_weights_nd

    def recording(values: Any, **kwargs: Any) -> Any:
        seen.append(kwargs["parallel"])
        return original(values, **kwargs)

    monkeypatch.setattr(apply_module, "apply_weights_nd", recording)
    weights = _weight_file(tmp_path / "w.nc")
    remap.apply_weight_file(
        _source(chunked), weights, cell_chunks=cell_chunks
    ).compute()
    assert seen and set(seen) == {expected}
