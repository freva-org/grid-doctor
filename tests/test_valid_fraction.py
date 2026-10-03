"""Optional ``<name>_valid_fraction`` companions on pyramid levels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import helpers
from grid_doctor.helpers import coarsen_healpix, create_healpix_pyramid, save_pyramid

LEVEL = 3
NPIX = 12 * 4**LEVEL


def _finest(chunked: bool = True) -> xr.Dataset:
    rng = np.random.default_rng(11)
    land = np.zeros(NPIX, bool)
    land[100:300] = True  # a continent
    land[rng.random(NPIX) < 0.2] = True  # islands
    sst = rng.normal(15, 5, (3, NPIX))
    sst[:, land] = np.nan
    ice = sst.copy()
    ice[1, 400:500] = np.nan  # mask that changes with time
    ds = xr.Dataset(
        {
            "sst": (("time", "cell"), sst, {"grid_mapping": "crs"}),
            "ice": (("time", "cell"), ice),
            "flag": (("time", "cell"), np.where(np.isnan(sst), np.nan, 1.0)),
        },
        coords={"time": np.arange(3), "cell": np.arange(NPIX)},
        attrs={
            "healpix_nside": 2**LEVEL,
            "healpix_level": LEVEL,
            "healpix_order": "nested",
            "grid_doctor_method": "conservative",
        },
    )
    return ds.chunk(time=1, cell=192) if chunked else ds


def _pyramid(
    monkeypatch: pytest.MonkeyPatch, chunked: bool = True, **kwargs: Any
) -> dict[int, xr.Dataset]:
    finest = _finest(chunked)
    monkeypatch.setattr(helpers, "regrid_to_healpix", lambda ds, level, **kw: finest)
    return create_healpix_pyramid(finest, max_level=LEVEL, min_level=0, **kwargs)


def _expected_fraction(values: np.ndarray, level: int) -> np.ndarray:
    grouped = np.isfinite(values).reshape(*values.shape[:-1], -1, 4 ** (LEVEL - level))
    return grouped.mean(axis=-1)


@pytest.mark.parametrize("chunked", [True, False])
def test_full_fractions_on_every_level(
    monkeypatch: pytest.MonkeyPatch, chunked: bool
) -> None:
    pyramid = _pyramid(monkeypatch, chunked, valid_fraction=True)
    source = _finest(False)
    for level, ds in pyramid.items():
        for name in ("sst", "ice", "flag"):
            frac = ds[f"{name}_valid_fraction"]
            assert frac.dims == ("time", "cell")
            assert frac.dtype == np.float32
            assert frac.attrs["units"] == "1"
            assert ds[name].attrs["ancillary_variables"] == f"{name}_valid_fraction"
            np.testing.assert_allclose(
                frac.values, _expected_fraction(source[name].values, level), rtol=1e-6
            )
    # the time-varying mask shows up only in the full-shape fraction
    ice = pyramid[1]["ice_valid_fraction"].values
    assert not np.array_equal(ice[0], ice[1])


def test_static_fraction_is_cell_only(monkeypatch: pytest.MonkeyPatch) -> None:
    pyramid = _pyramid(monkeypatch, valid_fraction="static")
    source = _finest(False)
    for level, ds in pyramid.items():
        frac = ds["sst_valid_fraction"]
        assert frac.dims == ("cell",)
        np.testing.assert_allclose(
            frac.values, _expected_fraction(source["sst"].values[0], level), rtol=1e-6
        )
        assert frac.attrs.get("grid_mapping") == "crs"


def test_selection_and_per_variable_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    by_list = _pyramid(monkeypatch, valid_fraction=["sst", "ice"])[1]
    assert {"sst_valid_fraction", "ice_valid_fraction"} <= set(by_list)
    assert "flag_valid_fraction" not in by_list
    assert "ancillary_variables" not in by_list["flag"].attrs

    mixed = _pyramid(monkeypatch, valid_fraction={"sst": "static", "ice": True})[1]
    assert mixed["sst_valid_fraction"].dims == ("cell",)
    assert mixed["ice_valid_fraction"].dims == ("time", "cell")
    assert "flag_valid_fraction" not in mixed


def test_default_adds_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    for ds in _pyramid(monkeypatch).values():
        assert not any(str(n).endswith("_valid_fraction") for n in ds.variables)
        assert "ancillary_variables" not in ds["sst"].attrs


@pytest.mark.parametrize(
    "spec",
    [["nope"], {"sst": "sometimes"}, "full"],
)
def test_invalid_specs(monkeypatch: pytest.MonkeyPatch, spec: Any) -> None:
    with pytest.raises(ValueError, match="valid_fraction"):
        _pyramid(monkeypatch, valid_fraction=spec)


def test_requires_nested_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="nested"):
        _pyramid(monkeypatch, valid_fraction=True, nest=False)


def test_weighted_means_match_finest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The promise behind the option: weighted means agree across levels."""
    pyramid = _pyramid(monkeypatch, valid_fraction=True, min_valid_fraction=0.0)
    reference = pyramid[LEVEL]["ice"].mean("cell").values
    for ds in pyramid.values():
        weights = ds["ice_valid_fraction"].fillna(0)
        weighted = ds["ice"].weighted(weights).mean("cell").values
        np.testing.assert_allclose(weighted, reference, rtol=1e-6)


def test_mode_variables_get_fractions(monkeypatch: pytest.MonkeyPatch) -> None:
    pyramid = _pyramid(monkeypatch, valid_fraction=True, coarsen_mode="mode")
    source = _finest(False)
    for level in range(LEVEL):
        np.testing.assert_allclose(
            pyramid[level]["flag_valid_fraction"].values,
            _expected_fraction(source["flag"].values, level),
            rtol=1e-6,
        )


def test_coarsen_healpix_fraction_relative_to_input() -> None:
    ds = _finest()
    coarse = coarsen_healpix(ds, 1, valid_fraction=["sst"])
    np.testing.assert_allclose(
        coarse["sst_valid_fraction"].values,
        _expected_fraction(_finest(False)["sst"].values, 1),
        rtol=1e-6,
    )
    assert "ice_valid_fraction" not in coarse


def test_fractions_are_written(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pyramid = _pyramid(monkeypatch, valid_fraction={"sst": "static", "ice": True})
    save_pyramid(pyramid, str(tmp_path), mode="w")
    for level, ds in pyramid.items():
        stored = xr.open_zarr(tmp_path / f"level_{level}.zarr")
        assert stored["ice_valid_fraction"].dtype == np.float32
        np.testing.assert_allclose(
            stored["ice_valid_fraction"].values, ds["ice_valid_fraction"].values
        )
        assert stored["sst"].attrs["ancillary_variables"] == "sst_valid_fraction"
