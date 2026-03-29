"""High-level regridding tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor.helpers import latlon_to_healpix_pyramid
from grid_doctor.remap import regrid_to_healpix


@pytest.mark.parametrize("method", ["nearest", "conservative"])
def test_latlon_to_healpix_pyramid_weight_methods(
    regular_ds: xr.Dataset,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_regrid(ds: xr.Dataset, level: int, **kwargs: Any) -> xr.Dataset:
        del kwargs
        return xr.Dataset(
            {
                name: (("cell",), np.arange(12 * (4**level), dtype=np.float64))
                for name in ds.data_vars
            },
            attrs=ds.attrs
            | {
                "healpix_level": level,
                "healpix_nside": 2**level,
                "healpix_order": "nested",
            },
        )

    monkeypatch.setattr("grid_doctor.helpers.regrid_to_healpix", fake_regrid)
    pyramid = latlon_to_healpix_pyramid(
        regular_ds,
        min_level=1,
        max_level=2,
        method=method,
        weights_path=str(tmp_path / "weights.nc"),
    )
    assert set(pyramid) == {1, 2}


@pytest.mark.parametrize(
    "missing_policy",
    ["renormalize", "propagate"],
)
def test_regrid_to_healpix_weight_methods_delegate(
    regular_ds: xr.Dataset,
    missing_policy: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    weight_file = tmp_path / "weights.nc"
    calls: dict[str, Any] = {}

    def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
        del ds, level
        calls["compute"] = kwargs
        weight_file.write_text("ok")
        return weight_file

    def fake_apply(
        ds: xr.Dataset,
        path: str | Path,
        *,
        missing_policy: str = "renormalize",
        **kwargs: Any,
    ) -> xr.Dataset:
        calls["apply"] = {"path": Path(path), "missing_policy": missing_policy}
        return xr.Dataset(
            {"temperature": (("cell",), np.arange(3, dtype=np.float64))},
            attrs=ds.attrs.copy(),
        )

    monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)
    monkeypatch.setattr(remap, "apply_weight_file", fake_apply)

    result = regrid_to_healpix(
        regular_ds, level=1, method="nearest", missing_policy=missing_policy
    )
    assert result.sizes["cell"] == 3
    assert calls["apply"]["missing_policy"] == missing_policy
