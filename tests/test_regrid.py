"""High-level regridding tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from grid_doctor import remap
from grid_doctor.remap import _regrid as regrid_module
from grid_doctor.remap import regrid_to_healpix

from .helpers import with_healpix_metadata


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

    monkeypatch.setattr(regrid_module, "compute_healpix_weights", fake_compute)
    monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)  # cached_weights
    monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)

    result = regrid_to_healpix(
        regular_ds, level=1, method="nearest", missing_policy=missing_policy
    )
    assert result.sizes["cell"] == 3
    assert calls["apply"]["missing_policy"] == missing_policy


def test_regrid_to_healpix_reuses_default_store(
    regular_ds: xr.Dataset, monkeypatch: pytest.MonkeyPatch, _isolated_weights_store: Path
) -> None:
    """Without ``weights_path`` weights come from the default store: computed once, then reused."""
    computed: list[Path] = []

    def fake_compute(ds: xr.Dataset, level: int, **kwargs: Any) -> Path:
        path = Path(kwargs["weights_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("weights")
        computed.append(path)
        return path

    applied: list[Path] = []

    def fake_apply(ds: xr.Dataset, path: str | Path, **kwargs: Any) -> xr.Dataset:
        applied.append(Path(path))
        return xr.Dataset({"temperature": (("cell",), np.zeros(3))})

    monkeypatch.setattr(regrid_module, "compute_healpix_weights", fake_compute)
    monkeypatch.setattr(remap, "compute_healpix_weights", fake_compute)  # cached_weights
    monkeypatch.setattr(regrid_module, "apply_weight_file", fake_apply)

    regrid_to_healpix(regular_ds)  # level derived from the grid resolution
    regrid_to_healpix(regular_ds)
    assert len(computed) == 1
    assert applied == computed * 2
    assert computed[0].is_relative_to(_isolated_weights_store / "weights")  # the default store
