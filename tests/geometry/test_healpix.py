"""HEALPix cell centres and coordinates (``grid_doctor._geometry.healpix``)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from grid_doctor._geometry import healpix as healpix_module
from grid_doctor._geometry.healpix import (
    _attach_healpix_coords,
    _healpix_centres,
)
from grid_doctor.metadata import HealpixGrid

from ..helpers import _FakeHealpixModule

# ===================================================================
# HEALPix centres
# ===================================================================


class TestHealpixCentres:
    def test_uses_healpix_geo_module(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            healpix_module,
            "_require_healpix_geo_module",
            lambda grid: (_FakeHealpixModule(), {}),
        )
        lat, lon = _healpix_centres(HealpixGrid(level=1))
        assert lat.shape == lon.shape
        assert lat.size == 48
# ===================================================================
# Attach HEALPix coords
# ===================================================================


class TestAttachHealpixCoords:
    def test_attach_coords(self) -> None:
        ds = xr.Dataset({"t": ("cell", np.arange(12.0))})
        result = _attach_healpix_coords(ds, HealpixGrid(level=0, indexing_scheme="ring"), method="nearest")
        assert "latitude" in result.coords
        assert "longitude" in result.coords
        np.testing.assert_array_equal(result["cell"].values, np.arange(12))
        assert result["cell"].attrs["standard_name"] == "healpix_index"
        assert result["crs"].attrs == {
            "grid_mapping_name": "healpix",
            "indexing_scheme": "ring",
            "refinement_level": 0,
        }
        assert result["t"].attrs["grid_mapping"] == "crs"
        assert result.attrs["grid_doctor_method"] == "nearest"

    def test_rejects_wrong_cell_count(self) -> None:
        ds = xr.Dataset({"t": ("cell", np.arange(2.0))})
        with pytest.raises(ValueError, match="conflicting sizes|cells"):
            _attach_healpix_coords(ds, HealpixGrid(level=2))
