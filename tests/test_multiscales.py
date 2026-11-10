"""Tests for the multiscales/dggs layout of datatree pyramids."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from grid_doctor import coarsen_healpix, init_pyramid_store, save_pyramid
from grid_doctor.io import maybe_drop_coordinates
from grid_doctor.metadata import (
    DGGS_CONVENTION,
    MULTISCALES_CONVENTION,
    HealpixGrid,
    HealpixMetadata,
    Provenance,
    PyramidLevel,
    PyramidMetadata,
    level_group,
)
from grid_doctor.metadata._pyramid import pyramid_root_attrs

SCHEMAS = Path(__file__).parent / "data" / "zarr_conventions"
FINEST = 3


def _finest(level: int = FINEST, *, nest: bool = True, method: str = "conservative") -> xr.Dataset:
    npix = 12 * 4**level
    ds = xr.Dataset(
        {"t": (("time", "cell"), np.random.default_rng(1).random((2, npix)))},
        coords={"time": [0, 1], "cell": np.arange(npix, dtype=np.int64)},
        attrs={"title": "pyramid test", "Conventions": "CF-1.8 ACDD-1.3"},
    )
    meta = HealpixMetadata(HealpixGrid.from_nest(level, nest=nest), Provenance(method=method))
    return meta.apply(ds)


def _pyramid(coarsen_mode: str = "auto", method: str = "conservative") -> dict[int, xr.Dataset]:
    finest = _finest(method=method)
    pyramid = {FINEST: finest}
    for level in range(FINEST):
        pyramid[level] = coarsen_healpix(finest, level, coarsen_mode=coarsen_mode)  # type: ignore[arg-type]
    return pyramid


def _validate(attrs: dict[str, object], schema_file: str) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((SCHEMAS / schema_file).read_text())
    document = {"zarr_format": 2, "node_type": "group", "attributes": attrs}
    # Round-trip through JSON: that is what ends up in the store.
    jsonschema.validate(json.loads(json.dumps(document)), schema)


# ===================================================================
# Layout
# ===================================================================


class TestFromPyramid:
    def test_levels_are_derived_from_the_finest(self) -> None:
        meta = PyramidMetadata.from_pyramid(_pyramid())
        assert [level.asset for level in meta.levels] == ["level_3", "level_2", "level_1", "level_0"]
        finest, *coarse = meta.levels
        assert finest.derived_from is None
        assert finest.grid == HealpixGrid(level=FINEST)
        assert all(level.derived_from == "level_3" for level in coarse)
        assert {level.resampling_method for level in coarse} == {"average"}

    def test_mode_coarsening(self) -> None:
        meta = PyramidMetadata.from_pyramid(_pyramid(method="nearest"))
        assert {level.resampling_method for level in meta.levels[1:]} == {"mode"}

    def test_independently_regridded_levels_are_not_derived(self) -> None:
        pyramid = {level: _finest(level, nest=False) for level in (1, 2)}
        meta = PyramidMetadata.from_pyramid(pyramid)
        assert all(level.derived_from is None for level in meta.levels)
        layout = meta.to_attrs()["multiscales"]["layout"]  # type: ignore[index]
        assert [entry["dggs"]["indexing_scheme"] for entry in layout] == ["ring", "ring"]

    def test_source_level_outside_pyramid_is_not_referenced(self) -> None:
        pyramid = _pyramid()
        del pyramid[FINEST]
        meta = PyramidMetadata.from_pyramid(pyramid)
        assert all(level.derived_from is None for level in meta.levels)

    def test_levels_without_metadata_are_listed_by_path(self) -> None:
        pyramid = {1: xr.Dataset({"t": ("cell", np.zeros(5))})}
        attrs = PyramidMetadata.from_pyramid(pyramid).to_attrs()
        assert attrs["multiscales"] == {"layout": [{"asset": "level_1"}]}
        assert attrs["zarr_conventions"] == [MULTISCALES_CONVENTION.to_dict()]


class TestToAttrs:
    def test_layout_follows_dggs_pr25(self) -> None:
        attrs = PyramidMetadata.from_pyramid(_pyramid()).to_attrs()
        assert attrs["zarr_conventions"] == [
            MULTISCALES_CONVENTION.to_dict(),
            DGGS_CONVENTION.to_dict(),
        ]
        multiscales = attrs["multiscales"]
        assert multiscales["resampling_method"] == "average"  # type: ignore[index]
        base, level_2, level_1, level_0 = multiscales["layout"]  # type: ignore[index]
        assert base == {
            "asset": "level_3",
            "dggs": {
                "name": "healpix",
                "refinement_level": 3,
                "indexing_scheme": "nested",
                "spatial_dimension": "cell",
                "coordinate": "cell",
                "compression": "none",
            },
        }
        assert level_2 == {
            "asset": "level_2",
            "derived_from": "level_3",
            "transform": {"scale": [2.0]},
            "dggs": {"refinement_level": 2},
        }
        assert level_1["transform"] == {"scale": [4.0]}
        assert level_0["transform"] == {"scale": [8.0]}

    def test_mixed_resampling_methods_are_per_level(self) -> None:
        grid = HealpixGrid(level=3)
        meta = PyramidMetadata(
            levels=(
                PyramidLevel("level_3", grid),
                PyramidLevel("level_2", HealpixGrid(level=2), derived_from="level_3", resampling_method="average"),
                PyramidLevel("level_1", HealpixGrid(level=1), derived_from="level_3", resampling_method="mode"),
            )
        )
        multiscales = meta.to_attrs()["multiscales"]
        assert "resampling_method" not in multiscales  # type: ignore[operator]
        methods = [entry.get("resampling_method") for entry in multiscales["layout"]]  # type: ignore[index]
        assert methods == [None, "average", "mode"]

    def test_implicit_finest_level(self) -> None:
        """A finest level without coordinates: coarse levels add theirs."""
        pyramid = _pyramid()
        pyramid[FINEST] = maybe_drop_coordinates(pyramid[FINEST], FINEST, write_coords=False)
        layout = PyramidMetadata.from_pyramid(pyramid).to_attrs()["multiscales"]["layout"]  # type: ignore[index]
        assert "coordinate" not in layout[0]["dggs"]
        assert layout[1]["dggs"] == {"refinement_level": 2, "coordinate": "cell", "compression": "none"}

    def test_coarse_level_without_coordinate_gets_full_object(self) -> None:
        """A reduced object cannot remove keys of its source."""
        meta = PyramidMetadata(
            levels=(
                PyramidLevel("level_3", HealpixGrid(level=3)),
                PyramidLevel("level_2", HealpixGrid(level=2), coordinate=None, derived_from="level_3"),
            )
        )
        layout = meta.to_attrs()["multiscales"]["layout"]  # type: ignore[index]
        assert layout[1]["dggs"]["name"] == "healpix"
        assert "coordinate" not in layout[1]["dggs"]

    def test_earth_radius_becomes_sphere(self) -> None:
        meta = PyramidMetadata(levels=(PyramidLevel("level_1", HealpixGrid(level=1, earth_radius=6371000.0)),))
        dggs = meta.to_attrs()["multiscales"]["layout"][0]["dggs"]  # type: ignore[index]
        assert dggs["ellipsoid"] == {"name": "sphere", "radius": 6371000.0}

    @pytest.mark.parametrize(
        "pyramid",
        [
            pytest.param(_pyramid(), id="mean"),
            pytest.param(_pyramid(method="nearest"), id="mode"),
            pytest.param({level: _finest(level, nest=False) for level in (1, 2)}, id="ring"),
        ],
    )
    @pytest.mark.parametrize("schema", ["multiscales-v0.1.schema.json", "dggs-pr25.schema.json"])
    def test_validates_against_published_schemas(self, pyramid: dict[int, xr.Dataset], schema: str) -> None:
        _validate(PyramidMetadata.from_pyramid(pyramid).to_attrs(), schema)

    def test_full_dggs_object_validates_against_dggs_v1(self) -> None:
        layout = PyramidMetadata.from_pyramid(_pyramid()).to_attrs()["multiscales"]["layout"]  # type: ignore[index]
        _validate(
            {"zarr_conventions": [DGGS_CONVENTION.to_dict()], "dggs": layout[0]["dggs"]},
            "dggs-v1.schema.json",
        )


class TestFromAttrs:
    @pytest.mark.parametrize(
        "pyramid",
        [
            pytest.param(_pyramid(), id="mean"),
            pytest.param({level: _finest(level, nest=False) for level in (1, 2)}, id="ring"),
        ],
    )
    def test_round_trip(self, pyramid: dict[int, xr.Dataset]) -> None:
        meta = PyramidMetadata.from_pyramid(pyramid)
        stored = json.loads(json.dumps(meta.to_attrs()))
        assert PyramidMetadata.from_attrs(stored) == meta

    def test_asset_lookup(self) -> None:
        meta = PyramidMetadata.from_pyramid(_pyramid())
        assert meta.asset(1) == level_group(1) == "level_1"
        with pytest.raises(KeyError):
            meta.asset(9)

    def test_requires_layout(self) -> None:
        with pytest.raises(ValueError, match="no `multiscales` layout"):
            PyramidMetadata.from_attrs({"title": "nope"})

    def test_rejects_other_dggs(self) -> None:
        attrs = {
            "multiscales": {
                "layout": [{"asset": "0", "dggs": {"name": "h3", "refinement_level": 2, "spatial_dimension": "cells"}}]
            }
        }
        with pytest.raises(ValueError, match="Not a HEALPix"):
            PyramidMetadata.from_attrs(attrs)

    def test_rejects_unresolvable_reduced_object(self) -> None:
        attrs = {
            "multiscales": {
                "layout": [
                    {
                        "asset": "1",
                        "derived_from": "missing",
                        "transform": {"scale": [2.0]},
                        "dggs": {"refinement_level": 1},
                    }
                ]
            }
        }
        with pytest.raises(ValueError, match="Cannot resolve"):
            PyramidMetadata.from_attrs(attrs)


class TestValidation:
    def test_derived_level_must_exist(self) -> None:
        with pytest.raises(ValueError, match="not a HEALPix level"):
            PyramidMetadata(levels=(PyramidLevel("level_2", HealpixGrid(level=2), derived_from="level_3"),))

    def test_derived_level_must_be_coarser(self) -> None:
        with pytest.raises(ValueError, match="coarser"):
            PyramidMetadata(
                levels=(
                    PyramidLevel("level_2", HealpixGrid(level=2)),
                    PyramidLevel("level_3", HealpixGrid(level=3), derived_from="level_2"),
                )
            )

    def test_assets_must_be_unique(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            PyramidMetadata(levels=(PyramidLevel("a", HealpixGrid(level=1)),) * 2)


# ===================================================================
# Root group of a datatree store
# ===================================================================


class TestRootAttrs:
    def test_pyramid_wide_attributes_only(self) -> None:
        attrs = pyramid_root_attrs(_pyramid())
        assert attrs["title"] == "pyramid test"
        assert attrs["Conventions"] == "CF-1.13 ACDD-1.3"
        assert attrs["grid_doctor_method"] == "conservative"
        assert "grid_doctor_coarsen_mode" not in attrs
        assert "grid_doctor_coarsened_from_level" not in attrs
        assert "multiscales" in attrs

    @pytest.mark.parametrize("writer", ["save", "init"])
    def test_written_store(self, tmp_path: Path, writer: str) -> None:
        pyramid = {level: ds.chunk(time=1) for level, ds in _pyramid().items()}
        store = tmp_path / "pyramid.zarr"
        if writer == "save":
            save_pyramid(pyramid, str(store), structure="multiscales")
        else:
            init_pyramid_store(pyramid, str(store), structure="multiscales")
            save_pyramid(
                pyramid,
                str(store),
                structure="multiscales",
                mode="r+",
                region={"time": slice(0, 1)},
            )
        dt = xr.open_datatree(store)
        assert PyramidMetadata.from_attrs(dt.attrs) == PyramidMetadata.from_pyramid(pyramid)
        assert "crs" not in dt.dataset.variables
        for level in pyramid:
            group = dt[level_group(level)].to_dataset()
            assert HealpixMetadata.from_dataset(group).grid.level == level
