"""Tests for grid_doctor.metadata: the single place metadata is written and read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

import grid_doctor
from grid_doctor.metadata import (
    CF_VERSION,
    METADATA_VERSION,
    HealpixGrid,
    HealpixMetadata,
    Provenance,
    grid_mapping_attrs,
    has_healpix_metadata,
    merge_conventions,
)
from grid_doctor.metadata._cf import grid_from_mapping

LEVEL = 2
NPIX = 12 * 4**LEVEL


def _dense(level: int = LEVEL, *, coords: bool = True) -> xr.Dataset:
    npix = 12 * 4**level
    ds = xr.Dataset(
        {
            "t": (("time", "cell"), np.zeros((2, npix))),
            "static": ("time", np.arange(2.0)),
        },
        coords={"time": [0, 1]},
        attrs={"title": "test"},
    )
    if coords:
        ds = ds.assign_coords(
            cell=np.arange(npix, dtype=np.int64),
            latitude=("cell", np.zeros(npix)),
            longitude=("cell", np.zeros(npix)),
        )
    return ds


def _sparse(cells: list[int]) -> xr.Dataset:
    return xr.Dataset(
        {"t": ("cell", np.zeros(len(cells)))},
        coords={"cell": np.asarray(cells, dtype=np.int64)},
    )


# ===================================================================
# HealpixGrid
# ===================================================================


class TestHealpixGrid:
    def test_derived_quantities(self) -> None:
        grid = HealpixGrid(level=3)
        assert (grid.nside, grid.npix, grid.nest) == (8, 768, True)
        assert HealpixGrid.from_nest(3, nest=False).indexing_scheme == "ring"

    def test_level_zero_is_valid(self) -> None:
        assert HealpixGrid(level=0).npix == 12

    def test_numpy_integers_are_normalised(self) -> None:
        grid = HealpixGrid(level=np.int64(4))  # type: ignore[arg-type]
        assert type(grid.level) is int
        assert grid == HealpixGrid(level=4)

    @pytest.mark.parametrize("level", [True, np.bool_(False), 1.5, "3", None])
    def test_rejects_non_integer_level(self, level: object) -> None:
        with pytest.raises(ValueError, match="integer"):
            HealpixGrid(level=level)  # type: ignore[arg-type]

    @pytest.mark.parametrize("level", [-1, 30])
    def test_rejects_level_out_of_range(self, level: int) -> None:
        with pytest.raises(ValueError, match="within"):
            HealpixGrid(level=level)

    @pytest.mark.parametrize("scheme", ["nest", "zuniq", "nuniq", ""])
    def test_rejects_unsupported_scheme(self, scheme: str) -> None:
        with pytest.raises(ValueError, match="indexing_scheme"):
            HealpixGrid(level=1, indexing_scheme=scheme)  # type: ignore[arg-type]

    @pytest.mark.parametrize("radius", [6371008.8, 6371000, np.float32(6.4e6)])
    def test_accepts_numeric_radius(self, radius: float) -> None:
        assert HealpixGrid(level=1, earth_radius=radius).earth_radius == float(radius)

    @pytest.mark.parametrize("radius", [0, -1.0, float("nan"), float("inf"), True, "6e6"])
    def test_rejects_invalid_radius(self, radius: object) -> None:
        with pytest.raises(ValueError, match="earth_radius"):
            HealpixGrid(level=1, earth_radius=radius)  # type: ignore[arg-type]

    def test_is_immutable(self) -> None:
        grid = HealpixGrid(level=1)
        with pytest.raises(AttributeError):
            grid.level = 2  # type: ignore[misc]


# ===================================================================
# CF spelling
# ===================================================================


class TestCF:
    def test_level_zero_keeps_refinement_level(self) -> None:
        assert grid_mapping_attrs(HealpixGrid(level=0)) == {
            "grid_mapping_name": "healpix",
            "indexing_scheme": "nested",
            "refinement_level": 0,
        }

    def test_earth_radius_only_when_set(self) -> None:
        attrs = grid_mapping_attrs(HealpixGrid(level=1, earth_radius=6371000))
        assert attrs["earth_radius"] == 6371000.0

    @pytest.mark.parametrize(
        "grid",
        [
            HealpixGrid(level=0),
            HealpixGrid(level=7, indexing_scheme="ring"),
            HealpixGrid(level=29, earth_radius=6371008.8),
        ],
    )
    def test_mapping_round_trip(self, grid: HealpixGrid) -> None:
        assert grid_from_mapping(grid_mapping_attrs(grid)) == grid

    def test_rejects_other_grid_mapping(self) -> None:
        with pytest.raises(ValueError, match="Not a HEALPix"):
            grid_from_mapping({"grid_mapping_name": "latitude_longitude"})

    def test_requires_refinement_level(self) -> None:
        with pytest.raises(ValueError, match="refinement_level"):
            grid_from_mapping({"grid_mapping_name": "healpix", "indexing_scheme": "nested"})

    @pytest.mark.parametrize(
        ("existing", "expected"),
        [
            (None, CF_VERSION),
            ("", CF_VERSION),
            ("CF-1.7", CF_VERSION),
            ("CF-1.7 CMIP-6.2", f"{CF_VERSION} CMIP-6.2"),
            ("ACDD-1.3, CF-1.8", f"{CF_VERSION} ACDD-1.3"),
            (CF_VERSION, CF_VERSION),
        ],
    )
    def test_merge_conventions(self, existing: str | None, expected: str) -> None:
        assert merge_conventions(existing) == expected


# ===================================================================
# Provenance
# ===================================================================


class TestProvenance:
    def test_defaults_to_running_version(self) -> None:
        assert Provenance().to_attrs() == {"grid_doctor_version": grid_doctor.__version__}

    def test_round_trip(self) -> None:
        prov = Provenance(
            version="1.0",
            method="conservative",
            coarsened_from_level=0,
            coarsen_mode="mean",
            min_valid_fraction=0.0,
        )
        assert Provenance.from_attrs(prov.to_attrs()) == prov

    def test_missing_version_is_unknown(self) -> None:
        assert Provenance.from_attrs({}).version == "unknown"


# ===================================================================
# HealpixMetadata.apply
# ===================================================================


class TestApply:
    def test_writes_cf_layout(self) -> None:
        out = HealpixMetadata(HealpixGrid(level=LEVEL)).apply(_dense())
        assert out["crs"].attrs == {
            "grid_mapping_name": "healpix",
            "indexing_scheme": "nested",
            "refinement_level": LEVEL,
        }
        assert out["cell"].attrs["standard_name"] == "healpix_index"
        assert out["cell"].attrs["units"] == "1"
        assert out["latitude"].attrs["units"] == "degrees_north"
        assert out["longitude"].attrs["units"] == "degrees_east"
        assert out["t"].attrs["grid_mapping"] == "crs"
        assert "grid_mapping" not in out["static"].attrs
        assert out.attrs["Conventions"] == CF_VERSION
        assert out.attrs["grid_doctor_metadata_version"] == METADATA_VERSION
        assert out.attrs["title"] == "test"

    def test_does_not_modify_input(self) -> None:
        ds = _dense()
        before = ds.copy(deep=True)
        HealpixMetadata(HealpixGrid(level=LEVEL)).apply(ds)
        xr.testing.assert_identical(ds, before)

    def test_replaces_stale_owned_attributes(self) -> None:
        ds = _dense().assign_attrs(
            grid_doctor_coarsen_mode="mode",
            grid_doctor_sparse=1,
            grid_doctor_version="0.0.1",
            Conventions="CF-1.6",
        )
        out = HealpixMetadata(HealpixGrid(level=LEVEL), Provenance(method="nearest")).apply(ds)
        assert "grid_doctor_coarsen_mode" not in out.attrs
        assert "grid_doctor_sparse" not in out.attrs
        assert out.attrs["grid_doctor_version"] == grid_doctor.__version__
        assert out.attrs["grid_doctor_method"] == "nearest"
        assert out.attrs["Conventions"] == CF_VERSION

    def test_replaces_existing_crs(self) -> None:
        first = HealpixMetadata(HealpixGrid(level=LEVEL)).apply(_dense())
        second = HealpixMetadata(HealpixGrid(level=LEVEL, indexing_scheme="ring")).apply(first)
        assert second["crs"].attrs["indexing_scheme"] == "ring"

    def test_is_idempotent(self) -> None:
        meta = HealpixMetadata(HealpixGrid(level=LEVEL), Provenance(method="nearest"))
        once = meta.apply(_dense())
        xr.testing.assert_identical(meta.apply(once), once)

    def test_implicit_coords_without_cell_coordinate(self) -> None:
        meta = HealpixMetadata(HealpixGrid(level=LEVEL), implicit_coords=True)
        out = meta.apply(_dense(coords=False))
        assert out.attrs["grid_doctor_implicit_coords"] == 1
        assert out["t"].attrs["grid_mapping"] == "crs"

    def test_sparse_subset(self) -> None:
        out = HealpixMetadata(HealpixGrid(level=LEVEL), sparse=True).apply(_sparse([3, 7]))
        assert out.attrs["grid_doctor_sparse"] == 1
        assert out["cell"].attrs["standard_name"] == "healpix_index"

    def test_rejects_wrong_dense_size(self) -> None:
        with pytest.raises(ValueError, match="Dense dataset has"):
            HealpixMetadata(HealpixGrid(level=LEVEL + 1)).apply(_dense())

    def test_rejects_sparse_larger_than_grid(self) -> None:
        with pytest.raises(ValueError, match="more than"):
            HealpixMetadata(HealpixGrid(level=0), sparse=True).apply(_sparse(list(range(13))))

    def test_rejects_sparse_without_cell_coordinate(self) -> None:
        with pytest.raises(ValueError, match="needs a `cell` coordinate"):
            HealpixMetadata(HealpixGrid(level=LEVEL), sparse=True).apply(_dense(coords=False))

    def test_rejects_implicit_with_cell_coordinate(self) -> None:
        with pytest.raises(ValueError, match="implicit_coords"):
            HealpixMetadata(HealpixGrid(level=LEVEL), implicit_coords=True).apply(_dense())

    def test_rejects_float_cell_coordinate(self) -> None:
        ds = _dense().assign_coords(cell=np.arange(NPIX, dtype=np.float64))
        with pytest.raises(ValueError, match="integer"):
            HealpixMetadata(HealpixGrid(level=LEVEL)).apply(ds)

    def test_rejects_sparse_and_implicit(self) -> None:
        with pytest.raises(ValueError, match="cannot have implicit"):
            HealpixMetadata(HealpixGrid(level=LEVEL), sparse=True, implicit_coords=True)


# ===================================================================
# HealpixMetadata.from_dataset
# ===================================================================


class TestFromDataset:
    @pytest.mark.parametrize(
        "meta",
        [
            HealpixMetadata(HealpixGrid(level=LEVEL)),
            HealpixMetadata(
                HealpixGrid(level=LEVEL, indexing_scheme="ring", earth_radius=6371000.0),
                Provenance(
                    method="conservative",
                    coarsened_from_level=LEVEL + 1,
                    coarsen_mode="mean",
                    min_valid_fraction=0.5,
                ),
            ),
        ],
    )
    def test_round_trip(self, meta: HealpixMetadata) -> None:
        assert HealpixMetadata.from_dataset(meta.apply(_dense())) == meta

    def test_round_trip_layout_flags(self) -> None:
        sparse = HealpixMetadata(HealpixGrid(level=LEVEL), sparse=True)
        implicit = HealpixMetadata(HealpixGrid(level=LEVEL), implicit_coords=True)
        assert HealpixMetadata.from_dataset(sparse.apply(_sparse([1, 2]))) == sparse
        assert HealpixMetadata.from_dataset(implicit.apply(_dense(coords=False))) == implicit

    @pytest.mark.parametrize("zarr_format", [2, 3])
    def test_round_trip_through_zarr(self, tmp_path: Path, zarr_format: int) -> None:
        meta = HealpixMetadata(
            HealpixGrid(level=0),
            Provenance(method="nearest", coarsened_from_level=1, min_valid_fraction=0.0),
        )
        store = tmp_path / "store.zarr"
        meta.apply(_dense(0)).to_zarr(store, zarr_format=zarr_format)
        opened = xr.open_zarr(store, zarr_format=zarr_format)
        assert HealpixMetadata.from_dataset(opened) == meta
        assert opened["cell"].attrs["standard_name"] == "healpix_index"

    def test_requires_grid_mapping(self) -> None:
        assert not has_healpix_metadata(_dense())
        with pytest.raises(ValueError, match="no HEALPix grid mapping"):
            HealpixMetadata.from_dataset(_dense())

    def test_finds_grid_mapping_under_another_name(self) -> None:
        ds = _dense().assign_coords(hp=xr.DataArray(0, attrs=grid_mapping_attrs(HealpixGrid(level=LEVEL))))
        meta = HealpixMetadata.from_dataset(ds)
        assert meta.grid == HealpixGrid(level=LEVEL)
        assert meta.provenance.version == "unknown"

    def test_rejects_ambiguous_grid_mappings(self) -> None:
        ds = HealpixMetadata(HealpixGrid(level=LEVEL)).apply(_dense())
        ds = ds.assign_coords(crs2=xr.DataArray(0, attrs=grid_mapping_attrs(HealpixGrid(level=LEVEL)))).rename_vars(
            crs="crs1"
        )
        with pytest.raises(ValueError, match="2 HEALPix grid mapping"):
            HealpixMetadata.from_dataset(ds)

    def test_rejects_newer_metadata_version(self) -> None:
        ds = HealpixMetadata(HealpixGrid(level=LEVEL)).apply(_dense())
        ds.attrs["grid_doctor_metadata_version"] = METADATA_VERSION + 1
        with pytest.raises(ValueError, match="Upgrade grid-doctor"):
            HealpixMetadata.from_dataset(ds)


# ===================================================================
# Interoperability
# ===================================================================


@pytest.mark.parametrize("scheme", ["nested", "ring"])
def test_xdggs_decodes_cf_output(scheme: str) -> None:
    """The written layout must be readable by xdggs' CF convention."""
    xdggs = pytest.importorskip("xdggs")
    meta = HealpixMetadata(HealpixGrid(level=LEVEL, indexing_scheme=scheme))  # type: ignore[arg-type]
    decoded = xdggs.decode(meta.apply(_dense()), convention="cf")
    info = decoded.dggs.grid_info
    assert info.level == LEVEL
    assert info.indexing_scheme == scheme


class TestMaybeDropCoordinates:
    def test_dense_store_drops_coords_and_is_marked(self) -> None:
        from grid_doctor.io import maybe_drop_coordinates

        ds = HealpixMetadata(HealpixGrid(level=LEVEL)).apply(_dense())
        out = maybe_drop_coordinates(ds, LEVEL, write_coords=False)
        assert not {"cell", "latitude", "longitude"} & set(out.variables)
        assert HealpixMetadata.from_dataset(out).implicit_coords

    def test_sparse_subset_keeps_cell_ids(self) -> None:
        from grid_doctor.io import maybe_drop_coordinates

        ds = HealpixMetadata(HealpixGrid(level=LEVEL), sparse=True).apply(_sparse([5, 9]))
        out = maybe_drop_coordinates(ds, LEVEL, write_coords=False)
        np.testing.assert_array_equal(out["cell"].values, [5, 9])
        assert not HealpixMetadata.from_dataset(out).implicit_coords
