"""``save_pyramid`` and ``init_pyramid_store`` (``grid_doctor.io._save``)."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import xarray as xr

from grid_doctor.io._save import (
    init_pyramid_store,
    save_pyramid,
)
from grid_doctor.metadata import PyramidMetadata

from ..helpers import with_healpix_metadata


class TestSavePyramid:
    def _make_pyramid(self, chunked: bool = False) -> dict[int, xr.Dataset]:
        def _create_dataset(level: int, chunked: bool) -> xr.Dataset:
            ds = with_healpix_metadata(
                xr.Dataset(
                    {"t": (("cell", "time"), np.random.randn(12 * (4**level), 100))},
                    coords={
                        "cell": np.arange(12 * (4**level), dtype=np.int64),
                        "time": np.arange(100, dtype=np.int64),
                    },
                ),
                level,
            )
            if chunked:
                return ds.chunk({"time": 5})
            return ds

        return {level: _create_dataset(level, chunked) for level in (0, 1)}

    def _make_pyramid_chunked_coord(self) -> dict[int, xr.Dataset]:
        def _create_dataset(level: int) -> xr.Dataset:
            lat = xr.DataArray(
                np.random.randn(12 * (4**level)),
                coords=[
                    np.arange(12 * (4**level)),
                ],
                dims=[
                    "cell",
                ],
            ).chunk({"cell": 12 * (4**level)})
            static = xr.DataArray(
                [1.0, 2.0],
                coords=[
                    [0, 1],
                ],
                dims=["level"],
            ).chunk({"level": 2})

            ds = xr.Dataset(
                data_vars={
                    "temp": (
                        ["time", "level", "cell"],
                        np.random.randn(4, 2, 12 * (4**level)),
                    ),
                    "static": static,
                },
                coords={
                    "time": ("time", range(4)),
                    "level": ("level", range(2)),
                    "cell": ("cell", range(12 * (4**level))),
                    "latitude": lat,
                },
            )
            ds["temp"] = ds.temp.chunk({"time": 2})
            return ds

        return {level: _create_dataset(level) for level in (0, 1)}

    @mock.patch("s3fs.S3FileSystem")
    @mock.patch("s3fs.S3Map")
    def test_calls_to_zarr(self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock) -> None:
        del mock_s3fs
        pyramid = self._make_pyramid()
        mock_s3map.return_value = mock.MagicMock()
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid(pyramid, "s3://bucket/test", s3_options={}, mode="w")
            assert mock_zarr.call_count == 2

    @mock.patch("s3fs.S3FileSystem")
    @mock.patch("s3fs.S3Map")
    def test_zarr_format_3_omits_consolidated(
        self, mock_s3map: mock.Mock, mock_s3fs: mock.Mock
    ) -> None:
        del mock_s3fs
        pyramid = self._make_pyramid()
        mock_s3map.return_value = mock.MagicMock()
        with mock.patch.object(xr.Dataset, "to_zarr") as mock_zarr:
            save_pyramid(pyramid, "s3://bucket/test", s3_options={}, zarr_format=3)
            for call in mock_zarr.call_args_list:
                assert "consolidated" not in call.kwargs

    @pytest.mark.parametrize("chunked", [True, False])
    def test_save_pyramid_local(self, tmp_path, chunked):
        pyramid = self._make_pyramid(chunked)

        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="w",
        )
        level0_path = tmp_path / "level_0.zarr"
        assert os.path.isdir(str(level0_path))
        level1_path = tmp_path / "level_1.zarr"
        assert os.path.isdir(str(level1_path))

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0, pyramid[0])
        xr.testing.assert_equal(ds1, pyramid[1])

    @pytest.mark.parametrize("chunked", [True, False])
    def test_save_pyramid_local_region_init(self, tmp_path, chunked):
        pyramid = self._make_pyramid(chunked)

        init_pyramid_store(pyramid=pyramid, path=str(tmp_path))

        level0_path = tmp_path / "level_0.zarr"
        assert os.path.isdir(str(level0_path))
        level1_path = tmp_path / "level_1.zarr"
        assert os.path.isdir(str(level1_path))

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        # Datasets should have the full coordinates, and data variables.
        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["t"]
        assert list(ds1.data_vars) == ["t"]

        if chunked:
            assert ds0.t.chunksizes == pyramid[0].t.chunksizes

        # Confirm that the data is nan.
        ds0t = ds0.t.data.compute()
        ds1t = ds1.t.data.compute()
        assert len(ds0t[~np.isnan(ds0t)]) == 0
        assert len(ds1t[~np.isnan(ds1t)]) == 0

    @pytest.mark.parametrize("chunked", [True, False])
    def test_save_pyramid_local_region_update(self, tmp_path, chunked):
        pyramid = self._make_pyramid(chunked)
        # Initialise zarr store
        init_pyramid_store(pyramid=pyramid, path=str(tmp_path))

        # Update region
        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="a",
            region={"time": slice(0, 5)},
        )

        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"
        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        # Confirm that the coordinates and data variables are unchanged in the datasets.
        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["t"]
        assert list(ds1.data_vars) == ["t"]

        ds0t = ds0.t.data.compute()
        ds1t = ds1.t.data.compute()

        # Confirm that the selected region has been written correctly.
        np.testing.assert_allclose(ds0.t.data[:, 0:5], pyramid[0].t.data[:, 0:5])
        np.testing.assert_allclose(ds1.t.data[:, 0:5], pyramid[1].t.data[:, 0:5])

        # Confirm that the rest of the data is nan.
        ds0_rest = ds0t[:, 5:]
        ds1_rest = ds1t[:, 5:]
        assert len(ds0_rest[~np.isnan(ds0_rest)]) == 0
        assert len(ds1_rest[~np.isnan(ds1_rest)]) == 0

        # Update second region
        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="a",
            region={"time": slice(5, 10)},
        )

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))
        # Confirm that both regions have been written correctly.
        np.testing.assert_allclose(ds0.t.data[:, 0:10], pyramid[0].t.data[:, 0:10])
        np.testing.assert_allclose(ds1.t.data[:, 0:10], pyramid[1].t.data[:, 0:10])
        # Confirm that the rest of the data is still nan.
        ds0_rest = ds0t[:, 10:]
        ds1_rest = ds1t[:, 10:]
        assert len(ds0_rest[~np.isnan(ds0_rest)]) == 0
        assert len(ds1_rest[~np.isnan(ds1_rest)]) == 0

    def test_save_pyramid_local_region_parallel(self, tmp_path):
        # Data has to be chunked for parallel  writes, and regions have to
        # be split along chunks.
        pyramid = self._make_pyramid(chunked=True)
        # Initialise zarr store
        init_pyramid_store(pyramid=pyramid, path=str(tmp_path))
        region_limits = [(i, i + 10) for i in range(0, 100, 10)]
        regions = [{"time": slice(*lim)} for lim in region_limits]

        def _save_section(region: dict):
            save_pyramid(pyramid, str(tmp_path), region=region)

        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            for future in [executor.submit(_save_section, i) for i in regions]:
                future.result()

        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"
        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        # Confirm that the coordinates and data variables are unchanged in the datasets.
        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["t"]
        assert list(ds1.data_vars) == ["t"]

        # Confirm that all regions have been written correctly.
        np.testing.assert_allclose(ds0.t.data, pyramid[0].t.data)
        np.testing.assert_allclose(ds1.t.data, pyramid[1].t.data)

    def test_save_pyramid_local_chunked_coord(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()

        save_pyramid(
            pyramid=pyramid,
            path=str(tmp_path),
            mode="w",
        )
        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0.compute(), pyramid[0].compute())
        xr.testing.assert_equal(ds1.compute(), pyramid[1].compute())

    def test_save_pyramid_local_chunked_coord_regions(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()
        init_pyramid_store(
            pyramid=pyramid,
            path=str(tmp_path),
            materialise=[
                "static",
            ],
        )

        level0_path = tmp_path / "level_0.zarr"
        level1_path = tmp_path / "level_1.zarr"

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0.coords, pyramid[0].coords)
        xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
        assert list(ds0.data_vars) == ["static", "temp"]
        assert list(ds1.data_vars) == ["static", "temp"]

        assert ds0.static.chunksizes == pyramid[0].static.chunksizes
        assert ds0.latitude.chunksizes == pyramid[0].latitude.chunksizes
        assert ds1.static.chunksizes == pyramid[1].static.chunksizes
        assert ds1.latitude.chunksizes == pyramid[1].latitude.chunksizes

        # Confirm that the data is nan.
        ds0t = ds0.temp.data.compute()
        ds1t = ds1.temp.data.compute()
        assert len(ds0t[~np.isnan(ds0t)]) == 0
        assert len(ds1t[~np.isnan(ds1t)]) == 0

        ds0stat = ds0.static.compute()
        ds1stat = ds1.static.compute()
        xr.testing.assert_equal(ds0stat, pyramid[0].static.compute())
        xr.testing.assert_equal(ds1stat, pyramid[1].static.compute())

        regions = [
            {"time": slice(i, i + 2)} for i in range(0, pyramid[0].sizes["time"], 2)
        ]

        def _save_section(region: dict):
            save_pyramid(pyramid, str(tmp_path), region=region)

        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            for future in [executor.submit(_save_section, i) for i in regions]:
                future.result()

        ds0 = xr.open_zarr(str(level0_path))
        ds1 = xr.open_zarr(str(level1_path))

        xr.testing.assert_equal(ds0.compute(), pyramid[0].compute())
        xr.testing.assert_equal(ds1.compute(), pyramid[1].compute())


def test_save_pyramid_region_init_huge_finely_chunked(tmp_path):
    """``region='init'`` on ~70 TiB in ~573k chunks writes metadata only, in seconds.

    Its cost must not grow with the number of chunks: a metadata-only
    init takes well under 1 s here, one that builds a write graph about a minute.
    """
    ncell = 12 * 4**10
    ntime = 70 * 2**40 // (ncell * 4)  # float32 -> ~70 TiB
    shape = xr.DataArray(
        np.broadcast_to(np.float32(0), (ncell, ntime)), dims=("cell", "time")
    )
    t = xr.zeros_like(
        shape, chunks=(4**9, 128), chunked_array_type="dask"
    )  # lazy, no memory
    ds = xr.Dataset({"t": t})
    with ThreadPoolExecutor(1) as pool:
        start = os.times().elapsed
        pool.submit(init_pyramid_store, {10: ds}, str(tmp_path)).result()
        duration = os.times().elapsed - start

    assert duration < 10, (
        f"region='init' took {duration:.0f} s: its cost grows with the number of chunks"
    )
    store = tmp_path / "level_10.zarr"
    assert xr.open_zarr(store)["t"].encoding["chunks"] == (4**9, 128), (
        "init must keep the chunk layout"
    )
    assert not [f for f in os.listdir(store / "t") if not f.startswith(".")], (
        "init must not write data"
    )


class TestSavePyramidMultiscales:
    def _make_pyramid_chunked_coord(self) -> dict[int, xr.Dataset]:
        def _create_dataset(level: int) -> xr.Dataset:
            num_cells = 3 * (4 ** int(level / 2))
            lat = xr.DataArray(
                np.random.randn(num_cells),
                coords=[
                    np.arange(num_cells),
                ],
                dims=[
                    "cell",
                ],
            ).chunk({"cell": num_cells})
            static = xr.DataArray(
                [1.0, 2.0],
                coords=[
                    [0, 1],
                ],
                dims=["level"],
            ).chunk({"level": 2})

            ds = xr.Dataset(
                data_vars={
                    "temp": (
                        ["time", "level", "cell"],
                        np.random.randn(4, 2, num_cells),
                    ),
                    "static": static,
                },
                coords={
                    "time": ("time", range(4)),
                    "level": ("level", range(2)),
                    "cell": ("cell", range(num_cells)),
                    "latitude": lat,
                },
            )
            ds["temp"] = ds.temp.chunk({"time": 2})
            return ds

        # Pick levels so one keeps the coordinates and one drops them.
        return {level: _create_dataset(level) for level in (7, 11)}

    def test_save_pyramid_local_multiscales(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()
        zarr_store = tmp_path / "multi.zarr"
        save_pyramid(
            pyramid=pyramid,
            path=str(zarr_store),
            structure="multiscales",
        )

        dt = xr.open_datatree(zarr_store, chunks="auto")
        ds0 = dt["/level_7"].dataset
        ds1 = dt["/level_11"].dataset

        layout = PyramidMetadata.from_attrs(dt.attrs).levels
        assert [level.asset for level in layout] == ["level_11", "level_7"]

        xr.testing.assert_allclose(ds0.compute(), pyramid[7].compute())
        xr.testing.assert_allclose(
            ds1.compute(), pyramid[11].compute().drop_vars(["cell", "latitude"])
        )

    def test_save_pyramid_local_multiscales_regions(self, tmp_path):
        pyramid = self._make_pyramid_chunked_coord()
        zarr_store = tmp_path / "multi.zarr"
        init_pyramid_store(
            pyramid=pyramid,
            path=str(zarr_store),
            materialise=[
                "static",
            ],
            structure="multiscales",
        )

        dt = xr.open_datatree(zarr_store, chunks="auto")
        ds0 = dt["/level_7"].dataset
        ds1 = dt["/level_11"].dataset

        xr.testing.assert_equal(ds0.coords, pyramid[7].coords)
        xr.testing.assert_equal(
            ds1.coords, pyramid[11].drop_vars(["cell", "latitude"]).coords
        )
        assert list(ds0.data_vars) == ["static", "temp"]
        assert list(ds1.data_vars) == ["static", "temp"]

        assert ds0.static.chunksizes == pyramid[7].static.chunksizes
        assert ds0.latitude.chunksizes == pyramid[7].latitude.chunksizes
        assert ds1.static.chunksizes == pyramid[11].static.chunksizes
        assert "latitude" not in ds1

        # Confirm that the data is nan.
        ds0t = ds0.temp.data.compute()
        ds1t = ds1.temp.data.compute()
        assert len(ds0t[~np.isnan(ds0t)]) == 0
        assert len(ds1t[~np.isnan(ds1t)]) == 0

        ds0stat = ds0.static.compute()
        ds1stat = ds1.static.compute()
        xr.testing.assert_equal(ds0stat, pyramid[7].static.compute())
        xr.testing.assert_equal(ds1stat, pyramid[11].static.compute())

        regions = [
            {"time": slice(i, i + 2)} for i in range(0, pyramid[7].sizes["time"], 2)
        ]

        def _save_section(region: dict):
            save_pyramid(
                pyramid,
                str(zarr_store),
                region=region,
                structure="multiscales",
            )

        with ThreadPoolExecutor(max_workers=len(regions)) as executor:
            for future in [executor.submit(_save_section, i) for i in regions]:
                future.result()

        dt = xr.open_datatree(zarr_store, chunks="auto")
        ds0 = dt["/level_7"].dataset
        ds1 = dt["/level_11"].dataset

        xr.testing.assert_allclose(ds0.compute(), pyramid[7].compute())
        xr.testing.assert_allclose(
            ds1.compute(), pyramid[11].compute().drop_vars(["cell", "latitude"])
        )


LEVEL = 3
NPIX = 12 * 4**LEVEL


def _values_with_nans(shape: tuple[int, ...], seed: int = 54) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=shape)
    values[..., : shape[-1] // 3] = np.nan  # a clustered "coastline"
    values[rng.random(shape) < 0.3] = np.nan
    return values


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
        init_pyramid_store(pyramid, str(tmp_path))
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

    def test_region_write_keeps_stored_packing(self, tmp_path: Path) -> None:
        """Region updates are packed with the store's scale and offset."""
        ds = xr.Dataset({"sst": (("time", "cell"), np.full((2, 12), 280.0))})
        ds = ds.chunk(time=1)
        packing = {
            "dtype": "int16",
            "scale_factor": 0.1,
            "add_offset": 270.0,
            "_FillValue": -9999,
        }
        save_pyramid({0: ds}, str(tmp_path), mode="w", encoding={0: {"sst": packing}})
        update = ds.assign(sst=xr.full_like(ds["sst"], 310.0))
        save_pyramid(
            {0: update}, str(tmp_path), mode="r+", region={"time": slice(1, 2)}
        )
        stored = xr.open_zarr(tmp_path / "level_0.zarr")["sst"].values
        np.testing.assert_allclose(stored[:, 0], [280.0, 310.0])

    def test_region_write_keeps_stored_time_units(self, tmp_path: Path) -> None:
        """Datetime variables are written in the units chosen at init."""
        stamps = np.array(
            ["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"],
            dtype="datetime64[ns]",
        )
        ds = xr.Dataset(
            {"obs_time": (("time", "cell"), np.repeat(stamps[:, None], 12, axis=1))}
        ).chunk(time=1)
        save_pyramid({0: ds}, str(tmp_path), mode="w")
        save_pyramid({0: ds}, str(tmp_path), mode="r+", region={"time": slice(2, 4)})
        stored = xr.open_zarr(tmp_path / "level_0.zarr")["obs_time"].values
        np.testing.assert_array_equal(stored[2:, 0], stamps[2:])


def _tiny_pyramid() -> dict[int, xr.Dataset]:
    """a two-level HEALPix-like pyramid for write tests."""

    def level_ds(npix: int) -> xr.Dataset:
        return xr.Dataset(
            {"tas": ("cell", np.arange(npix, dtype="float32"))},
            coords={"cell": np.arange(npix)},
        )

    return {0: level_ds(12), 1: level_ds(48)}


class TestSavePyramidToS3Local:
    def test_writes_levels_to_disk(self, tmp_path: Path) -> None:
        pyramid = _tiny_pyramid()
        out = tmp_path / "pyramid"
        save_pyramid(pyramid, str(out), mode="w")
        for level in pyramid:
            assert (out / f"level_{level}.zarr").is_dir()

    def test_round_trips_values(self, tmp_path: Path) -> None:
        pyramid = _tiny_pyramid()
        out = tmp_path / "pyramid"
        save_pyramid(pyramid, str(out), mode="w")
        for level, dataset in pyramid.items():
            reloaded = xr.open_zarr(out / f"level_{level}.zarr")
            xr.testing.assert_allclose(reloaded[["tas"]], dataset[["tas"]])

    def test_creates_missing_parent_directories(self, tmp_path: Path) -> None:
        pyramid = _tiny_pyramid()
        out = tmp_path / "nested" / "deeper" / "pyramid"
        save_pyramid(pyramid, str(out), mode="w")
        assert (out / "level_0.zarr").is_dir()

    def test_s3_options_optional_for_local(self, tmp_path: Path) -> None:
        # since no s3_options supplied a local write must
        # not construct an S3 client.
        out = tmp_path / "pyramid"
        with mock.patch("s3fs.S3FileSystem") as fs:
            save_pyramid(_tiny_pyramid(), str(out), mode="w")
        fs.assert_not_called()


class TestSavePyramidToS3Remote:
    def test_uses_s3_map_store_for_s3_path(self) -> None:
        class _FakeDataset:
            store: object = None
            variables: dict[str, object] = {}

            def to_zarr(self, store: object, **kwargs: object) -> None:
                self.store = store

        dataset = _FakeDataset()
        with (
            mock.patch("s3fs.S3FileSystem") as fs,
            mock.patch(
                "s3fs.S3Map", return_value="s3-store"
            ) as s3_map,
        ):
            save_pyramid(
                {0: dataset},  # type: ignore[dict-item]
                "s3://bucket/pyr",
                {"key": "x", "secret": "y"},
                mode="w",
            )
        fs.assert_called_once()
        s3_map.assert_called_once()
        assert s3_map.call_args.kwargs["root"] == "s3://bucket/pyr/level_0.zarr"
        assert dataset.store == "s3-store"
