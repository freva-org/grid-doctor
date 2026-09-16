import datetime
import os

import numpy as np
import pytest
import xarray as xr

from grid_doctor.multiscales import write_pyramid_to_multiscales_zarr


def _make_pyramid() -> dict[int, xr.Dataset]:
    return {
        level: xr.Dataset(
            {"t": (("cell",), np.zeros(12 * (4**level), dtype=np.float32))},
            coords={"cell": np.arange(12 * (4**level), dtype=np.int64)},
            attrs={
                "healpix_nside": 2**level,
                "healpix_level": level,
                "healpix_order": "nested",
            },
        )
        for level in (0, 1)
    }


def test_write_pyramid_to_multiscales_zarr(tmp_path):
    pyramid = _make_pyramid()
    root_attrs = {"source_attr_1": "attr1_val", "source_attr_2": 3}
    out_root = tmp_path / "multiscales_test_pyramid.zarr"
    additional_attrs = {"dataset_id": "test_dataset_id", "frequency": "monthly"}

    write_pyramid_to_multiscales_zarr(
        pyramid=pyramid,
        out_root=out_root,
        root_attrs=root_attrs,
        additional_attrs=additional_attrs,
    )

    assert os.path.isdir(out_root)
    dt = xr.open_datatree(out_root)

    assert dt.groups == (
        "/",
        "/multiscales",
        "/multiscales/zoom_0",
        "/multiscales/zoom_1",
    )

    ds_root = dt["/"]
    assert ds_root.attrs == {
        "healpix_zoom_min": 0,
        "healpix_zoom_max": 1,
        "source_dataset_attrs": {"source_attr_1": "attr1_val", "source_attr_2": "3"},
        "dataset_id": "test_dataset_id",
        "frequency": "monthly",
    }

    ds1 = dt["/multiscales/zoom_1"]
    assert ds1.attrs == {
        "healpix_nside": 2,
        "healpix_level": 1,
        "healpix_order": "nested",
        "healpix_zoom": 1,
        "dataset_id": "test_dataset_id",
        "frequency": "monthly",
    }

    xr.testing.assert_equal(ds1.coords, pyramid[1].coords)
    xr.testing.assert_equal(ds1.t, pyramid[1].t)
    xr.testing.assert_equal(ds1.cell, pyramid[1].cell)


def test_write_pyramid_to_multiscales_zarr_exists_raises(tmp_path):

    out_root = tmp_path / "multiscales_test_pyramid.zarr"
    os.mkdir(out_root)
    with pytest.raises(FileExistsError) as ffe:
        write_pyramid_to_multiscales_zarr(
            pyramid=_make_pyramid(),
            out_root=out_root,
            overwrite=False,
        )
    assert "multiscales zarr exists" in str(ffe.value)
    assert "multiscales_test_pyramid.zarr. Skipping creation." in str(ffe.value)


def test_write_pyramid_to_multiscales_zarr_exists_overwrite(tmp_path):

    out_root = tmp_path / "multiscales_test_pyramid.zarr"
    os.mkdir(out_root)
    # Set file access / modification times to the past so we can later confirm it was recreated.
    past = datetime.datetime(2001, 5, 11, tzinfo=datetime.timezone.utc)
    os.utime(out_root, (past.timestamp(), past.timestamp()))

    write_pyramid_to_multiscales_zarr(
        pyramid=_make_pyramid(),
        out_root=out_root,
        overwrite=True,
    )

    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(out_root), tz=datetime.timezone.utc)
    assert mod_time.year - past.year >= 25
