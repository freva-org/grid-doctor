import pytest

from dask.array import Array as DaskArray

from grid_doctor.helpers import regrid_to_healpix, latlon_to_healpix_pyramid


@pytest.mark.parametrize(
    "test_ds",
    ["regular", "curvilinear", "era5"],
    ids=["regular", "curvilinear", "era5"],
    indirect=True,
)
def test_regrid_to_healpix(test_ds):
    hp_ds = regrid_to_healpix(test_ds, level=9)

    print(test_ds)
    print(hp_ds)
    assert all((hp_ds.attrs[k] == v for k, v in test_ds.attrs.items()))

    assert set(test_ds.data_vars).issuperset(hp_ds.data_vars.keys())

    attrs_eq = lambda a, b: a.attrs == b.attrs
    var_check = lambda v, l: l(test_ds[v], hp_ds[v])

    dtype_eq = lambda a, b: a.dtype == b.dtype
    for v in hp_ds.data_vars:
        assert var_check(v, attrs_eq)
        assert var_check(v, dtype_eq)


@pytest.mark.parametrize(
    "test_ds",
    ["regular", "curvilinear", "era5"],
    ids=["regular", "curvilinear", "era5"],
    indirect=True,
)
def test_latlon_to_healpix_pyramid(test_ds):
    hp_p = latlon_to_healpix_pyramid(test_ds)
    for level, hp_ds in hp_p.items():
        assert all((hp_ds.attrs[k] == v for k, v in test_ds.attrs.items()))
        ## ensure all variables are dask.array.Array / lazy
        assert all((isinstance(v.data, DaskArray) for v in hp_ds.data_vars.values()))
