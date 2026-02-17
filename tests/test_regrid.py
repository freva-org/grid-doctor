import pytest

from grid_doctor.helpers import regrid_to_healpix

@pytest.mark.parametrize(
        "test_ds",
        ["regular", "curvilinear"],
        ids=["regular", "curvilinear"],
        indirect=True,
 )
def test_regrid_to_healpix(test_ds):
    hp_ds = regrid_to_healpix(test_ds, level=9)

    assert all((hp_ds.attrs[k] == v for k, v in test_ds.attrs.items()))

    assert len(test_ds.data_vars) == len(hp_ds.data_vars)

    attrs_eq = lambda a, b: a.attrs == b.attrs
    var_check = lambda v, l: l(test_ds[v], hp_ds[v])

    dtype_eq = lambda a, b: a.dtype == b.dtype
    for v in test_ds.data_vars:
        assert var_check(v, attrs_eq)
        assert var_check(v, dtype_eq)
