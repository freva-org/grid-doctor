"""Smoke tests for the installable ERA5-Land package layout."""

from heal_era5.resources import ASSETS_DIR, CMOR_TABLES_ROOT, LOCAL_TABLES_ROOT


def test_packaged_assets_are_available() -> None:
    """The package carries the static configuration required at runtime."""

    assert (ASSETS_DIR / "default_variables.csv").is_file()
    assert (ASSETS_DIR / "source_mapper.json").is_file()


def test_checkout_uses_external_tables_by_default() -> None:
    """CMOR tables remain outside the installed Python package."""

    assert CMOR_TABLES_ROOT == LOCAL_TABLES_ROOT
    assert "src" not in CMOR_TABLES_ROOT.parts
