"""Smoke tests for the installable ERA5-Land package layout."""

import logging

from heal_era5.helpers.logging_utils import log_debug_stage
from heal_era5.resources import ASSETS_DIR, CMOR_TABLES_ROOT, LOCAL_TABLES_ROOT


def test_packaged_assets_are_available() -> None:
    """The package carries the static configuration required at runtime."""

    assert (ASSETS_DIR / "default_variables.csv").is_file()
    assert (ASSETS_DIR / "source_mapper.json").is_file()


def test_checkout_uses_external_tables_by_default() -> None:
    """CMOR tables remain outside the installed Python package."""

    assert CMOR_TABLES_ROOT == LOCAL_TABLES_ROOT
    assert "src" not in CMOR_TABLES_ROOT.parts


def test_log_debug_stage_demotes_diagnostic_messages(caplog) -> None:
    logger = logging.getLogger("heal_era5.tests.logging")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_debug_stage(logger, "merge_plan", destination="test.zarr")
    assert not caplog.records

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_debug_stage(logger, "merge_plan", destination="test.zarr")
    assert caplog.records[-1].message == "stage=merge_plan destination=test.zarr"
