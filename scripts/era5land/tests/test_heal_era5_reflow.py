"""Tests for pressure-level selection in the Reflow workflow."""

import pandas as pd

from heal_era5.cli import reflow_workflow
from heal_era5.helpers.file_fetcher import SourceRecord


def _pressure_record() -> SourceRecord:
    return SourceRecord(
        variable="ta",
        table_variable="ta",
        dataset="era5",
        dataset_code="E5",
        frequency="1hr",
        stream="pl",
        type="an",
        parameter="130",
        level_type="pl_an",
        pattern="/tmp/*.grb",
        files=("/tmp/ta.grb",),
        conversion_factor=1.0,
        output_attrs={"level_type": "pl_an"},
    )


def test_pressure_level_selection_is_sent_to_each_worker(monkeypatch):
    """Configured selections should limit the complete level set sent to each worker."""

    monkeypatch.setattr(
        reflow_workflow,
        "cached_grib_inventory",
        lambda _: pd.DataFrame({"level": [1000, 850, 700, 500]}),
    )

    levels = reflow_workflow._pressure_levels_for_record(
        _pressure_record(),
        selected_pressure_levels=(1000, 700, 500),
    )

    assert levels == (1000, 700, 500)
