"""Unit tests for the direct heal-era5 command handlers."""

import sys
from argparse import Namespace
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from heal_era5 import main
from heal_era5.helpers.file_fetcher import SourceRecord, UnresolvedRecord


def _request(name: str = "tas") -> tuple[dict[str, object], list[SimpleNamespace]]:
    return {"remap_defaults": {"pressure_levels_hpa": [1000, 850]}}, [SimpleNamespace(name=name)]


def _record(*, files: tuple[str, ...] = ("/tmp/tas.grb",)) -> SourceRecord:
    return SourceRecord(
        variable="tas",
        table_variable="2t",
        dataset="era5land",
        dataset_code="EL",
        frequency="1hr",
        stream="sf",
        type="fc",
        parameter="167",
        level_type="sfc_fc_land",
        pattern="/tmp/*.grb",
        files=files,
        conversion_factor=1.0,
        output_attrs={"out_name": "tas"},
    )


def _remap_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "variables": "tas",
        "freq": "1hr",
        "interval": "20240101,20240101",
        "truncate_after": None,
        "coarsen_only": None,
        "pressure_levels": None,
        "dataset": "era5land",
        "root": None,
        "output_path": None,
        "from_scratch": False,
        "zarr_format": 2,
        "highest_level_only": False,
        "rechunk_only": False,
        "chunk_size": 32,
        "batch_months": None,
        "batch_files": None,
        "attrs_only": False,
        "clean": False,
    }
    values.update(overrides)
    return Namespace(**values)


# =============================================================================
# Tests for run_fetch
# =============================================================================


def test_fetch_prints_resolved_files(monkeypatch, capsys):
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "resolve_records", lambda **_: [_record()])
    monkeypatch.setattr(main, "unresolved_records", lambda *args: [])

    result = main.run_fetch(
        Namespace(
            variables="tas",
            freq="1hr",
            dataset="era5land",
            interval="20240101,20240101",
            root=None,
            show_patterns=False,
            strict=False,
            json=False,
        )
    )

    assert result == 0
    assert capsys.readouterr().out == "/tmp/tas.grb\n"


def test_fetch_strict_reports_missing_and_unresolved(monkeypatch, capsys):
    missing = _record(files=())
    unresolved = UnresolvedRecord("pr", "day", "not in the CMOR table")
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "resolve_records", lambda **_: [missing])
    monkeypatch.setattr(main, "unresolved_records", lambda *args: [unresolved])

    result = main.run_fetch(
        Namespace(
            variables="tas",
            freq="1hr",
            dataset="era5land",
            interval="20240101,20240101",
            root=None,
            show_patterns=False,
            strict=True,
            json=False,
        )
    )

    assert result == 1
    stderr = capsys.readouterr().err
    assert "missing: tas 1hr /tmp/*.grb" in stderr
    assert "unresolved: pr day: not in the CMOR table" in stderr


# =============================================================================
# Tests for run_remap
# =============================================================================


def test_remap_rechunk_only_uses_requested_settings(monkeypatch):
    from heal_era5.helpers import mapper

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(mapper, "rechunk_existing_healpix_stores", lambda **kwargs: calls.append(kwargs) or 2)

    result = main.run_remap(_remap_args(rechunk_only=True))

    assert result == 0
    assert calls == [
        {
            "dataset": "era5land",
            "frequencies": ("1hr",),
            "zarr_format": 2,
            "target_chunk_mb": 32,
            "highest_level_only": False,
            "output_path": None,
        }
    ]


def test_remap_attrs_only_updates_existing_outputs(monkeypatch):
    from heal_era5.helpers import mapper

    records = [_record()]
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "resolve_records", lambda **_: records)
    monkeypatch.setattr(mapper, "update_healpix_attrs_only", lambda *args, **kwargs: calls.append(kwargs))

    assert main.run_remap(_remap_args(attrs_only=True)) == 0
    assert calls == [
        {
            "dataset": "era5land",
            "frequencies": ("1hr",),
            "requested_variables": ("tas",),
            "output_path": None,
        }
    ]


def test_remap_maps_resolved_records(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "resolve_records", lambda **_: [_record()])
    monkeypatch.setattr(main, "map_records", lambda records, **kwargs: calls.append({"records": records, **kwargs}))

    assert main.run_remap(_remap_args(clean=True)) == 0
    assert calls[0]["records"] == [_record()]
    assert calls[0]["frequencies"] == ("1hr",)
    assert calls[0]["requested_variables"] == ("tas",)
    assert calls[0]["clean"] is True


def test_mapper_writes_finest_level_lazily_then_coarsens_from_zarr(monkeypatch):
    """The level-9 result must not stay materialised while making the pyramid."""

    from heal_era5.helpers import mapper

    class Dataset:
        def __init__(self):
            self.dims = {"time": 1, "cell": 1}
            self.sizes = {"time": 1, "cell": 1}
            self.attrs: dict[str, object] = {}
            self.coords: dict[str, object] = {}
            self.closed = False

        def chunk(self, _chunks):
            return self

        def close(self):
            self.closed = True

    class FinestDataset(Dataset):
        def load(self):
            raise AssertionError("The finest level must be written lazily, not loaded into memory.")

    source = Dataset()
    finest = FinestDataset()
    writes: list[object] = []
    coarsen_calls: list[dict[str, object]] = []
    monkeypatch.setattr(mapper, "merge_frequency_dataset", lambda *args, **kwargs: source)
    monkeypatch.setattr(mapper, "normalise_reduced_gaussian_dataset", lambda dataset, **kwargs: dataset)
    monkeypatch.setattr(mapper, "global_attrs_for_records", lambda records: {})
    monkeypatch.setattr(mapper.gd, "get_latlon_resolution", lambda dataset: 1.0)
    monkeypatch.setattr(mapper.gd, "resolution_to_healpix_level", lambda resolution: 2)
    monkeypatch.setattr(mapper.gd, "cached_weights", lambda *args, **kwargs: "/tmp/weights.nc")
    monkeypatch.setattr(mapper.gd, "regrid_to_healpix", lambda *args, **kwargs: finest)
    monkeypatch.setattr(mapper, "_write_zoom_level", lambda dataset, **kwargs: writes.append(dataset))
    monkeypatch.setattr(
        mapper,
        "_coarsen_existing_frequency",
        lambda **kwargs: coarsen_calls.append(kwargs) or (1, 0),
    )

    mapper.map_grib_to_healpix(
        [_record()],
        dataset="era5land",
        frequencies=("1hr",),
        requested_variables=("tas",),
        interval=(date(2024, 1, 1), date(2024, 1, 1)),
        output_path="/tmp/out",
    )

    assert writes == [finest]
    assert finest.closed
    assert source.closed
    assert coarsen_calls[0]["target_levels"] == (1, 0)


def test_remap_uses_pressure_level_override(monkeypatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "resolve_records", lambda **_: [_record()])
    monkeypatch.setattr(main, "map_records", lambda records, **kwargs: calls.append({"records": records, **kwargs}))

    args = _remap_args(pressure_levels="1000,850,500")
    assert main.run_remap(args) == 0
    assert args.pressure_levels == (1000, 850, 500)


# =============================================================================
# Tests for run_update
# =============================================================================


def test_update_preview_skips_missing_stores(monkeypatch):
    logged: list[tuple[object, str]] = []
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "_existing_frequency_variables", lambda *args, **kwargs: set())
    monkeypatch.setattr(main, "_log_update_preview", lambda rows, **kwargs: logged.append((rows, kwargs["batch_mode"])))

    result = main.run_update(
        Namespace(
            variables="tas",
            freq="1hr,fx",
            dataset="era5land",
            zarr_format=2,
            output_path=None,
            chunk_size=16,
            batch_files=None,
            batch_months=None,
            preview=True,
        )
    )

    assert result == 0
    assert logged == [([], "direct")]


def test_existing_variable_pressure_levels_reads_the_stored_coordinate(monkeypatch):
    class Variable(dict):
        dims = ("time", "plev")

    class Dataset(dict):
        def close(self):
            pass

    opened = {
        "/tmp/level_0.zarr": Dataset({"ta": Variable({"plev": SimpleNamespace(values=[1000, 850])})}),
        "/tmp/level_1.zarr": Dataset({"ta": Variable({"plev": SimpleNamespace(values=[1000, 850])})}),
    }
    monkeypatch.setattr(main, "existing_destinations_for_frequency", lambda *args, **kwargs: tuple(opened))
    monkeypatch.setitem(
        sys.modules,
        "xarray",
        SimpleNamespace(open_zarr=lambda destination, **kwargs: opened[destination]),
    )

    assert main._existing_variable_pressure_levels("era5land", "1hr", "ta", zarr_format=2, output_path=None) == (
        1000,
        850,
    )


def test_update_uses_existing_pressure_levels_for_batched_remaps(monkeypatch):
    calls: list[dict[str, object]] = []
    pressure_record = _record(files=("/tmp/tas_2024-01-01.grb",))._replace(variable="ta", level_type="pl")
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request("ta"))
    monkeypatch.setattr(main, "_existing_frequency_variables", lambda *args, **kwargs: {"ta"})
    monkeypatch.setattr(
        main,
        "_existing_variable_update_state",
        lambda *args, **kwargs: main.VariableUpdateState(None, date(2024, 1, 1), date(2024, 1, 1)),
    )
    monkeypatch.setattr(main, "_resolve_update_records", lambda **_: [pressure_record])
    monkeypatch.setattr(main, "_existing_variable_pressure_levels", lambda *args, **kwargs: (1000, 850))
    monkeypatch.setattr(
        main,
        "_select_permanent_records",
        lambda records, **_: main.UpdateSelection(records, (date(2024, 1, 1), date(2024, 1, 1)), 1),
    )
    monkeypatch.setattr(main, "_map_update_records", lambda *args, **kwargs: calls.append(kwargs))
    monkeypatch.setattr(main, "_persist_permanent_watermark", lambda *args, **kwargs: None)

    main.run_update(
        Namespace(
            variables="ta",
            freq="1hr",
            dataset="era5land",
            zarr_format=2,
            output_path=None,
            chunk_size=16,
            batch_files=8,
            batch_months=None,
            preview=False,
            use_inventory_cache=True,
            use_input_cache=False,
            fail_on_duplicate_times=False,
            weights_dir="/tmp/weights",
            highest_level_only=False,
            root=None,
        )
    )

    assert calls[0]["remap_args"].pressure_levels == (1000, 850)


def test_permanent_refresh_uses_last_data_update_not_permanent_watermark(monkeypatch):
    record = _record(files=("old", "recent", "temporary"))
    dates = {
        "old": date(2026, 8, 31),
        "recent": date(2026, 6, 30),
        "temporary": date(2026, 9, 10),
    }
    modified = {
        "old": datetime(2026, 9, 1, 12, tzinfo=main.UTC),
        "recent": datetime(2026, 9, 8, 12, tzinfo=main.UTC),
        "temporary": datetime(2026, 9, 10, 12, tzinfo=main.UTC),
    }
    monkeypatch.setattr(main, "file_interval", lambda source_file, frequency: (dates[source_file], dates[source_file]))
    monkeypatch.setattr(main, "_local_modification_time", lambda source_file: modified[source_file])
    monkeypatch.setattr(main, "_is_final_source_file", lambda source_file, **_: source_file != "temporary")

    selection = main._select_permanent_records(
        [record],
        dataset="era5land",
        frequency="1hr",
        permanent_watermark=date(2026, 6, 1),
        last_data_update=datetime(2026, 9, 10, 12, tzinfo=main.UTC),
    )

    assert selection.records[0].files == ("recent",)
    assert selection.interval == (date(2026, 6, 30), date(2026, 6, 30))


def test_era5land_finality_uses_coverage_end_month(monkeypatch):
    coverage = {
        "jun": (date(2026, 6, 30), date(2026, 6, 30)),
        "aug": (date(2026, 8, 31), date(2026, 8, 31)),
        "annual": (date(2026, 1, 1), date(2026, 12, 31)),
    }
    modified = {
        "jun": date(2026, 9, 8),
        "aug": date(2026, 9, 8),
        "annual": date(2026, 9, 8),
    }
    monkeypatch.setattr(main, "file_interval", lambda source_file, frequency: coverage[source_file])
    monkeypatch.setattr(main, "_local_modification_date", lambda source_file: modified[source_file])

    assert main._is_final_source_file("jun", dataset="era5land", frequency="1hr")
    assert not main._is_final_source_file("aug", dataset="era5land", frequency="1hr")
    assert not main._is_final_source_file("annual", dataset="era5land", frequency="mon")


def test_forced_permanent_selection_keeps_annual_file_overlapping_lookback(monkeypatch):
    annual_files = _record(files=("2024", "2025"))
    coverage = {
        "2024": (date(2024, 1, 1), date(2024, 12, 31)),
        "2025": (date(2025, 1, 1), date(2025, 12, 31)),
    }
    monkeypatch.setattr(main, "file_interval", lambda source_file, frequency: coverage[source_file])
    monkeypatch.setattr(main, "_is_final_source_file", lambda *args, **kwargs: True)

    selection = main._select_permanent_records(
        [annual_files],
        dataset="era5land",
        frequency="mon",
        permanent_watermark=date(2025, 11, 1),
        last_data_update=None,
        include_overlapping_watermark=True,
    )

    assert selection.records[0].files == ("2025",)
    assert selection.interval == coverage["2025"]


def test_temporary_monthly_selection_keeps_complete_annual_file(monkeypatch):
    annual = _record(files=("ELsf12_1M_2026_228.grb",))
    coverage = (date(2026, 1, 1), date(2026, 12, 31))
    monkeypatch.setattr(main, "file_interval", lambda source_file, frequency: coverage)
    monkeypatch.setattr(
        main,
        "overlaps_interval",
        lambda source_file, frequency, start, end: coverage[1] >= start and coverage[0] <= end,
    )

    selection = main._select_interval_records(
        [annual], frequency="mon", interval=(date(2026, 6, 26), date(2026, 9, 17))
    )

    assert selection.interval == coverage


def test_forced_update_starts_temporary_files_after_permanent_coverage(monkeypatch):
    planned_files: list[tuple[str, ...]] = []
    force_from = date(2026, 6, 26)
    record = _record(files=("permanent", "before", "temporary"))
    dates = {
        "permanent": (date(2026, 6, 26), date(2026, 6, 30)),
        "before": (date(2026, 6, 1), date(2026, 6, 25)),
        "temporary": (date(2026, 7, 1), date(2026, 9, 10)),
    }

    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "_existing_frequency_variables", lambda *args, **kwargs: {"tas"})
    monkeypatch.setattr(
        main,
        "_existing_variable_update_state",
        lambda *args, **kwargs: main.VariableUpdateState(None, date(2026, 9, 10), date(2026, 8, 31)),
    )
    monkeypatch.setattr(main, "_existing_variable_pressure_levels", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_resolve_update_records", lambda **kwargs: [record])
    monkeypatch.setattr(main, "file_interval", lambda source_file, frequency: dates[source_file])
    monkeypatch.setattr(
        main,
        "overlaps_interval",
        lambda source_file, frequency, start, end: dates[source_file][1] >= start and dates[source_file][0] <= end,
    )
    monkeypatch.setattr(
        main,
        "_select_permanent_records",
        lambda records, **kwargs: main.UpdateSelection([record._replace(files=("permanent",))], dates["permanent"], 1),
    )
    monkeypatch.setattr(
        main,
        "_map_update_records",
        lambda records, **kwargs: planned_files.extend(current.files for current in records),
    )
    monkeypatch.setattr(main, "_persist_real_data_watermark", lambda *args, **kwargs: None)
    monkeypatch.setattr(main, "_persist_permanent_watermark", lambda *args, **kwargs: None)

    main.run_update(
        Namespace(
            variables="tas",
            freq="1hr",
            dataset="era5land",
            zarr_format=2,
            output_path=None,
            chunk_size=16,
            batch_files=None,
            batch_months=None,
            preview=False,
            force_from=force_from,
            use_inventory_cache=True,
            use_input_cache=False,
            fail_on_duplicate_times=False,
            weights_dir="/tmp/weights",
            highest_level_only=False,
            root=None,
        )
    )

    assert planned_files == [("permanent", "temporary")]


def test_update_force_from_overrides_stored_update_boundaries(monkeypatch):
    resolved_intervals: list[tuple[date, date]] = []
    planned_intervals: list[tuple[date, date]] = []
    force_from = date(2026, 7, 1)

    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "_existing_frequency_variables", lambda *args, **kwargs: {"tas"})
    monkeypatch.setattr(
        main,
        "_existing_variable_update_state",
        lambda *args, **kwargs: main.VariableUpdateState(None, None, date(2026, 8, 31)),
    )
    monkeypatch.setattr(main, "_existing_variable_pressure_levels", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "_resolve_update_records",
        lambda **kwargs: resolved_intervals.append(kwargs["interval"]) or [_record(files=("/tmp/tas_2026-07-01.grb",))],
    )
    monkeypatch.setattr(
        main,
        "_select_permanent_records",
        lambda records, **kwargs: main.UpdateSelection([], None, 0),
    )
    monkeypatch.setattr(
        main, "_map_update_records", lambda records, **kwargs: planned_intervals.append(kwargs["interval"])
    )

    main.run_update(
        Namespace(
            variables="tas",
            freq="1hr",
            dataset="era5land",
            zarr_format=2,
            output_path=None,
            chunk_size=16,
            batch_files=None,
            batch_months=None,
            preview=False,
            force_from=force_from,
            use_inventory_cache=True,
            use_input_cache=False,
            fail_on_duplicate_times=False,
            weights_dir="/tmp/weights",
            highest_level_only=False,
            root=None,
        )
    )

    assert resolved_intervals == [(date(2026, 4, 1), datetime.now().astimezone().date())]
    assert planned_intervals == [(force_from, force_from)]


def test_update_snapshots_all_variable_coverage_before_writing(monkeypatch):
    events: list[str] = []
    monkeypatch.setattr(
        main,
        "selected_requests",
        lambda **_: ({}, [SimpleNamespace(name="tas"), SimpleNamespace(name="uas")]),
    )
    monkeypatch.setattr(main, "_existing_frequency_variables", lambda *args, **kwargs: {"tas", "uas"})
    monkeypatch.setattr(
        main,
        "_existing_variable_update_state",
        lambda *args, **kwargs: (
            events.append(f"coverage:{args[2]}") or main.VariableUpdateState(None, date(2026, 9, 4), date(2026, 8, 1))
        ),
    )
    monkeypatch.setattr(main, "_existing_variable_pressure_levels", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main,
        "_resolve_update_records",
        lambda **kwargs: [
            _record(files=(f"/tmp/{kwargs['variable']}_2026-09-04.grb",))._replace(variable=kwargs["variable"])
        ],
    )
    monkeypatch.setattr(main, "_select_permanent_records", lambda records, **_: main.UpdateSelection([], None, 0))
    monkeypatch.setattr(
        main, "_map_update_records", lambda records, **kwargs: events.append(f"map:{kwargs['variable']}")
    )
    monkeypatch.setattr(main, "_persist_real_data_watermark", lambda *args, **kwargs: None)

    main.run_update(
        Namespace(
            variables="tas,uas",
            freq="1hr",
            dataset="era5land",
            zarr_format=2,
            output_path=None,
            chunk_size=16,
            batch_files=None,
            batch_months=None,
            preview=False,
            force_from=None,
            use_inventory_cache=True,
            use_input_cache=False,
            fail_on_duplicate_times=False,
            weights_dir="/tmp/weights",
            highest_level_only=False,
            root=None,
        )
    )

    assert events == ["coverage:tas", "coverage:uas", "map:tas", "map:uas"]


# =============================================================================
# Tests for run_clean
# =============================================================================


def test_clean_deletes_the_dataset_root(monkeypatch):
    from heal_era5.helpers import cleanup

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cleanup, "delete_dataset_root", lambda **kwargs: calls.append(kwargs) or ["deleted root"])

    result = main.run_clean(
        Namespace(
            variables=None,
            levels=None,
            freq=None,
            dataset="era5land",
            output_path=Path("/tmp/output"),
            truncate_after=None,
            dry_run=True,
        )
    )

    assert result == 0
    assert calls == [{"dataset": "era5land", "output_path": Path("/tmp/output"), "dry_run": True}]


def test_clean_removes_selected_pressure_levels(monkeypatch):
    from heal_era5.helpers import cleanup

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cleanup,
        "clean_frequency_stores",
        lambda **kwargs: calls.append(kwargs) or ["removed levels"],
    )

    result = main.run_clean(
        Namespace(
            variables=None,
            levels="8-7",
            pressure_levels="1000,850",
            freq="1hr",
            dataset="era5land",
            output_path=None,
            truncate_after=None,
            dry_run=True,
        )
    )

    assert result == 0
    assert calls == [
        {
            "dataset": "era5land",
            "frequency": "1hr",
            "variable_names": None,
            "pressure_levels": (1000, 850),
            "levels": (8, 7),
            "output_path": None,
            "dry_run": True,
        }
    ]


def test_clean_pressure_levels_cannot_be_combined_with_variables():
    with pytest.raises(ValueError, match="cannot be combined"):
        main.run_clean(
            Namespace(
                variables="ta",
                levels=None,
                pressure_levels="1000",
                freq="1hr",
                dataset="era5land",
                output_path=None,
                truncate_after=None,
                dry_run=True,
            )
        )


def test_clean_pressure_levels_rejects_all():
    with pytest.raises(ValueError, match="does not accept 'all'"):
        main.run_clean(
            Namespace(
                variables=None,
                levels=None,
                pressure_levels="all",
                freq="1hr",
                dataset="era5land",
                output_path=None,
                truncate_after=None,
                dry_run=True,
            )
        )


def test_clean_pressure_levels_cannot_be_combined_with_truncation():
    with pytest.raises(ValueError, match="--pressure-levels"):
        main.run_clean(
            Namespace(
                variables=None,
                levels=None,
                pressure_levels="850",
                freq="1hr",
                dataset="era5land",
                output_path=None,
                truncate_after="2024-01-01",
                dry_run=False,
            )
        )


# =============================================================================
# Tests for run_merge
# =============================================================================


def test_merge_deletes_target_and_merges_sources(monkeypatch, tmp_path):
    source = tmp_path / "worker"
    target = tmp_path / "merged"
    target.mkdir()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [source])
    monkeypatch.setattr(main.shutil, "rmtree", lambda path: calls.append({"removed": path}))
    monkeypatch.setitem(
        sys.modules,
        "heal_era5.helpers.zarr_publisher",
        SimpleNamespace(merge_zarr_stores=lambda **kwargs: calls.append(kwargs) or [target / "level_4.zarr"]),
    )

    result = main.run_merge(
        Namespace(
            variables=None,
            dataset=None,
            freq=None,
            source_dirs=[str(source)],
            chunk_size=16,
            levels=None,
            interval=None,
            output_path=str(target),
            from_scratch=True,
            clean=False,
            zarr_format=2,
        )
    )

    assert result == 0
    assert calls[0] == {"removed": target}
    assert calls[1]["sources"] == [source]
    assert calls[1]["target_dir"] == target


def test_merge_requires_a_dataset_for_frequency_or_variable_selectors():
    with pytest.raises(ValueError, match="--dataset is required"):
        main.run_merge(
            Namespace(
                variables=None,
                dataset=None,
                freq="day",
                source_dirs=["/tmp/source"],
                chunk_size=16,
                levels=None,
                interval=None,
                output_path="/tmp/output",
                from_scratch=False,
                clean=False,
                zarr_format=2,
            )
        )


def test_merge_rejects_unmatched_sources(monkeypatch):
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [])

    with pytest.raises(ValueError, match="No matching merge source"):
        main.run_merge(
            Namespace(
                variables=None,
                dataset=None,
                freq=None,
                source_dirs=["/tmp/missing"],
                chunk_size=16,
                levels=None,
                interval=None,
                output_path="/tmp/output",
                from_scratch=False,
                clean=False,
                zarr_format=2,
            )
        )


def test_merge_rejects_nonpositive_chunk_size(monkeypatch):
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [Path("/tmp/source")])

    with pytest.raises(ValueError, match="chunk-size must be a positive integer"):
        main.run_merge(
            Namespace(
                variables=None,
                dataset=None,
                freq=None,
                source_dirs=["/tmp/source"],
                chunk_size=0,
                levels=None,
                interval=None,
                output_path="/tmp/output",
                from_scratch=False,
                clean=False,
                zarr_format=2,
            )
        )


def test_merge_uses_dataset_target_without_deleting_a_missing_directory(monkeypatch, tmp_path):
    source = tmp_path / "worker"
    target = tmp_path / "merged"
    target_dir = tmp_path / "era5land" / "day"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [source])
    monkeypatch.setattr(main, "merge_dataset_root", lambda *args, **kwargs: target_dir)
    monkeypatch.setattr(main.shutil, "rmtree", lambda path: calls.append({"removed": path}))
    monkeypatch.setitem(
        sys.modules,
        "heal_era5.helpers.zarr_publisher",
        SimpleNamespace(merge_zarr_stores=lambda **kwargs: calls.append(kwargs) or []),
    )

    assert (
        main.run_merge(
            Namespace(
                variables="tas",
                dataset="era5land",
                freq="day",
                source_dirs=[str(source)],
                chunk_size=16,
                levels="5-4",
                interval="20240101,20240131",
                output_path=str(target),
                from_scratch=True,
                clean=True,
                zarr_format=3,
            )
        )
        == 0
    )
    assert calls == [
        {
            "sources": [source],
            "target_dir": target_dir,
            "dataset": "era5land",
            "frequency": "day",
            "variable": ("tas",),
            "levels": (5, 4),
            "pressure_levels": None,
            "interval": (main.date(2024, 1, 1), main.date(2024, 1, 31)),
            "clean": True,
            "zarr_format": 3,
            "target_chunk_mb": 16,
        }
    ]


def test_merge_parses_pressure_level_selection(monkeypatch, tmp_path):
    """Merge should pass an explicit pressure-level subset to the publisher."""

    source = tmp_path / "worker"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [source])
    monkeypatch.setitem(
        sys.modules,
        "heal_era5.helpers.zarr_publisher",
        SimpleNamespace(merge_zarr_stores=lambda **kwargs: calls.append(kwargs) or []),
    )

    assert (
        main.run_merge(
            Namespace(
                variables=None,
                dataset=None,
                freq=None,
                source_dirs=[str(source)],
                chunk_size=16,
                levels=None,
                pressure_levels="1000,850",
                interval=None,
                output_path=str(tmp_path / "merged"),
                from_scratch=False,
                clean=False,
                zarr_format=2,
            )
        )
        == 0
    )
    assert calls[0]["pressure_levels"] == (1000, 850)


def test_merge_returns_success_when_no_stores_match(monkeypatch, tmp_path):
    source = tmp_path / "worker"
    target = tmp_path / "merged"
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [source])
    monkeypatch.setitem(
        sys.modules,
        "heal_era5.helpers.zarr_publisher",
        SimpleNamespace(merge_zarr_stores=lambda **kwargs: []),
    )

    assert (
        main.run_merge(
            Namespace(
                variables=None,
                dataset=None,
                freq=None,
                source_dirs=[str(source)],
                chunk_size=16,
                levels=None,
                interval=None,
                output_path=str(target),
                from_scratch=False,
                clean=False,
                zarr_format=2,
            )
        )
        == 0
    )
