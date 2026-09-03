"""Unit tests for the direct heal-era5 command handlers."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from heal_era5 import main
from heal_era5.helpers.file_fetcher import SourceRecord, UnresolvedRecord


def _request(name: str = "tas") -> tuple[tuple[object, ...], list[SimpleNamespace]]:
    return (), [SimpleNamespace(name=name)]


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


# =============================================================================
# Tests for run_update
# =============================================================================


def test_update_preview_skips_missing_stores(monkeypatch):
    logged: list[tuple[object, str]] = []
    monkeypatch.setattr(main, "selected_requests", lambda **_: _request())
    monkeypatch.setattr(main, "_existing_variable_last_date", lambda *args, **kwargs: (None, None))
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
    monkeypatch.setattr(
        main,
        "merge_zarr_stores",
        lambda **kwargs: calls.append(kwargs) or [target / "level_4.zarr"],
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
    monkeypatch.setattr(main, "merge_zarr_stores", lambda **kwargs: calls.append(kwargs) or [])

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
            "interval": (main.date(2024, 1, 1), main.date(2024, 1, 31)),
            "clean": True,
            "zarr_format": 3,
            "target_chunk_mb": 16,
        }
    ]


def test_merge_returns_success_when_no_stores_match(monkeypatch, tmp_path):
    source = tmp_path / "worker"
    target = tmp_path / "merged"
    monkeypatch.setattr(main, "expand_source_dirs", lambda values: [source])
    monkeypatch.setattr(main, "merge_zarr_stores", lambda **kwargs: [])

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
