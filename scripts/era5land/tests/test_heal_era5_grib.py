"""Tests for GRIB inventory caching."""

import json
from pathlib import Path

import pandas as pd

from heal_era5.helpers import grib


def test_cached_grib_inventory_reuses_overlapping_files(monkeypatch, tmp_path: Path):
    """Cache misses should scan only files absent from a later collection."""

    cache = tmp_path / "cache"
    cache.mkdir()
    source_files = [tmp_path / name for name in ("a.grb", "b.grb", "c.grb")]
    for source_file in source_files:
        source_file.touch()

    scanned: list[tuple[str, ...]] = []

    def fake_grib_inventory(files: list[str]) -> pd.DataFrame:
        scanned.append(tuple(files))
        return pd.DataFrame({"file": files, "message": [0]})

    monkeypatch.setattr(grib, "cache_dir", lambda: cache)
    monkeypatch.setattr(grib, "grib_inventory", fake_grib_inventory)

    first = grib.cached_grib_inventory(source_files[:2])
    second = grib.cached_grib_inventory(source_files[1:])

    resolved = [str(source_file.resolve()) for source_file in source_files]
    assert scanned == [(resolved[0],), (resolved[1],), (resolved[2],)]
    assert first["file"].tolist() == resolved[:2]
    assert second["file"].tolist() == resolved[1:]
    assert len(list(cache.glob("grib_inventory_file_*.pickle"))) == 3


def test_cached_grib_inventory_uses_provider_sidecar(monkeypatch, tmp_path: Path):
    """A valid provider sidecar should avoid the ecCodes and pickle fallback."""

    source_file = tmp_path / "surface.grb"
    source_file.write_bytes(b"x" * 100)
    index_file = source_file.with_suffix(".index")
    entries = [
        {
            "attrs": {"shortName": "rsn", "paramId": 33, "typeOfLevel": "surface", "stepType": "instant"},
            "extra": {"P1": 0, "P2": 0, "timeRangeIndicator": 0},
            "step": 0.0,
            "level": 0,
            "date": "20250701",
            "time": "0000",
            "_offset": 0,
            "_length": 50,
        },
        {
            "attrs": {"shortName": "rsn", "paramId": 33, "typeOfLevel": "surface", "stepType": "instant"},
            "extra": {"P1": 0, "P2": 0, "timeRangeIndicator": 0},
            "step": 0.0,
            "level": 0,
            "date": "20250701",
            "time": "0100",
            "_offset": 50,
            "_length": 50,
        },
    ]
    index_file.write_text("".join(f"{json.dumps(entry)}\n" for entry in entries), encoding="utf-8")

    monkeypatch.setattr(grib, "grib_inventory", lambda _: (_ for _ in ()).throw(AssertionError("ecCodes scan")))

    inventory = grib.cached_grib_inventory([source_file])

    assert inventory["message"].tolist() == [0, 1]
    assert inventory["shortName"].tolist() == ["rsn", "rsn"]
    assert inventory["valid_time"].tolist() == list(pd.to_datetime(["2025-07-01T00:00", "2025-07-01T01:00"]))
