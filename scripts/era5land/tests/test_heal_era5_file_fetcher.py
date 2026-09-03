import json
from pathlib import Path
from types import SimpleNamespace


def load_file_fetcher():
    """Import the packaged source-file resolver."""

    from heal_era5.helpers import file_fetcher

    return file_fetcher


def load_mapper():
    """Import the packaged remapping helpers."""

    from heal_era5.helpers import mapper

    return mapper


def write_table(path: Path, entries: dict[str, dict[str, str]]) -> None:
    path.write_text(
        json.dumps({"Header": {}, "variable_entry": entries}),
        encoding="utf-8",
    )


def test_resolves_era5land_paths_from_out_name_and_filters_interval(tmp_path: Path):
    fetcher = load_file_fetcher()
    var_table = tmp_path / "era5_era5land.csv"
    var_table.write_text("varname|reanalysis\ntas|E5, EL\n", encoding="utf-8")
    table_dir = tmp_path / "Tables"
    table_dir.mkdir()
    write_table(
        table_dir / "ERA5Land_1hr.json",
        {
            "2t": {
                "out_name": "tas",
                "frequency": "1hr",
                "level_type": "sfc_fc_land",
                "DKRZ_ID": "167",
            }
        },
    )
    root = tmp_path / "pool" / "data" / "ERA5"
    source_dir = root / "EL" / "sf" / "fc" / "1H" / "167"
    source_dir.mkdir(parents=True)
    kept = source_dir / "ELsf12_1H_2026-04-16_167.grb"
    kept.write_text("", encoding="utf-8")
    (source_dir / "ELsf12_1H_2026-05-16_167.grb").write_text("", encoding="utf-8")

    records = fetcher.resolve_records(
        var_table=var_table,
        cmor_tables_dir=table_dir,
        dataset="era5land",
        variables=("tas",),
        frequencies=("1hr",),
        interval=fetcher.parse_interval("20260401,20260430"),
        root=str(root),
        glob_files=True,
    )

    assert len(records) == 1
    assert records[0].variable == "tas"
    assert records[0].table_variable == "2t"
    assert records[0].pattern == str(source_dir / "*.grb")
    assert records[0].files == (str(kept),)


def test_interval_filter_matches_file_granularity():
    fetcher = load_file_fetcher()

    assert fetcher.overlaps_interval(
        "ELsf12_1M_2005_167.grb",
        "mon",
        *fetcher.parse_interval("20050501,20050601"),
    )
    assert not fetcher.overlaps_interval(
        "ELsf12_1M_2025_167.grb",
        "mon",
        *fetcher.parse_interval("20050501,20050601"),
    )
    assert fetcher.overlaps_interval(
        "ELsf12_1D_2022-08_228.grb",
        "day",
        *fetcher.parse_interval("20220815,20220816"),
    )
    assert not fetcher.overlaps_interval(
        "ELsf12_1D_2022-09_228.grb",
        "day",
        *fetcher.parse_interval("20220815,20220816"),
    )
    assert fetcher.overlaps_interval(
        "ELsf12_1H_2025-03-13_228.grb",
        "1hr",
        *fetcher.parse_interval("20250313,20250313"),
    )


def test_patterns_only_keeps_missing_source_visible(tmp_path: Path):
    fetcher = load_file_fetcher()
    var_table = tmp_path / "era5_era5land.csv"
    var_table.write_text("varname|reanalysis\npr|E5, EL\n", encoding="utf-8")
    table_dir = tmp_path / "Tables"
    table_dir.mkdir()
    write_table(
        table_dir / "ERA5Land_day.json",
        {
            "pr": {
                "out_name": "pr",
                "frequency": "day",
                "level_type": "sfc_fc_land",
                "DKRZ_ID": "228",
            }
        },
    )

    records = fetcher.resolve_records(
        var_table=var_table,
        cmor_tables_dir=table_dir,
        dataset="era5land",
        variables=None,
        frequencies=("day",),
        interval=(None, None),
        root=str(tmp_path / "pool" / "data" / "ERA5"),
        glob_files=False,
    )

    assert len(records) == 1
    assert records[0].files == ()
    assert records[0].pattern.endswith("/EL/sf/fc/1D/228/*.grb")


def test_era5_skips_model_levels_and_pl_invariant(tmp_path: Path):
    fetcher = load_file_fetcher()
    var_table = tmp_path / "era5_era5land.csv"
    var_table.write_text(
        "varname|reanalysis\nua|E5\norog|E5\n",
        encoding="utf-8",
    )
    table_dir = tmp_path / "Tables"
    table_dir.mkdir()
    write_table(
        table_dir / "ERA5_1hr.json",
        {
            "ua": {
                "out_name": "ua",
                "frequency": "1hr",
                "level_type": "ml_an",
                "DKRZ_ID": "131",
            }
        },
    )
    write_table(
        table_dir / "ERA5_fx.json",
        {
            "orog": {
                "out_name": "orog",
                "frequency": "fx",
                "level_type": "pl_an",
                "DKRZ_ID": "129",
            }
        },
    )

    records = fetcher.resolve_records(
        var_table=var_table,
        cmor_tables_dir=table_dir,
        dataset="era5",
        variables=None,
        frequencies=("1hr", "fx"),
        interval=(None, None),
        root=str(tmp_path / "pool" / "data" / "ERA5"),
        glob_files=False,
    )

    assert records == []


def test_global_attrs_use_table_header_and_cv_metadata():
    fetcher = load_file_fetcher()
    mapper = load_mapper()

    record = fetcher.SourceRecord(
        variable="tas",
        table_variable="2t",
        dataset="era5land",
        dataset_code="EL",
        frequency="1hr",
        stream="sf",
        type="fc",
        parameter="167",
        level_type="sfc_fc_land",
        pattern="/tmp/example/*.grb",
        files=("/tmp/example/ELsf12_1H_2026-04-16_167.grb",),
        conversion_factor=1.0,
        output_attrs={"out_name": "tas"},
    )

    attrs = mapper.global_attrs_for_records([record])

    assert attrs["table_id"] == "Table ERA5Land_1hr"
    assert attrs["source_id"] == "ERA-5-Land"
    assert attrs["frequency"] == "1hr"
    assert attrs["activity_id"] == "obs4MIPs"
    assert attrs["product"] == "reanalysis"
    assert attrs["institution_id"] == "ECMWF"
    assert attrs["institution"].startswith("The European Centre for Medium-Range")
    assert attrs["source_type"] == "reanalysis"


def test_replace_public_attrs_keeps_coordinates_and_grid_mapping():
    from heal_era5.helpers import zarr_publisher

    array = SimpleNamespace(
        attrs={
            "_ARRAY_DIMENSIONS": ["time"],
            "coordinates": "crs latitude longitude surface",
            "grid_mapping": "crs",
            "long_name": "old",
        }
    )

    changed = zarr_publisher._replace_public_attrs(array, {"long_name": "precipitation_flux"})

    assert changed is True
    assert array.attrs["coordinates"] == "crs latitude longitude surface"
    assert array.attrs["grid_mapping"] == "crs"
    assert array.attrs["long_name"] == "precipitation_flux"
