import importlib.util
import json
import sys
from pathlib import Path


def load_file_fetcher():
    path = Path(__file__).parents[1] / "scripts" / "era5land" / "helpers" / "file_fetcher.py"
    spec = importlib.util.spec_from_file_location("era5land_file_fetcher", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_mapper():
    scripts_dir = Path(__file__).parents[1] / "scripts" / "era5land"
    src_dir = Path(__file__).parents[1] / "src"
    for path in (scripts_dir, src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    from helpers import mapper  # type: ignore

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
    mapper = tmp_path / "source_mapper.json"
    mapper.write_text(
        json.dumps(
            {
                "path": "/pool/data/ERA5/{dataset}/{stream}/{type}/{time_freq}/{parameter}/*.grb",
                "datasets": {
                    "era5land": {"priority": ["EL"], "table_prefix": "ERA5Land"},
                    "era5": {"priority": ["E1", "E5", "ET"], "table_prefix": "ERA5"},
                },
                "level_type": {
                    "stream": {"sfc": "sf"},
                    "type": {"an": "an", "fc": "fc"},
                },
                "frequency": {"1hr": "1H", "day": "1D", "mon": "1M", "fx": "IV"},
            }
        ),
        encoding="utf-8",
    )
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
        mapper_path=mapper,
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
    mapper = tmp_path / "source_mapper.json"
    mapper.write_text(
        (Path(__file__).parents[1] / "scripts" / "era5land" / "source_mapper.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
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
        mapper_path=mapper,
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
    mapper = tmp_path / "source_mapper.json"
    mapper.write_text(
        (Path(__file__).parents[1] / "scripts" / "era5land" / "source_mapper.json").read_text(encoding="utf-8"),
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
        mapper_path=mapper,
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

    attrs = mapper._global_attrs_for_records([record])

    assert attrs["table_id"] == "Table ERA5Land_1hr"
    assert attrs["source_id"] == "ERA-5-Land"
    assert attrs["frequency"] == "1hr"
    assert attrs["activity_id"] == "obs4MIPs"
    assert attrs["product"] == "reanalysis"
    assert attrs["institution_id"] == "ECMWF"
    assert attrs["institution"].startswith("The European Centre for Medium-Range")
    assert attrs["source_type"] == "reanalysis"
    assert attrs["family"] == "final (EL)"
    assert attrs["creation_date"].endswith("Z")


def test_replace_public_attrs_keeps_coordinates_and_grid_mapping(tmp_path: Path):
    mapper = load_mapper()
    store = tmp_path / "attrs.zarr"
    root = mapper.zarr.open_group(store, mode="w")
    array = root.create_array(name="pr", shape=(1,), chunks=(1,), dtype="f8")
    array.attrs.update(
        {
            "_ARRAY_DIMENSIONS": ["time"],
            "coordinates": "crs latitude longitude surface",
            "grid_mapping": "crs",
            "long_name": "old",
        }
    )

    changed = mapper._replace_public_attrs(array, {"long_name": "precipitation_flux"})

    assert changed is True
    assert array.attrs["coordinates"] == "crs latitude longitude surface"
    assert array.attrs["grid_mapping"] == "crs"
    assert array.attrs["long_name"] == "precipitation_flux"
