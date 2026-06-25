import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Union

from dotenv import dotenv_values

DEFAULT_CONVERSION_TABLE_PATH = Path(__file__).with_name("variable_map.json")


def looks_like_url(value: str) -> bool:
    """Return whether a path-like string appears to be a URL."""
    return "://" in value


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON object from disk."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return data


def resolve_path(value: Union[str, Path], *, base_dir: Path) -> Path:
    """Resolve a path relative to a base directory."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_env_file(path: Path) -> Dict[str, str]:
    """Read environment values from a dotenv file."""
    if not path.exists():
        return {}
    values = dotenv_values(path)
    return {str(key): value for key, value in values.items() if value is not None}


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load conversion configuration and resolve referenced paths."""
    config_file = Path(config_path).resolve()
    config = load_json(config_file)
    env_file = config.get("_env_file")
    if env_file:
        env_path = resolve_path(env_file, base_dir=config_file.parent)
        config["destination"].update(load_env_file(env_path))
        config["_env_file"] = str(env_path)
    config["_config_path"] = str(config_file)
    config["_config_dir"] = str(config_file.parent)

    conversion_table = config.get("conversion_table")
    if conversion_table is None:
        config["conversion_table"] = str(DEFAULT_CONVERSION_TABLE_PATH.resolve())
    else:
        config["conversion_table"] = str(
            resolve_path(conversion_table, base_dir=config_file.parent)
        )
    return config


def load_conversion_rules(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Load and normalise conversion rules."""
    rules = load_json(config["conversion_table"])
    normalised: Dict[str, Dict[str, Any]] = {}
    for source_name, rule in rules.items():
        if not isinstance(rule, dict):
            raise TypeError(
                f"Conversion rule for {source_name!r} must be a JSON object."
            )
        merged = {"source": source_name}
        merged.update(rule)
        merged.setdefault("target", source_name)
        normalised[source_name] = merged
    return normalised


def resolve_requested_rules(
    config: Dict[str, Any],
    conversion_rules: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter conversion rules to the requested variables."""
    requested = config.get("variables")
    if requested is None or len(requested) == 0:
        return list(conversion_rules.values())
    if not isinstance(requested, list) or not all(
        isinstance(item, str) for item in requested
    ):
        raise TypeError("'variables' must be a list of strings.")

    requested_set = set(requested)
    selected: List[Dict[str, Any]] = []
    for rule in conversion_rules.values():
        source_name = str(rule["source"])
        target_name = str(rule["target"])
        if source_name in requested_set or target_name in requested_set:
            selected.append(rule)

    matched = {str(rule["source"]) for rule in selected} | {
        str(rule["target"]) for rule in selected
    }
    missing = sorted(requested_set - matched)
    if missing:
        raise KeyError(
            "No conversion rules defined for requested variables: " + ", ".join(missing)
        )
    return selected


def get_source_cfg(
    config: Dict[str, Any],
    *,
    kind_override: str | None = None,
) -> Dict[str, Any]:
    """Return the selected source configuration."""
    source = config["source"]
    kind_value = source.get("kind", "grib")
    if isinstance(kind_value, str):
        if kind_override is not None and kind_override != kind_value:
            raise KeyError(
                f"Requested source kind {kind_override!r} but config uses {kind_value!r}."
            )
        return dict(source)
    if not isinstance(kind_value, dict):
        raise TypeError("'source.kind' must be a string or a JSON object.")

    selected_kind = str(
        kind_override
        or config.get("_selected_source_kind")
        or source.get("default_kind", "grib")
    )
    if selected_kind not in kind_value:
        raise KeyError(f"Source configuration for kind {selected_kind!r} is missing.")

    selected_cfg = kind_value[selected_kind]
    if not isinstance(selected_cfg, dict):
        raise TypeError(f"'source.kind.{selected_kind}' must be a JSON object.")

    shared = {
        key: value
        for key, value in source.items()
        if key not in {"kind", "default_kind"}
    }
    return shared | selected_cfg | {"kind": selected_kind}


def _require_rule_attr(rule: Dict[str, Any], attr_name: str) -> str:
    """Fetch a required attribute from a rule."""
    attrs = rule.get("attrs", {})
    if not isinstance(attrs, dict):
        raise TypeError(f"'attrs' for {rule['source']!r} must be a JSON object.")
    value = attrs.get(attr_name)
    if value in (None, ""):
        raise KeyError(f"Missing attrs.{attr_name!r} for variable {rule['source']!r}.")
    return str(value)


def _parse_level_type(level_type: str) -> Dict[str, str]:
    """Split an ERA5 level-type string into source-path components."""
    parts = level_type.split("_")
    if len(parts) < 2:
        raise ValueError(f"Expected level_type like 'sfc_fc_land', got {level_type!r}.")

    stream_map = {
        "sfc": "sf",
    }
    dataset_map = {
        "land": "EL",
    }

    stream_key = parts[0]
    type_key = parts[1]
    domain_key = parts[2] if len(parts) > 2 else ""

    result = {
        "stream": stream_map.get(stream_key, stream_key),
        "type": type_key,
    }
    if domain_key:
        result["dataset"] = dataset_map.get(domain_key, domain_key)
    return result


def _map_frequency_to_source_dir(frequency: str) -> str:
    """Translate a rule frequency into the source directory code."""
    mapping = {
        "1hr": "1H",
        "day": "1D",
        "mon": "1M",
    }
    if frequency not in mapping:
        raise ValueError(f"Unsupported frequency {frequency!r} for source path.")
    return mapping[frequency]


def source_path_fields(rule: Dict[str, Any]) -> Dict[str, str]:
    """Build placeholder values for source path templates."""
    level_type = _require_rule_attr(rule, "level_type")
    fields = _parse_level_type(level_type)
    fields["time_freq"] = _map_frequency_to_source_dir(
        _require_rule_attr(rule, "frequency")
    )
    fields["parameter"] = _require_rule_attr(rule, "grib_paramID")
    return fields


def source_path_template(
    config: Dict[str, Any],
    rule: Dict[str, Any],
    *,
    path_key: str = "path",
    kind_override: str | None = None,
) -> str:
    """Render the source path template for a conversion rule."""
    source = get_source_cfg(config, kind_override=kind_override)
    rule_key = "source_path" if path_key == "path" else f"source_{path_key}"
    template = str(rule.get(rule_key, source[path_key]))
    fields = source_path_fields(rule)
    return template.format(
        variable=rule["source"],
        source=rule["source"],
        target=rule["target"],
        **fields,
    )


def kerchunk_reference_patterns(
    config: Dict[str, Any], rule: Dict[str, Any]
) -> List[str]:
    """Return candidate kerchunk reference paths for a rule."""
    source = get_source_cfg(config, kind_override="kerchunk")
    fields = source_path_fields(rule)
    raw_path = str(rule.get("source_path", source["path"])).rstrip("/")

    if looks_like_url(raw_path):
        return [raw_path]

    if any(token in raw_path for token in ("*", "?", "[")):
        return [
            raw_path.format(
                variable=rule["source"],
                source=rule["source"],
                target=rule["target"],
                **fields,
            )
        ]

    return [
        f"{raw_path}/{fields['dataset']}_{fields['stream']}_{fields['type']}_{fields['time_freq']}_{fields['parameter']}.parquet",
        f"{raw_path}/{fields['dataset']}_{fields['stream']}_{fields['type']}_{fields['time_freq']}*.parquet",
    ]


def resolve_source_files(config: Dict[str, Any], rule: Dict[str, Any]) -> List[str]:
    """Resolve source files for a conversion rule."""
    source = get_source_cfg(config)
    base_dir = Path(config["_config_dir"])

    def resolve_globbed(pattern_template: str) -> List[str]:
        if looks_like_url(pattern_template):
            return [pattern_template]
        pattern_path = Path(pattern_template).expanduser()
        pattern = str(
            pattern_path if pattern_path.is_absolute() else base_dir / pattern_template
        )
        return sorted(glob(pattern))

    if source.get("kind", "grib") == "kerchunk":
        for template in kerchunk_reference_patterns(config, rule):
            kerchunk_files = resolve_globbed(template)
            if kerchunk_files:
                return [kerchunk_files[0]]

        if not bool(config["source"].get("fallback_to_grib", True)):
            template = kerchunk_reference_patterns(config, rule)[-1]
            pattern_path = Path(template).expanduser()
            pattern = str(
                pattern_path if pattern_path.is_absolute() else base_dir / template
            )
            raise FileNotFoundError(
                f"No kerchunk source files matched {pattern!r} "
                f"for variable {rule['source']!r}."
            )

        fallback_template = source_path_template(config, rule, kind_override="grib")
        if looks_like_url(fallback_template):
            return [fallback_template]
        fallback_files = resolve_globbed(fallback_template)
        if fallback_files:
            return fallback_files
        pattern_path = Path(template).expanduser()
        pattern = str(
            pattern_path if pattern_path.is_absolute() else base_dir / template
        )
        raise FileNotFoundError(
            f"No kerchunk or grib source files matched {pattern!r} "
            f"for variable {rule['source']!r}."
        )

    template = source_path_template(config, rule)
    if looks_like_url(template):
        return [template]
    files = resolve_globbed(template)
    if not files:
        pattern_path = Path(template).expanduser()
        pattern = str(
            pattern_path if pattern_path.is_absolute() else base_dir / template
        )
        raise FileNotFoundError(
            f"No source files matched {pattern!r} for variable {rule['source']!r}."
        )
    return files


def override_source_kind(config: Dict[str, Any], kind: str | None) -> Dict[str, Any]:
    """Return a config override for the selected source kind."""
    if kind is None:
        return config
    return config | {"_selected_source_kind": kind}
