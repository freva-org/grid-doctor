import json
import shutil
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import xarray as xr
from tqdm import tqdm

from era5_tables.cfchecker import cf_check

from config import (  # isort: skip
    DEFAULT_DUMMY_OUTPUT_DIR,
    JSON_OUTPUT_PATH,
    LOG_FILE,
)

DEFAULT_TIME_UNITS = "hours since 2000-01-01 00:00:00"
DEFAULT_CALENDAR = "gregorian"
BOUNDS_DIM_NAME = "bnds"


def _metadata_root(json_dir: Union[Path, str]) -> Path:
    """
    Resolve the generated ERA5 metadata root for a tables directory or root.

    Parameters:
        json_dir (Union[Path, str]): ERA5 tables directory or its parent
            metadata directory.

    Returns:
        Path: Directory containing ancillary metadata files such as
            ``ERA5_coordinate.json`` and ``ERA5_source_id.json``.
    """
    path = Path(json_dir)
    return path.parent if path.name == "Tables" else path


@lru_cache(maxsize=None)
def _load_json_metadata(metadata_root: Path, filename: str) -> dict[str, Any]:
    """
    Load a generated ancillary JSON payload from the ERA5 metadata directory.

    Parameters:
        metadata_root (Path): Directory that contains ancillary ERA5 JSON
            files.
        filename (str): Ancillary JSON filename to load.

    Returns:
        dict[str, Any]: Parsed JSON payload, or an empty dictionary if the file
            is missing.
    """
    metadata_path = metadata_root / filename
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _axis_templates(json_dir: Union[Path, str]) -> dict[str, dict[str, Any]]:
    """
    Return coordinate templates from the generated ERA5 ancillary metadata.

    Parameters:
        json_dir (Union[Path, str]): ERA5 tables directory or parent metadata
            directory.

    Returns:
        dict[str, dict[str, Any]]: Coordinate template definitions keyed by
            token name.
    """
    coordinates = _load_json_metadata(
        _metadata_root(json_dir), "ERA5_coordinate.json"
    )
    return coordinates.get("axis_entry", {})


def _json_table_files(json_dir: Path) -> list[Path]:
    """
    List generated ERA5 JSON table files that should be converted to dummy data.

    The function scans a directory for ``.json`` files, sorts them for stable
    processing order, and skips ``ERA5_CV.json`` because that file contains
    controlled-vocabulary metadata rather than variable tables.

    Parameters:
        json_dir (Path): Directory containing generated ERA5 JSON artifacts.

    Returns:
        list[Path]: Sorted JSON table paths eligible for dummy NetCDF
            generation.
    """
    return sorted(
        path
        for path in Path(json_dir).glob("*.json")
        if path.name != "ERA5_CV.json"
    )


def _resolve_json_tables_dir(json_dir: Path) -> Path:
    """
    Resolve the directory that actually stores the generated table JSON files.

    The CLI sometimes points to the ERA5 output root, where the variable tables
    live in a nested ``Tables/`` directory. This helper normalizes that input
    so downstream logic can always iterate over the real table files.

    Parameters:
        json_dir (Path): User-provided JSON path or output root.

    Returns:
        Path: Directory containing the variable table JSON files.
    """
    json_dir = Path(json_dir)
    tables_dir = json_dir / "Tables"
    return tables_dir if tables_dir.is_dir() else json_dir


def _dimension_alias(token: str) -> str:
    """
    Normalize table-specific dimension tokens to canonical internal aliases.

    The generated ERA5 JSON tables may use names such as ``time1``,
    ``latitude1``, ``alevel``, or ``plevel``. This helper maps those variants
    onto a smaller internal vocabulary so coordinate templates and dummy-data
    logic can be shared across tables.

    Parameters:
        token (str): Raw dimension token from a JSON variable definition.

    Returns:
        str: Canonical alias used internally by the dummy generator.
    """
    aliases = {
        "latitude1": "latitude",
        "longitude1": "longitude",
        "time1": "time",
        "time2": "time",
        "alevel": "alevel137",
        "plevel": "plev37",
        "alevhalf": "standard_hybrid_sigma_half",
        "olevel": "depth",
        "olevhalf": "depth",
    }
    return aliases.get(token, token)


def _template_for_dimension(
    token: str, json_dir: Union[Path, str]
) -> dict[str, Any]:
    """
    Resolve the coordinate template associated with a dimension token.

    Most tokens map directly to an entry in ``ERA5_coordinate.json``.
    Hybrid model levels are a special case: the raw ``alevel137`` template is
    merged with the ``standard_hybrid_sigma`` metadata so the resulting dummy
    coordinate behaves like a CF hybrid-pressure vertical axis while keeping the
    configured 137-level structure.

    Parameters:
        token (str): Dimension token from the ERA5 JSON variable metadata.

    Returns:
        dict[str, Any]: Coordinate template used to construct the dummy
            coordinate variable.

    Raises:
        KeyError: If the token cannot be matched to any known coordinate
            template.
    """
    if _dimension_alias(token) == "alevel137":
        axis_templates = _axis_templates(json_dir)
        raw_template = axis_templates.get("alevel137", {})
        hybrid_template = axis_templates.get("standard_hybrid_sigma", {})
        if raw_template and hybrid_template:
            merged = dict(raw_template)
            merged.update(
                {
                    "out_name": hybrid_template.get("out_name", "lev"),
                    "long_name": hybrid_template.get(
                        "long_name", raw_template.get("long_name", "")
                    ),
                    "standard_name": hybrid_template.get(
                        "standard_name",
                        raw_template.get("standard_name", ""),
                    ),
                    "positive": hybrid_template.get(
                        "positive", raw_template.get("positive", "")
                    ),
                    "must_have_bounds": hybrid_template.get(
                        "must_have_bounds",
                        raw_template.get("must_have_bounds", ""),
                    ),
                    "formula": hybrid_template.get("formula", ""),
                    "coordinate_formula_terms": "",
                    "z_bounds_factors": hybrid_template.get(
                        "z_bounds_factors", ""
                    ),
                    "z_factors": hybrid_template.get("z_factors", ""),
                    "units": hybrid_template.get(
                        "units", raw_template.get("units", "")
                    ),
                }
            )
            return merged

    axis_templates = _axis_templates(json_dir)

    template = axis_templates.get(token)
    if template:
        return template

    alias = _dimension_alias(token)
    template = axis_templates.get(alias)
    if template:
        return template

    raise KeyError(f"Unsupported coordinate token: {token}")


def _coord_name(token: str, template: dict[str, Any]) -> str:
    """
    Derive the NetCDF coordinate variable name for a dimension token.

    The dummy files use CF-friendly short names such as ``time``, ``lat``, and
    ``lon`` for common axes, while other coordinates fall back to the template
    ``out_name`` or canonical alias.

    Parameters:
        token (str): Raw dimension token from the ERA5 JSON table.
        template (dict[str, Any]): Coordinate template resolved for the token.

    Returns:
        str: Coordinate variable name to use in the dummy dataset.
    """
    alias = _dimension_alias(token)

    if alias == "time":
        return "time"
    if alias == "latitude":
        return "lat"
    if alias == "longitude":
        return "lon"

    return str(template.get("out_name") or alias)


def _dimension_priority(token: str) -> tuple[int, str]:
    """
    Provide a stable sort key for dimension tokens.

    Dummy files are easier to inspect and compare when dimensions appear in a
    conventional order, with time first, then vertical axes, then horizontal
    axes. This helper assigns that priority before a final lexical tiebreaker.

    Parameters:
        token (str): Raw dimension token to prioritize.

    Returns:
        tuple[int, str]: Sort key used to order dimension tokens.
    """
    alias = _dimension_alias(token)

    if alias == "time":
        return (0, token)
    if alias in {
        "alevel137",
        "plev37",
        "height10m",
        "height2m",
        "depth",
        "sea_ice_depth",
        "standard_hybrid_sigma",
        "standard_hybrid_sigma_half",
    }:
        return (1, token)
    if alias == "latitude":
        return (2, token)
    if alias == "longitude":
        return (3, token)
    return (4, token)


def _normalized_dimension_tokens(dimensions: str) -> list[str]:
    """
    Split and normalize a dimension string into sorted individual tokens.

    Parameters:
        dimensions (str): Whitespace-separated ``dimensions`` string from a
            variable entry.

    Returns:
        list[str]: Dimension tokens ordered with the preferred CF-style
            priority used by this module.
    """
    tokens = str(dimensions).split()
    return sorted(tokens, key=_dimension_priority)


def _numpy_dtype(type_name: str):
    """
    Translate CMOR-like type names into NumPy dtypes for dummy arrays.

    Parameters:
        type_name (str): Type label from the variable or coordinate metadata.

    Returns:
        Any: NumPy dtype object suitable for array construction.
    """
    normalized = str(type_name).strip().lower()

    if normalized in {"double", "float64"}:
        return np.float64
    if normalized in {"integer", "int", "int32"}:
        return np.int32

    return np.float32


def _coord_dtype(template: dict[str, Any]):
    """
    Resolve the NumPy dtype for a coordinate template.

    Parameters:
        template (dict[str, Any]): Coordinate template metadata.

    Returns:
        Any: NumPy dtype used when creating coordinate values and bounds.
    """
    return _numpy_dtype(template.get("type", "double"))


def _token_values(token: str, template: dict[str, Any]) -> np.ndarray:
    """
    Build representative coordinate values for a dimension token.

    The dummy generator prefers explicit ``requested`` values from the config.
    When those are absent, it falls back to simple synthetic values that are
    sufficient for CF validation, such as a tiny two-point lat/lon grid or a
    short synthetic time axis.

    Parameters:
        token (str): Dimension token being materialized.
        template (dict[str, Any]): Coordinate template metadata.

    Returns:
        np.ndarray: Coordinate values for the dummy dataset.
    """
    alias = _dimension_alias(token)
    requested = template.get("requested", "")
    scalar_value = template.get("value", "")

    if isinstance(requested, list) and requested:
        return np.asarray([float(value) for value in requested], dtype="f8")

    if alias == "latitude":
        return np.asarray([0.0, 1.0], dtype="f8")

    if alias == "longitude":
        return np.asarray([0.0, 1.0], dtype="f8")

    if alias == "time":
        return np.asarray([0.0, 1.0, 2.0], dtype="f8")

    if scalar_value not in {"", None}:
        return np.asarray([float(scalar_value)], dtype="f8")

    return np.asarray([1.0], dtype="f8")


def _bounds_values(
    token: str,
    values: np.ndarray,
    template: dict[str, Any],
) -> np.ndarray:
    """
    Construct bounds for a coordinate variable when required by the template.

    Explicit bounds from the config take precedence. Otherwise the function
    derives simple bounds from the coordinate values using special handling for
    time and horizontal axes and a generic midpoint-based approach for other
    coordinates.

    Parameters:
        token (str): Dimension token whose bounds are being created.
        values (np.ndarray): Coordinate values already built for that token.
        template (dict[str, Any]): Coordinate template metadata.

    Returns:
        np.ndarray: Two-column bounds array aligned with the coordinate.
    """
    requested_bounds = template.get("requested_bounds", "")
    alias = _dimension_alias(token)

    if isinstance(requested_bounds, list) and requested_bounds:
        flat = np.asarray(
            [float(value) for value in requested_bounds], dtype="f8"
        )
        return flat.reshape(-1, 2)

    if alias == "latitude":
        return np.asarray([[-0.5, 0.5], [0.5, 1.5]], dtype="f8")

    if alias == "longitude":
        return np.asarray([[-0.5, 0.5], [0.5, 1.5]], dtype="f8")

    if alias == "time":
        starts = values - 0.5
        ends = values + 0.5
        return np.column_stack([starts, ends])

    if values.size == 1:
        return np.asarray([[values[0], values[0]]], dtype="f8")

    deltas = np.diff(values)
    lower = np.empty_like(values)
    upper = np.empty_like(values)

    lower[1:] = values[1:] - deltas / 2
    upper[:-1] = values[:-1] + deltas / 2
    lower[0] = values[0] - deltas[0] / 2
    upper[-1] = values[-1] + deltas[-1] / 2

    return np.column_stack([lower, upper])


def _is_dimension_coordinate(
    token: str, values: np.ndarray, template: dict[str, Any]
) -> bool:
    """
    Decide whether a token should become a full dimension coordinate.

    Some entries in the JSON metadata represent true dimension axes, while
    others behave more like scalar or auxiliary coordinates. This helper uses
    axis metadata, value cardinality, and selected known aliases to make that
    distinction for the dummy dataset builder.

    Parameters:
        token (str): Dimension token under consideration.
        values (np.ndarray): Candidate coordinate values for the token.
        template (dict[str, Any]): Coordinate template metadata.

    Returns:
        bool: ``True`` when the token should be used as a dataset dimension
            coordinate, otherwise ``False``.
    """
    axis = template.get("axis", "")
    alias = _dimension_alias(token)

    if axis in {"T", "X", "Y"}:
        return True

    if values.size > 1:
        return True

    if alias in {
        "depth",
        "sea_ice_depth",
        "alevel137",
        "standard_hybrid_sigma",
    }:
        return True

    return False


def _variable_fill_value(variable: dict[str, Any]) -> Any:
    """
    Choose an appropriate missing-value marker for a dummy data variable.

    The fill value is selected from the variable type so the resulting NetCDF
    file stays compatible with the generated data dtype.

    Parameters:
        variable (dict[str, Any]): Variable metadata from the JSON table.

    Returns:
        Any: Scalar fill value matching the variable dtype.
    """
    dtype = _numpy_dtype(variable.get("type", "real"))

    if dtype == np.int32:
        return np.int32(-999)
    if dtype == np.float64:
        return np.float64(1e20)

    return np.float32(1e20)


def _data_values(
    variable: dict[str, Any], shape: tuple[int, ...]
) -> np.ndarray:
    """
    Create deterministic synthetic data for a dummy variable payload.

    The actual numeric values are not scientifically meaningful; they only need
    to be well-formed and shaped correctly so the generated NetCDF files can be
    validated structurally.

    Parameters:
        variable (dict[str, Any]): Variable metadata from the JSON table.
        shape (tuple[int, ...]): Target array shape derived from the dimensions.

    Returns:
        np.ndarray: Dummy numeric payload cast to the configured variable dtype.
    """
    dtype = _numpy_dtype(variable.get("type", "real"))
    size = int(np.prod(shape)) if shape else 1
    values = np.arange(size, dtype=np.float64).reshape(shape or ())
    return values.astype(dtype)


def _load_cv(json_dir: Path) -> dict[str, Any]:
    """
    Load the ERA5 controlled vocabulary used to enrich global attributes.

    The CV may be stored either alongside the JSON table files or one directory
    above them, depending on which path the caller provides.

    Parameters:
        json_dir (Path): JSON tables directory or parent output directory.

    Returns:
        dict[str, Any]: Loaded CV payload, or an empty dictionary when no
            ``ERA5_CV.json`` file can be found.
    """
    return _load_json_metadata(_metadata_root(json_dir), "ERA5_CV.json").get(
        "CV", {}
    )


def _load_source_id_metadata(json_dir: Path) -> dict[str, Any]:
    """
    Load ancillary source-id metadata used to enrich dummy global attributes.

    Unlike ``ERA5_CV.json``, the generated ``ERA5_source_id.json`` carries
    fields such as ``family`` that are needed for dummy-file metadata.

    Parameters:
        json_dir (Path): JSON tables directory or parent output directory.

    Returns:
        dict[str, Any]: Loaded source-id payload, or an empty dictionary when
            ``ERA5_source_id.json`` cannot be found.
    """
    return _load_json_metadata(
        _metadata_root(json_dir), "ERA5_source_id.json"
    ).get("source_id", {})


def _pick_cv_value(value: Any, default: str = "") -> str:
    """
    Normalize a CV entry into a single string value for NetCDF attributes.

    The ERA5 CV mixes scalar values, lists, and dictionaries. This helper picks
    a representative string from those structures so the dummy global
    attributes remain simple and serializable.

    Parameters:
        value (Any): Raw value extracted from the CV payload.
        default (str): Fallback string to use when the value is empty.

    Returns:
        str: Normalized string suitable for a NetCDF attribute.
    """
    if isinstance(value, list):
        return str(value[0]) if value else default

    if isinstance(value, dict):
        first_key = next(iter(value), None)
        return str(first_key) if first_key else default

    if value not in {"", None}:
        return str(value)

    return default


def _source_id_from_table_id(table_id: str) -> str:
    """
    Infer the ERA5 source identifier from the generated table identifier.

    Parameters:
        table_id (str): Table identifier from the generated JSON header.

    Returns:
        str: ``ERA-5`` or ``ERA-5-Land`` depending on the table family.
    """
    return "ERA-5-Land" if "ERA5Land" in table_id else "ERA-5"


def _family_from_source_info(source_info: dict[str, Any]) -> str:
    """
    Build a human-readable family/status label from configured source metadata.

    The current ERA5 table-generation flow does not yet propagate a per-variable
    batch marker such as ``E5``, ``E1``, ``ET``, or ``EP`` into the JSON
    tables. As a result, dummy NetCDF files can only infer a family label from
    the broader configured source metadata. When more than one family code is
    listed for a source, the result is reported as a mixed family.

    Parameters:
        source_info (dict[str, Any]): Source metadata entry from ``SOURCE_ID``
            inside the ERA5 CV.

    Returns:
        str: Human-readable family/status description for the file metadata.
    """
    family_codes = source_info.get("family", [])
    if isinstance(family_codes, str):
        family_codes = [family_codes]

    labels = {
        "ET": "provisional",
        "E1": "improved",
        "E5": "final",
        "EL": "final",
        "EP": "partial",
    }
    normalized_codes = [
        str(code).strip() for code in family_codes if str(code).strip()
    ]

    if not normalized_codes:
        return ""

    if len(normalized_codes) == 1:
        code = normalized_codes[0]
        return f"{labels.get(code, 'unknown')} ({code})"

    return f"mixed ({'/'.join(normalized_codes)})"


def _global_attributes(
    table_payload: dict[str, Any],
    variable_name: str,
    variable: dict[str, Any],
    json_dir: Optional[Path] = None,
) -> dict[str, str]:
    """
    Build the global attribute dictionary for a dummy NetCDF file.

    The output starts from the generated table header and is then enriched with
    selected values from ``ERA5_CV.json`` so the dummy files resemble the
    structure of a real published product closely enough for CF validation.

    Parameters:
        table_payload (dict[str, Any]): Full JSON table payload containing the
            ``Header`` section.
        variable_name (str): Name of the variable being written.
        variable (dict[str, Any]): Variable metadata from the JSON table.
        json_dir (Optional[Path]): JSON directory used to locate the ERA5 CV.

    Returns:
        dict[str, str]: Global NetCDF attributes for the dummy file.
    """
    header = table_payload.get("Header", {})
    cv = _load_cv(json_dir) if json_dir else {}
    source_id_metadata = _load_source_id_metadata(json_dir) if json_dir else {}

    attrs = {
        key: str(value)
        for key, value in header.items()
        if value not in {"", None}
        and key not in {"missing_value", "int_missing_value"}
    }

    table_id = attrs.get("table_id", "")
    source_id = _source_id_from_table_id(table_id)
    source_info = {
        **cv.get("source_id", {}).get(source_id, {}),
        **source_id_metadata.get(source_id, {}),
    }
    institution_id = "ECMWF"

    attrs.update(
        {
            "Conventions": "CF-1.8",
            "activity_id": _pick_cv_value(cv.get("activity_id")),
            "contact": _pick_cv_value(cv.get("contact")),
            "creation_date": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "data_specs_version": _pick_cv_value(
                cv.get("data_specs_version"),
                attrs.get("data_specs_version", ""),
            ),
            "frequency": str(variable.get("frequency", "")),
            "grid": str(
                variable.get("orig_grid", "regular latitude-longitude grid")
            ),
            "grid_label": "gn",
            "family": _family_from_source_info(source_info),
            "institution_id": institution_id,
            "institution": cv.get("institution_id", {}).get(
                institution_id, ""
            ),
            "license": _pick_cv_value(cv.get("license")),
            "nominal_resolution": _pick_cv_value(cv.get("nominal_resolution")),
            "product": _pick_cv_value(cv.get("product")),
            "realm": str(
                variable.get("modeling_realm", attrs.get("realm", ""))
            ),
            "region": _pick_cv_value(source_info.get("region")),
            "source_id": source_id,
            "source": _pick_cv_value(source_info.get("source")),
            "source_type": _pick_cv_value(source_info.get("source_type")),
            "source_version_number": _pick_cv_value(
                source_info.get("source_version_number")
            ),
            "tracking_id": "hdl:21.14102/dummy-era5-cf-validation",
            "variant_label": _pick_cv_value(cv.get("variant_label")),
            "variable_id": variable_name,
            "title": f"Dummy CF validation file for {variable_name}",
            "history": "Generated by era5-tables for CF validation",
        }
    )

    return {
        key: str(value)
        for key, value in attrs.items()
        if value not in {"", None}
    }


def _cell_measure_name(cell_measures: str) -> str:
    """
    Extract the referenced area-measure variable name from ``cell_measures``.

    Parameters:
        cell_measures (str): ``cell_measures`` attribute string from a variable
            entry.

    Returns:
        str: Referenced measure variable name, or an empty string when no
            ``area: ...`` entry is present.
    """
    parts = str(cell_measures).split()
    if len(parts) >= 2 and parts[0] == "area:":
        return parts[1]
    return ""


def _coordinate_attrs(token: str, template: dict[str, Any]) -> dict[str, Any]:
    """
    Build the attribute dictionary for a coordinate variable.

    This helper translates the coordinate template into NetCDF-ready metadata,
    including axis labels, units, standard names, optional positive direction,
    and hybrid-coordinate formula terms. Time coordinates receive synthetic
    calendar metadata suitable for dummy validation files.

    Parameters:
        token (str): Dimension token being materialized.
        template (dict[str, Any]): Coordinate template metadata.

    Returns:
        dict[str, Any]: Coordinate attributes with empty values removed.
    """
    alias = _dimension_alias(token)

    attrs = {
        "axis": template.get("axis", ""),
        "long_name": template.get("long_name", ""),
        "standard_name": template.get("standard_name", ""),
        "units": template.get("units", ""),
        "positive": template.get("positive", ""),
        "formula_terms": (
            template.get("coordinate_formula_terms", "")
            or template.get("z_factors", "")
        ),
    }

    if alias == "time":
        attrs["units"] = DEFAULT_TIME_UNITS
        attrs["calendar"] = DEFAULT_CALENDAR

    return {
        key: value for key, value in attrs.items() if value not in {"", None}
    }


def _parse_formula_terms(spec: str) -> dict[str, str]:
    """
    Parse a CF-style ``formula_terms`` string into role-to-variable mappings.

    Parameters:
        spec (str): Raw ``formula_terms`` string such as
            ``"p0: p0 a: a b: b ps: ps"``.

    Returns:
        dict[str, str]: Mapping from formula-term role to referenced variable
            name.
    """
    parts = str(spec).split()
    formula_terms: dict[str, str] = {}

    for idx in range(0, len(parts) - 1, 2):
        key = parts[idx].rstrip(":")
        value = parts[idx + 1]
        formula_terms[key] = value

    return formula_terms


def _formula_term_dims(
    term_name: str,
    vertical_dim: str,
    dim_coords: dict[str, np.ndarray],
) -> tuple[str, ...]:
    """
    Determine the dimension signature for an auxiliary formula-term variable.

    Different hybrid-coordinate terms live on different domains: coefficients
    ``a`` and ``b`` vary with the vertical dimension, ``ps`` varies on the
    time-lat-lon grid, and ``orog`` is purely horizontal.

    Parameters:
        term_name (str): Canonical formula-term role such as ``a`` or ``ps``.
        vertical_dim (str): Name of the vertical coordinate dimension.
        dim_coords (dict[str, np.ndarray]): Existing dimension coordinates in
            the dataset under construction.

    Returns:
        tuple[str, ...]: Dimensions to assign to the formula-term variable.
    """
    if term_name in {"a", "b"}:
        return (vertical_dim,)
    if term_name in {"a_bnds", "b_bnds"}:
        return (vertical_dim, BOUNDS_DIM_NAME)
    if term_name == "ps":
        dims = tuple(
            dim for dim in ("time", "lat", "lon") if dim in dim_coords
        )
        return dims or tuple(
            dim for dim in ("lat", "lon") if dim in dim_coords
        )
    if term_name == "orog":
        return tuple(dim for dim in ("lat", "lon") if dim in dim_coords)
    return ()


def _formula_term_values(
    term_name: str,
    dims: tuple[str, ...],
    dim_coords: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Generate synthetic values for a formula-term auxiliary variable.

    The numeric content is intentionally simple; the goal is only to create
    shape-compatible placeholder arrays that satisfy CF structure checks.

    Parameters:
        term_name (str): Canonical formula-term role such as ``a`` or ``ps``.
        dims (tuple[str, ...]): Dimensions assigned to the formula-term
            variable.
        dim_coords (dict[str, np.ndarray]): Existing dimension coordinates in
            the dataset under construction.

    Returns:
        np.ndarray: Dummy values for the requested formula-term variable.
    """
    if term_name == "p0":
        return np.asarray(100000.0, dtype="f8")

    if term_name in {"a", "b"}:
        nlev = len(dim_coords[dims[0]])
        values = np.linspace(0.0, 1.0, nlev, dtype="f8")
        if term_name == "a":
            return values
        return values[::-1]

    if term_name in {"a_bnds", "b_bnds"}:
        nlev = len(dim_coords[dims[0]])
        bounds = np.linspace(0.0, 1.0, nlev + 1, dtype="f8")
        lower = bounds[:-1]
        upper = bounds[1:]
        values = np.column_stack([lower, upper])
        if term_name == "a_bnds":
            return values
        return values[::-1]

    shape = tuple(len(dim_coords[dim]) for dim in dims)
    return np.ones(shape, dtype="f8")


def _ensure_formula_term_variables(
    data_vars: dict[str, Any],
    dim_coords: dict[str, np.ndarray],
    coord_attrs: dict[str, dict[str, Any]],
) -> None:
    """
    Add any auxiliary variables required by coordinate ``formula_terms``.

    When a coordinate advertises hybrid or other formula terms, the dummy file
    must also contain placeholder variables for those referenced terms. This
    helper inspects all coordinate attributes and appends any missing auxiliary
    arrays directly into the in-progress dataset payload.

    Parameters:
        data_vars (dict[str, Any]): Dataset variable mapping being assembled.
        dim_coords (dict[str, np.ndarray]): Dimension coordinate arrays already
            created for the dataset.
        coord_attrs (dict[str, dict[str, Any]]): Coordinate attribute mapping
            used to discover formula-term requirements.

    Returns:
        None: The function mutates ``data_vars`` in place.
    """
    for coord_name, attrs in coord_attrs.items():
        formula_terms = _parse_formula_terms(attrs.get("formula_terms", ""))
        if not formula_terms:
            continue

        for term_role, term_name in formula_terms.items():
            if term_name in data_vars or term_name in dim_coords:
                continue

            dims = _formula_term_dims(term_role, coord_name, dim_coords)
            values = _formula_term_values(term_role, dims, dim_coords)

            term_attrs: dict[str, Any] = {"long_name": term_role}
            if term_role == "p0":
                term_attrs["units"] = "Pa"
            elif term_role in {"a", "a_bnds"}:
                term_attrs["units"] = "1"
            elif term_role in {"b", "b_bnds"}:
                term_attrs["units"] = "1"
            elif term_role == "ps":
                term_attrs["standard_name"] = "surface_air_pressure"
                term_attrs["units"] = "Pa"
            elif term_role == "orog":
                term_attrs["standard_name"] = "surface_altitude"
                term_attrs["units"] = "m"

            data_vars[term_name] = (
                dims,
                values,
                {k: v for k, v in term_attrs.items() if v not in {"", None}},
            )


def _ensure_cell_measure_variable(
    data_vars: dict[str, Any],
    dim_coords: dict[str, np.ndarray],
    variable: dict[str, Any],
) -> None:
    """
    Add a placeholder cell-measure variable referenced by the main data field.

    At present this is mainly used for ``areacella`` when a variable advertises
    ``cell_measures = "area: areacella"``. The created array is a simple field
    of ones with the correct horizontal shape.

    Parameters:
        data_vars (dict[str, Any]): Dataset variable mapping being assembled.
        dim_coords (dict[str, np.ndarray]): Dimension coordinate arrays already
            created for the dataset.
        variable (dict[str, Any]): Main variable metadata from the JSON table.

    Returns:
        None: The function mutates ``data_vars`` in place when needed.
    """
    measure_name = _cell_measure_name(variable.get("cell_measures", ""))

    if not measure_name or measure_name in data_vars:
        return

    dims = tuple(dim for dim in ("lat", "lon") if dim in dim_coords)

    if not dims:
        return

    shape = tuple(len(dim_coords[dim]) for dim in dims)

    data_vars[measure_name] = (
        dims,
        np.ones(shape, dtype="f4"),
        {
            "long_name": "grid cell area",
            "standard_name": "cell_area",
            "units": "m2",
        },
    )


def _build_dataset(
    table_payload: dict[str, Any],
    variable_name: str,
    variable: dict[str, Any],
    json_dir: Optional[Path] = None,
) -> xr.Dataset:
    """
    Build an in-memory dummy NetCDF dataset for a single ERA5 variable.

    The resulting dataset contains one target variable plus any required
    coordinate variables, bounds variables, cell-measure variables, and
    formula-term auxiliaries needed for structural CF validation.

    Parameters:
        table_payload (dict[str, Any]): Full JSON table payload containing
            header and variable metadata.
        variable_name (str): Name of the variable to materialize.
        variable (dict[str, Any]): Variable metadata from the JSON table.
        json_dir (Optional[Path]): JSON tables directory or metadata root used
            to locate ancillary ERA5 metadata.

    Returns:
        xr.Dataset: In-memory dummy dataset ready to be written to NetCDF.
    """
    tokens = _normalized_dimension_tokens(variable.get("dimensions", ""))

    dim_coords: dict[str, np.ndarray] = {}
    aux_coords: dict[str, Any] = {}
    coord_attrs: dict[str, dict[str, Any]] = {}
    data_dims: list[str] = []
    data_vars: dict[str, Any] = {}
    encoding: dict[str, dict[str, Any]] = {}
    metadata_dir = json_dir or JSON_OUTPUT_PATH

    for token in tokens:
        template = _template_for_dimension(token, metadata_dir)
        coord_name = _coord_name(token, template)

        values = _token_values(token, template).astype(_coord_dtype(template))
        attrs = _coordinate_attrs(token, template)

        if _is_dimension_coordinate(token, values, template):
            if coord_name not in dim_coords:
                dim_coords[coord_name] = values
                coord_attrs[coord_name] = attrs

            if coord_name not in data_dims:
                data_dims.append(coord_name)
        else:
            aux_coords[coord_name] = values[0].item()
            coord_attrs[coord_name] = attrs

    shape = tuple(len(dim_coords[dim]) for dim in data_dims)

    data_attrs = {
        "long_name": variable.get("long_name", ""),
        "standard_name": variable.get("standard_name", ""),
        "units": variable.get("units", ""),
        "cell_methods": variable.get("cell_methods", ""),
        "cell_measures": variable.get("cell_measures", ""),
        "comment": variable.get("comment", ""),
        "original_name": variable.get("orig_name", ""),
        "original_units": variable.get("orig_units", ""),
        "original_short_name": variable.get("orig_short_name", ""),
        "conversion": variable.get("conversion", ""),
        "grib_table": variable.get("grib_table", ""),
        "grib_paramID": variable.get("grib_paramID", ""),
    }

    coordinate_names = data_dims + list(aux_coords)
    if coordinate_names:
        data_attrs["coordinates"] = " ".join(coordinate_names)

    data_attrs = {
        key: value
        for key, value in data_attrs.items()
        if value not in {"", None}
    }

    data_vars[variable_name] = (
        tuple(data_dims),
        _data_values(variable, shape),
        data_attrs,
    )

    _ensure_cell_measure_variable(data_vars, dim_coords, variable)
    _ensure_formula_term_variables(data_vars, dim_coords, coord_attrs)

    coords = {name: (name, values) for name, values in dim_coords.items()}

    coords.update(aux_coords)

    for token in tokens:
        template = _template_for_dimension(token, metadata_dir)
        coord_name = _coord_name(token, template)

        if template.get("must_have_bounds", "") != "yes":
            continue

        if coord_name not in dim_coords:
            continue

        bounds_name = f"{coord_name}_bnds"
        bounds = _bounds_values(
            token,
            dim_coords[coord_name],
            template,
        ).astype(_coord_dtype(template))

        data_vars[bounds_name] = (
            (coord_name, BOUNDS_DIM_NAME),
            bounds,
            {"long_name": f"{coord_name} bounds"},
        )

        coord_attrs.setdefault(coord_name, {})["bounds"] = bounds_name
        encoding[bounds_name] = {"_FillValue": None}

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=_global_attributes(
            table_payload=table_payload,
            variable_name=variable_name,
            variable=variable,
            json_dir=json_dir,
        ),
    )

    for coord_name, attrs in coord_attrs.items():
        if coord_name in ds.coords:
            ds[coord_name].attrs = attrs
            ds[coord_name].encoding["_FillValue"] = None

    encoding[variable_name] = {
        "_FillValue": _variable_fill_value(variable),
        "zlib": False,
    }

    ds.attrs["_encoding"] = encoding
    return ds


def _write_dummy_file(
    table_payload: dict[str, Any],
    table_name: str,
    variable_name: str,
    variable: dict[str, Any],
    output_dir: Path,
    json_dir: Optional[Path] = None,
) -> Path:
    """
    Write a single dummy NetCDF file for one table variable.

    Parameters:
        table_payload (dict[str, Any]): Full JSON table payload containing
            header and variable metadata.
        table_name (str): Name of the source table, used in the output
            filename.
        variable_name (str): Name of the variable to write.
        variable (dict[str, Any]): Variable metadata from the JSON table.
        output_dir (Path): Destination directory for the NetCDF file.
        json_dir (Optional[Path]): JSON directory used to locate ERA5 CV
            metadata.

    Returns:
        Path: Absolute path to the created dummy NetCDF file.
    """
    output_path = (
        output_dir / f"{variable_name.lower()}_{table_name.lower()}.nc"
    )

    ds = _build_dataset(
        table_payload=table_payload,
        variable_name=variable_name,
        variable=variable,
        json_dir=json_dir,
    )

    encoding = ds.attrs.pop("_encoding", {})
    ds.to_netcdf(output_path, encoding=encoding)

    return output_path


def generate_dummy_netcdf_files(
    json_dir: Union[Path, str] = JSON_OUTPUT_PATH,
    output_dir: Union[Path, str] = DEFAULT_DUMMY_OUTPUT_DIR,
    clean_output: bool = True,
) -> list[Path]:
    """
    Generate dummy NetCDF files for every variable in the ERA5 JSON tables.

    The function resolves the actual JSON tables directory, optionally removes
    any previous dummy output, and then writes one NetCDF file per variable
    entry across all generated ERA5 tables.

    Parameters:
        json_dir (Union[Path, str]): JSON tables directory or ERA5 output root.
        output_dir (Union[Path, str]): Destination directory for the dummy
            NetCDF files.
        clean_output (bool): Whether to remove the existing output directory
            before generating new files.

    Returns:
        list[Path]: Paths to all dummy NetCDF files created during the run.
    """
    json_dir = Path(json_dir)
    json_dir = _resolve_json_tables_dir(json_dir)

    output_dir = Path(output_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")

    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    created_files: list[Path] = []

    for json_file in tqdm(
        _json_table_files(json_dir),
        desc="Generating dummy NetCDF from json files",
    ):
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)

        table_name = json_file.stem

        for variable_name, variable in payload.get(
            "variable_entry", {}
        ).items():
            created_files.append(
                _write_dummy_file(
                    table_payload=payload,
                    table_name=table_name,
                    variable_name=variable_name,
                    variable=variable,
                    output_dir=output_dir,
                    json_dir=json_dir,
                )
            )

    return created_files


def validate_dummy_netcdf(
    json_dir: Union[Path, str] = JSON_OUTPUT_PATH,
    output_dir: Union[Path, str] = DEFAULT_DUMMY_OUTPUT_DIR,
    clean_output: bool = True,
    log_file: Optional[Path] = LOG_FILE,
) -> list[Path]:
    """
    Generate dummy NetCDF files and run the CF checker on the output folder.

    This is the high-level validation entry point used by the CLI. It first
    creates fresh dummy NetCDF files from the generated ERA5 JSON tables and
    then invokes the CF checker on the resulting directory.

    Parameters:
        json_dir (Union[Path, str]): JSON tables directory or ERA5 output root.
        output_dir (Union[Path, str]): Destination directory for the dummy
            NetCDF files.
        clean_output (bool): Whether to remove existing dummy output before
            generation.
        log_file (Optional[Path]): Optional log destination for CF-checker
            output.

    Returns:
        list[Path]: Paths to all dummy NetCDF files created before validation.
    """
    created_files = generate_dummy_netcdf_files(
        json_dir=json_dir,
        output_dir=output_dir,
        clean_output=clean_output,
    )

    cf_check([Path(output_dir)], log_file=log_file)

    return created_files
