"""Metadata helpers for ERA5/ERA5-Land HEALPix outputs."""

from typing import Any

from ..resources import CMOR_TABLES_DIR, CMOR_TABLES_ROOT
from .file_fetcher import SOURCE_MAPPER, SourceRecord, load_json

OUTPUT_ATTR_KEYS = tuple(SOURCE_MAPPER.get("var_attrs", []))
LAST_DATA_UPDATE_ATTR = "last_data_update"
LAST_PERMANENT_UPDATE_ATTR = "last_permanent_update"
NOMINAL_RESOLUTION_INDEX_BY_SOURCE_ID = {
    "ERA-5": 1,
    "ERA-5-Land": 0,
}


def clean_output_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Keep only the curated output attrs for published variables."""

    cleaned = {key: value for key, value in attrs.items() if key in OUTPUT_ATTR_KEYS and value not in ("", None)}
    for key in (LAST_DATA_UPDATE_ATTR, LAST_PERMANENT_UPDATE_ATTR):
        if attrs.get(key) not in ("", None):
            cleaned[key] = attrs[key]
    return cleaned


def attrs_for_record(record: SourceRecord) -> dict[str, Any]:
    """Return the cleaned published attrs for one resolved variable."""

    attrs = dict(record.output_attrs)
    if record.conversion_factor != 1.0:
        attrs["conversion_factor"] = str(record.conversion_factor)
    return clean_output_attrs(attrs)


def _pick_cv_value(value: Any, default: str = "") -> str:
    """Normalise a CV value into one representative string."""

    if isinstance(value, list):
        return str(value[0]) if value else default
    if isinstance(value, dict):
        first_key = next(iter(value), None)
        return str(first_key) if first_key else default
    if value not in ("", None):
        return str(value)
    return default


def _source_id_from_table_id(table_id: str) -> str:
    """Infer the source identifier from a CMOR table identifier."""

    return "ERA-5-Land" if "ERA5Land" in table_id else "ERA-5"


def _nominal_resolution_for_source(source_id: str, cv: dict[str, Any]) -> str:
    """Return the nominal resolution for one ERA5 source identifier."""

    values = cv.get("nominal_resolution")
    if isinstance(values, list):
        index = NOMINAL_RESOLUTION_INDEX_BY_SOURCE_ID.get(source_id)
        if index is not None and 0 <= index < len(values):
            return str(values[index])
    return _pick_cv_value(values)


def global_attrs_for_dataset_frequency(dataset: str, frequency: str) -> dict[str, str]:
    """Build dataset-level attrs for one dataset/frequency CMOR table."""

    dataset_cfg = SOURCE_MAPPER["datasets"][dataset]
    table_prefix = str(dataset_cfg["table_prefix"])
    table_path = CMOR_TABLES_DIR / f"{table_prefix}_{frequency}.json"
    table_payload = load_json(table_path)
    header = table_payload.get("Header", {})
    cv = load_json(CMOR_TABLES_ROOT / "ERA5_CV.json").get("CV", {})

    attrs = {
        key: str(value)
        for key, value in header.items()
        if value not in ("", None) and key not in {"missing_value", "int_missing_value"}
    }

    table_id = attrs.get("table_id", "")
    source_id = _source_id_from_table_id(table_id)
    source_info = cv.get("source_id", {}).get(source_id, {})
    institution_id = str(source_info.get("institution_id", "ECMWF"))

    attrs.update(
        {
            "activity_id": _pick_cv_value(cv.get("activity_id")),
            "contact": _pick_cv_value(cv.get("contact")),
            "frequency": frequency,
            "institution_id": institution_id,
            "institution": _pick_cv_value(cv.get("institution_id", {}).get(institution_id)),
            "license": _pick_cv_value(cv.get("license")),
            "nominal_resolution": _nominal_resolution_for_source(source_id, cv),
            "product": _pick_cv_value(cv.get("product"), attrs.get("product", "")),
            "source_id": source_id,
            "source": _pick_cv_value(source_info.get("source")),
            "source_type": _pick_cv_value(source_info.get("source_type")),
            "variant_label": _pick_cv_value(cv.get("variant_label")),
        }
    )

    return {key: str(value) for key, value in attrs.items() if value not in ("", None)}


def global_attrs_for_records(records: list[SourceRecord]) -> dict[str, str]:
    """Build dataset-level attrs from the CMOR table header and ERA5 CV."""

    if not records:
        return {}

    first_record = records[0]
    return global_attrs_for_dataset_frequency(first_record.dataset, first_record.frequency)
