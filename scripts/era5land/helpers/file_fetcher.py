#! /usr/bin/env python3
"""Resolve ERA5 and ERA5-Land source GRIB files from the local CMOR tables."""

import csv
import json
import re
import ast
import operator
from datetime import date, datetime
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Union

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_MAPPER = SCRIPT_DIR / ".." / "assets" / "source_mapper.json"
DAY_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")
MONTH_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})(?!-\d{2})")
YEAR_RE = re.compile(r"(?<!\d)(?P<year>\d{4})(?!\d)")
DATE_VALUE_RE = re.compile(
    r"^(?P<year>\d{4})(?:-?(?P<month>\d{2})(?:-?(?P<day>\d{2}))?)?$"
)

def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON object from disk."""

    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data

SOURCE_MAPPER = load_json(DEFAULT_SOURCE_MAPPER)

class VariableRequest(NamedTuple):
    """One requested CMOR variable and the reanalysis sources allowed for it."""

    name: str
    reanalysis: Tuple[str, ...]


class SourceRecord(NamedTuple):
    """Resolved source metadata and matching files for one variable/frequency."""

    variable: str
    table_variable: str
    dataset: str
    dataset_code: str
    frequency: str
    stream: str
    type: str
    parameter: str
    level_type: str
    pattern: str
    files: Tuple[str, ...]
    conversion_factor: float
    output_attrs: Dict[str, str]


class UnresolvedRecord(NamedTuple):
    """A requested variable/frequency pair that could not be resolved."""

    variable: str
    frequency: str
    reason: str


def _safe_eval_numeric_expression(expression: str) -> float:
    """Evaluate a simple numeric expression containing only literals and */+-."""

    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return float(operators[type(node.op)](_eval(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return float(operators[type(node.op)](_eval(node.left), _eval(node.right)))
        raise ValueError(f"Unsupported numeric expression: {expression!r}")

    return _eval(ast.parse(expression, mode="eval"))


def parse_conversion_factor(entry: Dict[str, Any]) -> float:
    """Return the numeric multiplicative conversion factor for one CMOR entry."""

    raw = str(entry.get("conversion", "")).strip()
    if not raw:
        return 1.0
    return _safe_eval_numeric_expression(raw)


def extract_output_attrs(entry: Dict[str, Any]) -> Dict[str, str]:
    """Extract relevant output metadata from one CMOR entry."""

    keys = tuple(SOURCE_MAPPER.get("var_attrs", []))
    attrs: Dict[str, str] = {}
    for key in keys:
        value = str(entry.get(key, "")).strip()
        if value:
            attrs[key] = value
    return attrs


def split_csv_list(value: str) -> Tuple[str, ...]:
    """Split a comma-separated table value, trimming whitespace."""

    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_variable_requests(path: Union[str, Path]) -> List[VariableRequest]:
    """Read the variable selection CSV."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = [
            VariableRequest(
                name=str(row["varname"]).strip(),
                reanalysis=split_csv_list(str(row["reanalysis"])),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No variables found in {path}")
    return rows


def dataset_code_allowed(allowed_codes: Iterable[str], codes: Iterable[str]) -> bool:
    """Return whether a variable-table row permits the selected source codes."""

    allowed = set(codes)
    return bool(set(allowed_codes) & allowed)


def selected_variables(
    requests: List[VariableRequest],
    *,
    allowed_codes: Iterable[str],
    variables: Optional[Tuple[str, ...]],
) -> List[VariableRequest]:
    """Filter CSV requests by allowed source codes and optional variable names."""

    requested = set(variables or ())
    selected = [
        request
        for request in requests
        if dataset_code_allowed(allowed_codes, request.reanalysis)
        and (not requested or request.name in requested)
    ]
    missing = sorted(requested - {request.name for request in selected})
    if missing:
        raise KeyError(
            "Requested variables are not available for the selected source: "
            + ", ".join(missing)
        )
    return selected


def load_cmor_variable_entries(
    cmor_tables_dir: Union[str, Path],
    *,
    table_prefix: str,
    frequency: str,
) -> Dict[str, Dict[str, Any]]:
    """Load variable entries for one CMOR table frequency."""

    table_path = Path(cmor_tables_dir) / f"{table_prefix}_{frequency}.json"
    table = load_json(table_path)
    entries = table.get("variable_entry")
    if not isinstance(entries, dict):
        raise TypeError(f"Missing JSON object 'variable_entry' in {table_path}")
    return {
        str(name): entry
        for name, entry in entries.items()
        if isinstance(entry, dict)
    }


def find_variable_entry(
    entries: Dict[str, Dict[str, Any]],
    variable: str,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Find a CMOR table entry by entry name or by ``out_name``."""

    direct = entries.get(variable)
    if direct is not None:
        return variable, direct
    for table_name, entry in entries.items():
        if str(entry.get("out_name", "")).strip() == variable:
            return table_name, entry
    return None


def parse_interval(value: Optional[str]) -> Tuple[Optional[date], Optional[date]]:
    """Parse ``start,end`` where each value can be YYYY, YYYYMM, or YYYYMMDD."""

    if value in (None, ""):
        return None, None
    parts = value.split(",", maxsplit=1)
    start = parse_date_value(parts[0], bound="start") if parts[0].strip() else None
    if len(parts) == 1 or not parts[1].strip():
        end = date.today()
    else:
        end = parse_date_value(parts[1], bound="end")
    if start is not None and end is not None and start > end:
        raise ValueError("--interval start date must be before or equal to end date")
    return start, end


def parse_date(value: str) -> date:
    """Parse a compact or dashed date."""

    value = value.strip()
    fmt = "%Y-%m-%d" if "-" in value else "%Y%m%d"
    return datetime.strptime(value, fmt).date()


def parse_date_value(value: str, *, bound: str) -> date:
    """Parse YYYY, YYYYMM, or YYYYMMDD in compact or dashed form."""

    text = value.strip()
    match = DATE_VALUE_RE.match(text)
    if match is None:
        raise ValueError(
            "Unsupported date value {!r}; use YYYY, YYYYMM, YYYYMMDD, "
            "YYYY-MM, or YYYY-MM-DD.".format(value)
        )

    year = int(match.group("year"))
    month_text = match.group("month")
    day_text = match.group("day")

    if month_text is None:
        return date(year, 1, 1) if bound == "start" else date(year, 12, 31)

    month = int(month_text)
    if day_text is None:
        if bound == "start":
            return date(year, month, 1)
        next_month = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
        return date.fromordinal(next_month.toordinal() - 1)

    return date(year, month, int(day_text))


def file_interval(path: str, frequency: str) -> Optional[Tuple[date, date]]:
    """Extract the covered date interval from an ERA5/ERA5-Land file name."""

    name = Path(path).name
    if frequency == "1hr":
        match = DAY_RE.search(name)
        if match is None:
            return None
        current = date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
        return current, current

    if frequency == "day":
        match = MONTH_RE.search(name)
        if match is None:
            return None
        year = int(match.group("year"))
        month = int(match.group("month"))
        start = date(year, month, 1)
        end = date(year + int(month == 12), 1 if month == 12 else month + 1, 1)
        return start, date.fromordinal(end.toordinal() - 1)

    if frequency == "mon":
        match = YEAR_RE.search(name)
        if match is None:
            return None
        year = int(match.group("year"))
        return date(year, 1, 1), date(year, 12, 31)

    return None


def overlaps_interval(
    path: str,
    frequency: str,
    start: Optional[date],
    end: Optional[date],
) -> bool:
    """Return whether a file's covered date range overlaps the requested interval."""

    current = file_interval(path, frequency)
    if current is None:
        return True
    file_start, file_end = current
    if start is not None and file_end < start:
        return False
    if end is not None and file_start > end:
        return False
    return True


def parse_level_type(level_type: str, mapper: Dict[str, Any]) -> Dict[str, str]:
    """Convert a CMOR ``level_type`` value into source path fields."""

    parts = level_type.split("_")
    if len(parts) < 2:
        raise ValueError(f"Expected level_type like 'sfc_fc_land', got {level_type!r}")

    level_mapping = mapper.get("level_type", {})
    stream_map = level_mapping.get("stream", {})
    type_map = level_mapping.get("type", {})
    fields = {
        "stream": str(stream_map.get(parts[0], parts[0])),
        "type": str(type_map.get(parts[1], parts[1])),
    }
    return fields


def resolve_records(
    *,
    var_table: Union[str, Path],
    cmor_tables_dir: Union[str, Path],
    mapper_path: Union[str, Path],
    dataset: str,
    variables: Optional[Tuple[str, ...]],
    frequencies: Tuple[str, ...],
    interval: Tuple[Optional[date], Optional[date]],
    root: Optional[str],
    glob_files: bool,
) -> List[SourceRecord]:
    """Resolve source records and matching files."""

    mapper = load_json(mapper_path)
    dataset_cfg = mapper["datasets"][dataset]
    table_prefix = str(dataset_cfg["table_prefix"])
    dataset_priority = tuple(str(item) for item in dataset_cfg["priority"])
    allowed_streams = set(str(item) for item in dataset_cfg.get("allowed_streams", ()))
    frequencies_by_stream = {
        str(stream): set(str(freq) for freq in frequencies)
        for stream, frequencies in dataset_cfg.get("frequencies_by_stream", {}).items()
    }
    if dataset == "era5":
        dataset_priority = tuple(code for code in dataset_priority if code != "EL")
    requests = selected_variables(
        load_variable_requests(var_table),
        allowed_codes=dataset_priority,
        variables=variables,
    )

    start, end = interval
    records: List[SourceRecord] = []
    entry_cache: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for frequency in frequencies:
        entry_cache[frequency] = load_cmor_variable_entries(
            cmor_tables_dir,
            table_prefix=table_prefix,
            frequency=frequency,
        )

    for request in requests:
        for frequency in frequencies:
            match = find_variable_entry(entry_cache[frequency], request.name)
            if match is None:
                continue
            table_variable, entry = match
            parameter = str(entry.get("DKRZ_ID") or entry.get("grib_paramID") or "")
            if not parameter:
                continue
            parameter = parameter.zfill(3)
            level_type = str(entry["level_type"])
            fields = parse_level_type(level_type, mapper)
            if allowed_streams and fields["stream"] not in allowed_streams:
                continue
            allowed_frequencies = frequencies_by_stream.get(fields["stream"])
            if allowed_frequencies is not None and frequency not in allowed_frequencies:
                continue
            fields["time_freq"] = str(mapper["frequency"][frequency])
            fields["parameter"] = parameter

            for dataset_code in dataset_priority:
                path_fields = dict(fields)
                path_fields["dataset"] = dataset_code
                pattern_template = str(mapper["source_path"])
                if root:
                    pattern_template = pattern_template.replace("/pool/data/ERA5", root.rstrip("/"))
                pattern = pattern_template.format(**path_fields)
                files = (
                    tuple(
                        file
                        for file in sorted(glob(pattern))
                        if overlaps_interval(file, frequency, start, end)
                    )
                    if glob_files
                    else ()
                )
                records.append(
                    SourceRecord(
                        variable=request.name,
                        table_variable=table_variable,
                        dataset=dataset,
                        dataset_code=dataset_code,
                        frequency=frequency,
                        stream=path_fields["stream"],
                        type=path_fields["type"],
                        parameter=parameter,
                        level_type=level_type,
                        pattern=pattern,
                        files=files,
                        conversion_factor=parse_conversion_factor(entry),
                        output_attrs=extract_output_attrs(entry),
                    )
                )
                if files:
                    break
    return records


def unresolved_records(
    requests: List[VariableRequest],
    frequencies: Tuple[str, ...],
    records: List[SourceRecord],
    reason: str,
) -> List[UnresolvedRecord]:
    """Return requested variable/frequency pairs with no source record."""

    resolved_keys = {(record.variable, record.frequency) for record in records}
    return [
        UnresolvedRecord(
            variable=request.name,
            frequency=frequency,
            reason=reason,
        )
        for request in requests
        for frequency in frequencies
        if (request.name, frequency) not in resolved_keys
    ]
