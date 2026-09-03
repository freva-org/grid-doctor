#! /usr/bin/env python3
"""Resolve ERA5 and ERA5-Land source GRIB files from the local CMOR tables."""

import ast
import csv
import json
import re
from collections.abc import Callable, Iterable
from datetime import date, datetime
from glob import glob
from pathlib import Path
from typing import Any, NamedTuple

from ..resources import ASSETS_DIR

DEFAULT_SOURCE_MAPPER = ASSETS_DIR / "source_mapper.json"

DAY_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})")
MONTH_RE = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})(?!-\d{2})")
YEAR_RE = re.compile(r"(?P<year>\d{4})(?!\d)")
DATE_VALUE_RE = re.compile(r"^(?P<year>\d{4})(?:-?(?P<month>\d{2})(?:-?(?P<day>\d{2}))?)?$")


def load_json(path: str | Path) -> dict[str, Any]:
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
    reanalysis: tuple[str, ...]
    commentary: str | None = None


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
    files: tuple[str, ...]
    conversion_factor: float
    output_attrs: dict[str, str]


class UnresolvedRecord(NamedTuple):
    """A requested variable/frequency pair that could not be resolved."""

    variable: str
    frequency: str
    reason: str


def _safe_eval_numeric_expression(expression: str) -> float:
    """Evaluate a simple numeric expression containing only literals and */+-."""

    binary_operators: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
    }
    unary_operators: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.USub: lambda value: -value,
        ast.UAdd: lambda value: value,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operators:
            return unary_operators[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
            return float(
                binary_operators[type(node.op)](
                    _eval(node.left),
                    _eval(node.right),
                )
            )
        raise ValueError(f"Unsupported numeric expression: {expression!r}")

    return _eval(ast.parse(expression, mode="eval"))


def parse_conversion_factor(entry: dict[str, Any]) -> float:
    """Return the numeric multiplicative conversion factor for one CMOR entry."""

    raw = str(entry.get("conversion", "")).strip()
    if not raw:
        return 1.0
    return _safe_eval_numeric_expression(raw)


def extract_output_attrs(entry: dict[str, Any]) -> dict[str, str]:
    """Extract relevant output metadata from one CMOR entry."""

    keys = tuple(SOURCE_MAPPER.get("var_attrs", []))
    attrs: dict[str, str] = {}
    for key in keys:
        value = str(entry.get(key, "")).strip()
        if value:
            attrs[key] = value
    return attrs


def split_csv_list(value: str) -> tuple[str, ...]:
    """Split a comma-separated table value, trimming whitespace."""

    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_variable_requests(path: str | Path) -> list[VariableRequest]:
    """Read the variable selection CSV.

    The table must provide ``varname`` and ``reanalysis`` columns. Any
    additional columns are treated as optional human-facing commentary only.
    """

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        rows = [
            VariableRequest(
                name=str(row["varname"]).strip(),
                reanalysis=split_csv_list(str(row["reanalysis"])),
                commentary=(str(row.get("commentary", "")).strip() or None),
            )
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No variables found in {path}")
    return rows


def dataset_code_allowed(
    allowed_codes: Iterable[str],
    codes: Iterable[str],
) -> bool:
    """Return whether a variable-table row permits the selected source codes."""

    allowed = set(codes)
    return bool(set(allowed_codes) & allowed)


def selected_variables(
    requests: list[VariableRequest],
    *,
    allowed_codes: Iterable[str],
    variables: tuple[str, ...] | None,
) -> list[VariableRequest]:
    """Filter CSV requests by allowed source codes and optional variable names."""

    requested_filter = variables is not None
    requested = set(variables or ())
    selected = [
        request
        for request in requests
        if dataset_code_allowed(allowed_codes, request.reanalysis)
        and (not requested_filter or request.name in requested)
    ]
    missing = sorted(requested - {request.name for request in selected}) if requested_filter else []
    if missing:
        raise KeyError("Requested variables are not available for the selected source: " + ", ".join(missing))
    return selected


def load_cmor_variable_entries(
    cmor_tables_dir: str | Path,
    *,
    table_prefix: str,
    frequency: str,
) -> dict[str, dict[str, Any]]:
    """Load variable entries for one CMOR table frequency."""

    table_path = Path(cmor_tables_dir) / f"{table_prefix}_{frequency}.json"
    table = load_json(table_path)
    entries = table.get("variable_entry")
    if not isinstance(entries, dict):
        raise TypeError(f"Missing JSON object 'variable_entry' in {table_path}")
    return {str(name): entry for name, entry in entries.items() if isinstance(entry, dict)}


def find_variable_entry(
    entries: dict[str, dict[str, Any]],
    variable: str,
) -> tuple[str, dict[str, Any]] | None:
    """Find a CMOR table entry by entry name or by ``out_name``."""

    direct = entries.get(variable)
    if direct is not None:
        return variable, direct
    for table_name, entry in entries.items():
        if str(entry.get("out_name", "")).strip() == variable:
            return table_name, entry
    return None


def parse_interval(value: str | None) -> tuple[date | None, date | None]:
    """Parse ``start,end`` where each value can be YYYY, YYYYMM, or YYYYMMDD."""

    if value in (None, ""):
        return None, None
    parts = value.split(",", maxsplit=1)
    start = parse_date_value(parts[0], bound="start") if parts[0].strip() else None
    if len(parts) == 1 or not parts[1].strip():
        end = datetime.now().astimezone().date()
    else:
        end = parse_date_value(parts[1], bound="end")
    if start is not None and end is not None and start > end:
        raise ValueError("--interval start date must be before or equal to end date")
    return start, end


def parse_date(value: str) -> date:
    """Parse a compact or dashed date."""

    value = value.strip()
    if "-" in value:
        return date.fromisoformat(value)
    return date(int(value[:4]), int(value[4:6]), int(value[6:]))


def parse_date_value(value: str, *, bound: str) -> date:
    """Parse YYYY, YYYYMM, or YYYYMMDD in compact or dashed form."""

    text = value.strip()
    match = DATE_VALUE_RE.match(text)
    if match is None:
        raise ValueError(f"Unsupported date value {value!r}; use YYYY, YYYYMM, YYYYMMDD, YYYY-MM, or YYYY-MM-DD.")

    year = int(match.group("year"))
    month_text = match.group("month")
    day_text = match.group("day")

    if month_text is None:
        return date(year, 1, 1) if bound == "start" else date(year, 12, 31)

    month = int(month_text)
    if day_text is None:
        if bound == "start":
            return date(year, month, 1)
        next_month = date(
            year + int(month == 12),
            1 if month == 12 else month + 1,
            1,
        )
        return date.fromordinal(next_month.toordinal() - 1)

    return date(year, month, int(day_text))


def file_interval(path: str, frequency: str) -> tuple[date, date] | None:
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
        next_month = date(
            year + int(month == 12),
            1 if month == 12 else month + 1,
            1,
        )
        return start, date.fromordinal(next_month.toordinal() - 1)

    if frequency == "mon":
        match = YEAR_RE.search(name)
        if match is None:
            return None
        year = int(match.group("year"))
        return date(year, 1, 1), date(year, 12, 31)

    return None


def files_interval(
    files: Iterable[str | Path],
    *,
    frequency: str,
) -> tuple[date, date] | None:
    """Return the inclusive interval covered by one ordered file collection."""

    intervals: list[tuple[date, date]] = []
    for file in files:
        current_interval = file_interval(str(file), frequency)
        if current_interval is None:
            return None
        intervals.append(current_interval)

    if not intervals:
        return None

    starts = [start for start, _ in intervals]
    ends = [end for _, end in intervals]
    return min(starts), max(ends)


def batched_source_record_files(
    record: SourceRecord,
    *,
    batch_files: int | None,
    fallback_interval: tuple[date | None, date | None],
) -> tuple[tuple[SourceRecord, tuple[date | None, date | None]], ...]:
    """Split one resolved source record into file-count batches."""

    if batch_files is None or not record.files:
        return ((record, fallback_interval),)
    if batch_files <= 0:
        raise ValueError("--batch-files must be a positive integer.")

    batches: list[tuple[SourceRecord, tuple[date | None, date | None]]] = []
    for index in range(0, len(record.files), batch_files):
        current_files = tuple(record.files[index : index + batch_files])
        current_interval = files_interval(current_files, frequency=record.frequency)
        batches.append(
            (
                record._replace(files=current_files),
                current_interval if current_interval is not None else fallback_interval,
            )
        )
    return tuple(batches)


def overlaps_interval(
    path: str,
    frequency: str,
    start: date | None,
    end: date | None,
) -> bool:
    """Return whether a file's covered date range overlaps the requested interval."""

    current = file_interval(path, frequency)
    if current is None:
        return True

    file_start, file_end = current
    if start is not None and file_end < start:
        return False
    return not (end is not None and file_start > end)


def parse_level_type(
    level_type: str,
    mapper: dict[str, Any],
) -> dict[str, str]:
    """Convert a CMOR ``level_type`` value into source path fields."""

    parts = level_type.split("_")
    if len(parts) < 2:
        raise ValueError(f"Expected level_type like 'sfc_fc_land', got {level_type!r}")

    level_mapping = mapper.get("level_type", {})
    stream_map = level_mapping.get("stream", {})
    type_map = level_mapping.get("type", {})
    return {
        "stream": str(stream_map.get(parts[0], parts[0])),
        "type": str(type_map.get(parts[1], parts[1])),
    }


def source_pattern_template(
    mapper: dict[str, Any],
    *,
    root: str | None,
) -> str:
    """Return the configured source template, optionally rooted elsewhere.

    The default path comes exclusively from ``source_mapper.json``. When
    ``root`` is provided, it replaces the configured prefix before the
    ``{dataset}`` placeholder without embedding any site-specific path here.
    """

    template = str(mapper["source_path"])
    if root is None:
        return template

    prefix, marker, suffix = template.partition("{dataset}")
    if not marker:
        raise ValueError("The configured source_path must contain a {dataset} placeholder when --root is used.")

    del prefix
    return f"{root.rstrip('/')}/{{dataset}}{suffix}"


def resolve_priority_files(
    *,
    mapper: dict[str, Any],
    dataset_priority: tuple[str, ...],
    fields: dict[str, str],
    frequency: str,
    start: date | None,
    end: date | None,
    root: str | None,
    glob_files: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Resolve files by priority for each covered period.

    Earlier dataset codes have higher priority. Lower-priority collections fill
    only periods that are absent from all higher-priority collections.
    """

    template = source_pattern_template(mapper, root=root)
    patterns: list[str] = []

    if not glob_files:
        for dataset_code in dataset_priority:
            path_fields = dict(fields)
            path_fields["dataset"] = dataset_code
            patterns.append(template.format(**path_fields))
        return (), (), tuple(patterns)

    selected_by_interval: dict[tuple[date, date], str] = {}
    selected_codes: list[str] = []
    undated_files: tuple[str, ...] | None = None

    for dataset_code in dataset_priority:
        path_fields = dict(fields)
        path_fields["dataset"] = dataset_code
        pattern = template.format(**path_fields)
        patterns.append(pattern)

        candidates = tuple(path for path in sorted(glob(pattern)) if overlaps_interval(path, frequency, start, end))
        if not candidates:
            continue

        dated_candidates: list[tuple[tuple[date, date], str]] = []
        current_undated: list[str] = []

        for path in candidates:
            coverage = file_interval(path, frequency)
            if coverage is None:
                current_undated.append(path)
            else:
                dated_candidates.append((coverage, path))

        used_code = False

        for coverage, path in dated_candidates:
            if coverage not in selected_by_interval:
                selected_by_interval[coverage] = path
                used_code = True

        # Static/fixed fields have no temporal coverage. For those, select the
        # first priority collection containing matching files.
        if current_undated and undated_files is None:
            undated_files = tuple(current_undated)
            used_code = True

        if used_code:
            selected_codes.append(dataset_code)

    if selected_by_interval:
        files = tuple(
            path
            for _, path in sorted(
                selected_by_interval.items(),
                key=lambda item: (item[0][0], item[0][1], item[1]),
            )
        )
    else:
        files = undated_files or ()

    return files, tuple(selected_codes), tuple(patterns)


def resolve_records(
    *,
    var_table: str | Path,
    cmor_tables_dir: str | Path,
    dataset: str,
    variables: tuple[str, ...] | None,
    frequencies: tuple[str, ...],
    interval: tuple[date | None, date | None],
    root: str | None,
    glob_files: bool,
) -> list[SourceRecord]:
    """Resolve source records and matching files."""

    mapper = SOURCE_MAPPER
    dataset_cfg = mapper["datasets"][dataset]
    table_prefix = str(dataset_cfg["table_prefix"])
    dataset_priority = tuple(str(item) for item in dataset_cfg["priority"])
    allowed_streams = {str(item) for item in dataset_cfg.get("allowed_streams", ())}
    frequencies_by_stream = {
        str(stream): {str(freq) for freq in frequencies}
        for stream, frequencies in dataset_cfg.get(
            "frequencies_by_stream",
            {},
        ).items()
    }

    if dataset == "era5":
        dataset_priority = tuple(code for code in dataset_priority if code != "EL")

    requests = selected_variables(
        load_variable_requests(var_table),
        allowed_codes=dataset_priority,
        variables=variables,
    )
    start, end = interval
    records: list[SourceRecord] = []
    entry_cache: dict[str, dict[str, dict[str, Any]]] = {}

    for frequency in frequencies:
        entry_cache[frequency] = load_cmor_variable_entries(
            cmor_tables_dir,
            table_prefix=table_prefix,
            frequency=frequency,
        )

    for request in requests:
        for frequency in frequencies:
            match = find_variable_entry(
                entry_cache[frequency],
                request.name,
            )
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

            files, selected_codes, patterns = resolve_priority_files(
                mapper=mapper,
                dataset_priority=dataset_priority,
                fields=fields,
                frequency=frequency,
                start=start,
                end=end,
                root=root,
                glob_files=glob_files,
            )

            # Keep one SourceRecord per variable/frequency. Its file list may
            # combine multiple archive collections according to priority.
            if selected_codes:
                dataset_code = "+".join(selected_codes)
            else:
                dataset_code = dataset_priority[0]

            records.append(
                SourceRecord(
                    variable=request.name,
                    table_variable=table_variable,
                    dataset=dataset,
                    dataset_code=dataset_code,
                    frequency=frequency,
                    stream=fields["stream"],
                    type=fields["type"],
                    parameter=parameter,
                    level_type=level_type,
                    pattern=";".join(patterns),
                    files=files,
                    conversion_factor=parse_conversion_factor(entry),
                    output_attrs=extract_output_attrs(entry),
                )
            )

    return records


def unresolved_records(
    requests: list[VariableRequest],
    frequencies: tuple[str, ...],
    records: list[SourceRecord],
    reason: str,
) -> list[UnresolvedRecord]:
    """Return requested variable/frequency pairs with no source record."""

    resolved_keys = {(record.variable, record.frequency) for record in records if record.files}
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
