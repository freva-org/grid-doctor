#!/usr/bin/env python
"""Planning and download helpers for the Reflow-based ICON-DREAM pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

import grid_doctor as gd
import grid_doctor.cli as gd_cli

from .common import (DATE_TOKEN_RE, DEFAULT_GRID_URL, DEFAULT_INVARIANT_URL,
                     DEFAULT_SOURCE_ROOT, ICON_DREAM_VARIABLES, TIME_FREQUENCY,
                     UTC, HrefParser, build_paths, download_one, isoformat_utc,
                     load_existing_target_info, load_plan, open_grid_dataset,
                     parse_datetime, read_json_text, save_plan, target_root)


class IconDreamSource:
    """Discover source files on the DWD open data server."""

    def __init__(
        self,
        variables: Sequence[str],
        frequency: TIME_FREQUENCY,
        requested_time: tuple[str, str],
        source_root: str = DEFAULT_SOURCE_ROOT,
        grid_url: str = DEFAULT_GRID_URL,
        invariant_url: str = DEFAULT_INVARIANT_URL,
    ) -> None:
        self.variables = list(variables)
        self.frequency = frequency
        self.requested_time = requested_time
        self.source_root = source_root.rstrip("/")
        self.grid_url = grid_url
        self.invariant_url = invariant_url
        self.start_time = parse_datetime(requested_time[0])
        self.end_time = parse_datetime(requested_time[1])

    def _directory_url(self, variable: str) -> str:
        return (
            f"{self.source_root}/{self.frequency}/{variable.upper().replace('-', '_')}"
        )

    def _period_from_token(self, token: str) -> tuple[datetime, datetime]:
        if len(token) == 6:
            start = datetime.strptime(token, "%Y%m").replace(tzinfo=UTC)
            end = (
                start.replace(year=start.year + 1, month=1)
                if start.month == 12
                else start.replace(month=start.month + 1)
            )
            return start, end
        if len(token) == 8:
            start = datetime.strptime(token, "%Y%m%d").replace(tzinfo=UTC)
            return start, start + timedelta(days=1)
        raise ValueError(f"Unsupported date token: {token!r}")

    def _extract_token(self, href: str) -> str | None:
        match = DATE_TOKEN_RE.search(href)
        return None if match is None else match.group(1)

    def _should_keep(
        self, token: str | None, existing_max_time: datetime | None
    ) -> bool:
        if token is None:
            return True
        period_start, period_end = self._period_from_token(token)
        if period_end <= self.start_time or period_start > self.end_time:
            return False
        if existing_max_time is not None and period_end <= existing_max_time:
            return False
        return True

    def _list_variable_urls(
        self,
        variable: str,
        existing_max_time: datetime | None,
    ) -> list[dict[str, Any]]:
        parser = HrefParser(suffix=".grb")
        url = self._directory_url(variable)
        with gd_cli.AutoRaiseSession() as session:
            response = session.get(url, timeout=30)
            parser.feed(response.text)

        items: list[dict[str, Any]] = []
        for href in sorted(set(parser.hrefs)):
            token = self._extract_token(href)
            if not self._should_keep(token, existing_max_time):
                continue
            period_start: str | None = None
            period_end: str | None = None
            if token is not None:
                start, end = self._period_from_token(token)
                period_start = isoformat_utc(start)
                period_end = isoformat_utc(end)
            filename = Path(href).name
            items.append(
                {
                    "item_index": -1,
                    "variable": variable,
                    "frequency": self.frequency,
                    "url": f"{url}/{href}",
                    "filename": filename,
                    "relative_path": str(Path(variable) / filename),
                    "date_token": token,
                    "period_start": period_start,
                    "period_end": period_end,
                }
            )
        return items

    def list_items(
        self,
        *,
        existing_max_time: datetime | None = None,
        existing_variables: set[str] | None = None,
        update_only: bool = True,
    ) -> list[dict[str, Any]]:
        """List source files that still need processing."""
        if self.frequency == "fx":
            filename = Path(self.invariant_url).name
            return [
                {
                    "item_index": 0,
                    "variable": "fx",
                    "frequency": self.frequency,
                    "url": self.invariant_url,
                    "filename": filename,
                    "relative_path": str(Path("fx") / filename),
                    "date_token": None,
                    "period_start": None,
                    "period_end": None,
                }
            ]

        existing_variables = existing_variables or set()
        items: list[dict[str, Any]] = []
        for variable in self.variables:
            variable_max = (
                existing_max_time
                if update_only and variable in existing_variables
                else None
            )
            items.extend(self._list_variable_urls(variable, variable_max))

        for idx, item in enumerate(
            sorted(items, key=lambda item: (item["variable"], item["url"]))
        ):
            item["item_index"] = idx
        return items


def build_plan(
    *,
    s3_bucket: str,
    start: str,
    end: str,
    variables: Sequence[str],
    freq: TIME_FREQUENCY,
    source_root: str,
    s3_endpoint: str,
    s3_credentials_file: str,
    source_engine: str,
    source_backend_kwargs_json: str,
    update_only: bool,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Discover source files and create the persisted run plan."""
    if not variables:
        raise ValueError("At least one variable is required.")
    invalid = sorted(set(variables) - set(ICON_DREAM_VARIABLES))
    if invalid:
        raise ValueError(f"Unsupported variables: {invalid}")

    paths = build_paths(run_dir)
    s3_options = gd.get_s3_options(s3_endpoint, s3_credentials_file)
    existing = load_existing_target_info(target_root(s3_bucket, freq), s3_options)
    existing_max = (
        parse_datetime(existing["max_time"]) if existing["max_time"] else None
    )
    items = IconDreamSource(
        variables=variables,
        frequency=freq,
        requested_time=(start, end),
        source_root=source_root,
    ).list_items(
        existing_max_time=existing_max,
        existing_variables=set(existing["variables"]),
        update_only=update_only,
    )
    plan = {
        "run_dir": str(paths["run_dir"]),
        "target_root": target_root(s3_bucket, freq),
        "frequency": freq,
        "variables": list(variables),
        "requested_time": [start, end],
        "created_at": isoformat_utc(datetime.now(tz=UTC)),
        "source_root": source_root,
        "grid_url": DEFAULT_GRID_URL,
        "invariant_url": DEFAULT_INVARIANT_URL,
        "grid_path": str(paths["grid_path"]),
        "weights_path": str(paths["weights_path"]),
        "temp_root": str(paths["temp_root"]),
        "raw_root": str(paths["raw_root"]),
        "source_items": items,
        "source_engine": source_engine,
        "source_backend_kwargs": read_json_text(source_backend_kwargs_json),
        "existing_target": existing,
        "max_level": None,
    }
    save_plan(plan, paths["plan_path"])
    return plan


def prepare_shared_assets(
    *,
    max_level: int | None,
    download_timeout: int,
    download_chunk_size: int,
    overwrite_downloads: bool,
    run_dir: str | Path,
) -> str:
    """Download the grid once and cache the HEALPix weights."""
    plan = load_plan(run_dir)
    paths = build_paths(run_dir)
    downloaded_grid = gd_cli.download_file(
        plan["grid_url"],
        paths["grid_path"].parent,
        timeout=download_timeout,
        overwrite=overwrite_downloads,
        chunk_size=download_chunk_size,
    )
    grid_ds = open_grid_dataset(downloaded_grid)
    resolved_level = (
        gd.resolution_to_healpix_level(gd.get_latlon_resolution(grid_ds))
        if max_level is None
        else int(max_level)
    )
    weight_file = gd.cached_weights(
        grid_ds, level=resolved_level, cache_path=paths["weights_path"]
    )
    plan.update(
        grid_path=str(downloaded_grid),
        weights_path=str(weight_file),
        max_level=resolved_level,
    )
    save_plan(plan, paths["plan_path"])
    return str(weight_file)


def download_source_item(
    source_item: dict[str, Any],
    *,
    download_timeout: int,
    download_chunk_size: int,
    overwrite_downloads: bool,
    run_dir: str | Path,
) -> dict[str, Any]:
    """Download one raw source file."""
    plan = load_plan(run_dir)
    local_path = download_one(
        source_item["url"],
        Path(plan["raw_root"]) / source_item["relative_path"],
        timeout=download_timeout,
        overwrite=overwrite_downloads,
        chunk_size=download_chunk_size,
    )
    return {**source_item, "local_path": local_path}
