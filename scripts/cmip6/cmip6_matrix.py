#!/usr/bin/env python
"""Facet-matrix search for the Freva databrowser REST API.

Given a set of requested facets, this builds one output dataset per
``(experiment, time_frequency, model)`` combination such that *every*
requested variable is present, choosing a single ensemble member per
combination: the default if it carries all variables, otherwise the
first ensemble that does.

The realm is deliberately *not* constrained, so ocean variables (e.g.
``tos``) and atmosphere variables (e.g. ``tas``, ``pr``, ``uas``) can be
requested together.

This module also provides grid helpers used when regridding mixed-realm
datasets: ``grid_group_key`` / ``group_key_str`` group a dataset's files
by source grid (so one ESMF weight file is built per grid), and
``normalize_for_weights`` renames CMIP6 unstructured cell-corner
coordinates to the names grid_doctor expects.

REST API reference:
https://freva-nextgen.readthedocs.io/en/latest/developers/databrowser.html
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import requests

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE = "https://nextgems.dkrz.de/api/freva-nextgen/databrowser"
DEFAULT_ENSEMBLE = "r1i1p1f1"

# Query parameters that steer the endpoint rather than filter the search.
_CONTROL_PARAMS = frozenset({"facets", "multi-version", "translate", "start"})


CMOR_MAP = {
    "fx": None,
    "subhr": "PT1M",
    "subdaily": "PT1M",
    "30min": "PT30M",
    "15min": "PT15M",
    "1hr": "PT1H",
    "hour": "PT1H",
    "hr": "PT1H",
    "3hr": "PT3H",
    "6hr": "PT6H",
    "12hr": "PT12H",
    "day": "P1D",
    "1d": "P1D",
    "daily": "P1D",
    "mon": "P1M",
    "monthly": "P1M",
    "season": "P3M",
    "seasonal": "P3M",
    "yr": "P1Y",
    "year": "P1Y",
    "annual": "P1Y",
    "monClim": "P1M",
    "dayClim": "P1D",
}


def cmor_to_iso8601(freq: str) -> str | None:
    """Map a CMOR-style frequency token to an ISO-8601 duration."""
    f = str(freq).strip()
    if not f:
        return None
    key = f.lower()
    if key in CMOR_MAP:
        return CMOR_MAP[key]
    m = re.match(r"^(\d+)\s*([a-zA-Z]+)$", key)
    if m:
        n, u = m.groups()
        if u in ("min", "minute", "m", "minutes"):
            return f"PT{n}M"
        if u in ("s", "sec", "second", "seconds"):
            return f"PT{n}S"
        if u in ("h", "hr", "hour", "hours"):
            return f"PT{n}H"
        if u in ("d", "day", "days"):
            return f"P{n}D"
        if u in ("mo", "mon", "month", "months"):
            return f"P{n}M"
        if u in ("y", "yr", "year", "years"):
            return f"P{n}Y"
    if key.startswith("p") or key.startswith("pt"):
        return freq
    if "mon" in key:
        return "P1M"
    if "day" in key:
        return "P1D"
    if "hr" in key or "hour" in key:
        return "PT1H"
    return None


class DatabrowserClient:
    """Thin wrapper around the public Freva databrowser REST endpoints.

    The ``session`` is injectable so the client can be unit-tested
    without network access.
    """

    def __init__(
        self,
        instance: str = DEFAULT_INSTANCE,
        *,
        flavour: str = "freva",
        uniq_key: str = "file",
        timeout: int = 120,
        multi_version: bool = False,
        session: requests.Session | None = None,
    ) -> None:
        self.instance = instance.rstrip("/")
        self.flavour = flavour
        self.uniq_key = uniq_key
        self.timeout = timeout
        self.multi_version = multi_version
        self.session = session or requests.Session()

    def _url(self, kind: str) -> str:
        return f"{self.instance}/{kind}/{self.flavour}/{self.uniq_key}"

    def metadata_search(
        self, *, facets: Sequence[str] | None = None, **search: object
    ) -> dict:
        """Return the JSON metadata-search response (facets + counts)."""
        params: dict[str, object] = dict(search)
        params["multi-version"] = str(self.multi_version).lower()
        if facets is not None:
            params["facets"] = list(facets)
        resp = self.session.get(
            self._url("metadata-search"), params=params, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()

    def facet_values(self, facet: str, **search: object) -> list[str]:
        """Return facet values with a positive count.

        The API encodes facets as a flat ``[value, count, value, count,
        ...]`` list where the counts are *strings*; both are handled
        here. Order from the server is preserved.
        """
        data = self.metadata_search(facets=[facet], **search)
        raw = data.get("facets", {}).get(facet) or []
        values: list[str] = []
        for value, count in zip(raw[::2], raw[1::2]):
            try:
                positive = int(count) > 0
            except (TypeError, ValueError):
                positive = False
            if positive:
                values.append(value)
        return values

    def data_search(self, **search: object) -> list[str]:
        """Return a sorted list of file paths / URIs matching the search."""
        params: dict[str, object] = dict(search)
        params["multi-version"] = str(self.multi_version).lower()
        resp = self.session.get(
            self._url("data-search"),
            params=params,
            timeout=self.timeout,
            stream=True,
        )
        resp.raise_for_status()
        return sorted(
            line for line in resp.iter_lines(decode_unicode=True) if line
        )


@dataclass(frozen=True)
class DatasetEntry:
    """One complete output dataset: all variables, one chosen ensemble."""

    key: str
    experiment: str
    frequency: str
    model: str
    ensemble: str
    files_by_variable: dict[str, list[str]]

    @property
    def files(self) -> list[str]:
        """All variable files, grouped by variable in request order."""
        out: list[str] = []
        for variable in self.files_by_variable:
            out.extend(self.files_by_variable[variable])
        return out


@dataclass
class FacetMatrix:
    """Specification + builder for a facet-matrix dataset search.

    Parameters
    ----------
    variables:
        Variables that every output dataset must contain.
    experiments, frequencies:
        Each ``(experiment, frequency, model)`` triple becomes its own
        output dataset.
    models:
        Explicit model list. When given, only these models are
        considered (and still validated to carry all variables).
    max_models:
        When ``models`` is not given, cap discovery to the first N
        models (sorted). ``None`` means all discovered models.
    default_ensemble:
        Preferred ensemble member; falls back to the first member that
        carries all variables.
    extra_facets:
        Additional fixed search facets (e.g. ``project``/``product``).
        ``realm`` is intentionally left out by default.
    """

    variables: Sequence[str]
    experiments: Sequence[str]
    frequencies: Sequence[str] = ("mon",)
    models: Sequence[str] | None = None
    max_models: int | None = None
    default_ensemble: str = DEFAULT_ENSEMBLE
    extra_facets: Mapping[str, object] = field(default_factory=dict)
    path_template: str = "healpix/cmip6/{experiment}-{ensemble}/{model}/{frequency}"

    def _discover_models(
        self, client: DatabrowserClient, experiment: str, frequency: str
    ) -> list[str]:
        """Models that carry *every* variable (in some ensemble)."""
        per_variable: list[set[str]] = []
        for variable in self.variables:
            per_variable.append(
                set(
                    client.facet_values(
                        "model",
                        experiment=experiment,
                        time_frequency=frequency,
                        variable=variable,
                        **self.extra_facets,
                    )
                )
            )
        if not per_variable:
            return []
        return sorted(set.intersection(*per_variable))

    def _resolve_ensemble(
        self,
        client: DatabrowserClient,
        model: str,
        experiment: str,
        frequency: str,
    ) -> str | None:
        """Pick one ensemble carrying all variables (default preferred)."""
        per_variable: list[set[str]] = []
        for variable in self.variables:
            members = set(
                client.facet_values(
                    "ensemble",
                    model=model,
                    experiment=experiment,
                    time_frequency=frequency,
                    variable=variable,
                    **self.extra_facets,
                )
            )
            if not members:
                return None  # variable absent for this model/exp/freq
            per_variable.append(members)
        common = set.intersection(*per_variable)
        if not common:
            return None
        if self.default_ensemble in common:
            return self.default_ensemble
        return sorted(common)[0]

    def build(self, client: DatabrowserClient) -> list[DatasetEntry]:
        """Resolve the full matrix into complete dataset entries."""
        entries: list[DatasetEntry] = []
        for experiment in self.experiments:
            for frequency in self.frequencies:
                if self.models is not None:
                    models = list(self.models)
                else:
                    models = self._discover_models(client, experiment, frequency)
                    if self.max_models is not None:
                        models = models[: self.max_models]

                for model in models:
                    ensemble = self._resolve_ensemble(
                        client, model, experiment, frequency
                    )
                    if ensemble is None:
                        logger.warning(
                            "skip model=%s experiment=%s frequency=%s: "
                            "no single ensemble carries all variables %s",
                            model,
                            experiment,
                            frequency,
                            list(self.variables),
                        )
                        continue

                    files_by_variable: dict[str, list[str]] = {}
                    complete = True
                    for variable in self.variables:
                        files = client.data_search(
                            model=model,
                            experiment=experiment,
                            time_frequency=frequency,
                            ensemble=ensemble,
                            variable=variable,
                            **self.extra_facets,
                        )
                        if not files:
                            complete = False
                            break
                        files_by_variable[variable] = files
                    if not complete:
                        logger.warning(
                            "skip model=%s experiment=%s frequency=%s "
                            "ensemble=%s: data-search returned no files for %s",
                            model,
                            experiment,
                            frequency,
                            ensemble,
                            variable,
                        )
                        continue

                    key = self.path_template.format(
                        experiment=experiment,
                        ensemble=ensemble,
                        model=model,
                        frequency=cmor_to_iso8601(frequency) or frequency,
                    )
                    entries.append(
                        DatasetEntry(
                            key=key,
                            experiment=experiment,
                            frequency=frequency,
                            model=model,
                            ensemble=ensemble,
                            files_by_variable=files_by_variable,
                        )
                    )
        return entries

    def as_source_pairs(
        self, client: DatabrowserClient
    ) -> list[tuple[str, list[str]]]:
        """``[(output_key, [files...]), ...]`` for the regrid pipeline."""
        return [(entry.key, entry.files) for entry in self.build(client)]


# ===========================================================================
# Grid helpers for regridding mixed-realm CMIP6 datasets to HEALPix
# ===========================================================================
#
# A single output dataset can mix variables that live on *different* source
# grids (e.g. ``tos`` on an ocean grid, ``tas``/``pr``/``uas`` on an
# atmosphere grid). ESMF weights are grid-specific, so weights must be
# generated per grid and matched to each file at regrid time.

# CMIP6 unstructured cell-corner coordinate names, in priority order, each a
# 2-D array of shape (n_cell, n_vertex). Mapped to the grid_doctor names.
_VERTEX_ALIASES: tuple[tuple[str, str, str], ...] = (
    ("vertices_longitude", "vertices_latitude", "cmip6"),
    ("bounds_lon", "bounds_lat", "bounds"),
    ("lon_bnds", "lat_bnds", "bnds"),
    ("longitude_bnds", "latitude_bnds", "bnds_long"),
)

_GRID_DOCTOR_LON_VERTEX = "clon_vertices"
_GRID_DOCTOR_LAT_VERTEX = "clat_vertices"

# Matches CMIP6 grid_label DRS tokens: gn, gr, gr1, gr2, gm, ...
_GRID_LABEL_RE = re.compile(r"^g[a-z]\w*$")


def variable_of(path: str | Path) -> str | None:
    """Return the CMIP6 variable id from a file path.

    CMIP6 filenames are ``<variable_id>_<table_id>_<source_id>_...nc``.
    """
    name = Path(path).name
    if "_" in name:
        token = name.split("_", 1)[0]
        if token:
            return token
    return None


def grid_label_of(path: str | Path) -> str | None:
    """Return the CMIP6 grid_label from a file path.

    DRS layout: ``.../<table>/<variable>/<grid_label>/<version>/<file>``,
    so the grid_label is the third path component from the end.
    """
    parts = Path(path).parts
    if len(parts) >= 3:
        candidate = parts[-3]
        if _GRID_LABEL_RE.match(candidate):
            return candidate
    return None


def grid_group_key(path: str | Path) -> tuple[str, str]:
    """Return a key for files sharing one source grid.

    Uses ``(variable, grid_label)``. When the path cannot be parsed it
    falls back to the file path itself, which yields one weight file per
    file -- still correct, just less sharing (grid_doctor's weight cache
    deduplicates identical grids across keys anyway).
    """
    variable = variable_of(path)
    grid_label = grid_label_of(path)
    if variable is not None and grid_label is not None:
        return (variable, grid_label)
    return ("__file__", str(path))


def group_key_str(path: str | Path) -> str:
    """Stable string form of :func:`grid_group_key` for dict storage."""
    return "::".join(grid_group_key(path))


def _clean_vertex_values(values: "np.ndarray", fill: float | None) -> "np.ndarray":
    """Return float64 vertex array with fills / absurd values set to NaN."""
    data = np.asarray(values, dtype=np.float64)
    if fill is not None and np.isfinite(fill):
        data = np.where(data == fill, np.nan, data)
    # Guard against padding sentinels (e.g. 1e20) in ragged polygons.
    return np.where(np.abs(data) > 720.0, np.nan, data)


def _fill_value(var) -> float | None:
    for source in (var.attrs, getattr(var, "encoding", {})):
        for name in ("_FillValue", "missing_value"):
            if name in source:
                try:
                    return float(source[name])
                except (TypeError, ValueError):
                    return None
    return None


def normalize_for_weights(ds):
    """Return a dataset whose grid coords grid_doctor can read.

    Only unstructured (1-D cell) CMIP6 corner coordinates are remapped;
    curvilinear and regular grids are returned unchanged. The input is
    not mutated.
    """
    if _GRID_DOCTOR_LON_VERTEX in ds or "lon_vertices" in ds:
        return ds  # already in a recognised form

    lon_src = lat_src = None
    for lon_name, lat_name, _tag in _VERTEX_ALIASES:
        if lon_name in ds and lat_name in ds:
            # Unstructured corners are 2-D: (n_cell, n_vertex).
            if ds[lon_name].ndim == 2 and ds[lat_name].ndim == 2:
                lon_src, lat_src = lon_name, lat_name
                break
    if lon_src is None:
        return ds  # nothing to do (curvilinear/regular handled elsewhere)

    out = ds.copy()
    lon_dims = out[lon_src].dims
    lat_dims = out[lat_src].dims
    lon_vals = _clean_vertex_values(out[lon_src].values, _fill_value(out[lon_src]))
    lat_vals = _clean_vertex_values(out[lat_src].values, _fill_value(out[lat_src]))
    out = out.drop_vars([lon_src, lat_src], errors="ignore")
    out[_GRID_DOCTOR_LON_VERTEX] = (lon_dims, lon_vals)
    out[_GRID_DOCTOR_LAT_VERTEX] = (lat_dims, lat_vals)
    return out


def pick_target_level(native_levels: dict, override: int | None) -> int:
    """Choose one HEALPix level for a whole (multi-grid) dataset.

    A dataset is published as one pyramid, so every variable must target
    the same level. ``override > 0`` forces a fixed level (recommended
    for cross-model comparability); otherwise the finest native level
    among the dataset's grids is used so no variable is downsampled.
    """
    if override:
        return int(override)
    if not native_levels:
        raise ValueError("no native levels to choose a target from")
    return max(int(v) for v in native_levels.values())


def build_group_weights(
    source_paths,
    *,
    level: int,
    open_dataset,
    resolution_level,
    make_weights,
    normalize=normalize_for_weights,
):
    """Build one weight file per source grid for a single dataset.

    Pure orchestration (the grid_doctor / xarray calls are injected) so it
    is unit-testable and so a failure for one dataset can be caught by the
    caller and that dataset skipped without aborting the whole run.

    Parameters
    ----------
    source_paths:
        All files of one output dataset (mixed variables/grids).
    level:
        Target HEALPix level, or 0 for auto (finest native grid).
    open_dataset(path) -> ds:
        Opens a representative file per grid.
    resolution_level(ds) -> int:
        Native HEALPix level for a grid.
    make_weights(ds, target_level) -> path:
        Generates (or fetches cached) weights for a grid at *target_level*.
    normalize(ds) -> ds:
        Grid-coordinate normalization applied before weight generation.

    Returns
    -------
    (target_level, native_levels, group_weights)
        ``native_levels`` and ``group_weights`` are keyed by
        :func:`group_key_str`.
    """
    groups: dict[str, list[str]] = {}
    for src in sorted(source_paths):
        groups.setdefault(group_key_str(src), []).append(src)

    representatives = {key: files[0] for key, files in groups.items()}
    rep_datasets = {key: open_dataset(rep) for key, rep in representatives.items()}
    native_levels = {
        key: int(resolution_level(ds)) for key, ds in rep_datasets.items()
    }
    target_level = pick_target_level(native_levels, level)
    group_weights = {
        key: str(make_weights(normalize(ds), target_level))
        for key, ds in rep_datasets.items()
    }
    return target_level, native_levels, group_weights
