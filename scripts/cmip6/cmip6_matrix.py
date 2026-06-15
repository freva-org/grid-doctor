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

REST API reference:
https://freva-nextgen.readthedocs.io/en/latest/developers/databrowser.html
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

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
