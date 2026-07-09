#!/usr/bin/env python
"""Facet-matrix search for CORDEX-protocol data in the Freva databrowser.

Differences to the CMIP6 matrix:

- ``project`` and ``product`` are ordinary, flexible facets (NUKLEUS
  follows the CORDEX protocol but is its own project).
- The matrix has a **driving-model** dimension: one RCM driven by two
  GCMs is two distinct output datasets and must never be mixed.
- Variables are **discovered, not required**: each dataset carries all
  variables available for its combination, instead of being restricted
  to a fixed request list.
- Grid grouping is by the domain token of the CORDEX filename
  (``<variable>_<domain>_<driving>_...``): all variables of one domain
  share the rotated-pole grid, so one weight file serves the dataset.

The databrowser client is shared with the CMIP6 script (imported from
the sibling folder) so the REST handling lives in exactly one place.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cmip6"))
from cmip6_matrix import (  # noqa: E402
    DatabrowserClient,
    cmor_to_iso8601,
)

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE = "https://nextgems.dkrz.de/api/freva-nextgen/databrowser"
DEFAULT_ENSEMBLE = "r1i1p1"  # CORDEX-style member naming


# ---------------------------------------------------------------------------
# Dataset matrix
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CordexEntry:
    """One output dataset: one (experiment, freq, model, driver, member)."""

    key: str
    experiment: str
    frequency: str
    model: str
    ensemble: str
    files_by_variable: dict[str, list[str]]

    @property
    def files(self) -> list[str]:
        out: list[str] = []
        for variable in self.files_by_variable:
            out.extend(self.files_by_variable[variable])
        return out


@dataclass
class CordexMatrix:
    """Specification + builder for a CORDEX-style dataset search.

    Empty sequences mean "discover from the databrowser": experiments,
    models, driving models, and variables are all expanded from facet
    queries constrained by ``project``/``product``.  ``variables``
    given explicitly restricts the discovery instead.
    """

    project: str = "nukleus"
    product: str | None = None
    experiments: Sequence[str] = ()
    frequencies: Sequence[str] = ("1day",)
    models: Sequence[str] = ()
    driving_models: Sequence[str] = ()
    variables: Sequence[str] = ()
    exclude_variables: Sequence[str] = ()
    max_models: int | None = None
    default_ensemble: str = DEFAULT_ENSEMBLE
    extra_facets: Mapping[str, object] = field(default_factory=dict)
    path_template: str = (
        "healpix/{project}/{experiment}-{ensemble}/{driving_model}/{model}/{frequency}"
    )

    def _facets(self, **extra: object) -> dict[str, object]:
        base: dict[str, object] = {"project": self.project, **self.extra_facets}
        if self.product:
            base["product"] = self.product
        base.update({k: v for k, v in extra.items() if v is not None})
        return base

    def _values(
        self, client: DatabrowserClient, facet: str, **search: object
    ) -> list[str]:
        return client.facet_values(facet, **self._facets(**search))

    def build(self, client: DatabrowserClient) -> list[CordexEntry]:
        entries: list[CordexEntry] = []
        experiments = list(self.experiments or []) or self._values(client, "experiment")
        for experiment in experiments:
            for frequency in self.frequencies:
                combo = {"experiment": experiment, "time_frequency": frequency}
                if self.driving_models:
                    combo["driving_model"] = self.driving_models
                models = list(self.models or []) or self._values(
                    client, "model", **combo
                )
                if self.max_models is not None:
                    models = models[: self.max_models]
                for model in models:
                    driving_model = self._values(
                        client, "driving_model", model=model, **combo
                    )
                    driving_model = driving_model or ["self-driven"]
                    entry = self._build_entry(
                        client, experiment, frequency, model, driving_model[0]
                    )
                    if entry is not None:
                        entries.append(entry)
        return entries

    def _build_entry(
        self,
        client: DatabrowserClient,
        experiment: str,
        frequency: str,
        model: str,
        driving_model: str,
    ) -> CordexEntry | None:
        combo: dict[str, object] = {
            "experiment": experiment,
            "time_frequency": frequency,
            "model": model,
        }
        ensembles = self._values(client, "ensemble", **combo)
        if not ensembles:
            logger.warning("skip %s: no ensemble members", combo)
            return None
        ensemble = (
            self.default_ensemble
            if self.default_ensemble in ensembles
            else sorted(ensembles)[0]
        )

        variables = list(self.variables or []) or self._values(
            client, "variable", ensemble=ensemble, **combo
        )
        variables = [v for v in variables if v not in set(self.exclude_variables)]
        if not variables:
            logger.warning("skip %s: no variables", combo)
            return None

        files_by_variable: dict[str, list[str]] = {}
        for variable in sorted(variables):
            files = client.data_search(
                ensemble=ensemble, variable=variable, **self._facets(**combo)
            )
            if files:
                files_by_variable[variable] = files
        if not files_by_variable:
            logger.warning("skip %s: data-search returned no files", combo)
            return None

        key = self.path_template.format(
            project=self.project,
            product=self.product or "",
            experiment=experiment,
            ensemble=ensemble,
            model=model,
            driving_model=driving_model,
            frequency=cmor_to_iso8601(frequency) or frequency,
        )
        return CordexEntry(
            key=key,
            experiment=experiment,
            frequency=frequency,
            model=model,
            ensemble=ensemble,
            files_by_variable=files_by_variable,
        )


# ---------------------------------------------------------------------------
# Grid grouping
# ---------------------------------------------------------------------------
def cordex_group_key(path: str | Path) -> str:
    """Group files sharing one source grid.

    CORDEX-protocol filenames are ``<variable>_<domain>_<driving>_...``;
    all variables of one domain share the rotated-pole grid, so the
    domain token is the group.  Unparsable names fall back to a single
    shared group — grid_doctor's weight cache deduplicates identical
    grids anyway.
    """
    parts = Path(path).name.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "grid"


def build_group_weights(
    source_paths: Sequence[str],
    *,
    level: int,
    open_dataset: Callable[[str], object],
    resolution_level: Callable[[object], int],
    make_weights: Callable[[object, int], object],
    group_key: Callable[[str], str] = cordex_group_key,
) -> tuple[int, dict[str, str], dict[str, str], dict[str, str]]:
    """One weight file per source grid; injected callables for testability.

    Returns ``(target_level, native_levels, group_weights,
    representatives)``, all keyed by *group_key*.  Unlike the CMIP6
    variant this also returns the representative file per group, which
    the coverage computation reuses.
    """
    groups: dict[str, list[str]] = {}
    for src in sorted(source_paths):
        groups.setdefault(group_key(str(src)), []).append(str(src))

    representatives = {key: files[0] for key, files in groups.items()}
    rep_datasets = {key: open_dataset(rep) for key, rep in representatives.items()}
    native_levels = {key: int(resolution_level(ds)) for key, ds in rep_datasets.items()}
    target_level = int(level) if level else max(native_levels.values())
    group_weights = {
        key: str(make_weights(ds, target_level)) for key, ds in rep_datasets.items()
    }
    return target_level, native_levels, group_weights, representatives


# ---------------------------------------------------------------------------
# Regional helpers (coverage template, wrap-aware bounding box)
# ---------------------------------------------------------------------------
def make_ones_dataset(ds, template_var: str | None = None):
    """A ones-field on the source grid, for the coverage-fraction remap.

    Pushed through the same conservative weights as the data, the ones
    field arrives as the fraction of each HEALPix cell covered by the
    source domain (conservative weights are normalised by the full
    destination-cell area).
    """
    import xarray as xr

    if template_var is None:
        # The variable with the most dimensions is the safest template
        # (side-cars like time_bnds have fewer).
        template_var = max(
            (str(n) for n in ds.data_vars),
            key=lambda n: (ds[n].ndim, n),
        )
    template = ds[template_var]
    spatial: tuple[str, ...] = ()
    for name in ("lat", "latitude"):
        if name in ds.coords or name in ds.variables:
            if ds[name].ndim == 2:
                spatial = tuple(map(str, ds[name].dims))
                break
    if not spatial:
        spatial = tuple(map(str, template.dims[-2:]))
    squeeze = {d: 0 for d in map(str, template.dims) if d not in spatial}
    template = template.isel(squeeze, drop=True)
    return xr.Dataset(
        {"coverage_fraction": xr.ones_like(template, dtype=np.float64)},
        coords=ds.coords,
    )


def domain_bbox(
    latitude: np.ndarray,
    longitude: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Wrap-aware bounding box of the valid cells.

    Longitudes are evaluated in both the ``[0, 360)`` and the
    ``[-180, 180)`` frame and the frame with the smaller span wins, so a
    European domain crossing the Greenwich meridian gets
    ``(-12, 35)`` rather than ``(0, 360)``.
    """
    lat = np.asarray(latitude)[valid]
    lon = np.asarray(longitude)[valid]
    lon360 = np.mod(lon, 360.0)
    lon180 = np.mod(lon + 180.0, 360.0) - 180.0
    span360 = float(lon360.max() - lon360.min())
    span180 = float(lon180.max() - lon180.min())
    chosen = lon180 if span180 <= span360 else lon360
    return {
        "geospatial_lat_min": float(lat.min()),
        "geospatial_lat_max": float(lat.max()),
        "geospatial_lon_min": float(chosen.min()),
        "geospatial_lon_max": float(chosen.max()),
    }
