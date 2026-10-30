"""How a dataset was produced: package version, method and coarsening."""

from __future__ import annotations

import numbers
from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

from ._keys import (
    COARSEN_MODE_ATTR,
    COARSENED_FROM_ATTR,
    METHOD_ATTR,
    MIN_VALID_FRACTION_ATTR,
    VERSION_ATTR,
)


def _package_version() -> str:
    from .. import __version__

    return __version__


@dataclass(frozen=True, slots=True)
class Provenance:
    """Provenance attributes of a grid-doctor dataset.

    Parameters
    ----------
    version:
        grid-doctor version that produced the values.  Defaults to the
        running version.
    method:
        How the finest values were produced: a remapping method
        (``"conservative"``, ``"nearest"``, ...) or a binning method
        (``"binned-mean"``, ``"binned-mode"``).  Drives
        ``coarsen_mode="auto"``.
    coarsened_from_level:
        Level the values were coarsened from (coarsened levels only).
    coarsen_mode:
        ``"mean"`` or ``"mode"`` (coarsened levels only).
    min_valid_fraction:
        Threshold below which coarsened cells were set to NaN
        (coarsened levels only).
    """

    version: str = field(default_factory=_package_version)
    method: str | None = None
    coarsened_from_level: int | None = None
    coarsen_mode: str | None = None
    min_valid_fraction: float | None = None

    def to_attrs(self) -> dict[str, str | int | float]:
        """Return the provenance as global attributes, skipping unset fields."""
        attrs: dict[str, str | int | float] = {VERSION_ATTR: self.version}
        if self.method is not None:
            attrs[METHOD_ATTR] = self.method
        if self.coarsened_from_level is not None:
            attrs[COARSENED_FROM_ATTR] = int(self.coarsened_from_level)
        if self.coarsen_mode is not None:
            attrs[COARSEN_MODE_ATTR] = self.coarsen_mode
        if self.min_valid_fraction is not None:
            attrs[MIN_VALID_FRACTION_ATTR] = float(self.min_valid_fraction)
        return attrs

    @classmethod
    def from_attrs(cls, attrs: Mapping[Hashable, object]) -> Provenance:
        """Read provenance from global attributes; missing fields stay unset.

        A missing version is reported as ``"unknown"`` rather than the
        running version, which would misattribute foreign data.
        """
        return cls(
            version=str(attrs.get(VERSION_ATTR, "unknown")),
            method=_optional_str(attrs.get(METHOD_ATTR)),
            coarsened_from_level=_optional_int(attrs.get(COARSENED_FROM_ATTR)),
            coarsen_mode=_optional_str(attrs.get(COARSEN_MODE_ATTR)),
            min_valid_fraction=_optional_float(attrs.get(MIN_VALID_FRACTION_ATTR)),
        )


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"Expected an integer attribute, got {value!r}.")
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"Expected a numeric attribute, got {value!r}.")
    return float(value)


def variable_method(attrs: Mapping[Hashable, object]) -> str | None:
    """Return the per-variable production method, if recorded.

    Variables binned with different aggregations carry their own method
    next to the dataset-level one in [`Provenance`][grid_doctor.metadata.Provenance].
    """
    return _optional_str(attrs.get(METHOD_ATTR))


def variable_method_attrs(method: str) -> dict[str, str]:
    """Return the attributes recording a per-variable production method."""
    return {METHOD_ATTR: method}
