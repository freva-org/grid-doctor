"""High-level helpers for CF conventions.

Provides abstractions to convert remapping terms (level, order, grid type)
to the respective CF vocabolary.
"""

from enum import Enum
from typing import Mapping


class CFKey(Enum):
    """Supported CF attribute keys."""

    MAPPING = "grid_mapping_name"  # - "healpix"
    SCHEME = "indexing_scheme"  # - "nested", "ring", "nuniq", or "zuniq"
    LEVEL = "refinement_level"  # - The HEALPix refinement level


def healpix_cf_attrs(level: int, scheme: str) -> Mapping[CFKey, str]:
    """Given a level and an scheme, returns a dictionary with the values mapped as attributes."""
    if scheme not in {"nested", "ring", "nuniq", "zuniq"}:
        raise ValueError("Cannot attach unsupported scheme")

    if not isinstance(level, int):
        raise ValueError(
            f"Cannot set invalid refinement level: '{level}' (must be integer)."
        )

    if level < 0:
        raise ValueError(f"Cannot set negative refinement level: '{level}'")

    return {
        CFKey.MAPPING: "healpix",
        CFKey.SCHEME: str(scheme),
        CFKey.LEVEL: str(level),
    }
