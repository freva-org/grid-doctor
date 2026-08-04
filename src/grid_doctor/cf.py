"""High-level helpers for CF conventions.

Provides abstractions to convert remapping terms (level, order, grid type)
to the respective CF vocabolary.
"""

from typing import Mapping
from typing import get_args as get_type_args

from grid_doctor.types import HealpixIndexScheme


class CFKey:
    """Supported CF attribute keys."""

    MAPPING = "grid_mapping_name"  # - "healpix"
    SCHEME = "indexing_scheme"  # - "nested", "ring", "nuniq", or "zuniq"
    LEVEL = "refinement_level"  # - The HEALPix refinement level


def _healpix_cf_attrs(
    scheme: str | HealpixIndexScheme, level: int | None = None
) -> Mapping[str, str]:
    """Given a level and an scheme, returns a dictionary with the values mapped as attributes."""
    _schemes: tuple[str] = get_type_args(HealpixIndexScheme)
    if scheme not in _schemes:
        raise ValueError(
            f'Cannot attach unsupported scheme "{scheme}". Supported: {_schemes}',
        )

    if scheme in {"nuniq", "zuniq"}:
        return {
            CFKey.MAPPING: "healpix",
            CFKey.SCHEME: str(scheme),
        }

    # This is mandatory for nested and ring but must be omitted for *uniq
    # https://cfconventions.org/cf-conventions/cf-conventions.html#healpix
    if level is None:
        raise ValueError(
            f"Indexing scheme {scheme} requires a valid healpix refinement level."
        )

    if not isinstance(level, int):
        raise ValueError(
            f"Cannot set invalid refinement level: '{level}' (must be integer)."
        )

    if level < 0:
        raise ValueError(f"Cannot set negative refinement level: '{level}'")

    return {
        CFKey.LEVEL: str(level),
        CFKey.MAPPING: "healpix",
        CFKey.SCHEME: str(scheme),
    }
