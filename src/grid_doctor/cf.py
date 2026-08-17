"""High-level helpers for CF conventions.

Provides abstractions to convert remapping terms (level, order, grid type)
to the respective CF vocabulary.
"""

import operator
from dataclasses import (
    asdict,
    dataclass,
    field,
)
from typing import (
    Iterator,
    Literal,
    Mapping,
    cast,
)
from typing import get_args as get_type_args

HealpixIndexScheme = Literal["nested", "ring", "zuniq", "nuniq"]
"""Allowed Healpix indexing schemes."""

_ALLOWED_SCHEMES = frozenset(get_type_args(HealpixIndexScheme))
_LEVEL_NEEDED = frozenset(
    {
        "nested",
        "ring",
    }
)

_CONV_VERSION = 1.13


@dataclass(frozen=True, slots=True)
class CFConventions:
    """Model global `Conventions` CF attribute."""

    Conventions: str = f"CF-{_CONV_VERSION}"

    def to_dict(self) -> dict[str, str]:
        """Convert class to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class CFHealpixGridAttrs:
    """Models `healpix` grid mapping attribute according to CF conventions.

    - grid_mapping_name is always "healpix"
    - indexing_scheme can be one of "nested", "ring", "nuniq", "zuniq".
    - refinement_level is a positive int, required only for "nested" or "ring" indexing_schemes.
    - earth-radius is the optional radius of the sphere, in meters.
    """

    grid_mapping_name: Literal["healpix"] = field(
        default="healpix",
        init=False,
    )
    indexing_scheme: HealpixIndexScheme
    refinement_level: int | None = None
    earth_radius: int | None = 6371009

    def __post_init__(
        self,
    ) -> None:
        """Validate instantiations."""
        if self.indexing_scheme not in _ALLOWED_SCHEMES:
            raise ValueError(
                f"`indexing_scheme` must be one of {sorted(_ALLOWED_SCHEMES)}, not {self.indexing_scheme!r}"
            )

        if self.indexing_scheme in _LEVEL_NEEDED:
            # `refinement_level` is mandatory for `nested` and `ring` but must be omitted for *uniq
            # https://cfconventions.org/cf-conventions/cf-conventions.html#healpix
            if self.refinement_level is None:
                raise ValueError(f"Indexing scheme {self.indexing_scheme!r} requires a `refinement_level` to be set.")

            try:
                operator.index(self.refinement_level)
            except (
                TypeError,
                ValueError,
            ):
                raise ValueError(f"Invalid `refinement_level`: '{self.refinement_level}'; Must be integer.")

            if self.refinement_level < 0:
                raise ValueError(f"Cannot set negative `refinement_level`: '{self.refinement_level}'.")
        else:
            if self.refinement_level is not None:
                raise ValueError(
                    f"`refinement_level` cannot be specified when `indexing_scheme` is {self.indexing_scheme!r}."
                )

        # Validate earth_radius if provided
        if self.earth_radius:
            try:
                operator.index(self.earth_radius)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid `earth_radius`: {self.earth_radius!r}; Must be integer, in meters!")

    def __iter__(
        self,
    ) -> Iterator[
        tuple[
            str,
            str | int,
        ]
    ]:
        """Iterate through class fields, skipping optionals."""
        yield (
            "grid_mapping_name",
            self.grid_mapping_name,
        )
        yield (
            "indexing_scheme",
            self.indexing_scheme,
        )

        if self.refinement_level:
            yield (
                "refinement_level",
                self.refinement_level,
            )
        if self.earth_radius:
            yield ("earth_radius", self.earth_radius)

    def to_dict(self) -> dict[str, str | int]:
        """Convert class instance into an equivalent dictionary."""
        return dict(self)


def healpix_grid_mapping_attrs(
    scheme: str,
    level: int | None = None,
) -> Mapping[str, str | int]:
    """Given a level and a scheme, returns a dictionary with the values mapped as attributes."""
    return dict(
        CFHealpixGridAttrs(
            indexing_scheme=cast(
                HealpixIndexScheme,
                scheme,
            ),
            refinement_level=level,
        )
    )
