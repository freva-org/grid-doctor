"""High-level helpers for CF conventions.

Provides abstractions to convert remapping terms (level, order, grid type)
to the respective CF vocabulary.
"""

import operator
from dataclasses import (
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

from grid_doctor.types import (
    HealpixIndexScheme,
)

_ALLOWED_SCHEMES = frozenset(get_type_args(HealpixIndexScheme))
_LEVEL_NEEDED = frozenset(
    {
        "nested",
        "ring",
    }
)


@dataclass
class CFHealpixGridAttrs:
    """Models `healpix` grid mapping attribute according to CF conventions.

    - grid_mapping_name is always "healpix"
    - indexing_scheme can be one of "nested", "ring", "nuniq", "zuniq".
    - refinement_level is a positive int, required only for "nested" or "ring" indexing_schemes.
    """

    grid_mapping_name: Literal["healpix"] = field(
        default="healpix",
        init=False,
    )
    indexing_scheme: HealpixIndexScheme
    refinement_level: int | None = None

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
                raise ValueError(
                    f"Indexing scheme {self.indexing_scheme!r} requires a `refinement_level` to be set."
                )

            try:
                operator.index(self.refinement_level)
            except (
                TypeError,
                ValueError,
            ):
                raise ValueError(
                    f"Invalid `refinement_level`: '{self.refinement_level}'; Must be integer."
                )

            if self.refinement_level < 0:
                raise ValueError(f"Cannot set negative `refinement_level`: '{self.refinement_level}'.")
        else:
            if self.refinement_level is not None:
                raise ValueError(
                    f"`refinement_level` cannot be specified when `indexing_scheme` is {self.indexing_scheme!r}."
                )

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


def _healpix_cf_attrs(
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
