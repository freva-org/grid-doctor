"""Output formatting helpers for ERA5/ERA5-Land HEALPix Zarr products.

This module will own CF metadata normalization, frequency grouping decisions,
and writes to filesystem or S3 targets. The concrete dataset layout is still a
design choice: one consolidated store for all frequencies, or one store per
frequency.
"""

from typing import Iterable, Tuple


def normalise_frequencies(frequencies: Iterable[str]) -> Tuple[str, ...]:
    """Return a stable tuple of requested output frequencies."""

    return tuple(dict.fromkeys(frequencies))
