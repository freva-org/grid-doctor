"""Output formatting helpers for ERA5/ERA5-Land HEALPix Zarr products."""

from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .file_fetcher import SourceRecord, SOURCE_MAPPER

def normalise_frequencies(frequencies: Iterable[str]) -> Tuple[str, ...]:
    """Return a stable tuple of requested output frequencies."""

    return tuple(dict.fromkeys(str(frequency) for frequency in frequencies))


def group_records_by_frequency(
    records: Iterable[SourceRecord],
) -> Dict[str, List[SourceRecord]]:
    """Group only resolved records by source frequency."""

    grouped: Dict[str, List[SourceRecord]] = defaultdict(list)
    for record in records:
        if record.files:
            grouped[record.frequency].append(record)
    return dict(grouped)

def destination_for_level(dataset: str, frequency: str, zoom_number: int) -> str:
    """Return the concrete Zarr store path for one frequency and HEALPix level."""

    return SOURCE_MAPPER["output_path"].format(
        dataset=dataset,
        output_frequency=SOURCE_MAPPER["output_frequency"][frequency],
        zoom_number=zoom_number,
    )


def existing_destinations_for_frequency(dataset: str, frequency: str) -> Tuple[str, ...]:
    """Return existing Zarr stores for one output frequency."""

    output_frequency = SOURCE_MAPPER["output_frequency"][frequency]
    pattern = SOURCE_MAPPER["output_path"].format(
        dataset=dataset,
        output_frequency=output_frequency,
        zoom_number="*",
    )
    return tuple(sorted(glob(pattern)))
