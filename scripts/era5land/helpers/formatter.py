"""Output formatting helpers for ERA5/ERA5-Land HEALPix Zarr products."""

from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

def destination_for_level(
    dataset: str,
    frequency: str,
    zoom_number: int,
    *,
    output_path: str | Path | None = None,
) -> str:
    """Return the concrete Zarr store path for one frequency and HEALPix level."""

    if output_path is not None:
        root_path = Path(output_path)
        return str(
            root_path
            / str(dataset)
            / str(SOURCE_MAPPER["output_frequency"][frequency])
            / f"level_{zoom_number}.zarr"
        )

    return SOURCE_MAPPER["output_path"].format(
        dataset=dataset,
        output_frequency=SOURCE_MAPPER["output_frequency"][frequency],
        zoom_number=zoom_number,
    )


def dataset_output_root(
    dataset: str,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Return the dataset-level output root containing all frequencies."""

    sample_path = Path(destination_for_level(dataset, "fx", 0, output_path=output_path))
    return sample_path.parent.parent


def existing_destinations_for_frequency(
    dataset: str,
    frequency: str,
    *,
    output_path: str | Path | None = None,
) -> Tuple[str, ...]:
    """Return existing Zarr stores for one output frequency."""

    output_frequency = SOURCE_MAPPER["output_frequency"][frequency]
    if output_path is not None:
        pattern = str(Path(output_path) / str(dataset) / str(output_frequency) / "level_*.zarr")
    else:
        pattern = SOURCE_MAPPER["output_path"].format(
            dataset=dataset,
            output_frequency=output_frequency,
            zoom_number="*",
        )
    return tuple(sorted(glob(pattern)))
