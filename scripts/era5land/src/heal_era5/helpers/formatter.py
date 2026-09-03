"""Output formatting helpers for ERA5/ERA5-Land HEALPix Zarr products."""

from collections import defaultdict
from collections.abc import Iterable
from glob import glob
from pathlib import Path

from .file_fetcher import SOURCE_MAPPER, SourceRecord


def normalise_frequencies(frequencies: Iterable[str]) -> tuple[str, ...]:
    """Return a stable tuple of requested output frequencies."""

    return tuple(dict.fromkeys(str(frequency) for frequency in frequencies))


def group_records_by_frequency(
    records: Iterable[SourceRecord],
) -> dict[str, list[SourceRecord]]:
    """Group only resolved records by source frequency."""

    grouped: dict[str, list[SourceRecord]] = defaultdict(list)
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
            root_path / str(dataset) / str(SOURCE_MAPPER["output_frequency"][frequency]) / f"level_{zoom_number}.zarr"
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


def merge_dataset_root(
    dataset: str,
    *,
    output_path: str | Path,
    frequencies: Iterable[str] | str | None = None,
) -> Path:
    """Return the dataset root used by selector-based merge operations.

    ``output_path`` may already name the dataset root, or may name a single
    frequency directory for compatibility with the direct-store merge form.
    Otherwise it is treated as a shared publication root and ``dataset`` is
    appended to it.
    """

    root_path = Path(output_path)
    if root_path.name == str(dataset):
        return root_path

    if isinstance(frequencies, str):
        selected = tuple(item.strip() for item in frequencies.split(",") if item.strip())
    else:
        selected = tuple(frequencies or ())
    if (
        len(selected) == 1
        and selected[0] in SOURCE_MAPPER["output_frequency"]
        and root_path.name == SOURCE_MAPPER["output_frequency"][selected[0]]
    ):
        return root_path.parent

    return root_path / str(dataset)


def existing_destinations_for_frequency(
    dataset: str,
    frequency: str,
    *,
    output_path: str | Path | None = None,
) -> tuple[str, ...]:
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
