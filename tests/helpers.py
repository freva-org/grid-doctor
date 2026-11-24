"""Little helpers."""

import json
from pathlib import Path

import numpy as np
import xarray as xr


class _FakeHealpixModule:
    @staticmethod
    def vertices(
        ipix: np.ndarray, level: int, ellipsoid: str = "sphere"
    ) -> tuple[np.ndarray, np.ndarray]:
        del level, ellipsoid
        n = ipix.size
        lon = np.stack(
            [
                np.asarray(ipix, dtype=np.float64),
                np.asarray(ipix, dtype=np.float64) + 0.5,
                np.asarray(ipix, dtype=np.float64) + 0.5,
                np.asarray(ipix, dtype=np.float64),
            ],
            axis=1,
        )
        lat = np.tile(np.array([0.0, 0.0, 0.5, 0.5], dtype=np.float64), (n, 1))
        return lon, lat

    @staticmethod
    def healpix_to_lonlat(
        ipix: np.ndarray, level: int
    ) -> tuple[np.ndarray, np.ndarray]:
        del level
        n = ipix.size
        lon = np.linspace(-180.0, 180.0, n, endpoint=False)
        lat = np.linspace(-90.0, 90.0, n)
        return lon, lat


def with_healpix_metadata(
    ds: xr.Dataset,
    level: int,
    *,
    nest: bool = True,
    method: str | None = None,
    sparse: bool = False,
) -> xr.Dataset:
    """Write grid-doctor metadata onto a hand-built HEALPix test dataset."""
    from grid_doctor.metadata import HealpixGrid, HealpixMetadata, Provenance

    metadata = HealpixMetadata(
        HealpixGrid.from_nest(level, nest=nest),
        Provenance(method=method),
        sparse=sparse,
    )
    return metadata.apply(ds)


LEVEL = 3
NPIX = 12 * 4**LEVEL
NLAT, NLON = 8, 16


def _weight_file(path: Path, n_target: int = NPIX, seed: int = 7) -> Path:
    """Synthetic conservative-style weights: 3 normalised sources per target."""
    rng = np.random.default_rng(seed)
    n_source = NLAT * NLON
    rows = np.repeat(np.arange(n_target), 3)
    cols = rng.integers(0, n_source, size=rows.size)
    weights = rng.random(rows.size)
    weights /= np.bincount(rows, weights)[rows]
    xr.Dataset(
        {
            "row": ("n_s", rows + 1),
            "col": ("n_s", cols + 1),
            "S": ("n_s", weights),
        },
        attrs={
            "grid_doctor_level": LEVEL,
            "grid_doctor_order": "nested",
            "grid_doctor_method": "conservative",
            "grid_doctor_source_dims": json.dumps(["lat", "lon"]),
        },
    ).to_netcdf(path)
    return path
