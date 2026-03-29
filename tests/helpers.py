"""Little helpers."""

import numpy as np


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
