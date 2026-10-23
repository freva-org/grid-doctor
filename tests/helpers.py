"""Little helpers."""

from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

UNSTRUCTURED = {"unstructured_ds"}
STRUCTURED = {"regular_ds", "curvilinear_ds"}
MISC = {"era5_ds", "limited_area_ds"}

ALL = UNSTRUCTURED | STRUCTURED | MISC


# Usefull decorator to parametrize dataset fixures
def TEST_DS(datasets: Iterable[str] | Iterable[tuple[Any, ...]], argnames: str = "") -> pytest.MarkDecorator:
    """Parametrize the ``test_ds`` fixture over dataset fixture names.

    Pass a set of names, or ``(name, *values)`` tuples together with the
    extra *argnames*, e.g.:
      ``TEST_DS([("unstructured_ds", False)], "is_rad")``
    or simply:
        ```
        @TEST_DS(ALL)
        def test_mytest_name('test_ds'):
            pass
        ```
    """
    names = f"test_ds,{argnames}" if argnames else "test_ds"
    return pytest.mark.parametrize(names, sorted(datasets), indirect=["test_ds"])


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
