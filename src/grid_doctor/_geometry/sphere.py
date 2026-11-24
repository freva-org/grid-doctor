"""Vectorised spherical geometry: Cartesian conversion, centroids, winding."""

from __future__ import annotations

from typing import Literal, cast, overload

import numpy as np

from ..types import FloatArray
from .normalise import _canonical_lon


def _lonlat_to_xyz(lon_deg: FloatArray, lat_deg: FloatArray) -> FloatArray:
    """Convert lon/lat coordinates to Cartesian unit-sphere vectors.

    Both inputs are broadcast together before conversion, so any
    compatible shapes are accepted.

    Args:
        lon_deg: Longitude in degrees.
        lat_deg: Latitude in degrees.

    Returns:
        Array of shape ``(*broadcast_shape, 3)`` with
        ``(x, y, z)`` columns.
    """
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat)
    return np.stack(
        (cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)),
        axis=-1,
    )


@overload
def _xyz_to_lonlat(xyz: FloatArray, *, batch: Literal[False] = ...) -> tuple[float, float]: ...  # noqa: E501


@overload
def _xyz_to_lonlat(xyz: FloatArray, *, batch: Literal[True]) -> tuple[FloatArray, FloatArray]: ...  # noqa: E501


def _xyz_to_lonlat(
    xyz: FloatArray,
    *,
    batch: bool = False,
) -> tuple[float, float] | tuple[FloatArray, FloatArray]:
    """Convert Cartesian unit-sphere vectors back to lon/lat in degrees.

    Args:
        xyz: Either a single ``(3,)`` vector or, when *batch* is
            *True*, an array of shape ``(..., 3)``.
        batch: When *True* the function operates element-wise over the
            leading dimensions and returns arrays instead of scalars.

    Returns:
        ``(lon_deg, lat_deg)`` as scalars (default) or arrays.

    Raises:
        ValueError: When any vector has zero length.
    """
    if batch:
        norm = cast(float, np.linalg.norm(xyz, axis=-1, keepdims=True))
        if np.any(norm == 0.0):
            raise ValueError("Cannot convert zero-length vector to lon/lat.")
        unit = xyz / norm
        lon = np.rad2deg(np.arctan2(unit[..., 1], unit[..., 0]))
        lat = np.rad2deg(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
        return (
            lon.astype(np.float64, copy=False),
            lat.astype(np.float64, copy=False),
        )

    norm = cast(float, np.linalg.norm(xyz))
    if norm == 0.0:
        raise ValueError("Cannot convert zero-length vector to lon/lat.")
    unit = xyz / norm
    lon = np.rad2deg(np.arctan2(unit[1], unit[0]))
    lat = np.rad2deg(np.arcsin(np.clip(unit[2], -1.0, 1.0)))
    return float(lon), float(lat)


def _spherical_centroid(
    lon_deg: FloatArray,
    lat_deg: FloatArray,
) -> tuple[float, float]:
    """Return the spherical centroid of a set of lon/lat points.

    The centroid is computed as the mean Cartesian vector projected back
    onto the unit sphere.

    Args:
        lon_deg: Longitude values in degrees.
        lat_deg: Latitude values in degrees.

    Returns:
        ``(lon_deg, lat_deg)`` of the centroid.
    """
    xyz = _lonlat_to_xyz(lon_deg, lat_deg)
    return _xyz_to_lonlat(xyz.mean(axis=0))


def _ensure_ccw(
    lon_deg: FloatArray,
    lat_deg: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Ensure a polygon's vertices are wound counter-clockwise.

    The winding test is performed in a local tangent-plane projection
    centred on the polygon's spherical centroid.

    Args:
        lon_deg: Polygon vertex longitudes in degrees.
        lat_deg: Polygon vertex latitudes in degrees.

    Returns:
        ``(lon_deg, lat_deg)`` in counter-clockwise order.
    """
    pts = _lonlat_to_xyz(lon_deg, lat_deg)
    lon0, lat0 = _spherical_centroid(lon_deg, lat_deg)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)

    east = np.array(
        [-np.sin(lon0_rad), np.cos(lon0_rad), 0.0],
        dtype=np.float64,
    )
    north = np.array(
        [
            -np.sin(lat0_rad) * np.cos(lon0_rad),
            -np.sin(lat0_rad) * np.sin(lon0_rad),
            np.cos(lat0_rad),
        ],
        dtype=np.float64,
    )
    x = pts @ east
    y = pts @ north
    signed_area = float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if signed_area < 0.0:
        return lon_deg[::-1].copy(), lat_deg[::-1].copy()
    return lon_deg, lat_deg


# ===================================================================
# Polygon-centre computation (vectorised)
# ===================================================================


def _vectorized_polygon_centres(
    cell_lon: FloatArray,
    cell_lat: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute spherical polygon centres from padded corner arrays.

    The centre of each polygon is the normalised mean Cartesian vector
    of its valid corners, projected back to lon/lat.

    Args:
        cell_lon: Corner longitudes, shape ``(n_face, max_corners)``.
            Invalid slots contain ``NaN``.
        cell_lat: Corner latitudes, same shape as *cell_lon*.

    Returns:
        ``(face_lon, face_lat)`` in degrees, each shape ``(n_face,)``.
        Longitudes are canonicalised to ``[-180, 180)``.

    Raises:
        ValueError: When any polygon has no valid corners or the
            mean vector has zero length.
    """
    mask = np.isfinite(cell_lon) & np.isfinite(cell_lat)
    if np.any(mask.sum(axis=1) == 0):
        raise ValueError("Every polygon must contain at least one valid corner.")

    lon_rad = np.deg2rad(np.where(mask, cell_lon, 0.0))
    lat_rad = np.deg2rad(np.where(mask, cell_lat, 0.0))
    cos_lat = np.cos(lat_rad)

    x = np.where(mask, cos_lat * np.cos(lon_rad), 0.0).sum(axis=1)
    y = np.where(mask, cos_lat * np.sin(lon_rad), 0.0).sum(axis=1)
    z = np.where(mask, np.sin(lat_rad), 0.0).sum(axis=1)

    norm = np.sqrt(x * x + y * y + z * z)
    if np.any(norm == 0.0):
        raise ValueError("Cannot infer polygon centres from zero-length vectors.")

    lon = np.rad2deg(np.arctan2(y, x)).astype(np.float64, copy=False)
    lat = np.rad2deg(np.arcsin(np.clip(z / norm, -1.0, 1.0))).astype(np.float64, copy=False)
    return _canonical_lon(lon), lat
