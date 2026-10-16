"""Signed-distance fields and fire-front contours on a regular grid."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely import contains_xy
from shapely.geometry.base import BaseGeometry


def cell_size_m(x: np.ndarray, y: np.ndarray) -> float:
    """Mean cell side length. The grid is required to be square in meters."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("grid must have at least 2 coordinates on each axis")
    dx = float(np.mean(np.abs(np.diff(x))))
    dy = float(np.mean(np.abs(np.diff(y))))
    if dx <= 0 or dy <= 0:
        raise ValueError("grid coordinates must not be constant")
    if abs(dx - dy) / max(dx, dy) > 0.05:
        raise ValueError(
            f"signed distance expects square cells, got {dx:.2f} m by {dy:.2f} m"
        )
    return 0.5 * (dx + dy)


def signed_distance(geom: BaseGeometry, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Signed distance in meters. Negative inside the perimeter, positive outside.

    ``x`` and ``y`` are the cell-center coordinates of the model grid, and
    ``geom`` must use the same meter coordinates as ``x`` and ``y``
    (EPSG:3310 for a California fire).
    Rows of the returned array follow ``y``; columns follow ``x``.
    """
    if geom is None or geom.is_empty:
        raise ValueError("perimeter geometry is empty")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pixel = cell_size_m(x, y)
    xx, yy = np.meshgrid(x, y)
    inside = np.asarray(contains_xy(geom, xx, yy), dtype=bool)
    if not inside.any():
        raise ValueError(
            "No grid cell falls inside the perimeter. "
            "The polygon and the grid are in different coordinate systems, "
            "or the perimeter misses the study window."
        )
    outside_px = distance_transform_edt(~inside)
    inside_px = distance_transform_edt(inside)
    sdf = outside_px * pixel
    sdf[inside] = -inside_px[inside] * pixel
    if float(sdf.max()) <= 0:
        raise ValueError(
            "Every grid cell is inside the perimeter, so the domain has no fire front."
        )
    return sdf


def zero_contours(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Return phi=0 polylines as arrays of shape (vertex, 2) in grid coordinates.

    ``field`` is indexed ``[row, col]`` with ``row`` along ``y`` and ``col`` along ``x``.
    Vertex positions are interpolated between cell centers, which is the zero
    crossing marching squares finds. Snapping those vertices back to the nearest
    cell (as the notebook did) adds up to half a cell of error.
    """
    from skimage.measure import find_contours

    field = np.asarray(field, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if field.shape != (len(y), len(x)):
        raise ValueError(
            f"field shape {field.shape} does not match y, x lengths {(len(y), len(x))}"
        )
    rows = np.arange(len(y), dtype=np.float64)
    cols = np.arange(len(x), dtype=np.float64)
    polylines = []
    for contour in find_contours(field, level=0.0):
        xs = np.interp(contour[:, 1], cols, x)
        ys = np.interp(contour[:, 0], rows, y)
        polylines.append(np.column_stack([xs, ys]))
    return polylines


def mean_distance_m(vertices: np.ndarray, geom: BaseGeometry) -> float:
    """Mean distance from front vertices to an observed perimeter boundary."""
    from shapely import distance, points

    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) == 0 or geom is None or geom.is_empty:
        return float("nan")
    return float(np.mean(distance(points(vertices), geom)))
