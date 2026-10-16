"""Irregular initial fire as a signed-distance grid, negative inside.

The polar radius r(theta) = r0 * (1 + 0.15 cos(3 theta) + 0.1 sin(5 theta))
is a closed curve, not a signed-distance function. The grid field is the
Euclidean distance to that polygon. Bilinear sampling supplies derivatives
for the hard initial condition away from the medial axis.
"""

from __future__ import annotations

import numpy as np
import torch
from shapely.geometry import Polygon

from wildfire_pinn.geometry import signed_distance
from wildfire_pinn.pinn import sample_grid


def polar_radius(theta, r0: float = 400.0) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return float(r0) * (1.0 + 0.15 * np.cos(3.0 * theta) + 0.1 * np.sin(5.0 * theta))


def irregular_polygon(x0: float, y0: float, r0: float = 400.0, n_vertices: int = 720) -> Polygon:
    theta = np.linspace(0.0, 2.0 * np.pi, int(n_vertices), endpoint=False)
    radius = polar_radius(theta, r0)
    coords = np.column_stack([x0 + radius * np.cos(theta), y0 + radius * np.sin(theta)])
    polygon = Polygon(coords)
    if not polygon.is_valid or polygon.is_empty:
        raise RuntimeError("irregular fire polygon is not a valid closed region")
    return polygon


def irregular_sdf(x: np.ndarray, y: np.ndarray, x0: float, y0: float, r0: float = 400.0) -> np.ndarray:
    """Signed distance in meters on the cell-center grid. Negative inside."""
    return signed_distance(irregular_polygon(x0, y0, r0), x, y)


def sample_sdf_normalized(x_norm, y_norm, sdf: np.ndarray, grid_x, grid_y, length: float, center_x: float, center_y: float):
    """Bilinear phi0 / L. Coordinates stay in the autograd graph."""
    x_phys = center_x + x_norm * length
    y_phys = center_y + y_norm * length
    meters = sample_grid(sdf, grid_x, grid_y, x_phys, y_phys)
    return meters / float(length)
