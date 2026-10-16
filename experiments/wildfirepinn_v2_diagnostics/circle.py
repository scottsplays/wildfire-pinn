"""Analytical expanding circle in meters and seconds.

phi(x, y, t) = distance to (x0, y0) - (r0 + R t).
Negative inside the burned disk. The zero contour is a circle of radius r0 + R t.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CircleProblem:
    """Domain is the square [-half_extent, half_extent]^2 in meters.

    The ignition center is offset from the origin so the gradient kink is not
    assumed to sit on the domain center. L is the side length used by the same
    normalization as the Kincade model: X = (x - xc) / L, tau = t / T.
    """

    x0: float = 200.0
    y0: float = -150.0
    r0: float = 400.0
    t_final: float = 4000.0
    half_extent: float = 2000.0
    n: int = 401
    pde_exclusion_m: float = 50.0

    @property
    def length(self) -> float:
        return 2.0 * float(self.half_extent)

    @property
    def center_x(self) -> float:
        return 0.0

    @property
    def center_y(self) -> float:
        return 0.0

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(-self.half_extent, self.half_extent, int(self.n), dtype=np.float64)
        y = np.linspace(-self.half_extent, self.half_extent, int(self.n), dtype=np.float64)
        return x, y

    def clearance_m(self, rate: float) -> float:
        """Smallest gap between the final circle and the edge of the square."""
        radius = float(self.r0) + float(rate) * float(self.t_final)
        left = (self.x0 - radius) - (-self.half_extent)
        right = self.half_extent - (self.x0 + radius)
        bottom = (self.y0 - radius) - (-self.half_extent)
        top = self.half_extent - (self.y0 + radius)
        return float(min(left, right, bottom, top))

    def to_dict(self) -> dict:
        return asdict(self)


def analytical_phi(x, y, t, problem: CircleProblem, rate: float) -> np.ndarray:
    """Signed distance in meters. ``t`` and ``rate`` are seconds and m/s."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    time = np.asarray(t, dtype=np.float64)
    radius = float(problem.r0) + float(rate) * time
    return np.hypot(xx - problem.x0, yy - problem.y0) - radius


def true_radius(problem: CircleProblem, rate: float, t: float) -> float:
    return float(problem.r0) + float(rate) * float(t)


def mask_iou(predicted: np.ndarray, truth: np.ndarray) -> float:
    """IoU of the burned sets phi < 0."""
    pred = np.asarray(predicted) < 0
    true = np.asarray(truth) < 0
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, true).sum() / union)


def front_radius_stats(phi: np.ndarray, x: np.ndarray, y: np.ndarray, x0: float, y0: float, radius: float) -> dict:
    """Radius and mean radial displacement of the longest zero contour."""
    from wildfire_pinn.geometry import zero_contours

    lines = zero_contours(np.asarray(phi, dtype=np.float64), x, y)
    if not lines:
        return {
            "contour": False,
            "radius_m": None,
            "radius_error_m": None,
            "mean_radial_displacement_m": None,
            "n_vertices": 0,
        }
    pts = max(lines, key=lambda item: len(item))
    radial = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0)
    return {
        "contour": True,
        "radius_m": float(np.mean(radial)),
        "radius_error_m": float(np.mean(radial) - radius),
        "mean_radial_displacement_m": float(np.mean(np.abs(radial - radius))),
        "n_vertices": int(len(pts)),
    }
