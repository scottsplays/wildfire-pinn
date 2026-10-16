"""Perimeters and local front speeds. These functions never receive R."""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path
from scipy.ndimage import distance_transform_edt

from wildfire_pinn.geometry import zero_contours

FORBIDDEN_TIMES = {3000.0, 4000.0}


def longest_contour(phi, x, y) -> np.ndarray | None:
    lines = [line for line in zero_contours(np.asarray(phi), x, y) if len(line) >= 8]
    if not lines:
        return None
    return max(lines, key=lambda item: len(item))


def contour_center(phi, x, y) -> tuple[float, float]:
    pts = longest_contour(phi, x, y)
    if pts is None:
        raise RuntimeError("the observed perimeter has no contour")
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def mean_radius(phi, x, y, x0: float, y0: float) -> float:
    pts = longest_contour(phi, x, y)
    if pts is None:
        raise RuntimeError("the observed perimeter has no contour")
    return float(np.hypot(pts[:, 0] - x0, pts[:, 1] - y0).mean())


def training_view(fields: dict, times: list[float], x, y) -> dict:
    """Observed fields at training times. The rate array is not copied in."""
    clean = [float(t) for t in times]
    if any(t in FORBIDDEN_TIMES for t in clean):
        raise ValueError("training observations include a forecast time")
    if 0.0 not in clean:
        raise ValueError("training observations must include the initial perimeter")
    return {
        "times": clean,
        "fields": {float(t): np.asarray(fields[float(t)], dtype=np.float64) for t in clean},
        "phi0": np.asarray(fields[0.0], dtype=np.float64),
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
    }


def angular_radii(phi, x, y, x0: float, y0: float, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    centers = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)
    radii = np.full(n_bins, np.nan)
    pts = longest_contour(phi, x, y)
    if pts is None:
        return centers, radii
    ang = np.arctan2(pts[:, 1] - y0, pts[:, 0] - x0)
    rad = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0)
    bins = np.floor((ang + np.pi) / (2.0 * np.pi) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    for index in range(n_bins):
        chosen = rad[bins == index]
        if chosen.size:
            radii[index] = float(np.max(chosen))
    good = np.isfinite(radii)
    if int(good.sum()) >= max(4, n_bins // 8):
        slots = np.arange(n_bins)
        radii = np.interp(slots, slots[good], radii[good], period=n_bins)
    return centers, radii


def segment_speeds(view: dict, x0: float, y0: float, n_bins: int) -> list[dict]:
    """Radial speed of the observed front on each training interval."""
    segments = []
    times = view["times"]
    for left, right in zip(times[:-1], times[1:]):
        _, r0 = angular_radii(view["fields"][left], view["x"], view["y"], x0, y0, n_bins)
        centers, r1 = angular_radii(view["fields"][right], view["x"], view["y"], x0, y0, n_bins)
        speed = (r1 - r0) / (right - left)
        segments.append({
            "t0": left,
            "t1": right,
            "angle_rad": centers,
            "radius_mid_m": 0.5 * (r0 + r1),
            "speed_m_s": speed,
        })
    return segments


def sdf_from_mask(mask: np.ndarray, spacing: float) -> np.ndarray:
    inside = np.asarray(mask, dtype=bool)
    outer = distance_transform_edt(~inside)
    inner = distance_transform_edt(inside)
    return (outer - inner) * float(spacing)


def perturb_view(view: dict, amplitude: float, harmonic: int, phase: float) -> dict:
    """Move each observed perimeter radially and rebuild a signed distance."""
    x = view["x"]
    y = view["y"]
    spacing = float(x[1] - x[0])
    fields = {}
    for time, phi in view["fields"].items():
        pts = longest_contour(phi, x, y)
        if pts is None:
            raise RuntimeError("cannot perturb a missing perimeter")
        center = np.array([pts[:, 0].mean(), pts[:, 1].mean()])
        offset = pts - center
        radius = np.hypot(offset[:, 0], offset[:, 1])
        direction = offset / np.maximum(radius[:, None], 1e-6)
        theta = np.arctan2(offset[:, 1], offset[:, 0])
        moved = pts + (amplitude * np.sin(harmonic * theta + phase))[:, None] * direction
        if np.linalg.norm(moved[0] - moved[-1]) > 1e-6:
            moved = np.vstack([moved, moved[0]])
        yy, xx = np.meshgrid(y, x, indexing="ij")
        flat = np.column_stack([xx.ravel(), yy.ravel()])
        mask = Path(moved, closed=True).contains_points(flat).reshape(xx.shape)
        fields[time] = sdf_from_mask(mask, spacing)
    return training_view(fields, view["times"], x, y)
