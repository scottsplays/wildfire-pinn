"""Contour roughness, angular error, and front sampling.

Sampling uses the model's own zero contour. Coordinates are detached.
The numerical front is not an input to the sampler.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label

from experiments.wildfirepinn_v2_diagnostics.experiment1 import _sample_pde
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import R0_M_S, spatial_rate_np
from wildfire_pinn.geometry import zero_contours

HIGH_MODE = 6
FRONT_JITTER_M = 100.0
# |grad R| at half its analytic maximum. R = R0 (1 + 0.5 sin(kx) cos(ky)).
STEEP_FRACTION_OF_MAX = 0.5


def rate_gradient_magnitude(x, y) -> np.ndarray:
    kx = 2.0 * np.pi / 1500.0
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    drdx = R0_M_S * 0.5 * kx * np.cos(kx * xx) * np.cos(kx * yy)
    drdy = -R0_M_S * 0.5 * kx * np.sin(kx * xx) * np.sin(kx * yy)
    return np.sqrt(drdx ** 2 + drdy ** 2)


def steep_threshold() -> float:
    kx = 2.0 * np.pi / 1500.0
    return STEEP_FRACTION_OF_MAX * R0_M_S * 0.5 * kx


def longest_contour(phi, x, y) -> np.ndarray | None:
    lines = [line for line in zero_contours(phi, x, y) if len(line) >= 8]
    if not lines:
        return None
    return max(lines, key=len)


def radius_by_angle(points: np.ndarray, x0: float, y0: float, n: int = 360) -> np.ndarray:
    """Outermost radius in each angular bin around the ignition point. NaN if empty."""
    ang = np.arctan2(points[:, 1] - y0, points[:, 0] - x0)
    rad = np.hypot(points[:, 0] - x0, points[:, 1] - y0)
    bins = np.floor((ang + np.pi) / (2.0 * np.pi) * n).astype(int) % n
    profile = np.full(n, np.nan, dtype=np.float64)
    for index, radius in zip(bins, rad):
        if np.isnan(profile[index]) or radius > profile[index]:
            profile[index] = float(radius)
    good = np.isfinite(profile)
    if int(good.sum()) < n // 4:
        return profile
    grid = np.arange(n)
    profile[~good] = np.interp(grid[~good], grid[good], profile[good], period=n)
    return profile


def spectrum_fraction(profile: np.ndarray, high_mode: int = HIGH_MODE) -> dict:
    """Share of demeaned angular power in modes at and above ``high_mode``."""
    finite = np.isfinite(profile)
    if int(finite.sum()) < 16:
        return {"high_frequency_fraction": None, "rms_radius_m": None}
    centered = profile - np.mean(profile[finite])
    centered = np.where(finite, centered, 0.0)
    power = np.abs(np.fft.rfft(centered)) ** 2
    alternating = float(power[1:].sum())
    if alternating <= 0:
        fraction = 0.0
    else:
        fraction = float(power[high_mode:].sum() / alternating)
    return {
        "high_frequency_fraction": fraction,
        "rms_radius_m": float(np.sqrt(np.mean(centered ** 2))),
    }


def polyline_curvature_rms(points: np.ndarray) -> float | None:
    """RMS curvature of a closed polyline, in 1/m."""
    if len(points) < 8:
        return None
    closed = np.vstack([points, points[0]])
    edge = np.diff(closed, axis=0)
    length = np.hypot(edge[:, 0], edge[:, 1])
    keep = length > 1e-6
    if int(keep.sum()) < 8:
        return None
    tangent = edge[keep] / length[keep, None]
    turn = np.arctan2(
        tangent[:-1, 0] * tangent[1:, 1] - tangent[:-1, 1] * tangent[1:, 0],
        tangent[:-1, 0] * tangent[1:, 0] + tangent[:-1, 1] * tangent[1:, 1],
    )
    step = 0.5 * (length[keep][:-1] + length[keep][1:])
    curvature = turn / np.maximum(step, 1e-6)
    return float(np.sqrt(np.mean(curvature ** 2)))


def component_areas(phi, x, y) -> dict:
    burned = np.asarray(phi) < 0
    labels, count = label(burned)
    cell = float(abs(x[1] - x[0]) * abs(y[1] - y[0]))
    areas = sorted((float((labels == index).sum() * cell) for index in range(1, count + 1)), reverse=True)
    return {
        "burned_components": int(count),
        "component_areas_m2": areas,
        "largest_component_m2": areas[0] if areas else 0.0,
        "other_components_m2": float(sum(areas[1:])) if len(areas) > 1 else 0.0,
    }


def angular_comparison(predicted_phi, reference_phi, x, y, x0: float, y0: float, r0: float, t_final: float) -> dict:
    pred = longest_contour(predicted_phi, x, y)
    ref = longest_contour(reference_phi, x, y)
    if pred is None or ref is None:
        return {"available": False}
    pred_r = radius_by_angle(pred, x0, y0)
    ref_r = radius_by_angle(ref, x0, y0)
    signed = pred_r - ref_r
    angles = np.linspace(-np.pi, np.pi, len(signed), endpoint=False)
    local_r = spatial_rate_np(x0 + ref_r * np.cos(angles), y0 + ref_r * np.sin(angles))
    pred_speed = (pred_r - r0) / t_final
    ref_speed = (ref_r - r0) / t_final
    over = signed > 0
    return {
        "available": True,
        "angles_rad": angles,
        "signed_displacement_m": signed,
        "predicted_radius_m": pred_r,
        "reference_radius_m": ref_r,
        "mean_signed_displacement_m": float(np.nanmean(signed)),
        "mean_overshoot_m": float(np.nanmean(signed[over])) if np.any(over) else 0.0,
        "mean_undershoot_m": float(np.nanmean(signed[~over])) if np.any(~over) else 0.0,
        "overshoot_fraction": float(np.mean(over)),
        "mean_speed_error_m_s": float(np.nanmean(pred_speed - ref_speed)),
        "mean_speed_minus_local_R_m_s": float(np.nanmean(pred_speed - local_r)),
        "high_R_mean_displacement_m": float(np.nanmean(signed[local_r >= R0_M_S])),
        "low_R_mean_displacement_m": float(np.nanmean(signed[local_r < R0_M_S])),
        "predicted_spectrum": spectrum_fraction(pred_r),
        "reference_spectrum": spectrum_fraction(ref_r),
        "predicted_curvature_rms_per_m": polyline_curvature_rms(pred),
        "reference_curvature_rms_per_m": polyline_curvature_rms(ref),
    }


class PredictedFrontCache:
    """Detached samples of this model's zero contour. No reference contour."""

    def __init__(self):
        self.xy = None
        self.time = None

    def refresh(self, model, problem, phi0_fn, grid_x, grid_y, times, predict):
        coords = []
        stamps = []
        for moment in times:
            field = predict(model, problem, phi0_fn, moment, grid_x, grid_y)
            for line in zero_contours(field, grid_x, grid_y):
                if len(line) >= 2:
                    coords.append(line)
                    stamps.append(np.full(len(line), moment, dtype=np.float64))
        if not coords:
            self.xy = None
            self.time = None
            return
        self.xy = torch.tensor(np.concatenate(coords), dtype=torch.float32).detach()
        self.time = torch.tensor(np.concatenate(stamps), dtype=torch.float32).detach()


def _reject_center(x, y, problem):
    return torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m


def _steep_points(n: int, problem, device):
    threshold = steep_threshold()
    xs, ys, ts = [], [], []
    got = 0
    while got < n:
        x, y, t = _sample_pde(max(n * 4, 1024), problem, device)
        grad = torch.tensor(rate_gradient_magnitude(x.cpu().numpy(), y.cpu().numpy()), dtype=torch.float32, device=device)
        keep = grad >= threshold
        xs.append(x[keep])
        ys.append(y[keep])
        ts.append(t[keep])
        got += int(keep.sum())
    return torch.cat(xs)[:n], torch.cat(ys)[:n], torch.cat(ts)[:n]


def sample_mixture(n: int, problem, cache: PredictedFrontCache, device):
    """50% uniform, 40% near the predicted front, 10% where |grad R| is large."""
    n_steep = max(1, int(round(0.10 * n)))
    n_uniform = int(round(0.50 * n))
    n_front = n - n_uniform - n_steep
    x_u, y_u, t_u = _sample_pde(n_uniform, problem, device)
    x_s, y_s, t_s = _steep_points(n_steep, problem, device)
    if cache.xy is None or len(cache.xy) < 8:
        x_f, y_f, t_f = _sample_pde(n_front, problem, device)
    else:
        choice = torch.randint(0, len(cache.xy), (n_front,))
        base = cache.xy[choice].to(device)
        jitter = (torch.rand(n_front, 2, device=device) * 2.0 - 1.0) * FRONT_JITTER_M
        xy = (base + jitter).detach()
        x_f = xy[:, 0]
        y_f = xy[:, 1]
        t_f = cache.time[choice].to(device).detach()
        inside = (
            (x_f >= -problem.half_extent) & (x_f <= problem.half_extent)
            & (y_f >= -problem.half_extent) & (y_f <= problem.half_extent)
            & _reject_center(x_f, y_f, problem)
        )
        if int(inside.sum()) < n_front // 4:
            x_f, y_f, t_f = _sample_pde(n_front, problem, device)
        else:
            x_f, y_f, t_f = x_f[inside], y_f[inside], t_f[inside]
            while x_f.shape[0] < n_front:
                x_f = torch.cat([x_f, x_f])
                y_f = torch.cat([y_f, y_f])
                t_f = torch.cat([t_f, t_f])
            x_f, y_f, t_f = x_f[:n_front], y_f[:n_front], t_f[:n_front]
    return (
        torch.cat([x_u, x_f, x_s]).detach(),
        torch.cat([y_u, y_f, y_s]).detach(),
        torch.cat([t_u, t_f, t_s]).detach(),
    )
