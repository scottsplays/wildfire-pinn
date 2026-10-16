"""Fit a rate from observed perimeters, then leave forecasting to the solver.

The hidden true rate is not an argument of any function in this module.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy.optimize import minimize

from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import contour_distances
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from experiments.wildfirepinn_v3.experiment3.models import RateMLP, pack_rate, parametric_rate, unpack_rate
from experiments.wildfirepinn_v3.experiment3.observations import FORBIDDEN_TIMES, mean_radius, segment_speeds
from experiments.wildfirepinn_v3.experiment3.torch_solver import adjoint_rate_gradient, dt_for_interval, observe_rollout


def constant_rate(view: dict, x0: float, y0: float) -> float:
    """One global speed from the first and last training perimeters."""
    times = view["times"]
    start = mean_radius(view["fields"][times[0]], view["x"], view["y"], x0, y0)
    end = mean_radius(view["fields"][times[-1]], view["x"], view["y"], x0, y0)
    duration = float(times[-1] - times[0])
    if duration <= 0:
        raise ValueError("the training window has no duration")
    return float(np.clip((end - start) / duration, 1e-4, 0.15))


def _mean_symmetric(rate: np.ndarray, view: dict) -> float:
    times = [time for time in view["times"] if time > 0.0]
    solved = propagate(view["phi0"], view["x"], view["y"], rate, times)
    total = 0.0
    for moment in times:
        distances = contour_distances(solved[moment], view["fields"][moment], view["x"], view["y"])
        if distances["symmetric_m"] is None:
            return 1.0e6
        total += float(distances["symmetric_m"])
    return total / len(times)


def fit_parametric(view: dict, xx, yy, config: dict, mode: str, r0_init: float, amplitude_init: float) -> dict:
    wavelength = float(config["diagnostic_wavelength_m"])
    r_min = float(config["r_min_m_s"])
    r_max = float(config["r_max_m_s"])
    amplitude_max = float(config["amplitude_max"])
    theta0 = pack_rate(r0_init, amplitude_init, r_min, r_max, amplitude_max)
    evaluations = {"count": 0}

    def objective(theta: np.ndarray) -> float:
        evaluations["count"] += 1
        rate = parametric_rate(theta, xx, yy, wavelength, mode, r_min, r_max, amplitude_max)
        if np.any(rate < 0):
            return 1.0e6
        return _mean_symmetric(rate, view)

    started = time.perf_counter()
    result = minimize(
        objective,
        theta0,
        method="Nelder-Mead",
        options={"maxfev": int(config["nelder_mead_maxfev"]), "xatol": 1e-3, "fatol": 0.05},
    )
    elapsed = time.perf_counter() - started
    r0, amplitude = unpack_rate(result.x, r_min, r_max, amplitude_max)
    rate = parametric_rate(result.x, xx, yy, wavelength, mode, r_min, r_max, amplitude_max)
    return {
        "rate": rate,
        "r0": r0,
        "amplitude": amplitude,
        "train_symmetric_m": float(result.fun),
        "evaluations": int(evaluations["count"]),
        "train_seconds": elapsed,
        "success": bool(result.success),
        "mode": mode,
    }


def ray_rate(view: dict, x0: float, y0: float, n_bins: int) -> dict:
    """Paint each ray with the observed radial speeds, extrapolated outside the swept radii."""
    segments = segment_speeds(view, x0, y0, n_bins)
    yy, xx = np.meshgrid(view["y"], view["x"], indexing="ij")
    angle = np.arctan2(yy - y0, xx - x0)
    radius = np.hypot(xx - x0, yy - y0)
    bins = np.floor((angle + np.pi) / (2.0 * np.pi) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    painted = np.empty(xx.shape, dtype=np.float64)
    clipped = 0
    for index in range(n_bins):
        samples_r = []
        samples_v = []
        for segment in segments:
            radius_mid = segment["radius_mid_m"][index]
            speed = segment["speed_m_s"][index]
            if np.isfinite(radius_mid) and np.isfinite(speed):
                samples_r.append(float(radius_mid))
                samples_v.append(float(speed))
        mask = bins == index
        if not samples_r:
            painted[mask] = 1.0e-3
            clipped += int(mask.sum())
            continue
        order = np.argsort(samples_r)
        radii = np.asarray(samples_r, dtype=np.float64)[order]
        speeds = np.asarray(samples_v, dtype=np.float64)[order]
        # Identical radii (a stationary bin) collapse to one sample.
        keep = np.concatenate([[True], np.diff(radii) > 1e-6])
        values = np.interp(radius[mask], radii[keep], speeds[keep])
        low = values < 1.0e-3
        high = values > 0.15
        clipped += int(np.sum(low | high))
        painted[mask] = np.clip(values, 1.0e-3, 0.15)
    return {
        "rate": painted,
        "segments": segments,
        "clipped_cells": int(clipped),
        "train_seconds": 0.0,
    }


def _sample_phi(phi: torch.Tensor, pts: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    gx = (pts[:, 0] - x[0]) / (x[-1] - x[0]) * 2.0 - 1.0
    gy = (pts[:, 1] - y[0]) / (y[-1] - y[0]) * 2.0 - 1.0
    grid = torch.stack((gx, gy), dim=-1).view(1, -1, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        phi.view(1, 1, phi.shape[0], phi.shape[1]),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.view(-1)


def train_rate_network(view: dict, config: dict, r_init: float, seed: int) -> dict:
    """Unroll phi_t + R_theta|grad phi| = 0 and match the observed training perimeters."""
    from wildfire_pinn.geometry import zero_contours

    times = [float(t) for t in view["times"]]
    if any(t in FORBIDDEN_TIMES for t in times):
        raise ValueError("refusing to train on a forecast time")
    x = torch.tensor(view["x"], dtype=torch.float64)
    y = torch.tensor(view["y"], dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    contours = {}
    for moment in times:
        if moment == 0.0:
            continue
        lines = [line for line in zero_contours(view["fields"][moment], view["x"], view["y"]) if len(line) >= 8]
        if not lines:
            raise RuntimeError(f"no observed contour at t = {moment}")
        pts = max(lines, key=lambda item: len(item))
        contours[moment] = torch.tensor(pts, dtype=torch.float64)
    dx = float(view["x"][1] - view["x"][0])
    dy = float(view["y"][1] - view["y"][0])
    # Training times in this experiment are integer multiples of 500 s.
    dt = dt_for_interval(500.0, dx, dy, float(config["r_max_m_s"]), float(config["cfl"]))
    model = RateMLP(
        int(config["mlp_hidden"]),
        int(config["mlp_layers"]),
        r_init,
        float(config["r_max_m_s"]),
        float(view["x"][-1] - view["x"][0]),
        seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["mlp_lr"]))
    phi0 = torch.tensor(view["phi0"], dtype=torch.float64)
    scale = float(config["loss_length_m"])
    history = []
    started = time.perf_counter()
    model.train()
    positive = sorted(moment for moment in times if moment > 0.0)
    n_steps = int(round(positive[-1] / dt))
    observed = {int(round(moment / dt)): moment for moment in positive}

    def loss_grad(step: int, phi: torch.Tensor):
        moment = observed.get(step)
        if moment is None:
            return None
        phi_var = phi.detach().requires_grad_(True)
        loss = torch.mean((_sample_phi(phi_var, contours[moment], x, y) / scale) ** 2) / len(contours)
        return torch.autograd.grad(loss, phi_var)[0]

    for epoch in range(1, int(config["mlp_epochs"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        rate = model(xx, yy)
        gradient = adjoint_rate_gradient(phi0, rate, dx, dy, dt, n_steps, loss_grad)
        if not bool(torch.isfinite(gradient).all()):
            print(f"mlp seed {seed} stopped on a non-finite adjoint at epoch {epoch}", flush=True)
            break
        rate.backward(gradient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
        optimizer.step()
        with torch.no_grad():
            solved = observe_rollout(phi0, rate, dx, dy, dt, observed)
            pieces = [
                torch.mean((_sample_phi(solved[moment], contours[moment], x, y) / scale) ** 2)
                for moment in positive
            ]
            history.append(float(torch.stack(pieces).mean()))
        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["mlp_epochs"]):
            print(
                f"mlp seed {seed} epoch {epoch:3d} | contour loss {history[-1]:.4e} | "
                f"R mean {float(rate.detach().mean()):.4f} std {float(rate.detach().std()):.4f}",
                flush=True,
            )
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        learned = model(xx, yy).cpu().numpy()
    return {"model": model, "rate": learned, "history": history, "train_seconds": elapsed, "dt_s": dt}
