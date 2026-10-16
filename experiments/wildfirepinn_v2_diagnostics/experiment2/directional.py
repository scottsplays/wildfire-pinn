"""Directional level-set step: isotropic expansion plus advection.

The Experiment 1 solver advances phi_t + R |grad phi| = 0 for R >= 0.
A wind-aligned speed with c < 1 and a unit blowing-toward vector w is

    phi_t + R |grad phi| + (c R w) · grad phi = 0.

The extra term advects level sets at velocity v = c R w. Upwinding looks
opposite that velocity. This is not the isotropic solver with a substituted
normal speed: the normal is not known as a grid field before the differences
are chosen. c >= 1 is refused because the upwind side then changes where
1 + c w · n drops through zero.
"""

from __future__ import annotations

import numpy as np

from experiments.wildfirepinn_v2_diagnostics.solver import (
    _as_rate,
    _one_sided,
    cfl_timestep,
    godunov_gradient_magnitude,
)


def _require_unit(wind: tuple[float, float]) -> tuple[float, float]:
    east, north = float(wind[0]), float(wind[1])
    norm = float(np.hypot(east, north))
    if abs(norm - 1.0) > 1e-3:
        raise ValueError(f"wind must be a unit blowing-toward vector, got norm {norm}")
    return east, north


def directional_timestep(dx: float, dy: float, rate_max: float, c_wind: float, cfl: float) -> float:
    """CFL step for expansion at ``rate_max`` plus advection at speed c * rate_max."""
    if c_wind < 0 or c_wind >= 1:
        raise ValueError("directional advection split requires 0 <= c < 1")
    base = cfl_timestep(dx, dy, rate_max, cfl)
    if c_wind == 0.0 or not np.isfinite(base):
        return base
    advection = c_wind * float(rate_max) * (1.0 / float(dx) + 1.0 / float(dy))
    expansion_rate = cfl / base
    return float(cfl) / (expansion_rate + advection)


def propagate_directional(
    phi0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rate,
    times,
    wind_at,
    c_wind: float,
    cfl: float = 0.4,
    switch_times=(),
) -> dict[float, np.ndarray]:
    """Advance phi_t + R|grad phi| + c R w(t) · grad phi = 0.

    ``wind_at(t)`` returns the blowing-toward unit vector at time t, in seconds.
    ``switch_times`` are instants where that vector jumps; steps stop on them.
    """
    if c_wind < 0 or c_wind >= 1:
        raise ValueError("directional advection split requires 0 <= c < 1")
    phi0 = np.asarray(phi0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if phi0.shape != (len(y), len(x)):
        raise ValueError(f"phi shape {phi0.shape} does not match grid {(len(y), len(x))}")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    speed = _as_rate(rate, phi0.shape)
    _require_unit(wind_at(0.0))
    events = {float(t) for t in times} | {float(t) for t in switch_times}
    if any(t < 0 for t in events):
        raise ValueError("times must be nonnegative")
    step = directional_timestep(dx, dy, float(speed.max()), c_wind, cfl)
    stored: dict[float, np.ndarray] = {}
    targets = set(float(t) for t in times)
    if 0.0 in targets:
        stored[0.0] = phi0.copy()
    field = phi0.copy()
    current = 0.0
    ordered = [t for t in sorted(events) if t > 0]
    for event in ordered:
        while current < event - 1e-12:
            dt = min(step, event - current) if np.isfinite(step) else event - current
            if not np.isfinite(step):
                current = event
                break
            east, north = _require_unit(wind_at(current))
            dx_m, dx_p = _one_sided(field, dx, axis=1)
            dy_m, dy_p = _one_sided(field, dy, axis=0)
            magnitude = godunov_gradient_magnitude(field, dx, dy, positive_speed=True)
            velocity_x = c_wind * speed * east
            velocity_y = c_wind * speed * north
            up_x = np.where(velocity_x >= 0.0, dx_m, dx_p)
            up_y = np.where(velocity_y >= 0.0, dy_m, dy_p)
            field = field - dt * (speed * magnitude + velocity_x * up_x + velocity_y * up_y)
            current += dt
        if event in targets:
            stored[event] = field.copy()
    return stored
