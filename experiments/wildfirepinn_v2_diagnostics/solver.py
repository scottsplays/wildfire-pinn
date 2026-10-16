"""First-order Godunov upwind integrator for phi_t + R |grad phi| = 0.

R is a nonnegative outward normal speed in m/s. phi is negative inside the
burned region, so the zero contour moves in the direction of grad phi.
Coordinates are meters. The same R definition is what the PINN residual uses.

The scheme is the Osher-Sethian upwind Hamiltonian for positive speed:

    |grad phi|_up = sqrt(
        max(D-x, 0)^2 + min(D+x, 0)^2 + max(D-y, 0)^2 + min(D+y, 0)^2
    )
    phi <- phi - dt * R * |grad phi|_up

At a discrete minimum, such as the center of a signed-distance circle, this
upwind gradient is zero, so that single cell does not deepen. The front still
moves where the gradient is resolved. Do not read a small front error as a
small pointwise error at that kink.
"""

from __future__ import annotations

import numpy as np


def _one_sided(phi: np.ndarray, spacing: float, axis: int) -> tuple[np.ndarray, np.ndarray]:
    backward = np.empty_like(phi)
    forward = np.empty_like(phi)
    if axis == 1:
        backward[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / spacing
        forward[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / spacing
        backward[:, 0] = forward[:, 0]
        forward[:, -1] = backward[:, -1]
    elif axis == 0:
        backward[1:, :] = (phi[1:, :] - phi[:-1, :]) / spacing
        forward[:-1, :] = (phi[1:, :] - phi[:-1, :]) / spacing
        backward[0, :] = forward[0, :]
        forward[-1, :] = backward[-1, :]
    else:
        raise ValueError("axis must be 0 or 1")
    return backward, forward


def godunov_gradient_magnitude(phi: np.ndarray, dx: float, dy: float, positive_speed: bool = True) -> np.ndarray:
    """Upwind |grad phi| in 1/m. ``positive_speed`` is the outward (R >= 0) switch."""
    dx_m, dx_p = _one_sided(phi, dx, axis=1)
    dy_m, dy_p = _one_sided(phi, dy, axis=0)
    if positive_speed:
        x_term = np.maximum(dx_m, 0.0) ** 2 + np.minimum(dx_p, 0.0) ** 2
        y_term = np.maximum(dy_m, 0.0) ** 2 + np.minimum(dy_p, 0.0) ** 2
    else:
        x_term = np.maximum(dx_p, 0.0) ** 2 + np.minimum(dx_m, 0.0) ** 2
        y_term = np.maximum(dy_p, 0.0) ** 2 + np.minimum(dy_m, 0.0) ** 2
    return np.sqrt(x_term + y_term)


def cfl_timestep(dx: float, dy: float, rate_max: float, cfl: float) -> float:
    if rate_max < 0:
        raise ValueError("R must be nonnegative for this upwind switch")
    if rate_max == 0:
        return float("inf")
    if cfl <= 0 or cfl > 1:
        raise ValueError("CFL number must be in (0, 1]")
    return float(cfl) * min(float(dx), float(dy)) / (np.sqrt(2.0) * float(rate_max))


def _as_rate(rate, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(rate, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(shape, float(values), dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f"R shape {values.shape} does not match phi shape {shape}")
    if np.any(values < 0):
        raise ValueError("R must be nonnegative")
    return values


def propagate(
    phi0: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    rate,
    times,
    cfl: float = 0.4,
) -> dict[float, np.ndarray]:
    """Advance phi to each requested time. Times are seconds from the initial field.

    ``x`` and ``y`` must be increasing, equally spaced, and in meters.
    Returns a dict of copies keyed by the requested times, including 0 when asked.
    """
    phi0 = np.asarray(phi0, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if phi0.shape != (len(y), len(x)):
        raise ValueError(f"phi shape {phi0.shape} does not match grid {(len(y), len(x))}")
    if len(x) < 3 or len(y) < 3:
        raise ValueError("grid must have at least 3 nodes on each axis")
    dx = np.diff(x)
    dy = np.diff(y)
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("x and y must be strictly increasing")
    if np.max(np.abs(dx - dx[0])) > 1e-6 * dx[0] or np.max(np.abs(dy - dy[0])) > 1e-6 * dy[0]:
        raise ValueError("this solver expects uniform spacing")
    spacing_x = float(dx[0])
    spacing_y = float(dy[0])
    speed = _as_rate(rate, phi0.shape)
    targets = sorted({float(t) for t in times})
    if any(t < 0 for t in targets):
        raise ValueError("times must be nonnegative")
    step = cfl_timestep(spacing_x, spacing_y, float(speed.max()), cfl)
    stored: dict[float, np.ndarray] = {}
    if 0.0 in targets:
        stored[0.0] = phi0.copy()
    field = phi0.copy()
    current = 0.0
    for target in targets:
        if target == 0.0:
            continue
        if not np.isfinite(step):
            stored[target] = field.copy()
            continue
        while current < target - 1e-12:
            dt = min(step, target - current)
            magnitude = godunov_gradient_magnitude(field, spacing_x, spacing_y, positive_speed=True)
            field = field - dt * speed * magnitude
            current += dt
        stored[target] = field.copy()
    return stored
