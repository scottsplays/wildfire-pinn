"""Differentiable Godunov step for phi_t + R|grad phi| = 0.

R is a nonnegative field in m/s. The update matches solver.godunov_gradient_magnitude
for one step. The public NumPy integrator chooses its own time step, so a
multi-step Torch rollout is the same scheme with a caller-chosen dt, not a
bitwise copy of every NumPy step.
"""

from __future__ import annotations

import torch


class _ZeroAtOriginSqrt(torch.autograd.Function):
    """sqrt(u) with derivative 0 at u = 0.

    The upwind magnitude is zero at a discrete minimum. The usual derivative
    1/(2 sqrt(u)) is infinite there and turns the adjoint into NaN.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(value)
        return torch.sqrt(value)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (value,) = ctx.saved_tensors
        derivative = torch.zeros_like(value)
        mask = value > 0
        derivative[mask] = grad[mask] / (2.0 * torch.sqrt(value[mask]))
        return derivative


def godunov_magnitude(phi: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Upwind |grad phi| for R >= 0. phi is (ny, nx) in meters."""
    dx_m = torch.zeros_like(phi)
    dx_p = torch.zeros_like(phi)
    dx_m[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / dx
    dx_p[:, :-1] = (phi[:, 1:] - phi[:, :-1]) / dx
    dx_m[:, 0] = dx_p[:, 0]
    dx_p[:, -1] = dx_m[:, -1]
    dy_m = torch.zeros_like(phi)
    dy_p = torch.zeros_like(phi)
    dy_m[1:, :] = (phi[1:, :] - phi[:-1, :]) / dy
    dy_p[:-1, :] = (phi[1:, :] - phi[:-1, :]) / dy
    dy_m[0, :] = dy_p[0, :]
    dy_p[-1, :] = dy_m[-1, :]
    x_term = torch.clamp(dx_m, min=0.0) ** 2 + torch.clamp(dx_p, max=0.0) ** 2
    y_term = torch.clamp(dy_m, min=0.0) ** 2 + torch.clamp(dy_p, max=0.0) ** 2
    return _ZeroAtOriginSqrt.apply(x_term + y_term)


def godunov_step(phi: torch.Tensor, rate: torch.Tensor, dx: float, dy: float, dt: float) -> torch.Tensor:
    return phi - dt * rate * godunov_magnitude(phi, dx, dy)


def rollout(phi0: torch.Tensor, rate: torch.Tensor, dx: float, dy: float, times: list[float], dt: float) -> dict[float, torch.Tensor]:
    """Advance from t = 0 to each positive time. dt must divide every interval."""
    stored = {0.0: phi0}
    phi = phi0
    current = 0.0
    for target in times:
        if target <= current + 1e-12:
            continue
        span = float(target) - current
        steps = int(round(span / dt))
        if steps < 1 or abs(steps * dt - span) > 1e-6:
            raise ValueError(f"dt {dt} does not divide the interval ending at {target}")
        for _ in range(steps):
            phi = godunov_step(phi, rate, dx, dy, dt)
        current = float(target)
        stored[current] = phi
    return stored


def stable_dt(dx: float, dy: float, rate_max: float, cfl: float) -> float:
    import math

    return float(cfl) * min(dx, dy) / (math.sqrt(2.0) * float(rate_max))


def dt_for_interval(interval: float, dx: float, dy: float, rate_max: float, cfl: float) -> float:
    limit = stable_dt(dx, dy, rate_max, cfl)
    steps = max(1, int(math_ceil(interval / limit)))
    return float(interval) / steps


def observe_rollout(phi0: torch.Tensor, rate: torch.Tensor, dx: float, dy: float, dt: float, observed: dict[int, float]) -> dict[float, torch.Tensor]:
    """Forward Godunov states at the requested step indices. No autograd graph."""
    with torch.no_grad():
        current = phi0.detach()
        saved: dict[float, torch.Tensor] = {}
        last = max(observed)
        for step in range(1, last + 1):
            current = godunov_step(current, rate.detach(), dx, dy, dt)
            if step in observed:
                saved[observed[step]] = current
        return saved


def adjoint_rate_gradient(phi0: torch.Tensor, rate: torch.Tensor, dx: float, dy: float, dt: float, n_steps: int, loss_grad) -> torch.Tensor:
    """Exact discrete adjoint of the Godunov rollout.

    ``loss_grad(step, phi)`` returns dL/dphi at that saved step, or None.
    Step 0 is the initial field. The graph never holds more than one update.
    """
    with torch.no_grad():
        current = phi0.detach()
        saved = [current]
        for _ in range(n_steps):
            current = godunov_step(current, rate.detach(), dx, dy, dt)
            saved.append(current)
    lam = torch.zeros_like(phi0)
    gradient = torch.zeros_like(rate)
    if not torch.isfinite(saved[-1]).all():
        raise RuntimeError("forward field is not finite")
    for step in range(n_steps - 1, -1, -1):
        incoming = loss_grad(step + 1, saved[step + 1])
        if incoming is not None:
            if not torch.isfinite(incoming).all():
                raise RuntimeError(f"observation gradient is not finite at step {step + 1}")
            lam = lam + incoming
        phi_in = saved[step].detach().requires_grad_(True)
        rate_in = rate.detach().requires_grad_(True)
        phi_next = godunov_step(phi_in, rate_in, dx, dy, dt)
        grad_phi, grad_rate = torch.autograd.grad(phi_next, (phi_in, rate_in), grad_outputs=lam)
        if grad_phi is None or not torch.isfinite(grad_phi).all() or not torch.isfinite(grad_rate).all():
            raise RuntimeError(f"adjoint is not finite at step {step}")
        lam = grad_phi.detach()
        gradient = gradient + grad_rate.detach()
    return gradient


def math_ceil(value: float) -> int:
    import math

    return int(math.ceil(value - 1e-12))
