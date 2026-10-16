"""Hard initial-condition parameterizations for the circular benchmark.

Normalized output, matching Experiment 1:

    soft:     Phi = network(tau, X, Y)
    hard IC:  Phi = phi0 / L + tau * network(tau, X, Y)
    R-scaled: Phi = phi0 / L + tau * (R T / L) * network(tau, X, Y)

phi0 is the exact initial signed distance. Its spatial derivatives stay in the
graph. The ignition center is excluded from PDE samples because phi0 is not
differentiable there. The R-scaled form is a structural constraint: R = 0 makes
Phi identically phi0, so a perfect stationary score is not evidence that the
network learned the PDE.
"""

from __future__ import annotations

import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem


def initial_phi_normalized(x_norm: torch.Tensor, y_norm: torch.Tensor, problem: CircleProblem) -> torch.Tensor:
    """Exact phi(x, y, 0) / L. Depends on the coordinate tensors so autograd sees it."""
    length = problem.length
    dx = x_norm + (problem.center_x - problem.x0) / length
    dy = y_norm + (problem.center_y - problem.y0) / length
    return torch.sqrt(dx * dx + dy * dy) - problem.r0 / length


def normalized_phi(raw: torch.Tensor, tau, x_norm, y_norm, problem: CircleProblem, kind: str, rate: float) -> torch.Tensor:
    if kind == "soft":
        return raw
    phi0 = initial_phi_normalized(x_norm, y_norm, problem)
    if kind == "hard_ic":
        return phi0 + tau * raw
    if kind == "r_scaled":
        scale = float(rate) * problem.t_final / problem.length
        return phi0 + tau * scale * raw
    raise ValueError(f"unknown ansatz {kind}")
