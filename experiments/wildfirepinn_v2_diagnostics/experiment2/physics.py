"""Hard initial condition and the nondimensional level-set residual.

Phi = phi0 / L + tau * g_theta. phi0 stays in the autograd graph.
The coded residual is (T / L) times phi_t + R_n |grad phi|, in the same
normalization as the Kincade trainer. For c < 1,

    R_n |grad phi| = R |grad phi| + c R w · grad phi.
"""

from __future__ import annotations

import torch

from experiments.wildfirepinn_v2_diagnostics.ansatz import initial_phi_normalized
from experiments.wildfirepinn_v2_diagnostics.experiment2.sdf import sample_sdf_normalized


def hard_ic(raw: torch.Tensor, tau: torch.Tensor, phi0_normalized: torch.Tensor) -> torch.Tensor:
    return phi0_normalized + tau * raw


def circle_phi0(x_norm, y_norm, problem):
    return initial_phi_normalized(x_norm, y_norm, problem)


def grid_phi0(x_norm, y_norm, sdf, grid_x, grid_y, problem):
    return sample_sdf_normalized(
        x_norm, y_norm, sdf, grid_x, grid_y, problem.length, problem.center_x, problem.center_y
    )


def coded_residual(phi_t, phi_x, phi_y, rate, t_over_l: float, wind_east=None, wind_north=None, c_wind: float = 0.0):
    """Dimensionless residual. ``rate`` is m/s and may vary per point. ``t_over_l`` is T/L."""
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)
    coded = phi_t + (rate * t_over_l) * magnitude
    if c_wind != 0.0:
        if wind_east is None or wind_north is None:
            raise ValueError("wind components are required when c is nonzero")
        coded = coded + (c_wind * rate * t_over_l) * (wind_east * phi_x + wind_north * phi_y)
    physical = coded / t_over_l
    return coded, physical, magnitude
