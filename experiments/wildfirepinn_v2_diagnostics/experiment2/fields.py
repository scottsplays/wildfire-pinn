"""Prescribed propagation speeds for Experiment 2.

R is known and nonnegative. These fields are inputs, not learned rates.
The spatial field is

    R(x, y) = R0 * (1 + a * sin(2 pi x / wavelength) * cos(2 pi y / wavelength))

with R0 = 0.05 m/s, a = 0.5, and wavelength = 1500 m. Then R stays in
[0.025, 0.075] m/s. Wind, when present, is the direction the wind blows
toward, matching the Kincade residual, and the factor is

    R_n = R_base * max(0, 1 + c * w_hat · n),  c = 0.6.

For a unit wind and c < 1 the max does not bind, and

    R_n |grad phi| = R_base |grad phi| + c R_base w_hat · grad phi.
"""

from __future__ import annotations

import math

import numpy as np
import torch

R0_M_S = 0.05
SPATIAL_AMPLITUDE = 0.5
WAVELENGTH_M = 1500.0
WIND_C = 0.6


def _wave_numbers() -> tuple[float, float]:
    k = 2.0 * math.pi / WAVELENGTH_M
    return k, k


def spatial_rate_np(x, y) -> np.ndarray:
    kx, ky = _wave_numbers()
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    return R0_M_S * (1.0 + SPATIAL_AMPLITUDE * np.sin(kx * xx) * np.cos(ky * yy))


def spatial_rate_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    kx, ky = _wave_numbers()
    return R0_M_S * (1.0 + SPATIAL_AMPLITUDE * torch.sin(kx * x) * torch.cos(ky * y))


def constant_wind(east: float, north: float):
    """Return a schedule w(t) = (east, north). The vector must be a unit vector."""

    def schedule(t: float) -> tuple[float, float]:
        del t
        return float(east), float(north)

    return schedule


def switching_wind(t_switch: float, first: tuple[float, float], second: tuple[float, float]):
    """Use ``first`` for t < t_switch and ``second`` afterward. Times are seconds."""

    def schedule(t: float) -> tuple[float, float]:
        return first if float(t) < float(t_switch) else second

    return schedule


def wind_torch(t: torch.Tensor, t_switch: float | None, first, second) -> tuple[torch.Tensor, torch.Tensor]:
    if t_switch is None:
        return torch.full_like(t, first[0]), torch.full_like(t, first[1])
    east = torch.where(t < t_switch, torch.full_like(t, first[0]), torch.full_like(t, second[0]))
    north = torch.where(t < t_switch, torch.full_like(t, first[1]), torch.full_like(t, second[1]))
    return east, north
