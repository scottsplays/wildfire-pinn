"""Positive rate models. None of these modules reads the hidden true field."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from wildfire_pinn.pinn import set_seed


def sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def unpack_rate(theta: np.ndarray, r_min: float, r_max: float, amplitude_max: float) -> tuple[float, float]:
    fraction = float(sigmoid(theta[0]))
    r0 = float(r_min + (r_max - r_min) * fraction)
    amplitude = float(amplitude_max * np.tanh(theta[1]))
    return r0, amplitude


def pack_rate(r0: float, amplitude: float, r_min: float, r_max: float, amplitude_max: float) -> np.ndarray:
    fraction = (float(r0) - r_min) / (r_max - r_min)
    fraction = float(np.clip(fraction, 1e-4, 1.0 - 1e-4))
    raw_r = math.log(fraction / (1.0 - fraction))
    raw_a = math.atanh(float(np.clip(amplitude / amplitude_max, -0.999, 0.999)))
    return np.array([raw_r, raw_a], dtype=np.float64)


def parametric_rate(theta, xx, yy, wavelength: float, mode: str, r_min: float, r_max: float, amplitude_max: float) -> np.ndarray:
    """Low-dimensional family. ``product`` is sin(kx) cos(ky). ``x-only`` drops y."""
    r0, amplitude = unpack_rate(np.asarray(theta, dtype=np.float64), r_min, r_max, amplitude_max)
    wave = 2.0 * math.pi / float(wavelength)
    if mode == "product":
        shape = np.sin(wave * xx) * np.cos(wave * yy)
    elif mode == "x-only":
        shape = np.sin(wave * xx)
    else:
        raise ValueError(mode)
    return r0 * (1.0 + amplitude * shape)


class RateMLP(nn.Module):
    """R = R_max * sigmoid(net(x/L, y/L)). The output stays in (0, R_max)."""

    def __init__(self, hidden: int, layers: int, r_init: float, r_max: float, length: float, seed: int):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        set_seed(seed)
        blocks: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks).double()
        self.r_max = float(r_max)
        self.length = float(length)
        last = self.net[-1]
        if not isinstance(last, nn.Linear):
            raise TypeError("the rate head must be linear")
        last.weight.data.zero_()
        fraction = float(np.clip(r_init / r_max, 1e-4, 1.0 - 1e-4))
        last.bias.data.fill_(math.log(fraction / (1.0 - fraction)))
        for module in self.net:
            if module is last or not isinstance(module, nn.Linear):
                continue
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain("tanh"))
            module.bias.data.zero_()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.stack((x / self.length, y / self.length), dim=-1))
        return self.r_max * torch.sigmoid(raw.squeeze(-1))
