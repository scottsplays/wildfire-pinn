"""Hard-IC Fourier network whose frequency bank is chosen by the experiment."""

from __future__ import annotations

import math

import torch
from torch import nn


def fourier_features(x_norm: torch.Tensor, y_norm: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    coords = torch.cat([x_norm.reshape(-1, 1), y_norm.reshape(-1, 1)], dim=1)
    projected = (2.0 * math.pi) * (coords @ bank.T)
    return torch.cat([torch.sin(projected), torch.cos(projected)], dim=1)


class BankFourierPINN(nn.Module):
    def __init__(self, bank: torch.Tensor, variant: str, hidden: int = 64, layers: int = 4):
        super().__init__()
        if layers < 1 or hidden < 1:
            raise ValueError("hidden width and depth must be positive")
        if bank.ndim != 2 or bank.shape[1] != 2:
            raise ValueError("frequency bank must have shape (frequencies, 2)")
        self.variant = str(variant)
        self.register_buffer("frequency_bank", bank.detach().float().clone())
        width_in = 3 + int(bank.shape[0]) * 2
        blocks: list[nn.Module] = [nn.Linear(width_in, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            blocks.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.reset_parameters()

    @property
    def arch(self) -> dict:
        return {
            "hidden": self.hidden,
            "layers": self.layers,
            "fourier_frequencies": int(self.frequency_bank.shape[0]),
            "variant": self.variant,
        }

    def reset_parameters(self) -> None:
        linear = [module for module in self.net if isinstance(module, nn.Linear)]
        gain = nn.init.calculate_gain("tanh")
        for module in linear[:-1]:
            nn.init.xavier_normal_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)
        nn.init.xavier_normal_(linear[-1].weight, gain=1.0)
        nn.init.zeros_(linear[-1].bias)

    def forward(self, t, x, y) -> torch.Tensor:
        device = self.frequency_bank.device
        columns = []
        for value in (t, x, y):
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.float32, device=device)
            columns.append(value.reshape(-1, 1).to(device=device, dtype=torch.float32))
        features = fourier_features(columns[1], columns[2], self.frequency_bank)
        return self.net(torch.cat([*columns, features], dim=1)).squeeze(-1)
