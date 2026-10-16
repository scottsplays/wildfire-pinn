"""Frequency banks for the stabilization study.

F1 keeps the two diagonal modes of sin(kx) cos(ky).
F2 adds the axis-aligned fundamentals.
F3 is the eight-frequency bank from Experiment 2A.2.
F4 keeps the axis-aligned fundamental and its first harmonic, with no diagonals.
Rows are (cycles per unit X, cycles per unit Y), X = x / L.
"""

from __future__ import annotations

import torch

from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.models import frequency_bank, r_frequency


def _rows(length_m: float, specs: list[tuple[float, float, float]]) -> torch.Tensor:
    frequency = r_frequency(length_m)
    rows = [(scale_x * frequency, scale_y * frequency) for scale_x, scale_y, _harmonic in specs]
    return torch.tensor(rows, dtype=torch.float32)


def bank_f1(length_m: float) -> torch.Tensor:
    """Diagonal modes of the R product, at the fundamental wavelength."""
    return _rows(length_m, [(1.0, 1.0, 1.0), (1.0, -1.0, 1.0)])


def bank_f2(length_m: float) -> torch.Tensor:
    """F1 plus axis-aligned fundamentals."""
    return _rows(
        length_m,
        [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, -1.0, 1.0)],
    )


def bank_f3(length_m: float) -> torch.Tensor:
    """Original Experiment 2A.2 bank."""
    return frequency_bank(length_m)


def bank_f4(length_m: float) -> torch.Tensor:
    """Axis-aligned fundamental and first harmonic. No diagonal modes."""
    return _rows(
        length_m,
        [(1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (2.0, 0.0, 2.0), (0.0, 2.0, 2.0)],
    )


BANKS = {"F1": bank_f1, "F2": bank_f2, "F3": bank_f3, "F4": bank_f4}
