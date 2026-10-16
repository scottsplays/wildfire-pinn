"""Banks, contour spectrum, detached front sampling, and the Eikonal scale.

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2a3
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem
from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.models import frequency_bank
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import circle_phi0, hard_ic
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.banks import BANKS, bank_f1, bank_f4
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.net import BankFourierPINN
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.sampling import (
    PredictedFrontCache,
    sample_mixture,
    spectrum_fraction,
)
from wildfire_pinn.pinn import set_seed


class Experiment2A3Test(unittest.TestCase):
    def test_banks_match_the_stated_modes(self):
        length = 4000.0
        frequency = length / 1500.0
        f1 = bank_f1(length)
        self.assertEqual(tuple(f1.shape), (2, 2))
        self.assertTrue(torch.allclose(f1[0], torch.tensor([frequency, frequency])))
        self.assertTrue(torch.allclose(f1[1], torch.tensor([frequency, -frequency])))
        f4 = bank_f4(length)
        self.assertTrue(torch.all(f4[:, 0] * f4[:, 1] == 0))
        self.assertTrue(torch.allclose(BANKS["F3"](length), frequency_bank(length)))

    def test_hard_ic_exact_at_tau_zero(self):
        set_seed(0)
        problem = CircleProblem(n=21)
        model = BankFourierPINN(bank_f1(problem.length), "F1", hidden=16, layers=2)
        x = torch.linspace(-0.2, 0.2, 5)
        y = torch.linspace(-0.2, 0.2, 5)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        tau = torch.zeros(xx.numel())
        phi0 = circle_phi0(xx.reshape(-1), yy.reshape(-1), problem)
        phi = hard_ic(model(tau, xx.reshape(-1), yy.reshape(-1)), tau, phi0)
        self.assertTrue(torch.allclose(phi, phi0, atol=1e-6))

    def test_constant_radius_has_no_high_frequency_power(self):
        smooth = spectrum_fraction(np.full(360, 600.0))
        self.assertEqual(smooth["high_frequency_fraction"], 0.0)
        wiggly = 600.0 + 8.0 * np.sin(np.linspace(0, 12 * np.pi, 360, endpoint=False))
        rough = spectrum_fraction(wiggly)
        self.assertGreater(rough["high_frequency_fraction"], 0.5)

    def test_front_samples_are_detached_and_cluster_on_the_cache(self):
        problem = CircleProblem(n=21)
        cache = PredictedFrontCache()
        angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        cache.xy = torch.tensor(np.stack([800.0 + 5.0 * np.cos(angles), 5.0 * np.sin(angles)], axis=1), dtype=torch.float32)
        cache.time = torch.full((24,), 1000.0)
        x, y, t = sample_mixture(400, problem, cache, torch.device("cpu"))
        self.assertFalse(x.requires_grad)
        self.assertFalse(y.requires_grad)
        self.assertFalse(t.requires_grad)
        front = torch.stack([x[200:360], y[200:360]], dim=1)
        distance = torch.cdist(front, cache.xy).min(dim=1).values
        self.assertGreater(float((distance < 120.0).float().mean()), 0.7)

    def test_unit_ramp_gradient_is_one(self):
        x = torch.tensor([0.1, -0.2], requires_grad=True)
        y = torch.zeros(2, requires_grad=True)
        phi = x + 0.0 * y
        phi_x, phi_y = torch.autograd.grad(phi.sum(), (x, y))
        magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2)
        self.assertTrue(torch.allclose(magnitude, torch.ones(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
