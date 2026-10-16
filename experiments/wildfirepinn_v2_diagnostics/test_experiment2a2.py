"""Fourier features, curriculum, regional loss, and PCGrad checks.

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2a2
"""

from __future__ import annotations

import math
import unittest

import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem
from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.models import (
    FourierPINN,
    fourier_features,
    frequency_bank,
    r_frequency,
)
from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.optim import (
    lambda_at,
    pcgrad,
    rate_lambda,
    uniform_and_balanced,
)
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import R0_M_S, WAVELENGTH_M, spatial_rate_torch
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import circle_phi0, hard_ic
from wildfire_pinn.pinn import set_seed


class Experiment2A2Test(unittest.TestCase):
    def test_fourier_frequency_matches_the_r_wavelength(self):
        length = 4000.0
        frequency = r_frequency(length)
        self.assertAlmostEqual(frequency, length / WAVELENGTH_M)
        x_m = torch.tensor([250.0, -750.0, 1500.0])
        x_norm = x_m / length
        bank = frequency_bank(length)
        features = fourier_features(x_norm, torch.zeros_like(x_norm), bank)
        expected = torch.sin(2.0 * math.pi * x_m / WAVELENGTH_M)
        self.assertTrue(torch.allclose(features[:, 0], expected, atol=1e-5))
        # sin(kx) cos(ky) = 0.5 * [sin(kx+ky) + sin(kx-ky)], the diagonal rows.
        y_m = torch.tensor([100.0, -400.0, 900.0])
        y_norm = y_m / length
        both = fourier_features(x_norm, y_norm, bank)
        kx = 2.0 * math.pi * x_m / WAVELENGTH_M
        ky = 2.0 * math.pi * y_m / WAVELENGTH_M
        product = torch.sin(kx) * torch.cos(ky)
        diagonal = 0.5 * (both[:, 2] + both[:, 3])
        self.assertTrue(torch.allclose(diagonal, product, atol=1e-5))

    def test_feature_derivative_stays_on_the_coordinate_graph(self):
        length = 4000.0
        x = torch.tensor([0.2, -0.1], requires_grad=True)
        y = torch.tensor([0.05, 0.3], requires_grad=True)
        bank = frequency_bank(length)
        features = fourier_features(x, y, bank)
        grad_x, = torch.autograd.grad(features[:, 0].sum(), x)
        frequency = r_frequency(length)
        expected = torch.cos(2.0 * math.pi * frequency * x.detach()) * (2.0 * math.pi * frequency)
        self.assertTrue(torch.allclose(grad_x, expected, atol=1e-4))

    def test_hard_ic_is_exact_at_tau_zero_with_fourier_features(self):
        set_seed(0)
        problem = CircleProblem(n=21)
        model = FourierPINN(hidden=16, layers=2, length_m=problem.length)
        x = torch.linspace(-0.4, 0.4, 8)
        y = torch.linspace(-0.3, 0.3, 8)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        tau = torch.zeros_like(xx).reshape(-1)
        xn = xx.reshape(-1)
        yn = yy.reshape(-1)
        phi0 = circle_phi0(xn, yn, problem)
        phi = hard_ic(model(tau, xn, yn), tau, phi0)
        self.assertTrue(torch.allclose(phi, phi0, atol=1e-6))

    def test_lambda_schedule_ramps_through_the_middle_half(self):
        self.assertEqual(lambda_at(1, 1500), 0.0)
        self.assertEqual(lambda_at(375, 1500), 0.0)
        self.assertAlmostEqual(lambda_at(750, 1500), 0.5)
        self.assertEqual(lambda_at(1125, 1500), 1.0)
        self.assertEqual(lambda_at(1500, 1500), 1.0)
        x = torch.tensor([0.0, 400.0])
        y = torch.tensor([0.0, -200.0])
        self.assertTrue(torch.allclose(rate_lambda(x, y, 0.0), torch.full((2,), R0_M_S)))
        self.assertTrue(torch.allclose(rate_lambda(x, y, 1.0), spatial_rate_torch(x, y)))
        mid = rate_lambda(x, y, 0.5)
        blend = 0.5 * R0_M_S + 0.5 * spatial_rate_torch(x, y)
        self.assertTrue(torch.allclose(mid, blend))

    def test_equal_region_loss_matches_the_uniform_mean(self):
        squared = torch.tensor([1.0, 2.0, 3.0, 4.0])
        high = torch.tensor([True, True, False, False])
        uniform, balanced = uniform_and_balanced(squared, high)
        self.assertAlmostEqual(float(uniform), float(balanced))
        skewed = torch.tensor([True, True, True, False])
        uniform_s, balanced_s = uniform_and_balanced(squared, skewed)
        self.assertGreater(abs(float(uniform_s - balanced_s)), 0.1)

    def test_pcgrad_removes_a_collinear_conflict_and_keeps_an_agreement(self):
        base = torch.tensor([1.0, 0.0, -2.0])
        cancelled = pcgrad([base, -base])
        self.assertLess(float(cancelled.norm()), 1e-5)
        agreed = pcgrad([base, base])
        self.assertTrue(torch.allclose(agreed, 2.0 * base))


if __name__ == "__main__":
    unittest.main()
