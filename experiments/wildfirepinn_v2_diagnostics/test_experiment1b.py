"""Tests for hard initial-condition parameterizations.

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment1b
"""

from __future__ import annotations

import unittest

import torch

from experiments.wildfirepinn_v2_diagnostics.ansatz import initial_phi_normalized, normalized_phi
from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem
from wildfire_pinn.pinn import isotropic_residual


class AnsatzTest(unittest.TestCase):
    def setUp(self):
        self.problem = CircleProblem()

    def test_hard_ic_matches_phi0_at_tau_zero(self):
        problem = self.problem
        tau = torch.zeros(5)
        x_norm = torch.tensor([0.25, -0.4, 0.1, 0.0, -0.2])
        y_norm = torch.tensor([0.1, 0.3, -0.35, 0.2, 0.05])
        raw = torch.randn(5)
        phi = normalized_phi(raw, tau, x_norm, y_norm, problem, "hard_ic", 0.1)
        phi0 = initial_phi_normalized(x_norm, y_norm, problem)
        self.assertTrue(torch.allclose(phi, phi0, atol=1e-6))

    def test_hard_ic_time_derivative_is_the_network_and_gradient_includes_phi0(self):
        problem = self.problem
        tau = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
        x_norm = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
        y_norm = torch.tensor([-0.2], dtype=torch.float64, requires_grad=True)
        raw = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
        phi = normalized_phi(raw, tau, x_norm, y_norm, problem, "hard_ic", 0.1)
        phi_t, phi_x, phi_y, phi_raw = torch.autograd.grad(phi, (tau, x_norm, y_norm, raw), torch.ones_like(phi))
        self.assertAlmostEqual(float(phi_t), 0.4, places=6)
        self.assertAlmostEqual(float(phi_raw), 0.0, places=6)
        magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2)
        self.assertAlmostEqual(float(magnitude), 1.0, places=5)
        eps = 1e-4
        plus = normalized_phi(
            raw.detach(), tau.detach(), x_norm.detach() + eps, y_norm.detach(), problem, "hard_ic", 0.1
        )
        minus = normalized_phi(
            raw.detach(), tau.detach(), x_norm.detach() - eps, y_norm.detach(), problem, "hard_ic", 0.1
        )
        self.assertAlmostEqual(float(phi_x), float((plus - minus) / (2 * eps)), places=4)

    def test_exact_hard_ic_solution_has_zero_residual(self):
        problem = self.problem
        rate = 0.1
        tau = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
        x_norm = torch.tensor([0.35, -0.15], dtype=torch.float64, requires_grad=True)
        y_norm = torch.tensor([0.05, 0.25], dtype=torch.float64, requires_grad=True)
        # phi = phi0 - R t, so the dimensionless network term is -R T / L.
        raw = torch.full_like(tau, -rate * problem.t_final / problem.length)
        phi = normalized_phi(raw, tau, x_norm, y_norm, problem, "hard_ic", rate)
        phi_t, phi_x, phi_y = torch.autograd.grad(phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=False)
        coded, magnitude = isotropic_residual(phi_t, phi_x, phi_y, rate * problem.t_final / problem.length)
        self.assertTrue(torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5))
        self.assertTrue(torch.allclose(coded, torch.zeros_like(coded), atol=1e-5))

    def test_r_scaled_is_exactly_stationary_when_r_is_zero(self):
        problem = self.problem
        tau = torch.tensor([0.0, 0.4, 1.0], dtype=torch.float64, requires_grad=True)
        x_norm = torch.tensor([0.2, -0.3, 0.15], dtype=torch.float64, requires_grad=True)
        y_norm = torch.tensor([-0.1, 0.25, 0.05], dtype=torch.float64, requires_grad=True)
        raw = torch.randn(3, dtype=torch.float64, requires_grad=True)
        phi = normalized_phi(raw, tau, x_norm, y_norm, problem, "r_scaled", 0.0)
        phi0 = initial_phi_normalized(x_norm, y_norm, problem)
        self.assertTrue(torch.allclose(phi, phi0, atol=1e-8))
        phi_t, phi_x, phi_y, phi_raw = torch.autograd.grad(
            phi, (tau, x_norm, y_norm, raw), torch.ones_like(phi)
        )
        self.assertTrue(torch.allclose(phi_t, torch.zeros_like(phi_t), atol=1e-8))
        self.assertTrue(torch.allclose(phi_raw, torch.zeros_like(phi_raw), atol=1e-8))
        magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2)
        self.assertTrue(torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5))

    def test_r_scaled_constant_network_solves_nonzero_r(self):
        problem = self.problem
        rate = 0.01
        tau = torch.tensor([0.3, 0.9], dtype=torch.float64, requires_grad=True)
        x_norm = torch.tensor([0.4, -0.2], dtype=torch.float64, requires_grad=True)
        y_norm = torch.tensor([0.1, -0.3], dtype=torch.float64, requires_grad=True)
        raw = torch.full_like(tau, -1.0)
        phi = normalized_phi(raw, tau, x_norm, y_norm, problem, "r_scaled", rate)
        phi_t, phi_x, phi_y = torch.autograd.grad(phi, (tau, x_norm, y_norm), torch.ones_like(phi))
        coded, _magnitude = isotropic_residual(phi_t, phi_x, phi_y, rate * problem.t_final / problem.length)
        self.assertTrue(torch.allclose(coded, torch.zeros_like(coded), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
