"""Unit tests for the inverse-rate experiment. They do not train on forecast times."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import spatial_rate_np
from experiments.wildfirepinn_v2_diagnostics.solver import godunov_gradient_magnitude
from experiments.wildfirepinn_v3.experiment3.inverse import constant_rate
from experiments.wildfirepinn_v3.experiment3.models import RateMLP, pack_rate, parametric_rate, unpack_rate
from experiments.wildfirepinn_v3.experiment3.observations import angular_radii, training_view
from experiments.wildfirepinn_v3.experiment3.torch_solver import adjoint_rate_gradient, dt_for_interval, godunov_step, rollout


class InverseRateTest(unittest.TestCase):
    def test_one_godunov_step_matches_numpy(self):
        problem = CircleProblem(n=31, half_extent=200.0, r0=40.0, x0=10.0, y0=-15.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        phi = analytical_phi(xx, yy, 0.0, problem, 0.0)
        rate = np.full_like(phi, 0.04)
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        updated = phi - 1.5 * rate * godunov_gradient_magnitude(phi, dx, dy)
        torch_updated = godunov_step(torch.tensor(phi), torch.tensor(rate), dx, dy, 1.5)
        np.testing.assert_allclose(torch_updated.numpy(), updated, atol=1e-8)

    def test_training_view_rejects_forecast_times(self):
        phi = np.zeros((5, 5))
        x = np.linspace(-1, 1, 5)
        y = np.linspace(-1, 1, 5)
        with self.assertRaises(ValueError):
            training_view({0.0: phi, 3000.0: phi}, [0.0, 3000.0], x, y)

    def test_constant_rate_recovers_a_circle(self):
        problem = CircleProblem(n=121, half_extent=600.0, r0=100.0, x0=0.0, y0=0.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        fields = {0.0: analytical_phi(xx, yy, 0.0, problem, 0.05), 2000.0: analytical_phi(xx, yy, 2000.0, problem, 0.05)}
        view = training_view(fields, [0.0, 2000.0], x, y)
        self.assertNotIn("rate", view)
        estimate = constant_rate(view, 0.0, 0.0)
        self.assertAlmostEqual(estimate, 0.05, delta=0.002)

    def test_product_family_matches_the_benchmark_at_the_true_coefficients(self):
        theta = pack_rate(0.05, 0.5, 0.005, 0.15, 0.9)
        r0, amplitude = unpack_rate(theta, 0.005, 0.15, 0.9)
        self.assertAlmostEqual(r0, 0.05, places=8)
        self.assertAlmostEqual(amplitude, 0.5, places=8)
        x = np.linspace(-2000, 2000, 21)
        y = np.linspace(-2000, 2000, 21)
        xx, yy = np.meshgrid(x, y)
        learned = parametric_rate(theta, xx, yy, 1500.0, "product", 0.005, 0.15, 0.9)
        np.testing.assert_allclose(learned, spatial_rate_np(xx, yy), atol=1e-8)

    def test_rate_network_is_positive_and_starts_at_the_constant(self):
        model = RateMLP(8, 2, 0.05, 0.15, 4000.0, seed=0)
        coords = torch.linspace(-2000, 2000, 9, dtype=torch.float64)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        rate = model(xx, yy).detach()
        self.assertGreater(float(rate.min()), 0.0)
        self.assertLess(float(rate.max()), 0.15)
        self.assertAlmostEqual(float(rate.mean()), 0.05, places=6)

    def test_training_module_does_not_name_the_hidden_rate(self):
        source = Path("experiments/wildfirepinn_v3/experiment3/inverse.py").read_text(encoding="utf-8")
        self.assertNotIn("spatial_rate", source)
        self.assertNotIn("rate_true", source)

    def test_rollout_advances_the_front(self):
        problem = CircleProblem(n=21, half_extent=80.0, r0=20.0, x0=0.0, y0=0.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        phi0 = torch.tensor(analytical_phi(xx, yy, 0.0, problem, 0.0))
        rate = torch.full_like(phi0, 0.05)
        dx = float(x[1] - x[0])
        dt = dt_for_interval(500.0, dx, dx, 0.15, 0.4)
        self.assertAlmostEqual(1000.0 / dt, round(1000.0 / dt), places=6)
        solved = rollout(phi0, rate, dx, dx, [100.0], dt_for_interval(100.0, dx, dx, 0.15, 0.4))
        centers, radii = angular_radii(solved[100.0].numpy(), x, y, 0.0, 0.0, 36)
        self.assertGreater(float(np.nanmean(radii)), 24.0)
        self.assertEqual(len(centers), 36)

    def test_outward_rate_has_a_negative_level_set_derivative(self):
        problem = CircleProblem(n=15, half_extent=40.0, r0=12.0, x0=0.0, y0=0.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        phi0 = torch.tensor(analytical_phi(xx, yy, 0.0, problem, 0.0))
        rate = torch.full_like(phi0, 0.04, requires_grad=True)
        dx = float(x[1] - x[0])
        dt = dt_for_interval(20.0, dx, dx, 0.15, 0.4)
        solved = rollout(phi0, rate, dx, dx, [20.0], dt)
        solved[20.0].mean().backward()
        unroll = rate.grad.detach().clone()
        rate.grad = None

        def loss_grad(step, phi):
            if step != int(round(20.0 / dt)):
                return None
            variable = phi.detach().requires_grad_(True)
            return torch.autograd.grad(variable.mean(), variable)[0]

        adjoint = adjoint_rate_gradient(phi0, rate.detach(), dx, dx, dt, int(round(20.0 / dt)), loss_grad)
        self.assertTrue(torch.allclose(adjoint, unroll, rtol=1e-5, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
