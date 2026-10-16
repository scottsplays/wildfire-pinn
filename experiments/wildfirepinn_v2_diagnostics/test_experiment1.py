"""Derivative, unit, and numerical-solver checks for the circular benchmark.

Run from the repository root:

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment1
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.circle import (
    CircleProblem,
    analytical_phi,
    front_radius_stats,
    true_radius,
)
from experiments.wildfirepinn_v2_diagnostics.solver import (
    cfl_timestep,
    godunov_gradient_magnitude,
    propagate,
)
from wildfire_pinn.pinn import isotropic_residual


class CircleUnitsTest(unittest.TestCase):
    def test_analytical_field_satisfies_the_normalized_residual(self):
        problem = CircleProblem()
        rate = 0.1
        length = problem.length
        duration = problem.t_final
        x = torch.tensor([900.0, -400.0, 200.0], dtype=torch.float64)
        y = torch.tensor([700.0, 800.0, 400.0], dtype=torch.float64)
        time = torch.tensor([0.0, 1500.0, 4000.0], dtype=torch.float64)
        tau = (time / duration).detach().requires_grad_(True)
        x_norm = ((x - problem.center_x) / length).detach().requires_grad_(True)
        y_norm = ((y - problem.center_y) / length).detach().requires_grad_(True)
        radius = problem.r0 + rate * (tau * duration)
        phi_m = torch.sqrt((x_norm * length - problem.x0) ** 2 + (y_norm * length - problem.y0) ** 2) - radius
        phi = phi_m / length
        phi_tau, phi_x, phi_y = torch.autograd.grad(phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=False)
        coded, magnitude = isotropic_residual(phi_tau, phi_x, phi_y, rate * duration / length)
        physical = coded * (length / duration)
        self.assertTrue(torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5))
        self.assertTrue(torch.allclose(coded, torch.zeros_like(coded), atol=1e-5))
        self.assertTrue(torch.allclose(physical, torch.zeros_like(physical), atol=1e-6))

    def test_physical_time_derivative_is_minus_r(self):
        problem = CircleProblem()
        rate = 0.01
        x = torch.tensor([1200.0], dtype=torch.float64, requires_grad=True)
        y = torch.tensor([-800.0], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([1000.0], dtype=torch.float64, requires_grad=True)
        phi = torch.sqrt((x - problem.x0) ** 2 + (y - problem.y0) ** 2) - (problem.r0 + rate * t)
        phi_t, phi_x, phi_y = torch.autograd.grad(phi, (t, x, y), torch.ones_like(phi))
        self.assertAlmostEqual(float(phi_t), -rate, places=6)
        magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2)
        self.assertAlmostEqual(float(magnitude), 1.0, places=6)

    def test_final_circle_stays_inside_the_domain(self):
        problem = CircleProblem()
        for rate in (0.0, 0.01, 0.1):
            self.assertGreater(problem.clearance_m(rate), 500.0)

    def test_zero_rate_does_not_move_the_numerical_front(self):
        problem = CircleProblem(n=81, t_final=1000.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
        advanced = propagate(phi0, x, y, 0.0, [0.0, 1000.0])
        self.assertTrue(np.array_equal(advanced[0.0], phi0))
        self.assertTrue(np.array_equal(advanced[1000.0], phi0))
        stats = front_radius_stats(advanced[1000.0], x, y, problem.x0, problem.y0, problem.r0)
        self.assertTrue(stats["contour"])
        self.assertLess(abs(stats["radius_error_m"]), 1.0)

    def test_planar_front_advances_at_the_prescribed_speed(self):
        x = np.linspace(0.0, 1000.0, 201)
        y = np.linspace(0.0, 200.0, 21)
        xx, yy = np.meshgrid(x, y)
        rate = 0.2
        duration = 500.0
        phi0 = xx - 200.0
        final = propagate(phi0, x, y, rate, [duration])[duration]
        # Burned is x < 200 + R T. The zero crossing of the row-mean field.
        row = final[len(y) // 2]
        sign = np.where(row[:-1] <= 0, 1, 0) - np.where(row[1:] <= 0, 1, 0)
        crossing = int(np.flatnonzero(sign > 0)[0])
        x_left, x_right = x[crossing], x[crossing + 1]
        z_left, z_right = row[crossing], row[crossing + 1]
        x_zero = x_left - z_left * (x_right - x_left) / (z_right - z_left)
        self.assertAlmostEqual(x_zero, 200.0 + rate * duration, delta=float(x[1] - x[0]))

    def test_circle_radius_grows_by_r_times_t(self):
        problem = CircleProblem(n=201, t_final=2000.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        rate = 0.1
        phi0 = analytical_phi(xx, yy, 0.0, problem, rate)
        final = propagate(phi0, x, y, rate, [problem.t_final])[problem.t_final]
        expected = true_radius(problem, rate, problem.t_final)
        stats = front_radius_stats(final, x, y, problem.x0, problem.y0, expected)
        self.assertTrue(stats["contour"])
        self.assertLess(abs(stats["radius_error_m"]), 15.0)
        self.assertLess(stats["mean_radial_displacement_m"], 15.0)

    def test_faster_region_moves_the_front_farther(self):
        x = np.linspace(-800.0, 800.0, 161)
        y = np.linspace(-800.0, 800.0, 161)
        xx, yy = np.meshgrid(x, y)
        phi0 = np.hypot(xx, yy) - 200.0
        rate = np.where(xx >= 0.0, 0.1, 0.02)
        final = propagate(phi0, x, y, rate, [1000.0])[1000.0]
        burned = final < 0
        right_reach = float(xx[burned].max())
        left_reach = float((-xx[burned]).max())
        self.assertGreater(right_reach, 200.0 + 70.0)
        self.assertLess(left_reach, 200.0 + 40.0)

    def test_upwind_gradient_of_a_unit_ramp_is_one(self):
        x = np.linspace(0.0, 100.0, 21)
        y = np.linspace(0.0, 40.0, 9)
        xx, yy = np.meshgrid(x, y)
        phi = xx.copy()
        magnitude = godunov_gradient_magnitude(phi, x[1] - x[0], y[1] - y[0], positive_speed=True)
        self.assertTrue(np.allclose(magnitude[:, 1:-1], 1.0, atol=1e-8))

    def test_cfl_scales_with_grid_and_speed(self):
        coarse = cfl_timestep(20.0, 20.0, 0.1, 0.4)
        fine = cfl_timestep(10.0, 10.0, 0.1, 0.4)
        fast = cfl_timestep(10.0, 10.0, 0.2, 0.4)
        self.assertAlmostEqual(coarse, 2.0 * fine, places=8)
        self.assertAlmostEqual(fine, 2.0 * fast, places=8)
        self.assertTrue(np.isinf(cfl_timestep(10.0, 10.0, 0.0, 0.4)))
        with self.assertRaises(ValueError):
            cfl_timestep(10.0, 10.0, -0.1, 0.4)


if __name__ == "__main__":
    unittest.main()
