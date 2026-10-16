"""Solver, wind identity, irregular SDF, and sampling checks for Experiment 2.

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2
"""

from __future__ import annotations

import unittest

import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.experiment2.directional import propagate_directional
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import WIND_C, spatial_rate_np
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import coded_residual, grid_phi0, hard_ic
from experiments.wildfirepinn_v2_diagnostics.experiment2.runner import FrontCache, sample_mixed
from experiments.wildfirepinn_v2_diagnostics.experiment2.sdf import irregular_polygon, irregular_sdf
from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from shapely import distance, points
from wildfire_pinn.geometry import zero_contours
from wildfire_pinn.pinn import sample_grid


def _crossing(row, x) -> float:
    change = np.where(row[:-1] <= 0.0, 1, 0) - np.where(row[1:] <= 0.0, 1, 0)
    index = int(np.flatnonzero(change > 0)[0])
    span = x[index + 1] - x[index]
    return float(x[index] - row[index] * span / (row[index + 1] - row[index]))


class Experiment2Test(unittest.TestCase):
    def test_spatial_rate_stays_positive_and_bounded(self):
        x = np.linspace(-2000, 2000, 80)
        y = np.linspace(-2000, 2000, 80)
        xx, yy = np.meshgrid(x, y)
        rate = spatial_rate_np(xx, yy)
        self.assertGreater(float(rate.min()), 0.024)
        self.assertLess(float(rate.max()), 0.076)

    def test_zero_rate_directional_solver_does_not_move(self):
        x = np.linspace(-400, 400, 41)
        y = np.linspace(-400, 400, 41)
        xx, yy = np.meshgrid(x, y)
        phi0 = np.hypot(xx, yy) - 120.0
        final = propagate_directional(phi0, x, y, 0.0, [500.0], lambda _t: (1.0, 0.0), WIND_C)[500.0]
        self.assertTrue(np.allclose(final, phi0))

    def test_zero_wind_factor_matches_the_isotropic_solver(self):
        x = np.linspace(-500, 500, 51)
        y = np.linspace(-500, 500, 51)
        xx, yy = np.meshgrid(x, y)
        phi0 = np.hypot(xx, yy) - 150.0
        rate = np.full(phi0.shape, 0.05)
        iso = propagate(phi0, x, y, rate, [400.0])[400.0]
        directional = propagate_directional(phi0, x, y, rate, [400.0], lambda _t: (0.0, 1.0), 0.0)[400.0]
        self.assertTrue(np.allclose(directional, iso, atol=1e-8))

    def test_planar_front_speed_follows_the_wind_identity(self):
        x = np.linspace(0.0, 1200.0, 241)
        y = np.linspace(0.0, 80.0, 9)
        xx, yy = np.meshgrid(x, y)
        phi0 = xx - 400.0
        rate = 0.1
        duration = 300.0
        expected = {
            (1.0, 0.0): 400.0 + rate * (1.0 + WIND_C) * duration,
            (-1.0, 0.0): 400.0 + rate * (1.0 - WIND_C) * duration,
            (0.0, 1.0): 400.0 + rate * duration,
        }
        cell = float(x[1] - x[0])
        for wind, target in expected.items():
            final = propagate_directional(phi0, x, y, rate, [duration], lambda _t, wind=wind: wind, WIND_C)[duration]
            found = _crossing(final[len(y) // 2], x)
            self.assertAlmostEqual(found, target, delta=cell)

    def test_circle_moves_farther_downwind_than_upwind(self):
        x = np.linspace(-800.0, 800.0, 161)
        y = np.linspace(-800.0, 800.0, 161)
        xx, yy = np.meshgrid(x, y)
        phi0 = np.hypot(xx, yy) - 150.0
        final = propagate_directional(phi0, x, y, 0.05, [1000.0], lambda _t: (1.0, 0.0), WIND_C)[1000.0]
        burned = final < 0
        down = float(xx[burned].max())
        up = float((-xx[burned]).max())
        self.assertGreater(down - up, 30.0)

    def test_switching_wind_changes_the_later_axis(self):
        x = np.linspace(-800.0, 800.0, 121)
        y = np.linspace(-800.0, 800.0, 121)
        xx, yy = np.meshgrid(x, y)
        phi0 = np.hypot(xx, yy) - 150.0
        constant = propagate_directional(phi0, x, y, 0.05, [1000.0], lambda _t: (1.0, 0.0), WIND_C)[1000.0]
        switched = propagate_directional(
            phi0, x, y, 0.05, [1000.0],
            lambda t: (1.0, 0.0) if t < 500.0 else (0.0, 1.0),
            WIND_C,
            switch_times=(500.0,),
        )[1000.0]
        self.assertGreater(float(xx[constant < 0].max()), float(xx[switched < 0].max()) + 10.0)
        self.assertGreater(float(yy[switched < 0].max()), float(yy[constant < 0].max()) + 10.0)

    def test_c_at_least_one_is_refused(self):
        x = np.linspace(0, 10, 5)
        y = np.linspace(0, 10, 5)
        phi = np.zeros((5, 5))
        with self.assertRaises(ValueError):
            propagate_directional(phi, x, y, 0.1, [1.0], lambda _t: (1.0, 0.0), 1.0)

    def test_spatial_rate_advances_the_fast_side_of_a_planar_front(self):
        x = np.linspace(0.0, 800.0, 161)
        y = np.linspace(-150.0, 150.0, 61)
        xx, yy = np.meshgrid(x, y)
        phi0 = xx - 300.0
        rate = np.where(yy >= 0.0, 0.08, 0.02)
        final = propagate(phi0, x, y, rate, [500.0])[500.0]
        top = _crossing(final[-3], x)
        bottom = _crossing(final[2], x)
        self.assertGreater(top, bottom + 20.0)

    def test_wind_identity_residual_is_zero_for_the_exact_speed(self):
        rate = torch.tensor([0.05], dtype=torch.float64)
        phi_x = torch.tensor([1.0], dtype=torch.float64)
        phi_y = torch.tensor([0.0], dtype=torch.float64)
        phi_t = -(rate * (1.0 + WIND_C))
        _coded, physical, _mag = coded_residual(
            phi_t, phi_x, phi_y, rate, 1.0,
            torch.tensor([1.0], dtype=torch.float64), torch.tensor([0.0], dtype=torch.float64), WIND_C,
        )
        self.assertAlmostEqual(float(physical), 0.0, places=5)

    def test_irregular_sdf_matches_the_polygon_and_is_negative_inside(self):
        x = np.linspace(-1200.0, 1200.0, 161)
        y = np.linspace(-1200.0, 1200.0, 161)
        sdf = irregular_sdf(x, y, 0.0, 0.0, 400.0)
        self.assertLess(float(sdf[len(y) // 2, len(x) // 2]), 0.0)
        polygon = irregular_polygon(0.0, 0.0, 400.0)
        lines = [line for line in zero_contours(sdf, x, y) if len(line) >= 2]
        self.assertTrue(lines)
        miss = float(np.mean(distance(points(np.concatenate(lines)), polygon.boundary)))
        self.assertLess(miss, 1.5 * float(x[1] - x[0]))

    def test_sdf_interpolation_derivative_matches_a_finite_difference(self):
        x = np.linspace(-1600.0, 1600.0, 161)
        y = np.linspace(-1600.0, 1600.0, 161)
        sdf = irregular_sdf(x, y, 0.0, 0.0, 400.0)
        xq = torch.tensor([900.0], dtype=torch.float32, requires_grad=True)
        yq = torch.tensor([40.0], dtype=torch.float32, requires_grad=True)
        meters = sample_grid(sdf, x, y, xq, yq)
        grad_x, grad_y = torch.autograd.grad(meters, (xq, yq), torch.ones_like(meters))
        eps = 0.5
        plus = float(sample_grid(sdf, x, y, xq.detach() + eps, yq.detach()))
        minus = float(sample_grid(sdf, x, y, xq.detach() - eps, yq.detach()))
        self.assertAlmostEqual(float(grad_x), (plus - minus) / (2 * eps), delta=0.05)
        self.assertAlmostEqual(float(torch.sqrt(grad_x ** 2 + grad_y ** 2)), 1.0, delta=0.15)

    def test_hard_ic_matches_the_interpolated_field_at_tau_zero(self):
        problem = CircleProblem(n=81, half_extent=1200.0)
        x, y = problem.grid()
        sdf = irregular_sdf(x, y, problem.x0, problem.y0, problem.r0)
        x_norm = torch.tensor([0.15, -0.2], dtype=torch.float32, requires_grad=True)
        y_norm = torch.tensor([0.05, 0.1], dtype=torch.float32, requires_grad=True)
        phi0 = grid_phi0(x_norm, y_norm, sdf, x, y, problem)
        raw = torch.randn(2)
        phi = hard_ic(raw, torch.zeros(2), phi0)
        self.assertTrue(torch.allclose(phi, phi0, atol=1e-6))

    def test_predicted_front_samples_are_detached_and_cluster_on_the_cache(self):
        problem = CircleProblem()
        cache = FrontCache()
        cache.xy = torch.tensor([[800.0 + 5.0 * i, 20.0] for i in range(10)], dtype=torch.float32)
        x, y, _t = sample_mixed(400, problem, cache, torch.device("cpu"))
        self.assertFalse(x.requires_grad)
        self.assertFalse(y.requires_grad)
        nearest = []
        cached = cache.xy.numpy()
        for px, py in zip(x.numpy(), y.numpy()):
            nearest.append(np.min(np.hypot(cached[:, 0] - px, cached[:, 1] - py)))
        self.assertGreater(float(np.mean(np.asarray(nearest) < 120.0)), 0.3)


if __name__ == "__main__":
    unittest.main()
