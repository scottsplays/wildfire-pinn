"""Estimator guards that do not refit the held-out study."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import experiments.wildfirepinn_v3.experiment3.inverse  # noqa: F401
from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from deployment.wildfirepinn_api.interface import fit_constant_rate, fit_ray_rate
from deployment.wildfirepinn_api.real_fire import evaluation_status


class EstimatorTest(unittest.TestCase):
    def test_constant_and_ray_recover_a_circular_expansion(self):
        problem = CircleProblem(n=81, half_extent=800.0, r0=80.0, x0=0.0, y0=0.0, t_final=600.0)
        x, y = problem.grid()
        xx, yy = np.meshgrid(x, y)
        phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
        fields = propagate(phi0, x, y, np.full(phi0.shape, 0.05), [0.0, 200.0, 400.0, 600.0])
        view = {
            "times": [0.0, 200.0, 400.0],
            "fields": {moment: fields[moment] for moment in (0.0, 200.0, 400.0)},
            "phi0": fields[0.0],
            "x": x,
            "y": y,
        }
        rate, diag, _elapsed = fit_constant_rate(view, None)
        self.assertAlmostEqual(diag["rate_m_s"], 0.05, delta=0.005)
        ray, _ray_diag, _ray_s = fit_ray_rate(view)
        self.assertGreater(float(np.min(ray)), 0.0)
        forecast = propagate(view["phi0"], x, y, ray, [600.0])[600.0]
        truth = fields[600.0]
        union = np.logical_or(forecast < 0, truth < 0).sum()
        overlap = np.logical_and(forecast < 0, truth < 0).sum()
        self.assertGreater(float(overlap) / float(union), 0.9)

    def test_interface_does_not_read_the_hidden_rate(self):
        source = (Path(__file__).resolve().parents[1] / "interface.py").read_text(encoding="utf-8")
        self.assertNotIn("spatial_rate", source)
        self.assertNotIn("rate_true", source)

    def test_real_fire_hook_has_no_fabricated_metrics(self):
        status = evaluation_status()
        self.assertEqual(status["status"], "not_executed")
        self.assertIsNone(status["metrics"])


if __name__ == "__main__":
    unittest.main()
