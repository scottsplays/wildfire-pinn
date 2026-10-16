"""PINN adapter against the published Experiment 2A.3 F1 seed-0 score."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import experiments.wildfirepinn_v3.experiment3.inverse  # noqa: F401  torch before matplotlib
from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import spatial_rate_np
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import score_against_reference
from experiments.wildfirepinn_v2_diagnostics.experiment2.runner import predict_phi
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import circle_phi0
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from deployment.wildfirepinn_api.pinn_adapter import predict_phi_meters, status

METRICS = Path(__file__).resolve().parents[3] / "experiments" / "wildfirepinn_v2_diagnostics" / "experiment2" / "stabilization" / "metrics.json"


class PinnAdapterTest(unittest.TestCase):
    def test_checkpoint_loads_and_matches_the_experiment_field(self):
        info = status()
        self.assertTrue(info["available"], info["reason"])
        self.assertEqual(len(info["sha256"]), 64)
        problem = CircleProblem(n=201)
        published = predict_phi(self._model(), problem, self._phi0(), 4000.0, *problem.grid())
        adapted, x, y = predict_phi_meters(4000.0)
        self.assertTrue(abs(float((published - adapted).max())) < 1e-5)
        xx, yy = __import__("numpy").meshgrid(x, y)
        phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
        reference = propagate(phi0, x, y, spatial_rate_np(xx, yy), [4000.0])[4000.0]
        scored = score_against_reference(adapted, reference, x, y)
        rows = json.loads(METRICS.read_text(encoding="utf-8"))
        official = next(row for row in rows if row["name"] == "F1" and row["seed"] == 0)
        self.assertAlmostEqual(scored["iou"], official["final_iou"], places=3)
        self.assertAlmostEqual(scored["symmetric_distance_m"], official["final_symmetric_m"], delta=0.05)

    def _model(self):
        from deployment.wildfirepinn_api.pinn_adapter import _load

        return _load()

    def _phi0(self):
        problem = CircleProblem(n=201)

        def phi0(x_norm, y_norm, problem=problem):
            return circle_phi0(x_norm, y_norm, problem)

        return phi0


if __name__ == "__main__":
    unittest.main()
