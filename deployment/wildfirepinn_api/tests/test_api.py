"""Unit tests for validation, the research API, and the hard-IC PINN adapter."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("WILDFIREPINN_API_KEY", "test-key-0123456789")

import numpy as np
from fastapi.testclient import TestClient

import experiments.wildfirepinn_v3.experiment3.inverse  # noqa: F401  torch before matplotlib
from deployment.wildfirepinn_api.perimeter import geometry_is_valid
from deployment.wildfirepinn_api.service.app import app

KEY = {"X-API-Key": os.environ["WILDFIREPINN_API_KEY"]}


def _ring(radius: float, n: int = 48) -> list[list[float]]:
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    points = [[float(radius * np.cos(angle)), float(radius * np.sin(angle))] for angle in theta]
    points.append(points[0])
    return points


def _body(model: str, forecast: float = 600.0) -> dict:
    return {
        "model": model,
        "coordinate_system": "local_cartesian_m",
        "grid": {
            "x_min_m": -400.0,
            "x_max_m": 400.0,
            "y_min_m": -400.0,
            "y_max_m": 400.0,
            "resolution_m": 20.0,
        },
        "observations": [
            {"time_s": 0.0, "perimeter": {"type": "Polygon", "coordinates": [_ring(80.0)]}},
            {"time_s": 200.0, "perimeter": {"type": "Polygon", "coordinates": [_ring(90.0)]}},
            {"time_s": 400.0, "perimeter": {"type": "Polygon", "coordinates": [_ring(100.0)]}},
        ],
        "forecast_time_s": forecast,
    }


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_and_version_do_not_require_a_key(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertFalse(health.json()["operational_wildfire_service"])
        version = self.client.get("/version")
        self.assertEqual(version.status_code, 200)
        body = version.json()
        self.assertEqual(body["default_model"], "ray_inverse_solver")
        self.assertEqual(body["cloud_deployment"], "not_deployed")

    def test_missing_and_invalid_keys_are_rejected(self):
        self.assertEqual(self.client.get("/models").status_code, 401)
        self.assertEqual(self.client.get("/models", headers={"X-API-Key": "wrong-key-but-long"}).status_code, 401)
        denied = self.client.post("/forecast", headers={"Authorization": "Bearer nope-nope-nope"}, json=_body("ray_inverse_solver"))
        self.assertEqual(denied.status_code, 401)
        self.assertEqual(denied.json()["detail"], "invalid api key")

    def test_ray_forecast_schema_and_horizon(self):
        response = self.client.post("/forecast", headers=KEY, json=_body("ray_inverse_solver"))
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["forecast_time_s"], 600.0)
        self.assertEqual(body["model_id"], "ray_inverse_solver")
        self.assertEqual(body["coordinate_system"], "local_cartesian_m")
        self.assertIsNone(body["crs"])
        self.assertIsNone(body["uncertainty"])
        self.assertIsNone(body["mask_artifact"])
        self.assertTrue(body["experimental"])
        self.assertFalse(body["operational_wildfire_service"])
        self.assertGreater(body["burned_area_m2"], 0.0)
        self.assertNotEqual(body["burned_area_m2"], body["perimeter_polygon_area_m2"])
        self.assertTrue(geometry_is_valid(body["perimeter"]))
        self.assertIn("Not an operational", " ".join(body["warnings"]))
        self.assertEqual(body["units"]["length"], "m")
        self.assertGreater(body["elapsed_inference_s"], 0.0)

    def test_geographic_crs_and_unknown_family_are_rejected(self):
        geographic = _body("ray_inverse_solver")
        geographic["coordinate_system"] = "projected_meters"
        geographic["crs"] = "EPSG:4326"
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=geographic).status_code, 422)
        fourier = _body("fourier_inverse_solver")
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=fourier).status_code, 422)
        fourier["rate_parameters"] = {"family": "sin_kx_cos_ky", "wavelength_m": 2200}
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=fourier).status_code, 422)
        leaked = _body("ray_inverse_solver", forecast=400.0)
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=leaked).status_code, 422)
        huge = _body("ray_inverse_solver", forecast=20_000.0)
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=huge).status_code, 422)
        smuggled = _body("constant_rate_solver")
        smuggled["rate_parameters"] = {"family": "constant", "rate_m_s": 0.05, "expression": "x + y"}
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=smuggled).status_code, 422)

    def test_constant_prescribed_rate_and_projected_code(self):
        body = _body("constant_rate_solver")
        body["observations"] = body["observations"][:1]
        body["rate_parameters"] = {"family": "constant", "rate_m_s": 0.05}
        body["coordinate_system"] = "projected_meters"
        body["crs"] = "EPSG:3310"
        response = self.client.post("/forecast", headers=KEY, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["crs"], "EPSG:3310")
        self.assertGreater(response.json()["burned_area_m2"], 3.1416 * 80.0 ** 2 * 0.5)

    def test_pinn_benchmark_and_rejection(self):
        rejected = _body("direct_pinn_experimental")
        self.assertEqual(self.client.post("/forecast", headers=KEY, json=rejected).status_code, 422)
        accepted = {
            "model": "direct_pinn_experimental",
            "coordinate_system": "local_cartesian_m",
            "supported_benchmark": "experiment2a3_f1_spatial_product",
            "forecast_time_s": 4000.0,
        }
        response = self.client.post("/forecast", headers=KEY, json=accepted)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["forecast_time_s"], 4000.0)
        self.assertEqual(body["grid"]["shape"], [201, 201])
        self.assertTrue(geometry_is_valid(body["perimeter"]))
        self.assertIn("F1", " ".join(body["warnings"]))

    def test_oversized_content_length_is_rejected(self):
        response = self.client.post(
            "/forecast",
            headers={**KEY, "content-length": "2000000"},
            json=_body("ray_inverse_solver"),
        )
        self.assertEqual(response.status_code, 413)

    def test_models_marks_the_ray_solver_as_default(self):
        body = self.client.get("/models", headers=KEY).json()
        self.assertEqual(body["default_model"], "ray_inverse_solver")
        self.assertFalse(body["models"]["ray_inverse_solver"]["genuine_pde_residual_pinn"])
        self.assertTrue(body["models"]["direct_pinn_experimental"]["genuine_pde_residual_pinn"])
        self.assertTrue(body["models"]["direct_pinn_experimental"]["available"])


if __name__ == "__main__":
    unittest.main()
