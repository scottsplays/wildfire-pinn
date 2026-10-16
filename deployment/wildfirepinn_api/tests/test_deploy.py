"""Deployment checks that do not need Docker or a cloud project."""

from __future__ import annotations

import inspect
import os
import unittest
from pathlib import Path

os.environ.setdefault("WILDFIREPINN_API_KEY", "test-key-0123456789")

from fastapi.testclient import TestClient

from deployment.wildfirepinn_api.client import (
    ForecastClientError,
    build_forecast_body,
    forecast,
    forecast_benchmark,
    synthetic_circle,
)
from deployment.wildfirepinn_api.perimeter import geometry_is_valid
from deployment.wildfirepinn_api.service.app import app
from deployment.wildfirepinn_api.tests.test_api import KEY, _body

ROOT = Path(__file__).resolve().parents[3]


class DeployTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_container_sources_exist(self):
        required = [
            "wildfire_pinn/geometry.py",
            "wildfire_pinn/pinn.py",
            "experiments/wildfirepinn_v2_diagnostics/solver.py",
            "experiments/wildfirepinn_v2_diagnostics/experiment2/physics.py",
            "experiments/wildfirepinn_v2_diagnostics/experiment2/stabilization/runs/F1_seed0/levelset_pinn.pt",
            "experiments/wildfirepinn_v3/experiment3/inverse.py",
            "experiments/wildfirepinn_v3/experiment3/configs.json",
            "deployment/wildfirepinn_api/service/app.py",
        ]
        missing = [path for path in required if not (ROOT / path).is_file()]
        self.assertEqual(missing, [])
        checkpoint = ROOT / required[4]
        self.assertLess(checkpoint.stat().st_size, 1_000_000)

    def test_startup_does_not_train(self):
        from deployment.wildfirepinn_api.service.app import lifespan

        source = inspect.getsource(lifespan)
        self.assertNotIn("fit_", source)
        self.assertNotIn("train", source)

    def test_env_example_is_a_placeholder(self):
        text = (ROOT / "deployment" / "wildfirepinn_api" / ".env.example").read_text(encoding="utf-8")
        self.assertIn("replace-with-a-long-random-string", text)
        self.assertFalse((ROOT / "deployment" / "wildfirepinn_api" / ".env").exists())

    def test_vertex_predict_envelope_matches_forecast(self):
        instance = _body("ray_inverse_solver")
        denied = self.client.post("/predict", json={"instances": [instance]})
        self.assertEqual(denied.status_code, 401)
        response = self.client.post("/predict", headers=KEY, json={"instances": [instance], "parameters": {}})
        self.assertEqual(response.status_code, 200, response.text)
        prediction = response.json()["predictions"][0]
        self.assertEqual(prediction["model_id"], "ray_inverse_solver")
        self.assertEqual(prediction["units"]["length"], "m")
        self.assertEqual(prediction["forecast_time_s"], 600.0)
        self.assertTrue(geometry_is_valid(prediction["perimeter"]))
        self.assertFalse(prediction["operational_wildfire_service"])
        smuggled = self.client.post(
            "/predict",
            headers=KEY,
            json={"instances": [instance], "parameters": {"code": "print(1)"}},
        )
        self.assertEqual(smuggled.status_code, 422)

    def test_local_mode_refuses_startup_without_a_long_key(self):
        previous_key = os.environ.pop("WILDFIREPINN_API_KEY", None)
        previous_mode = os.environ.pop("WILDFIREPINN_PREDICT_AUTH", None)
        try:
            with self.assertRaises(RuntimeError) as caught:
                with TestClient(app):
                    pass
            self.assertIn("at least 16 characters", str(caught.exception))
            os.environ["WILDFIREPINN_API_KEY"] = "short-key"
            with self.assertRaises(RuntimeError):
                with TestClient(app):
                    pass
        finally:
            if previous_key is None:
                os.environ.pop("WILDFIREPINN_API_KEY", None)
            else:
                os.environ["WILDFIREPINN_API_KEY"] = previous_key
            if previous_mode is None:
                os.environ.pop("WILDFIREPINN_PREDICT_AUTH", None)
            else:
                os.environ["WILDFIREPINN_PREDICT_AUTH"] = previous_mode

    def test_vertex_iam_starts_without_a_key_and_keeps_other_routes_closed(self):
        previous_key = os.environ.pop("WILDFIREPINN_API_KEY", None)
        previous_mode = os.environ.get("WILDFIREPINN_PREDICT_AUTH")
        os.environ["WILDFIREPINN_PREDICT_AUTH"] = "vertex_iam"
        try:
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 200)
                predicted = client.post("/predict", json={"instances": [_body("constant_rate_solver")]})
                self.assertEqual(predicted.status_code, 200, predicted.text)
                self.assertEqual(predicted.json()["predictions"][0]["model_id"], "constant_rate_solver")
                self.assertIn("predictions", predicted.json())
                forecast = client.post("/forecast", json=_body("constant_rate_solver"))
                self.assertEqual(forecast.status_code, 401)
                models = client.get("/models")
                self.assertEqual(models.status_code, 401)
                presented = client.post(
                    "/forecast",
                    headers={"X-API-Key": "test-key-0123456789"},
                    json=_body("constant_rate_solver"),
                )
                self.assertEqual(presented.status_code, 401)
        finally:
            if previous_key is None:
                os.environ.pop("WILDFIREPINN_API_KEY", None)
            else:
                os.environ["WILDFIREPINN_API_KEY"] = previous_key
            if previous_mode is None:
                os.environ.pop("WILDFIREPINN_PREDICT_AUTH", None)
            else:
                os.environ["WILDFIREPINN_PREDICT_AUTH"] = previous_mode

    def test_invalid_predict_auth_refuses_startup(self):
        previous_mode = os.environ.get("WILDFIREPINN_PREDICT_AUTH")
        os.environ["WILDFIREPINN_PREDICT_AUTH"] = "public"
        try:
            with self.assertRaises(RuntimeError) as caught:
                with TestClient(app):
                    pass
            self.assertIn("WILDFIREPINN_PREDICT_AUTH", str(caught.exception))
        finally:
            if previous_mode is None:
                os.environ.pop("WILDFIREPINN_PREDICT_AUTH", None)
            else:
                os.environ["WILDFIREPINN_PREDICT_AUTH"] = previous_mode

    def test_vertex_iam_mode_does_not_open_forecast(self):
        previous = os.environ.get("WILDFIREPINN_PREDICT_AUTH")
        os.environ["WILDFIREPINN_PREDICT_AUTH"] = "vertex_iam"
        try:
            open_predict = self.client.post("/predict", json={"instances": [_body("constant_rate_solver")]})
            self.assertEqual(open_predict.status_code, 200, open_predict.text)
            self.assertEqual(open_predict.json()["predictions"][0]["model_id"], "constant_rate_solver")
            closed = self.client.post("/forecast", json=_body("constant_rate_solver"))
            self.assertEqual(closed.status_code, 401)
        finally:
            if previous is None:
                os.environ.pop("WILDFIREPINN_PREDICT_AUTH", None)
            else:
                os.environ["WILDFIREPINN_PREDICT_AUTH"] = previous

    def test_missing_pinn_is_disabled(self):
        from deployment.wildfirepinn_api.service import app as app_module

        original_status = app_module.pinn_status
        original_forecast = app_module.forecast_pinn
        app_module.pinn_status = lambda: {
            "available": False,
            "reason": "checkpoint missing",
            "sha256": None,
            "version": "experiment2a3-f1-seed0",
        }

        def _fail(_time):
            raise RuntimeError("checkpoint missing")

        app_module.forecast_pinn = _fail
        try:
            listed = self.client.get("/models", headers=KEY).json()
            pinn = listed["models"]["direct_pinn_experimental"]
            self.assertFalse(pinn["available"])
            self.assertIn("checkpoint missing", pinn["disabled_reason"])
            rejected = self.client.post(
                "/forecast",
                headers=KEY,
                json={
                    "model": "direct_pinn_experimental",
                    "coordinate_system": "local_cartesian_m",
                    "supported_benchmark": "experiment2a3_f1_spatial_product",
                    "forecast_time_s": 1000.0,
                },
            )
            self.assertEqual(rejected.status_code, 409)
        finally:
            app_module.pinn_status = original_status
            app_module.forecast_pinn = original_forecast

    def test_client_builds_a_ray_request_and_refuses_the_pinn_shortcut(self):
        body = build_forecast_body(
            [synthetic_circle(80), synthetic_circle(90), synthetic_circle(100)],
            [0, 200, 400],
            200,
        )
        self.assertEqual(body["model"], "ray_inverse_solver")
        self.assertEqual(body["forecast_time_s"], 600.0)
        self.assertEqual(body["coordinate_system"], "local_cartesian_m")
        self.assertIsNone(body["crs"])
        response = self.client.post("/forecast", headers=KEY, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["units"]["time"], "s")
        with self.assertRaises(ForecastClientError):
            build_forecast_body([synthetic_circle(80)], [0], 100, model="direct_pinn_experimental")
        with self.assertRaises(ForecastClientError):
            build_forecast_body([synthetic_circle(80)], [0], 9000)

    def test_client_transport_surfaces_http_errors(self):
        def transport(_base, _key, _path, _body):
            return 401, {"detail": "invalid api key"}

        with self.assertRaises(ForecastClientError) as caught:
            forecast(
                [synthetic_circle(80), synthetic_circle(90)],
                [0, 100],
                50,
                api_key="test-key-0123456789",
                transport=transport,
            )
        self.assertEqual(caught.exception.status_code, 401)

    def test_benchmark_client_posts_the_supported_checkpoint(self):
        seen = {}

        def transport(_base, _key, path, body):
            seen["path"] = path
            seen["body"] = body
            return 200, {"model_id": body["model"], "operational_wildfire_service": False}

        payload = forecast_benchmark(4000, api_key="test-key-0123456789", transport=transport)
        self.assertEqual(seen["path"], "/forecast")
        self.assertEqual(seen["body"]["supported_benchmark"], "experiment2a3_f1_spatial_product")
        self.assertEqual(payload["model_id"], "direct_pinn_experimental")
        self.assertNotIn("observations", seen["body"])


if __name__ == "__main__":
    unittest.main()
