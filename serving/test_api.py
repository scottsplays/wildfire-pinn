"""Local smoke tests for the frozen v1 API. No training and no cloud deploy."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

from infer_wildfirepinn_v1 import EXPECTED_SHA256 as SCRIPT_SHA256
from serving.auth import UNAUTHORIZED, AuthError, assert_auth_configured, require_invoker
from serving.inference import EXPECTED_SHA256, file_sha256
from wildfire_pinn.pinn import predict_phi_meters

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "outputs" / "kincade_final_candidates" / "observation" / "levelset_pinn.pt"
TOKEN = "local-test-token-not-a-cloud-secret"


def _auth_units() -> None:
    previous = {name: os.environ.get(name) for name in (
        "WILDFIREPINN_AUTH_MODE",
        "WILDFIREPINN_AUDIENCE",
        "WILDFIREPINN_LOCAL_TOKEN",
        "K_SERVICE",
    )}
    try:
        os.environ["WILDFIREPINN_AUTH_MODE"] = "iam"
        os.environ.pop("WILDFIREPINN_AUDIENCE", None)
        try:
            assert_auth_configured()
        except RuntimeError:
            pass
        else:
            raise AssertionError("iam mode started without an audience")

        os.environ["WILDFIREPINN_AUTH_MODE"] = "local"
        os.environ["WILDFIREPINN_LOCAL_TOKEN"] = "short"
        os.environ["K_SERVICE"] = "wildfirepinn-v1"
        try:
            assert_auth_configured()
        except RuntimeError:
            pass
        else:
            raise AssertionError("Cloud Run accepted local auth mode")
        os.environ.pop("K_SERVICE", None)

        os.environ["WILDFIREPINN_AUTH_MODE"] = "iam"
        os.environ["WILDFIREPINN_AUDIENCE"] = "https://wildfirepinn-v1.example.run.app"
        try:
            require_invoker(None)
        except AuthError as exc:
            assert exc.detail == UNAUTHORIZED
        else:
            raise AssertionError("missing token was accepted")
        try:
            require_invoker("Bearer not-a-google-identity-token")
        except AuthError as exc:
            assert exc.detail == UNAUTHORIZED
        else:
            raise AssertionError("invalid identity token was accepted")

        from serving import auth as auth_module

        original_verify = auth_module._verify_google_identity
        auth_module._verify_google_identity = lambda token, audience: {
            "sub": "test-subject",
            "aud": audience,
        }
        try:
            claims = require_invoker("Bearer header.payload.signature")
            assert claims["mode"] == "iam"
            assert claims["subject_present"] is True
        finally:
            auth_module._verify_google_identity = original_verify
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _http() -> None:
    os.environ["WILDFIREPINN_AUTH_MODE"] = "local"
    os.environ["WILDFIREPINN_LOCAL_TOKEN"] = TOKEN
    os.environ.pop("K_SERVICE", None)
    before = file_sha256(CHECKPOINT)
    assert before == EXPECTED_SHA256 == SCRIPT_SHA256

    from fastapi.testclient import TestClient

    from serving.app import app

    headers = {"Authorization": f"Bearer {TOKEN}"}
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200, health.text
        assert health.json()["training_enabled"] is False
        assert client.get("/docs").status_code == 404
        assert client.get("/openapi.json").status_code == 404

        hidden = client.get("/model-info")
        assert hidden.status_code == 401
        assert hidden.json()["detail"] == UNAUTHORIZED
        assert client.post("/predict", json={"timestamp": "2019-10-24T09:05:00"}).status_code == 401
        assert client.post(
            "/predict",
            headers={"Authorization": "Bearer wrong-token-value"},
            json={"timestamp": "2019-10-24T09:05:00"},
        ).status_code == 401

        info = client.get("/model-info", headers=headers)
        assert info.status_code == 200, info.text
        body = info.json()
        assert body["model_version"] == "wildfirepinn-v1"
        assert body["model_sha256"] == EXPECTED_SHA256
        assert body["w_pde"] == 0
        assert body["training_enabled"] is False
        assert body["device"] == "cpu"
        assert "not an operational wildfire forecast" in body["warning"]
        start = body["time_window"]["start"]
        end = body["time_window"]["end"]

        service = client.app.state.model
        direct = predict_phi_meters(service.model, service.case, float(service.pass_tau[0]))
        predicted = client.post("/predict", headers=headers, json={"timestamp": start})
        assert predicted.status_code == 200, predicted.text
        assert len(predicted.content) < 20000
        meta = predicted.json()
        assert meta["model_sha256"] == EXPECTED_SHA256
        assert meta["crs"] == "EPSG:3310"
        assert meta["timestamp"] == start
        assert meta["grid"]["height"] == direct.shape[0]
        assert meta["grid"]["width"] == direct.shape[1]
        assert meta["grid"]["cell_size_m"] == 90 or abs(meta["grid"]["cell_size_m"] - 90) < 1e-6
        assert isinstance(meta["mask"], dict)
        assert meta["mask"]["path"] == "/predict/mask"
        assert "not an operational wildfire forecast" in meta["warning"]
        burned = direct < 0
        assert meta["burned_cells"] == int(np.count_nonzero(burned))
        assert abs(meta["phi_min_m"] - float(direct.min())) < 1e-3
        assert abs(meta["phi_max_m"] - float(direct.max())) < 1e-3
        assert abs(meta["tau"] - float(service.pass_tau[0])) < 1e-8
        area = meta["burned_cells"] * (meta["grid"]["cell_size_m"] ** 2) / 1e6
        assert abs(meta["burned_area_km2"] - area) < 1e-9
        assert 42.2 < meta["burned_area_km2"] < 42.5

        via_service = service.predict(start)
        assert np.allclose(via_service.phi, direct)

        mask = client.post("/predict/mask", headers=headers, json={"timestamp": start})
        assert mask.status_code == 200, mask.text
        assert "attachment;" in mask.headers["content-disposition"]
        assert len(mask.content) < direct.nbytes
        with np.load(io.BytesIO(mask.content)) as payload:
            assert payload["burned"].dtype == np.uint8
            assert np.array_equal(payload["burned"], burned.astype(np.uint8))
            assert np.allclose(payload["x"], service.x)
            assert np.allclose(payload["y"], service.y)

        end_response = client.post("/predict", headers=headers, json={"timestamp": end})
        assert end_response.status_code == 200, end_response.text
        assert abs(end_response.json()["tau"] - 1.0) < 1e-8

        before_window = (pd.Timestamp(start) - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
        after_window = (pd.Timestamp(end) + pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
        for bad in (before_window, after_window, start + "Z", "not-a-timestamp", "2024-01-08T00:00:00"):
            rejected = client.post("/predict", headers=headers, json={"timestamp": bad})
            assert rejected.status_code == 422, (bad, rejected.status_code, rejected.text)
            assert "warning" in rejected.json()
        extra = client.post(
            "/predict",
            headers=headers,
            json={"timestamp": start, "w_pde": 1},
        )
        assert extra.status_code == 422, extra.text

    assert file_sha256(CHECKPOINT) == before


def main() -> None:
    _auth_units()
    _http()
    print("api smoke tests passed")
    print(f"sha256 {EXPECTED_SHA256}")


if __name__ == "__main__":
    main()
