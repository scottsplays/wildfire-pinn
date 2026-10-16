"""Call a running research API. Start the server before this script."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

from deployment.wildfirepinn_api.examples.client import example_request
from deployment.wildfirepinn_api.perimeter import geometry_is_valid


def _request(url: str, key: str | None, payload: dict | None = None, method: str | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    chosen = method or ("POST" if payload is not None else "GET")
    request = urllib.request.Request(url, data=data, method=chosen)
    if payload is not None:
        request.add_header("content-type", "application/json")
    if key is not None:
        request.add_header("x-api-key", key)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def main() -> None:
    base = os.environ.get("WILDFIREPINN_API_URL", "http://127.0.0.1:8080")
    key = os.environ["WILDFIREPINN_API_KEY"]
    status, health = _request(f"{base}/health", None)
    if status != 200 or health.get("status") != "ok":
        raise SystemExit(f"health failed: {status} {health}")
    status, models = _request(f"{base}/models", key)
    if status != 200 or "ray_inverse_solver" not in models["models"]:
        raise SystemExit(f"models failed: {status}")
    status, forecast = _request(f"{base}/forecast", key, example_request())
    if status != 200:
        raise SystemExit(f"forecast failed: {status} {forecast}")
    if forecast["forecast_time_s"] != 600.0:
        raise SystemExit("forecast horizon was not returned")
    if not geometry_is_valid(forecast["perimeter"]):
        raise SystemExit("perimeter is not a valid geometry")
    if forecast["operational_wildfire_service"]:
        raise SystemExit("service claimed to be operational")
    if forecast["model_id"] != "ray_inverse_solver":
        raise SystemExit(f"default example did not select the ray hybrid: {forecast['model_id']}")
    if forecast["units"]["length"] != "m" or forecast["coordinate_system"] != "local_cartesian_m":
        raise SystemExit(f"units or coordinates were unexpected: {forecast['units']} {forecast['coordinate_system']}")
    status, version = _request(f"{base}/version", None)
    if status != 200 or version.get("cloud_deployment") != "not_deployed":
        raise SystemExit(f"version failed: {status} {version}")
    status, predicted = _request(f"{base}/predict", key, {"instances": [example_request()]})
    if status != 200 or predicted["predictions"][0]["model_id"] != "ray_inverse_solver":
        raise SystemExit(f"vertex predict failed: {status} {predicted}")
    pinn_body = {
        "model": "direct_pinn_experimental",
        "coordinate_system": "local_cartesian_m",
        "supported_benchmark": "experiment2a3_f1_spatial_product",
        "forecast_time_s": 4000,
    }
    status, pinn = _request(f"{base}/forecast", key, pinn_body)
    if status != 200 or not geometry_is_valid(pinn["perimeter"]) or pinn["grid"]["shape"] != [201, 201]:
        raise SystemExit(f"PINN benchmark failed: {status} {pinn}")
    geographic = example_request()
    geographic["coordinate_system"] = "projected_meters"
    geographic["crs"] = "EPSG:4326"
    status, _geo = _request(f"{base}/forecast", key, geographic)
    if status != 422:
        raise SystemExit(f"geographic CRS was not rejected: {status}")
    too_far = example_request()
    too_far["forecast_time_s"] = 20000
    status, _horizon = _request(f"{base}/forecast", key, too_far)
    if status != 422:
        raise SystemExit(f"horizon limit was not enforced: {status}")
    status, _denied = _request(f"{base}/forecast", "wrong-key-0123456789", example_request())
    if status != 401:
        raise SystemExit(f"invalid key was not rejected: {status}")
    print("smoke ok", forecast["model_id"], forecast["burned_area_m2"], "pinn", pinn["model_version"])


if __name__ == "__main__":
    try:
        main()
    except KeyError:
        sys.exit("Set WILDFIREPINN_API_KEY to the same value the server is using.")
