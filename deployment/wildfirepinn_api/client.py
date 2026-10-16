"""Teammate client for the research forecast API.

ray_inverse_solver is the default. It is a numerical hybrid: radial speeds
go into the Godunov level-set solver. It is not a PINN.

direct_pinn_experimental is a separate call, forecast_benchmark(). That
checkpoint accepts only the Experiment 2A.3 circle.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import numpy as np

MODEL_SUPPORT = {
    "ray_inverse_solver": {
        "kind": "numerical hybrid",
        "genuine_pde_residual_pinn": False,
        "minimum_perimeters": 2,
        "prescribed_rate": False,
        "notes": "Default. Estimates radial speeds. Extrapolates outside the observed annulus.",
    },
    "constant_rate_solver": {
        "kind": "numerical hybrid",
        "genuine_pde_residual_pinn": False,
        "minimum_perimeters": 1,
        "prescribed_rate": "optional constant rate_m_s",
        "notes": "One speed everywhere. Two perimeters are required when rate_m_s is omitted.",
    },
    "fourier_inverse_solver": {
        "kind": "numerical hybrid",
        "genuine_pde_residual_pinn": False,
        "minimum_perimeters": 2,
        "prescribed_rate": "family sin_kx_cos_ky and wavelength_m 1500",
        "notes": "Specialized. The 1500 m product family must be named in the request.",
    },
    "direct_pinn_experimental": {
        "kind": "PDE-residual PINN",
        "genuine_pde_residual_pinn": True,
        "minimum_perimeters": 0,
        "prescribed_rate": "training benchmark only",
        "notes": "Use forecast_benchmark(). Perimeters, grids, and other rate fields are rejected.",
    },
}

PINN_BENCHMARK = "experiment2a3_f1_spatial_product"


class ForecastClientError(Exception):
    def __init__(self, message: str, status_code: int | None = None, detail=None):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail


def synthetic_circle(radius_m: float, n: int = 48) -> list[list[float]]:
    """Closed ring in local Cartesian meters, centered at the origin."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = [[float(radius_m * np.cos(angle)), float(radius_m * np.sin(angle))] for angle in theta]
    ring.append(ring[0])
    return ring


def _ring(perimeter) -> list[list[float]]:
    if isinstance(perimeter, dict):
        if perimeter.get("type") != "Polygon":
            raise ForecastClientError("each perimeter must be a GeoJSON Polygon in meters")
        coordinates = perimeter.get("coordinates")
        if not coordinates or len(coordinates) != 1:
            raise ForecastClientError("each perimeter must have one exterior ring")
        raw = coordinates[0]
    else:
        raw = perimeter
    points = []
    for point in raw:
        if len(point) != 2:
            raise ForecastClientError("each perimeter position is [x_m, y_m]")
        x_m = float(point[0])
        y_m = float(point[1])
        if not np.isfinite(x_m) or not np.isfinite(y_m):
            raise ForecastClientError("perimeter coordinates must be finite meters")
        points.append([x_m, y_m])
    if len(points) < 4:
        raise ForecastClientError("a perimeter ring needs at least three positions plus the closing point")
    if points[0] != points[-1]:
        points.append(points[0])
    return points


def infer_grid(perimeters, resolution_m: float = 20.0, pad_m: float = 400.0) -> dict:
    """Square-cell meter grid around the rings, padded so a short forecast can grow."""
    rings = [_ring(perimeter) for perimeter in perimeters]
    xs = [point[0] for ring in rings for point in ring]
    ys = [point[1] for ring in rings for point in ring]
    spans = (
        (min(xs) - pad_m, max(xs) + pad_m),
        (min(ys) - pad_m, max(ys) + pad_m),
    )
    resolution = float(resolution_m)
    for low, high in spans:
        if (high - low) / resolution + 1 > 201:
            resolution = (high - low) / 200.0
    if resolution < 10.0 or resolution > 200.0:
        raise ForecastClientError("the inferred cell size falls outside 10 m to 200 m. Pass grid= explicitly.")
    grid = {"resolution_m": float(resolution)}
    for name, (low, high) in zip(("x", "y"), spans):
        cells = max(2, int(np.ceil((high - low) / resolution - 1e-9)))
        if cells + 1 > 201:
            raise ForecastClientError("the padded domain exceeds 201 nodes. Pass a coarser grid.")
        grid[f"{name}_min_m"] = float(low)
        grid[f"{name}_max_m"] = float(low + cells * resolution)
    return grid


def build_forecast_body(
    observed_perimeters,
    timestamps,
    forecast_horizon,
    model: str = "ray_inverse_solver",
    coordinate_system: str = "local_cartesian_m",
    crs: str | None = None,
    grid: dict | None = None,
    rate_parameters: dict | None = None,
) -> dict:
    """JSON body for POST /forecast. forecast_horizon is seconds after the last perimeter."""
    if model == "direct_pinn_experimental":
        raise ForecastClientError(
            "direct_pinn_experimental does not accept observed perimeters. "
            "Call forecast_benchmark(forecast_time_s) for the Experiment 2A.3 circle."
        )
    if model not in MODEL_SUPPORT:
        raise ForecastClientError(f"unknown model {model}")
    if len(observed_perimeters) != len(timestamps):
        raise ForecastClientError("observed_perimeters and timestamps must have the same length")
    needed = 2 if model in {"ray_inverse_solver", "fourier_inverse_solver"} else 1
    if model == "constant_rate_solver" and rate_parameters and rate_parameters.get("rate_m_s") is not None:
        needed = 1
    if len(observed_perimeters) < needed:
        raise ForecastClientError(f"{model} needs at least {needed} perimeters")
    horizon = float(forecast_horizon)
    if horizon <= 0.0 or horizon > 8000.0:
        raise ForecastClientError("forecast_horizon must be in (0, 8000] seconds")
    times = [float(moment) for moment in timestamps]
    if len(set(times)) != len(times) or times != sorted(times):
        raise ForecastClientError("timestamps must be strictly increasing")
    forecast_time = times[-1] + horizon
    if forecast_time > 20000.0:
        raise ForecastClientError("forecast time exceeds 20000 s")
    rings = [_ring(perimeter) for perimeter in observed_perimeters]
    body = {
        "model": model,
        "coordinate_system": coordinate_system,
        "crs": crs,
        "grid": grid if grid is not None else infer_grid(rings),
        "observations": [
            {"time_s": moment, "perimeter": {"type": "Polygon", "coordinates": [ring]}}
            for moment, ring in zip(times, rings)
        ],
        "forecast_time_s": forecast_time,
    }
    if rate_parameters is not None:
        body["rate_parameters"] = rate_parameters
    return body


def _endpoint(url: str | None) -> str:
    return (url or os.environ.get("WILDFIREPINN_API_URL") or "http://127.0.0.1:8080").rstrip("/")


def _key(api_key: str | None) -> str:
    key = api_key if api_key is not None else os.environ.get("WILDFIREPINN_API_KEY", "")
    if len(key) < 16:
        raise ForecastClientError(
            "Set WILDFIREPINN_API_KEY to at least 16 characters. Do not put the key in source code."
        )
    return key


def _post(path: str, body: dict, url: str | None, api_key: str | None, transport) -> dict:
    base = _endpoint(url)
    key = _key(api_key)
    if transport is not None:
        status, payload = transport(base, key, path, body)
    else:
        status, payload = _urllib_post(base, key, path, body)
    if status >= 400:
        detail = payload.get("detail") if isinstance(payload, dict) else payload
        raise ForecastClientError(f"forecast request failed ({status}): {detail}", status_code=status, detail=detail)
    return payload


def _urllib_post(base: str, key: str, path: str, body: dict) -> tuple[int, dict]:
    request = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "x-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return exc.code, payload


def forecast(
    observed_perimeters,
    timestamps,
    forecast_horizon,
    model: str = "ray_inverse_solver",
    coordinate_system: str = "local_cartesian_m",
    crs: str | None = None,
    grid: dict | None = None,
    rate_parameters: dict | None = None,
    url: str | None = None,
    api_key: str | None = None,
    transport=None,
) -> dict:
    """Forecast from observed perimeters.

    Returns the API JSON: perimeter geometry, burned area, model id, units,
    warnings, and unsupported-region notes. uncertainty is null.
    """
    body = build_forecast_body(
        observed_perimeters,
        timestamps,
        forecast_horizon,
        model=model,
        coordinate_system=coordinate_system,
        crs=crs,
        grid=grid,
        rate_parameters=rate_parameters,
    )
    return _post("/forecast", body, url, api_key, transport)


def forecast_benchmark(
    forecast_time_s: float,
    url: str | None = None,
    api_key: str | None = None,
    transport=None,
) -> dict:
    """Evaluate the F1 seed-0 PINN on its training circle. Time is seconds from ignition, in [0, 4000]."""
    moment = float(forecast_time_s)
    if moment < 0.0 or moment > 4000.0:
        raise ForecastClientError("the PINN benchmark accepts forecast_time_s in [0, 4000]")
    body = {
        "model": "direct_pinn_experimental",
        "coordinate_system": "local_cartesian_m",
        "supported_benchmark": PINN_BENCHMARK,
        "forecast_time_s": moment,
    }
    return _post("/forecast", body, url, api_key, transport)
