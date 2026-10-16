"""Research forecast API.

Import the inverse module before any matplotlib import. On this Windows
workstation the opposite order has aborted the interpreter.
"""

from __future__ import annotations

import copy
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import experiments.wildfirepinn_v3.experiment3.inverse  # noqa: F401  torch before matplotlib
from deployment.wildfirepinn_api import API_VERSION
from deployment.wildfirepinn_api.interface import (
    REGISTRY,
    forecast_constant,
    forecast_fourier,
    forecast_pinn,
    forecast_ray,
)
from deployment.wildfirepinn_api.limits import DISCLAIMER, FORECAST_TIMEOUT_S, MAX_BODY_BYTES
from deployment.wildfirepinn_api.perimeter import rasterize_polygon
from deployment.wildfirepinn_api.pinn_adapter import status as pinn_status
from deployment.wildfirepinn_api.service.auth import (
    assert_key_configured,
    predict_uses_platform_iam,
    require_api_key,
)
from deployment.wildfirepinn_api.service.schemas import ForecastRequest, PredictEnvelope
from deployment.wildfirepinn_api.service.validate import build_axes, validate_forecast

log = logging.getLogger("wildfirepinn_api")


class _RedactKey(logging.Filter):
    """Drop the API key if a log line ever contains it."""

    def filter(self, record: logging.LogRecord) -> bool:
        key = os.environ.get("WILDFIREPINN_API_KEY", "")
        if not key:
            return True
        message = record.getMessage()
        if key not in message:
            return True
        record.msg = message.replace(key, "[redacted]")
        record.args = ()
        return True


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def observation_view(fields: dict, x: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    items = sorted((float(moment), np.asarray(phi, dtype=np.float64)) for moment, phi in fields.items())
    origin = items[0][0]
    relative = {moment - origin: phi for moment, phi in items}
    return {
        "times": [moment - origin for moment, _phi in items],
        "fields": relative,
        "phi0": relative[0.0],
        "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64),
    }, origin


def run_model(body: ForecastRequest):
    validate_forecast(body)
    if body.model == "direct_pinn_experimental":
        return forecast_pinn(float(body.forecast_time_s)), "local_cartesian_m", None
    x, y = build_axes(body.grid)
    fields = {}
    for observation in body.observations:
        fields[float(observation.time_s)] = rasterize_polygon(observation.perimeter.model_dump(), x, y)
    view, origin = observation_view(fields, x, y)
    horizon = float(body.forecast_time_s) - origin
    if body.model == "constant_rate_solver":
        prescribed = None
        if body.rate_parameters is not None and body.rate_parameters.rate_m_s is not None:
            prescribed = float(body.rate_parameters.rate_m_s)
        return forecast_constant(view, horizon, prescribed), body.coordinate_system, body.crs
    if body.model == "ray_inverse_solver":
        return forecast_ray(view, horizon), body.coordinate_system, body.crs
    if body.model == "fourier_inverse_solver":
        return forecast_fourier(view, horizon), body.coordinate_system, body.crs
    raise ValueError("unsupported model")


def _response(result, forecast_time_s: float, coordinate_system: str, crs, elapsed: float) -> dict:
    mask = _write_mask(result)
    warnings = list(result.warnings)
    warnings.append(
        "burned_area_m2 counts cells with phi < 0. perimeter_polygon_area_m2 is the zero-contour polygon. "
        "Both use meters. Neither quantity is a longitude-latitude area."
    )
    payload = {
        "forecast_time_s": float(forecast_time_s),
        "perimeter": result.geometry,
        "geometry_kind": "cartesian_perimeter_meters",
        "coordinate_order": ["x_m", "y_m"],
        "burned_area_m2": result.burned_area_m2,
        "burned_area_km2": result.burned_area_m2 / 1.0e6,
        "perimeter_polygon_area_m2": result.polygon_area_m2,
        "mask_artifact": mask,
        "model_id": result.model_id,
        "model_version": result.model_version,
        "units": {"length": "m", "time": "s", "area": "m^2", "rate": "m/s"},
        "coordinate_system": coordinate_system,
        "crs": crs,
        "elapsed_inference_s": float(elapsed),
        "warnings": warnings,
        "uncertainty": None,
        "unsupported_region": result.unsupported_region,
        "diagnostics": result.diagnostics,
        "grid": {
            "x_min_m": float(result.x[0]),
            "x_max_m": float(result.x[-1]),
            "y_min_m": float(result.y[0]),
            "y_max_m": float(result.y[-1]),
            "resolution_m": float(result.x[1] - result.x[0]),
            "shape": [int(result.phi.shape[0]), int(result.phi.shape[1])],
        },
        "experimental": True,
        "operational_wildfire_service": False,
    }
    return _plain(payload)


def _write_mask(result) -> str | None:
    directory = os.environ.get("WILDFIREPINN_MASK_DIR", "").strip()
    if not directory:
        return None
    folder = Path(directory)
    if not folder.is_dir():
        log.warning("WILDFIREPINN_MASK_DIR is not a directory; no mask was written")
        return None
    name = f"mask-{uuid.uuid4().hex}.npz"
    np.savez_compressed(
        folder / name,
        burned=(np.asarray(result.phi) < 0).astype(np.uint8),
        x=np.asarray(result.x),
        y=np.asarray(result.y),
    )
    return name


def public_registry() -> dict:
    payload = copy.deepcopy(REGISTRY)
    info = pinn_status()
    pinn = payload["models"]["direct_pinn_experimental"]
    pinn["available"] = bool(info["available"])
    pinn["disabled_reason"] = info["reason"]
    pinn["sha256"] = info["sha256"]
    payload["experimental"] = True
    payload["operational_wildfire_service"] = False
    payload["disclaimer"] = DISCLAIMER
    return payload


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    log.addFilter(_RedactKey())
    assert_key_configured()
    if predict_uses_platform_iam():
        if len(os.environ.get("WILDFIREPINN_API_KEY", "")) < 16:
            log.info(
                "Vertex IAM serving started without an API key. "
                "POST /predict uses platform IAM. /models and /forecast reject every caller."
            )
        else:
            log.info("POST /predict relies on platform IAM. POST /forecast still requires the API key.")
    info = pinn_status()
    if info["available"]:
        log.info("direct PINN checkpoint loaded version=%s", info["version"])
    else:
        log.info("direct PINN disabled: %s", info["reason"])
    yield


docs_url = "/docs" if os.environ.get("WILDFIREPINN_ENABLE_DOCS") == "1" else None
app = FastAPI(
    title="WildfirePINN research API",
    version=API_VERSION,
    description=DISCLAIMER,
    lifespan=lifespan,
    docs_url=docs_url,
    redoc_url=None,
    openapi_url="/openapi.json" if docs_url else None,
)


@app.middleware("http")
async def limit_body(request: Request, call_next):
    raw = request.headers.get("content-length")
    if raw is not None:
        try:
            size = int(raw)
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
        if size > MAX_BODY_BYTES:
            return JSONResponse(status_code=413, content={"detail": "request body exceeds 1 MB"})
    return await call_next(request)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "wildfirepinn-research-api",
        "experimental": True,
        "operational_wildfire_service": False,
    }


@app.get("/version")
def version():
    return {
        "api_version": API_VERSION,
        "service": "wildfirepinn-research-api",
        "default_model": REGISTRY["default_model"],
        "experimental": True,
        "operational_wildfire_service": False,
        "cloud_deployment": "not_deployed",
        "disclaimer": DISCLAIMER,
    }


@app.get("/models")
def models(request: Request):
    require_api_key(request)
    return public_registry()


def _execute_forecast(body: ForecastRequest) -> dict:
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(run_model, body)
            result, coordinate_system, crs = future.result(timeout=FORECAST_TIMEOUT_S)
    except FuturesTimeout as exc:
        raise HTTPException(status_code=504, detail="forecast exceeded the 60 second job limit") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    elapsed = time.perf_counter() - started
    return _response(result, body.forecast_time_s, coordinate_system, crs, elapsed)


@app.post("/forecast")
def forecast(body: ForecastRequest, request: Request):
    require_api_key(request)
    log.info(
        "forecast requested model=%s time_s=%s observations=%s",
        body.model,
        body.forecast_time_s,
        len(body.observations),
    )
    payload = _execute_forecast(body)
    log.info("forecast completed model=%s elapsed_s=%.3f", payload["model_id"], payload["elapsed_inference_s"])
    return payload


def _register_predict(route: str) -> None:
    @app.post(route, name=f"predict:{route}")
    def predict(body: PredictEnvelope, request: Request):
        if not predict_uses_platform_iam():
            require_api_key(request)
        log.info("vertex predict instances=%s", len(body.instances))
        predictions = [_execute_forecast(instance) for instance in body.instances]
        return {"predictions": predictions}

    return None


_register_predict("/predict")
_alias = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
if _alias != "/predict":
    _register_predict(_alias)
_health_alias = os.environ.get("AIP_HEALTH_ROUTE", "/health")
if _health_alias != "/health":
    app.add_api_route(_health_alias, health, methods=["GET"])
