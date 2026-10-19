"""WildfirePINN inference service (port 8082) for the Fight Fire With AI backend.

    python pinn_service/serve.py --port 8082          # from the repo root
    python serve.py --port 8082                       # from inside pinn_service/

Inputs: INGESTION=native (default; FIRMS + Open-Meteo + gridMET + bundled LANDFIRE)
or INGESTION=npz (offline replay from $FEATURE_DIR/YYYY-MM-DD.npz).

Routes (Vertex AI custom-container compatible):
    GET  /health    -> {"status": "ok", "models_loaded": true}
    POST /predict   -> GeoJSON FeatureCollection (raw request)
                       {"predictions": [FeatureCollection, ...]} (Vertex {"instances": [...]})

Vertex AI sets AIP_HTTP_PORT, AIP_HEALTH_ROUTE and AIP_PREDICT_ROUTE; they are honoured.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # allow `python serve.py` from inside the folder
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "pinn_service"

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from pinn_service.contract import DEFAULT_PORT, MODEL_VERSION, PredictionRequest, ServiceError, unwrap
from pinn_service.features import NpzSource
from pinn_service.forecaster import forecast
from pinn_service.native_ingestion import NativeSource
from pinn_service.postprocessing import active_grid, build_feature_collection

log = logging.getLogger("pinn_service")

HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")

app = FastAPI(title="WildfirePINN inference", version=MODEL_VERSION, docs_url=None, redoc_url=None)


def make_source():
    mode = os.environ.get("INGESTION", "native").lower()
    if mode == "npz":
        return NpzSource()
    if mode == "native":
        return NativeSource()
    raise ServiceError(500, f"INGESTION must be 'native' or 'npz', not {mode!r}")


def predict_one(instance: dict) -> dict:
    try:
        req = PredictionRequest.model_validate(instance)
    except ValidationError as exc:
        messages = "; ".join(f"{'.'.join(str(p) for p in e['loc']) or 'request'}: {e['msg']}" for e in exc.errors())
        raise ServiceError(422, messages) from exc
    grid = active_grid()
    probabilities, meta = forecast(req, make_source(), grid)
    params = req.additional_params
    return build_feature_collection(
        probabilities,
        ensemble_size=params.members,
        bbox=req.bbox,
        grid=grid,
        threshold=params.confidence_threshold,
        extra_metadata=meta,
        high_risk=params.high_risk,
        moderate_risk=params.moderate_risk,
    )


def handle(payload) -> tuple[int, dict]:
    """Pure function behind POST /predict; returns (status, body). Used by tests too."""
    try:
        instances, vertex = unwrap(payload)
        results = [predict_one(inst if isinstance(inst, dict) else {}) for inst in instances]
    except ServiceError as exc:
        return exc.status, {"detail": exc.detail}
    return 200, ({"predictions": results} if vertex else results[0])


@app.get(HEALTH_ROUTE)
def health() -> dict:
    grid = active_grid()
    try:
        source = make_source().describe()
    except ServiceError as exc:
        source = {"ingestion_error": exc.detail}
    from pinn_service.calibrated import learned_model

    model = learned_model()
    learned = {"learned_alpha_model": bool(model)} | ({"learned_alpha_sha256_12": model["_sha"]} if model else {})
    return {"status": "ok", "models_loaded": True, "model_version": MODEL_VERSION, "grid": grid.name} | learned | source


@app.post(PREDICT_ROUTE)
async def predict(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except ValueError:
        return JSONResponse({"detail": "body is not valid JSON"}, status_code=400)
    status, body = handle(payload)
    if status != 200:
        log.warning("predict -> %s: %s", status, body.get("detail"))
    return JSONResponse(body, status_code=status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIP_HTTP_PORT", DEFAULT_PORT)))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
