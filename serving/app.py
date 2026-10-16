"""HTTPS API for frozen WildfirePINN v1.

Routes:
  GET  /health
  GET  /model-info
  POST /predict
  POST /predict/mask

Prediction routes require a bearer token. /health stays open inside the
process so a local probe can see that the model loaded. On Cloud Run the
whole service, including /health, stays private because the service is
deployed with --no-allow-unauthenticated.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from serving.auth import UNAUTHORIZED, AuthError, assert_auth_configured, require_invoker
from serving.inference import EXPERIMENTAL_WARNING, MODEL_VERSION, FrozenV1, RequestError

log = logging.getLogger("wildfirepinn.serving")
ROOT_HEADER = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "X-Experimental-Model": "wildfirepinn-v1",
}


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: str = Field(min_length=10, max_length=40)


def _caller(authorization: Annotated[str | None, Header()] = None) -> dict:
    return require_invoker(authorization)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    assert_auth_configured()
    app.state.model = FrozenV1.load(Path(__file__).resolve().parents[1])
    log.info("serving %s training_enabled=false", MODEL_VERSION)
    yield
    app.state.model = None


app = FastAPI(
    title="WildfirePINN v1",
    version=MODEL_VERSION,
    description=EXPERIMENTAL_WARNING,
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def _secure_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.update(ROOT_HEADER)
    if request.url.path != "/health":
        log.info("%s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.exception_handler(AuthError)
async def _auth_error(_request: Request, exc: AuthError):
    return JSONResponse(
        status_code=401,
        content={"detail": exc.detail},
        headers={"WWW-Authenticate": "Bearer", **ROOT_HEADER},
    )


@app.exception_handler(RequestError)
async def _request_error(_request: Request, exc: RequestError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "warning": EXPERIMENTAL_WARNING},
        headers=ROOT_HEADER,
    )


@app.exception_handler(Exception)
async def _unexpected(_request: Request, exc: Exception):
    log.exception("inference request failed: %s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content={"detail": "Inference failed.", "warning": EXPERIMENTAL_WARNING},
        headers=ROOT_HEADER,
    )


@app.exception_handler(RequestValidationError)
async def _validation_error(_request: Request, _exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request body must be JSON with one field, timestamp, inside the Kincade model window.",
            "warning": EXPERIMENTAL_WARNING,
        },
        headers=ROOT_HEADER,
    )


@app.get("/health")
def health(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        return JSONResponse(status_code=503, content={"status": "loading"}, headers=ROOT_HEADER)
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "experimental": True,
        "training_enabled": False,
    }


@app.get("/model-info")
def model_info(request: Request, _caller: Annotated[dict, Depends(_caller)]):
    return request.app.state.model.info()


@app.post("/predict")
def predict(body: PredictRequest, request: Request, _caller: Annotated[dict, Depends(_caller)]):
    model: FrozenV1 = request.app.state.model
    prediction = model.predict(body.timestamp)
    return model.metadata(prediction)


@app.post("/predict/mask")
def predict_mask(body: PredictRequest, request: Request, _caller: Annotated[dict, Depends(_caller)]):
    model: FrozenV1 = request.app.state.model
    prediction = model.predict(body.timestamp)
    return Response(
        content=model.mask_bytes(prediction),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{model.mask_filename(prediction)}"',
            **ROOT_HEADER,
        },
    )
