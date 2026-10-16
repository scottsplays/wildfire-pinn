"""Request contract shared with the backend FastAPI ML Adapter.

Mirrors 'Fight Fire With AI - AI/ML Model Local Integration' (sponsor guide,
Oct 2026): raw JSON or Vertex AI ``{"instances": [...]}``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_HORIZONS = (0, 1, 3, 6)
MODEL_TYPE = "pinn"
MODEL_VERSION = "0.1.0-physics-baseline"
BAY_AREA_BBOX = (-122.9, 36.9, -121.5, 38.3)


class ServiceError(Exception):
    """An error that maps to an HTTP status and a JSON ``detail``."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class AdditionalParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    confidence_threshold: float = Field(0.0, ge=0.0, le=1.0)
    members: int = Field(10, ge=1, le=32)
    fire_threshold: float = Field(0.5, gt=0.0, le=1.0)
    resolution_m: float = Field(90.0, ge=30.0, le=1000.0)
    margin_m: float = Field(6000.0, ge=1000.0, le=30000.0)
    rate_log_sigma: float = Field(0.5, ge=0.0, le=2.0)
    wind_dir_sigma_deg: float = Field(20.0, ge=0.0, le=90.0)
    seed: int | None = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bbox: list[float]
    time_horizon: int
    model_type: str
    date: str
    additional_params: AdditionalParams = Field(default_factory=AdditionalParams)

    @field_validator("bbox")
    @classmethod
    def _bbox(cls, value: list[float]) -> list[float]:
        if len(value) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
        min_lon, min_lat, max_lon, max_lat = (float(v) for v in value)
        if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat] in EPSG:4326 with min < max")
        return [min_lon, min_lat, max_lon, max_lat]

    @field_validator("time_horizon")
    @classmethod
    def _horizon(cls, value: int) -> int:
        if not 0 <= value <= max(SUPPORTED_HORIZONS):
            raise ValueError(f"time_horizon must be between 0 and {max(SUPPORTED_HORIZONS)} hours")
        return value

    @field_validator("model_type")
    @classmethod
    def _model(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"unet", "pinn", "rl", "ensemble"}:
            raise ValueError("model_type must be one of unet, pinn, rl, ensemble")
        return value

    @field_validator("date")
    @classmethod
    def _date(cls, value: str) -> str:
        text = value.strip()
        try:
            return dt.date.fromisoformat(text[:10]).isoformat()
        except ValueError as exc:
            raise ValueError("date must be ISO-8601, e.g. 2026-10-16") from exc

    @model_validator(mode="after")
    def _only_pinn(self) -> "PredictionRequest":
        if self.model_type != MODEL_TYPE:
            raise ValueError(
                f"this container serves model_type 'pinn' only; got '{self.model_type}'"
            )
        return self

    @property
    def horizons(self) -> list[int]:
        return [h for h in SUPPORTED_HORIZONS if h <= self.time_horizon]


def unwrap(payload: Any) -> tuple[list[dict], bool]:
    """Return (instances, came_from_vertex)."""
    if isinstance(payload, dict) and "instances" in payload:
        instances = payload["instances"]
        if not isinstance(instances, list) or not instances:
            raise ServiceError(400, "'instances' must be a non-empty list")
        return instances, True
    if isinstance(payload, dict):
        return [payload], False
    raise ServiceError(400, "body must be a JSON object")
