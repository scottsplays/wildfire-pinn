"""Request contract shared with the backend FastAPI ML Adapter.

Mirrors 'Fight Fire With AI - AI/ML Model Local Integration' (sponsor guide,
revision 2, Oct 16 2026): raw JSON or Vertex AI ``{"instances": [...]}``,
PINN served on port 8082, inputs ingested natively by the service.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SUPPORTED_HORIZONS = (0, 1, 3, 6)
MODEL_TYPE = "pinn"
MODEL_VERSION = "0.2.2-physics-baseline"
DEFAULT_PORT = 8082
DEFAULT_HOUR_UTC = 21          # sponsor FAQ Q2: date-only requests mean 21:00 UTC
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
    hour: int | None = Field(None, ge=0, le=23, description="UTC hour; overrides the time in `date`")
    members: int = Field(10, ge=1, le=32, description="internal ensemble size (not the contract's `members`)")
    lookback_hours: float = Field(24.0, gt=0.0, le=120.0, description="FIRMS detections this far before the reference time form the current fire")
    search_buffer_m: float = Field(5000.0, ge=0.0, le=50000.0, description="look for the fire this far outside the bbox")
    fire_threshold: float = Field(0.5, gt=0.0, le=1.0)
    min_burnable_share: float = Field(0.25, ge=0.0, le=1.0, description="drop detection groups with less burnable fuel than this within 500 m (refineries, flares, crop burns)")
    resolution_m: float = Field(90.0, ge=30.0, le=1000.0)
    margin_m: float = Field(6000.0, ge=1000.0, le=30000.0)
    rate_log_sigma: float = Field(0.5, ge=0.0, le=2.0)
    wind_dir_sigma_deg: float = Field(20.0, ge=0.0, le=90.0)
    high_risk: float = Field(0.6, ge=0.0, le=1.0)
    moderate_risk: float = Field(0.3, ge=0.0, le=1.0)
    seed: int | None = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    bbox: list[float]
    time_horizon: int
    model_type: str
    date: str | None = None   # contract says required; the team adapter can send None -> "now"
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
        # The team adapter's PredictionRequest defaults to 24; horizons above 6 are served as 0/1/3/6.
        if not 0 <= value <= 72:
            raise ValueError("time_horizon must be between 0 and 72 hours (forecasts go to 6 h)")
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
    def _date(cls, value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        parse_reference_time(value)  # raises ValueError on garbage
        return value.strip()

    @model_validator(mode="after")
    def _only_pinn(self) -> "PredictionRequest":
        if self.model_type != MODEL_TYPE:
            raise ValueError(
                f"this service serves model_type 'pinn' only; got '{self.model_type}' "
                "(the adapter routes unet to :8081)"
            )
        return self

    @property
    def horizons(self) -> list[int]:
        return [h for h in SUPPORTED_HORIZONS if h <= self.time_horizon]

    @property
    def reference_time(self) -> dt.datetime:
        """Forecast start (horizon 0) in UTC."""
        if self.date is None:
            ref = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
        else:
            ref = parse_reference_time(self.date)
        if self.additional_params.hour is not None:
            ref = ref.replace(hour=self.additional_params.hour, minute=0, second=0, microsecond=0)
        return ref

    @property
    def day(self) -> str:
        return self.reference_time.date().isoformat()


def parse_reference_time(text: str) -> dt.datetime:
    """'2026-10-16' -> 2026-10-16T21:00Z; '2026-10-16T18:30:00Z' (or +hh:mm) -> that instant in UTC."""
    text = text.strip()
    if len(text) <= 10:
        try:
            day = dt.date.fromisoformat(text)
        except ValueError as exc:
            raise ValueError("date must be ISO-8601, e.g. 2026-10-16 or 2026-10-16T21:00:00Z") from exc
        return dt.datetime(day.year, day.month, day.day, DEFAULT_HOUR_UTC, tzinfo=dt.timezone.utc)
    try:
        moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("date must be ISO-8601, e.g. 2026-10-16 or 2026-10-16T21:00:00Z") from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=dt.timezone.utc)
    return moment.astimezone(dt.timezone.utc)


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
