"""FastAPI models. Extra fields are rejected so a request cannot smuggle a rate function."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PolygonGeometry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    time_s: float = Field(ge=0)
    perimeter: PolygonGeometry


class GridSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    resolution_m: float = Field(ge=10, le=200)


class RateParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    family: Literal["constant", "sin_kx_cos_ky"]
    rate_m_s: float | None = None
    wavelength_m: float | None = None


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Literal[
        "constant_rate_solver",
        "ray_inverse_solver",
        "fourier_inverse_solver",
        "direct_pinn_experimental",
    ]
    coordinate_system: Literal["local_cartesian_m", "projected_meters"]
    crs: str | None = None
    grid: GridSpec | None = None
    observations: list[Observation] = Field(default_factory=list)
    forecast_time_s: float = Field(ge=0)
    rate_parameters: RateParameters | None = None
    supported_benchmark: str | None = None


class PredictParameters(BaseModel):
    """Vertex may send an empty parameters object. Keys are rejected."""

    model_config = ConfigDict(extra="forbid")


class PredictEnvelope(BaseModel):
    """Vertex AI custom-prediction body. Each instance is one /forecast request."""

    model_config = ConfigDict(extra="forbid")
    instances: list[ForecastRequest] = Field(min_length=1, max_length=4)
    parameters: PredictParameters | None = None
