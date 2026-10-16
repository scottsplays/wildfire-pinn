"""Request checks. Geographic coordinates are rejected rather than treated as meters."""

from __future__ import annotations

import re

import numpy as np

from deployment.wildfirepinn_api.limits import (
    FOURIER_WAVELENGTH_M,
    GEOGRAPHIC_CRS,
    MAX_CELLS_PER_AXIS,
    MAX_DOMAIN_SIDE_M,
    MAX_FORECAST_TIME_S,
    MAX_HORIZON_S,
    MAX_OBSERVATIONS,
    MAX_RATE_M_S,
    MAX_VERTICES,
    MIN_CELLS_PER_AXIS,
    MIN_RATE_M_S,
    PINN_BENCHMARK,
)


def build_axes(grid) -> tuple[np.ndarray, np.ndarray]:
    x = _axis(grid.x_min_m, grid.x_max_m, grid.resolution_m, "x")
    y = _axis(grid.y_min_m, grid.y_max_m, grid.resolution_m, "y")
    if abs(float(x[1] - x[0]) - float(y[1] - y[0])) > 1e-6:
        raise ValueError("grid cells must be square")
    return x, y


def _axis(lo: float, hi: float, resolution: float, name: str) -> np.ndarray:
    span = float(hi) - float(lo)
    if span <= 0:
        raise ValueError(f"{name} max must be greater than {name} min")
    if span > MAX_DOMAIN_SIDE_M:
        raise ValueError(f"{name} span exceeds {MAX_DOMAIN_SIDE_M:.0f} m")
    count = int(round(span / float(resolution))) + 1
    if count < MIN_CELLS_PER_AXIS or count > MAX_CELLS_PER_AXIS:
        raise ValueError(
            f"{name} axis would have {count} nodes; the limit is {MIN_CELLS_PER_AXIS} to {MAX_CELLS_PER_AXIS}"
        )
    spacing = span / (count - 1)
    if abs(spacing - float(resolution)) / float(resolution) > 0.02:
        raise ValueError(
            f"resolution {resolution} m does not divide the {name} span {span} m. "
            "Choose bounds that are an integer number of cells apart."
        )
    return np.linspace(float(lo), float(hi), count, dtype=np.float64)


def validate_forecast(body) -> None:
    _check_crs(body.coordinate_system, body.crs)
    if body.model == "direct_pinn_experimental":
        _validate_pinn(body)
        return
    if body.supported_benchmark is not None:
        raise ValueError("supported_benchmark is only valid for direct_pinn_experimental")
    if body.grid is None:
        raise ValueError("grid is required for numerical forecasts")
    if len(body.observations) > MAX_OBSERVATIONS:
        raise ValueError(f"at most {MAX_OBSERVATIONS} perimeters are accepted")
    if body.forecast_time_s > MAX_FORECAST_TIME_S:
        raise ValueError(f"forecast_time_s exceeds {MAX_FORECAST_TIME_S:.0f} s")
    times = [float(item.time_s) for item in body.observations]
    if len(times) != len(set(times)):
        raise ValueError("observation times must be unique")
    if times != sorted(times):
        raise ValueError("observations must be in increasing time order")
    vertices = 0
    for item in body.observations:
        vertices += _check_ring(item.perimeter.coordinates)
    if vertices > MAX_VERTICES:
        raise ValueError(f"perimeters exceed {MAX_VERTICES} vertices")
    if not times:
        raise ValueError("at least one perimeter is required")
    if float(body.forecast_time_s) <= times[-1]:
        raise ValueError("forecast_time_s must be later than every observation. Future perimeters are not training data.")
    if float(body.forecast_time_s) - times[-1] > MAX_HORIZON_S:
        raise ValueError(f"forecast horizon exceeds {MAX_HORIZON_S:.0f} s")
    _validate_rate(body, len(times))


def _check_crs(coordinate_system: str, crs: str | None) -> None:
    if coordinate_system == "local_cartesian_m":
        if crs is not None:
            raise ValueError("local_cartesian_m does not accept a CRS. Coordinates are meters, not degrees.")
        return
    if coordinate_system != "projected_meters":
        raise ValueError("unsupported coordinate system")
    if crs is None or re.fullmatch(r"EPSG:\d+", crs) is None:
        raise ValueError("projected_meters requires a CRS like EPSG:3310. This service does not reproject.")
    label = crs.upper()
    if label in GEOGRAPHIC_CRS or "4326" in label or "WGS" in label or "GEOG" in label:
        raise ValueError(
            "geographic CRS inputs are rejected. Degrees are not meters, and this service does not reproject."
        )


def _check_ring(coordinates) -> int:
    if len(coordinates) != 1:
        raise ValueError("each perimeter must be one exterior ring. Holes are not accepted.")
    ring = coordinates[0]
    if len(ring) < 4:
        raise ValueError("a perimeter ring needs at least four positions, including the repeated close")
    if list(ring[0]) != list(ring[-1]):
        raise ValueError("perimeter ring must be closed: the first and last positions must match")
    for point in ring:
        if len(point) != 2:
            raise ValueError("each perimeter position is [x_m, y_m]")
        if not np.isfinite(point[0]) or not np.isfinite(point[1]):
            raise ValueError("perimeter coordinates must be finite meters")
    return len(ring)


def _validate_pinn(body) -> None:
    if body.supported_benchmark != PINN_BENCHMARK:
        raise ValueError(
            "direct_pinn_experimental accepts only "
            f"supported_benchmark={PINN_BENCHMARK}. It is disabled for every other request."
        )
    if body.observations:
        raise ValueError("this PINN checkpoint does not accept a user perimeter")
    if body.rate_parameters is not None:
        raise ValueError("this PINN checkpoint does not accept a rate field")
    if body.grid is not None:
        raise ValueError("this PINN checkpoint uses its training grid only")
    if body.crs is not None:
        raise ValueError("this PINN checkpoint uses local Cartesian meters")
    if body.coordinate_system != "local_cartesian_m":
        raise ValueError("this PINN checkpoint uses local Cartesian meters")
    if float(body.forecast_time_s) < 0.0 or float(body.forecast_time_s) > 4000.0:
        raise ValueError("this PINN checkpoint accepts forecast_time_s in [0, 4000]")


def _validate_rate(body, n_times: int) -> None:
    params = body.rate_parameters
    if body.model == "ray_inverse_solver":
        if params is not None:
            raise ValueError("ray_inverse_solver does not accept a prescribed rate")
        if n_times < 2:
            raise ValueError("ray_inverse_solver needs at least two perimeters")
        return
    if body.model == "fourier_inverse_solver":
        if n_times < 2:
            raise ValueError("fourier_inverse_solver needs at least two perimeters")
        if params is None or params.family != "sin_kx_cos_ky":
            raise ValueError(
                "fourier_inverse_solver requires rate_parameters.family = sin_kx_cos_ky "
                "and wavelength_m = 1500. It is a specialized option, not the default."
            )
        if params.wavelength_m is None or abs(float(params.wavelength_m) - FOURIER_WAVELENGTH_M) > 1e-6:
            raise ValueError("fourier_inverse_solver only accepts wavelength_m = 1500")
        if params.rate_m_s is not None:
            raise ValueError("fourier_inverse_solver estimates its coefficients. Do not pass rate_m_s.")
        return
    if body.model != "constant_rate_solver":
        raise ValueError("unsupported model")
    if params is None:
        if n_times < 2:
            raise ValueError("constant_rate_solver needs two perimeters, or one perimeter and rate_m_s")
        return
    if params.family != "constant":
        raise ValueError("constant_rate_solver only accepts a constant rate family")
    if params.wavelength_m is not None:
        raise ValueError("constant_rate_solver does not take a wavelength")
    if params.rate_m_s is None:
        if n_times < 2:
            raise ValueError("omit rate_m_s only when at least two perimeters are present")
        return
    if not (MIN_RATE_M_S <= float(params.rate_m_s) <= MAX_RATE_M_S):
        raise ValueError(f"rate_m_s must lie in [{MIN_RATE_M_S}, {MAX_RATE_M_S}]")
    if n_times < 1:
        raise ValueError("a prescribed constant rate still needs the initial perimeter")
