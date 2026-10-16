"""One forecast path for the constant, ray, Fourier, and hard-IC PINN models.

The hidden true rate is never an argument. Forecast times are not placed in the
fitting view. Fourier keeps the start with the smaller training-contour error.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import experiments.wildfirepinn_v3.experiment3.inverse  # torch before matplotlib
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import contour_distances
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from experiments.wildfirepinn_v3.experiment3.inverse import constant_rate, fit_parametric, ray_rate
from experiments.wildfirepinn_v3.experiment3.observations import contour_center
from deployment.wildfirepinn_api.limits import DISCLAIMER, FOURIER_WAVELENGTH_M
from deployment.wildfirepinn_api.perimeter import burned_area_m2, perimeter_geometry

HERE = Path(__file__).resolve().parent
REGISTRY = json.loads((HERE / "model_registry.json").read_text(encoding="utf-8"))
EXPERIMENT3_CONFIG = json.loads(
    (HERE.parents[1] / "experiments" / "wildfirepinn_v3" / "experiment3" / "configs.json").read_text(encoding="utf-8")
)


@dataclass
class ForecastOutput:
    phi: np.ndarray
    x: np.ndarray
    y: np.ndarray
    model_id: str
    model_version: str
    geometry: dict
    burned_area_m2: float
    polygon_area_m2: float
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    unsupported_region: dict = field(default_factory=dict)
    estimation_s: float = 0.0
    propagation_s: float = 0.0
    rate: np.ndarray | None = None


def model_record(model_id: str) -> dict:
    try:
        return REGISTRY["models"][model_id]
    except KeyError as exc:
        raise KeyError(model_id) from exc


def default_model_id() -> str:
    return str(REGISTRY["default_model"])


def _mesh(x, y):
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return xx, yy


def _training_symmetric(rate: np.ndarray, view: dict) -> float | None:
    times = [moment for moment in view["times"] if moment > 0.0]
    if not times:
        return None
    solved = propagate(view["phi0"], view["x"], view["y"], rate, times)
    values = []
    for moment in times:
        distances = contour_distances(solved[moment], view["fields"][moment], view["x"], view["y"])
        if distances["symmetric_m"] is None:
            return None
        values.append(float(distances["symmetric_m"]))
    return float(np.mean(values))


def _finish(model_id: str, view_or_none, phi, x, y, rate, estimation_s, propagation_s, extra_warnings, diagnostics, unsupported) -> ForecastOutput:
    record = model_record(model_id)
    geometry = perimeter_geometry(phi, x, y)
    from shapely.geometry import shape

    polygon_area = float(shape(geometry).area)
    warnings = [DISCLAIMER, *record["limitations"], *extra_warnings]
    diagnostics = dict(diagnostics)
    diagnostics["rate_min_m_s"] = None if rate is None else float(np.min(rate))
    diagnostics["rate_max_m_s"] = None if rate is None else float(np.max(rate))
    diagnostics["rate_mean_m_s"] = None if rate is None else float(np.mean(rate))
    if view_or_none is not None and rate is not None and "training_mean_symmetric_distance_m" not in diagnostics:
        diagnostics["training_mean_symmetric_distance_m"] = _training_symmetric(rate, view_or_none)
    return ForecastOutput(
        phi=phi,
        x=np.asarray(x),
        y=np.asarray(y),
        model_id=model_id,
        model_version=record["version"],
        geometry=geometry,
        burned_area_m2=burned_area_m2(phi, x, y),
        polygon_area_m2=polygon_area,
        warnings=warnings,
        limitations=list(record["limitations"]),
        diagnostics=diagnostics,
        unsupported_region=unsupported,
        estimation_s=float(estimation_s),
        propagation_s=float(propagation_s),
        rate=None if rate is None else np.asarray(rate, dtype=np.float64),
    )


def _exterior_fraction(view: dict) -> float:
    last = view["fields"][view["times"][-1]]
    return float(np.mean(np.asarray(last) >= 0.0))


def fit_constant_rate(view: dict, prescribed_rate: float | None) -> tuple[np.ndarray, dict, float]:
    x0, y0 = contour_center(view["phi0"], view["x"], view["y"])
    started = time.perf_counter()
    if prescribed_rate is None:
        value = constant_rate(view, x0, y0)
        how = "estimated_from_mean_radial_growth"
    else:
        value = float(prescribed_rate)
        how = "prescribed"
    elapsed = time.perf_counter() - started
    rate = np.full(view["phi0"].shape, value, dtype=np.float64)
    return rate, {"rate_m_s": value, "rate_source": how, "estimated_front_center_m": [x0, y0]}, elapsed


def forecast_constant(view: dict, forecast_s: float, prescribed_rate: float | None) -> ForecastOutput:
    rate, diagnostics, estimation_s = fit_constant_rate(view, prescribed_rate)
    started = time.perf_counter()
    field = propagate(view["phi0"], view["x"], view["y"], rate, [forecast_s])[float(forecast_s)]
    propagation_s = time.perf_counter() - started
    unsupported = {
        "kind": "spatially_uniform_rate",
        "exterior_cell_fraction": _exterior_fraction(view),
        "detail": "The same speed is used inside, on, and outside the observed front.",
    }
    return _finish(
        "constant_rate_solver",
        view,
        field,
        view["x"],
        view["y"],
        rate,
        estimation_s,
        propagation_s,
        [],
        diagnostics,
        unsupported,
    )


def fit_ray_rate(view: dict) -> tuple[np.ndarray, dict, float]:
    x0, y0 = contour_center(view["phi0"], view["x"], view["y"])
    started = time.perf_counter()
    painted = ray_rate(view, x0, y0, int(EXPERIMENT3_CONFIG["ray_bins"]))
    elapsed = time.perf_counter() - started
    return painted["rate"], {
        "clipped_cells": int(painted["clipped_cells"]),
        "estimated_front_center_m": [x0, y0],
        "ray_bins": int(EXPERIMENT3_CONFIG["ray_bins"]),
    }, elapsed


def forecast_ray(view: dict, forecast_s: float) -> ForecastOutput:
    rate, diagnostics, estimation_s = fit_ray_rate(view)
    started = time.perf_counter()
    field = propagate(view["phi0"], view["x"], view["y"], rate, [forecast_s])[float(forecast_s)]
    propagation_s = time.perf_counter() - started
    exterior = _exterior_fraction(view)
    unsupported = {
        "kind": "radial_extrapolation",
        "exterior_cell_fraction": exterior,
        "detail": (
            "Cells outside the last observed burned set take the endpoint radial speed "
            "of their angle bin. That region was not crossed by the training front."
        ),
    }
    return _finish(
        "ray_inverse_solver",
        view,
        field,
        view["x"],
        view["y"],
        rate,
        estimation_s,
        propagation_s,
        [f"Clipped {int(diagnostics['clipped_cells'])} cells into [0.001, 0.15] m/s."],
        diagnostics,
        unsupported,
    )


def fit_fourier_rate(view: dict) -> tuple[np.ndarray, dict, float]:
    """Fit the frozen 1500 m product family. Callers must keep forecast times out of ``view``."""
    if abs(float(EXPERIMENT3_CONFIG["diagnostic_wavelength_m"]) - FOURIER_WAVELENGTH_M) > 1e-6:
        raise RuntimeError("Experiment 3 wavelength config drifted from the validated 1500 m family")
    x0, y0 = contour_center(view["phi0"], view["x"], view["y"])
    xx, yy = _mesh(view["x"], view["y"])
    baseline = constant_rate(view, x0, y0)
    starts = []
    for raw_r0, amplitude in EXPERIMENT3_CONFIG["fourier_starts"]:
        starts.append((baseline if raw_r0 is None else float(raw_r0), float(amplitude)))
    candidates = []
    started = time.perf_counter()
    for r0_init, amplitude_init in starts:
        fitted = fit_parametric(view, xx, yy, EXPERIMENT3_CONFIG, "product", r0_init, amplitude_init)
        candidates.append({
            "r0_init": r0_init,
            "amplitude_init": amplitude_init,
            "r0": fitted["r0"],
            "amplitude": fitted["amplitude"],
            "training_mean_symmetric_distance_m": fitted["train_symmetric_m"],
            "evaluations": fitted["evaluations"],
            "train_seconds": fitted["train_seconds"],
            "optimizer_success": fitted["success"],
            "rate": fitted["rate"],
        })
    estimation_s = time.perf_counter() - started
    selected = min(candidates, key=lambda item: item["training_mean_symmetric_distance_m"])
    rate = selected["rate"]
    public = []
    for item in candidates:
        public.append({key: value for key, value in item.items() if key != "rate"})
    diagnostics = {
        "selected_by": "minimum_training_symmetric_distance",
        "selected_r0_m_s": selected["r0"],
        "selected_amplitude": selected["amplitude"],
        "training_mean_symmetric_distance_m": selected["training_mean_symmetric_distance_m"],
        "candidates": public,
        "wavelength_m": FOURIER_WAVELENGTH_M,
        "estimated_front_center_m": [x0, y0],
    }
    return rate, diagnostics, estimation_s


def forecast_fourier(view: dict, forecast_s: float) -> ForecastOutput:
    rate, diagnostics, estimation_s = fit_fourier_rate(view)
    started = time.perf_counter()
    field = propagate(view["phi0"], view["x"], view["y"], rate, [forecast_s])[float(forecast_s)]
    propagation_s = time.perf_counter() - started
    unsupported = {
        "kind": "prescribed_functional_family",
        "exterior_cell_fraction": _exterior_fraction(view),
        "detail": (
            "R is the 1500 m product formula at the fitted coefficients everywhere, "
            "including cells the training front never reached."
        ),
    }
    return _finish(
        "fourier_inverse_solver",
        None,
        field,
        view["x"],
        view["y"],
        rate,
        estimation_s,
        propagation_s,
        ["Family and wavelength were prescribed. They were not inferred from the perimeters."],
        diagnostics,
        unsupported,
    )


def forecast_pinn(forecast_s: float) -> ForecastOutput:
    from deployment.wildfirepinn_api.pinn_adapter import predict_phi_meters, status

    info = status()
    if not info["available"]:
        raise RuntimeError(info["reason"] or "direct PINN is disabled")
    started = time.perf_counter()
    phi, x, y = predict_phi_meters(float(forecast_s))
    elapsed = time.perf_counter() - started
    unsupported = {
        "kind": "training_benchmark_only",
        "exterior_cell_fraction": None,
        "detail": (
            "This checkpoint forecasts only the Experiment 2A.3 spatial-product circle. "
            "It does not consume the request perimeter."
        ),
    }
    return _finish(
        "direct_pinn_experimental",
        None,
        phi,
        x,
        y,
        None,
        0.0,
        elapsed,
        [
            "Inference only. R is not estimated. Seed 0 of the F1 hard-IC PINN.",
            f"Checkpoint SHA-256 {info['sha256']}.",
        ],
        {
            "sha256": info["sha256"],
            "benchmark": "experiment2a3_f1_spatial_product",
            "normalization": "X=(x-0)/4000, Y=(y-0)/4000, tau=t/4000, phi_meters = L * Phi",
            "domain_m": [-2000.0, 2000.0],
            "ignition_m": [200.0, -150.0],
            "initial_radius_m": 400.0,
        },
        unsupported,
    )
