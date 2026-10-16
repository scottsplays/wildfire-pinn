"""Held-out synthetic rate fields for the inverse solvers.

Training perimeters are t = 0, 1000, 2000 s. Scores use t = 3000 and 4000 s.
The Fourier estimator always uses its prescribed 1500 m product family.
Cases B and C do not match that family. Case D is not run.

    python -m deployment.wildfirepinn_api.generalization
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import experiments.wildfirepinn_v3.experiment3.inverse  # noqa: F401  torch before matplotlib
from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import spatial_rate_np
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import contour_distances, score_against_reference
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from deployment.wildfirepinn_api.interface import fit_constant_rate, fit_fourier_rate, fit_ray_rate
from wildfire_pinn.geometry import zero_contours

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TRAIN_S = (0.0, 1000.0, 2000.0)
FORECAST_S = (3000.0, 4000.0)
GRID_N = 201
IGNITIONS = {
    "primary": (200.0, -150.0),
    "second": (-400.0, 300.0),
}


def rate_product(xx, yy):
    return spatial_rate_np(xx, yy)


def rate_shifted(xx, yy):
    """Same product shape, different wavelength and phase. Not the Fourier family."""
    wave = 2.0 * np.pi / 2200.0
    return 0.05 * (1.0 + 0.35 * np.sin(wave * xx + 0.8) * np.cos(wave * yy - 1.1))


def rate_logistic(xx, yy):
    """Smooth, positive, and not periodic."""
    return 0.03 + 0.035 / (1.0 + np.exp(-(xx + 0.25 * yy) / 700.0))


CASES = (
    {
        "id": "A_product",
        "label": "original 1500 m product",
        "family_matches_fourier_estimator": True,
        "rate": rate_product,
    },
    {
        "id": "B_shifted",
        "label": "2200 m product with a phase shift",
        "family_matches_fourier_estimator": False,
        "rate": rate_shifted,
    },
    {
        "id": "C_logistic",
        "label": "smooth nonperiodic logistic",
        "family_matches_fourier_estimator": False,
        "rate": rate_logistic,
    },
)


def _points(phi, x, y):
    lines = [line for line in zero_contours(phi, x, y) if len(line) >= 2]
    if not lines:
        return None
    return np.concatenate(lines, axis=0)


def _growth(phi, x, y, x0, y0, r0, rate_fn, threshold):
    pts = _points(phi, x, y)
    if pts is None:
        return {"high_growth_m": None, "low_growth_m": None, "gap_m": None}
    radial = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0) - r0
    local = np.asarray(rate_fn(pts[:, 0], pts[:, 1]), dtype=np.float64)
    high = local >= threshold
    low = ~high
    if not np.any(high) or not np.any(low):
        return {"high_growth_m": None, "low_growth_m": None, "gap_m": None, "threshold_m_s": float(threshold)}
    high_growth = float(np.mean(radial[high]))
    low_growth = float(np.mean(radial[low]))
    return {
        "high_growth_m": high_growth,
        "low_growth_m": low_growth,
        "gap_m": high_growth - low_growth,
        "threshold_m_s": float(threshold),
    }


def _rate_block(learned, truth, mask):
    count = int(np.sum(mask))
    if count < 10:
        return {"rmse_m_s": None, "corr": None, "cells": count}
    left = learned[mask]
    right = truth[mask]
    corr = None
    if float(np.std(left)) > 1e-8 and float(np.std(right)) > 1e-8:
        corr = float(np.corrcoef(left, right)[0, 1])
    return {
        "rmse_m_s": float(np.sqrt(np.mean((left - right) ** 2))),
        "corr": corr,
        "cells": count,
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _reference(case, ignition_name):
    x0, y0 = IGNITIONS[ignition_name]
    problem = CircleProblem(n=GRID_N, x0=x0, y0=y0, r0=400.0, t_final=4000.0, half_extent=2000.0)
    x, y = problem.grid()
    yy, xx = np.meshgrid(y, x, indexing="ij")
    phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
    truth = np.asarray(case["rate"](xx, yy), dtype=np.float64)
    if float(np.min(truth)) <= 0.0 or float(np.max(truth)) > 0.15:
        raise RuntimeError(f"{case['id']} rate leaves the positive range used by the solvers")
    started = time.perf_counter()
    fields = propagate(phi0, x, y, truth, list(TRAIN_S) + list(FORECAST_S))
    elapsed = time.perf_counter() - started
    view = {
        "times": list(TRAIN_S),
        "fields": {float(moment): fields[float(moment)] for moment in TRAIN_S},
        "phi0": fields[0.0],
        "x": x,
        "y": y,
    }
    if any(moment in view["times"] for moment in FORECAST_S):
        raise RuntimeError("forecast times leaked into the training view")
    return {
        "problem": problem,
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "truth": truth,
        "fields": fields,
        "view": view,
        "solver_seconds": elapsed,
        "clearance_m": problem.clearance_m(float(np.max(truth))),
    }


def _score_model(name, rate, estimation_s, reference, rate_fn):
    x, y = reference["x"], reference["y"]
    problem = reference["problem"]
    started = time.perf_counter()
    predicted = propagate(reference["view"]["phi0"], x, y, rate, list(FORECAST_S))
    propagation_s = time.perf_counter() - started
    train_solved = propagate(reference["view"]["phi0"], x, y, rate, [1000.0, 2000.0])
    train_values = []
    for moment in (1000.0, 2000.0):
        distances = contour_distances(train_solved[moment], reference["view"]["fields"][moment], x, y)
        train_values.append(distances["symmetric_m"])
    masks = {
        "interior": reference["fields"][0.0] < 0,
        "annulus": (reference["fields"][2000.0] < 0) & ~(reference["fields"][0.0] < 0),
        "exterior": ~(reference["fields"][2000.0] < 0),
    }
    rows = {}
    for moment in FORECAST_S:
        row = score_against_reference(predicted[moment], reference["fields"][moment], x, y)
        ref_pts = _points(reference["fields"][moment], x, y)
        threshold = float(np.median(rate_fn(ref_pts[:, 0], ref_pts[:, 1])))
        truth_growth = _growth(reference["fields"][moment], x, y, problem.x0, problem.y0, problem.r0, rate_fn, threshold)
        pred_growth = _growth(predicted[moment], x, y, problem.x0, problem.y0, problem.r0, rate_fn, threshold)
        gap_error = None
        correct_sign = None
        if truth_growth["gap_m"] is not None and pred_growth["gap_m"] is not None:
            gap_error = float(pred_growth["gap_m"] - truth_growth["gap_m"])
            correct_sign = bool(pred_growth["gap_m"] * truth_growth["gap_m"] > 0.0)
        row["growth"] = pred_growth
        row["reference_growth"] = truth_growth
        row["growth_asymmetry_error_m"] = gap_error
        row["correct_asymmetry_sign"] = correct_sign
        rows[f"{moment:g}"] = row
    return {
        "model": name,
        "forecast": rows,
        "rate_error": {mask_name: _rate_block(rate, reference["truth"], mask) for mask_name, mask in masks.items()},
        "training_mean_symmetric_distance_m": None if any(value is None for value in train_values) else float(np.mean(train_values)),
        "estimation_seconds": float(estimation_s),
        "inference_seconds": float(propagation_s),
        "final_iou": rows["4000"]["iou"],
        "final_symmetric_m": rows["4000"]["symmetric_distance_m"],
    }


def run() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    runs = []
    print("anisotropic: unsupported by the isotropic Godunov solver", flush=True)
    for case in CASES:
        for ignition_name in IGNITIONS:
            print(f"{case['id']} {ignition_name}", flush=True)
            reference = _reference(case, ignition_name)
            view = reference["view"]
            rate_fn = case["rate"]
            fits = []
            constant_rate_grid, constant_diag, constant_s = fit_constant_rate(view, None)
            fits.append(("constant_rate_solver", constant_rate_grid, constant_s, constant_diag))
            ray_grid, ray_diag, ray_s = fit_ray_rate(view)
            fits.append(("ray_inverse_solver", ray_grid, ray_s, ray_diag))
            fourier_grid, fourier_diag, fourier_s = fit_fourier_rate(view)
            fits.append(("fourier_inverse_solver", fourier_grid, fourier_s, fourier_diag))
            for name, rate, seconds, diag in fits:
                scored = _score_model(name, rate, seconds, reference, rate_fn)
                public_diag = {key: value for key, value in diag.items() if key != "rate"}
                scored["diagnostics"] = public_diag
                scored["case"] = case["id"]
                scored["ignition"] = ignition_name
                scored["family_matches_fourier_estimator"] = case["family_matches_fourier_estimator"]
                scored["clearance_m"] = reference["clearance_m"]
                scored["reference_solver_seconds"] = reference["solver_seconds"]
                runs.append(scored)
                final = scored["forecast"]["4000"]
                print(
                    f"  {name} IoU {final['iou']:.3f} sym {final['symmetric_distance_m']:.2f} m "
                    f"gap_error {final['growth_asymmetry_error_m']}",
                    flush=True,
                )
    payload = {
        "grid_n": GRID_N,
        "train_times_s": list(TRAIN_S),
        "forecast_times_s": list(FORECAST_S),
        "selection_rule": "Fourier keeps the Experiment 3 start with the smaller training-contour error. Held-out times are not used to choose it.",
        "anisotropic": {
            "status": "unsupported",
            "reason": (
                "The validated solver advances phi_t + R(x, y) |grad phi| = 0. "
                "A normal-dependent rate R(x, y, n) is a different Hamiltonian. "
                "directional.py implements a wind-advection split and was not part of the inverse fits. "
                "No anisotropic case was run."
            ),
        },
        "direct_pinn_experimental": {
            "status": "not_entered_in_the_inverse_comparison",
            "reason": (
                "The F1 seed-0 checkpoint forecasts only its training circle and does not estimate R. "
                "Its benchmark check is the unit test, not these unseen fields."
            ),
        },
        "runs": runs,
    }
    _plot(runs, plt)
    (RESULTS / "metrics.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote {RESULTS / 'metrics.json'}", flush=True)
    return payload


def _plot(runs, plt) -> None:
    labels = []
    series = {"constant_rate_solver": [], "ray_inverse_solver": [], "fourier_inverse_solver": []}
    for case in CASES:
        for ignition_name in IGNITIONS:
            labels.append(f"{case['id']}\n{ignition_name}")
            for model in series:
                match = next(item for item in runs if item["case"] == case["id"] and item["ignition"] == ignition_name and item["model"] == model)
                series[model].append(match["final_iou"])
    positions = np.arange(len(labels))
    width = 0.25
    figure, axis = plt.subplots(figsize=(10, 4.5))
    colors = {"constant_rate_solver": "#4c78a8", "ray_inverse_solver": "#f58518", "fourier_inverse_solver": "#54a24b"}
    for index, (model, values) in enumerate(series.items()):
        axis.bar(positions + (index - 1) * width, values, width=width, label=model, color=colors[model])
    axis.set_xticks(positions, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("IoU at 4000 s")
    axis.set_title("Held-out forecast IoU at 4000 s")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(RESULTS / "figures" / "forecast_iou.png", dpi=120)
    plt.close(figure)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
