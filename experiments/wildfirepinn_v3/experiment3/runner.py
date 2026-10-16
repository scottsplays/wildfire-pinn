"""Inverse rate experiment: estimate R from early perimeters and forecast with the solver.

    python -m unittest experiments.wildfirepinn_v3.test_experiment3
    python -m experiments.wildfirepinn_v3.experiment3 --phase all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import R0_M_S, spatial_rate_np
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import score_against_reference
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from experiments.wildfirepinn_v3.experiment3.inverse import constant_rate, fit_parametric, ray_rate, train_rate_network
from experiments.wildfirepinn_v3.experiment3.observations import contour_center, perturb_view, segment_speeds, training_view
from experiments.wildfirepinn_v3.experiment3.reference import circle_problem, make_reference
from wildfire_pinn.geometry import zero_contours

HERE = Path(__file__).resolve().parent
FORECAST = (3000.0, 4000.0)


def load_config() -> dict:
    return json.loads((HERE / "configs.json").read_text(encoding="utf-8"))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _contour_points(phi, x, y) -> np.ndarray | None:
    lines = [line for line in zero_contours(phi, x, y) if len(line) >= 8]
    if not lines:
        return None
    return np.concatenate(lines, axis=0)


def growth_split(phi, x, y, x0: float, y0: float, r0: float) -> dict:
    pts = _contour_points(phi, x, y)
    if pts is None:
        return {"high_r_growth_m": None, "low_r_growth_m": None}
    radial = np.hypot(pts[:, 0] - x0, pts[:, 1] - y0) - r0
    local = spatial_rate_np(pts[:, 0], pts[:, 1])
    high = local >= R0_M_S
    low = ~high
    return {
        "high_r_growth_m": float(radial[high].mean()) if high.any() else None,
        "low_r_growth_m": float(radial[low].mean()) if low.any() else None,
    }


def _gap(growth: dict, reference: dict) -> dict:
    if None in (growth["high_r_growth_m"], growth["low_r_growth_m"], reference["high_r_growth_m"], reference["low_r_growth_m"]):
        return {"gap_m": None, "reference_gap_m": None, "gap_error_m": None, "correct_sign": False}
    gap = growth["high_r_growth_m"] - growth["low_r_growth_m"]
    truth = reference["high_r_growth_m"] - reference["low_r_growth_m"]
    return {
        "gap_m": float(gap),
        "reference_gap_m": float(truth),
        "gap_error_m": float(gap - truth),
        "correct_sign": bool(gap > 0.0 and truth > 0.0),
    }


def score_fields(predicted: dict, reference: dict, times: list[float], x0: float, y0: float, r0: float) -> dict:
    scored = {}
    x, y = reference["x"], reference["y"]
    for moment in times:
        row = score_against_reference(predicted[moment], reference["fields"][moment], x, y)
        growth = growth_split(predicted[moment], x, y, x0, y0, r0)
        truth = growth_split(reference["fields"][moment], x, y, x0, y0, r0)
        row.update(growth)
        row.update(_gap(growth, truth))
        scored[f"{moment:g}"] = row
    return scored


def region_masks(phi0, phi_end) -> dict[str, np.ndarray]:
    interior = np.asarray(phi0) < 0
    burned = np.asarray(phi_end) < 0
    return {"interior": interior, "annulus": burned & ~interior, "exterior": ~burned}


def rate_accuracy(learned: np.ndarray, truth: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    def block(mask: np.ndarray) -> dict:
        if int(mask.sum()) < 10:
            return {"rmse_m_s": None, "corr": None, "cells": int(mask.sum())}
        error = learned[mask] - truth[mask]
        left = learned[mask]
        right = truth[mask]
        corr = None
        if float(np.std(left)) > 1e-8 and float(np.std(right)) > 1e-8:
            corr = float(np.corrcoef(left, right)[0, 1])
        return {"rmse_m_s": float(np.sqrt(np.mean(error ** 2))), "corr": corr, "cells": int(mask.sum())}

    stats = {name: block(mask) for name, mask in masks.items()}
    stats["all"] = block(np.ones(truth.shape, dtype=bool))
    stats["min_m_s"] = float(np.min(learned))
    stats["max_m_s"] = float(np.max(learned))
    stats["mean_m_s"] = float(np.mean(learned))
    return stats


def propagate_rate(phi0, x, y, rate, times) -> tuple[dict, float]:
    started = time.perf_counter()
    fields = propagate(phi0, x, y, rate, times)
    return fields, time.perf_counter() - started


def _record(name: str, study: str, seed: int, forecast: dict, train_fit: dict | None, rate_stats: dict | None, train_seconds: float, forecast_seconds: float, extra: dict | None = None) -> dict:
    row = {
        "name": name,
        "study": study,
        "seed": seed,
        "forecast": forecast,
        "train_fit": train_fit,
        "rate": rate_stats,
        "train_seconds": train_seconds,
        "forecast_seconds": forecast_seconds,
    }
    if extra:
        row.update(extra)
    return row


def evaluate_learned(name, study, seed, rate, view, reference, train_seconds, extra=None) -> tuple[dict, dict]:
    times = sorted(set(view["times"]) | set(FORECAST))
    fields, elapsed = propagate_rate(view["phi0"], view["x"], view["y"], rate, times)
    x0, y0 = reference["problem"].x0, reference["problem"].y0
    r0 = reference["problem"].r0
    forecast = score_fields(fields, reference, list(FORECAST), x0, y0, r0)
    train_times = [moment for moment in view["times"] if moment > 0.0]
    # Training fit is against the perimeters that were shown to the learner.
    shown = {"fields": view["fields"], "x": view["x"], "y": view["y"]}
    train_fit = {}
    for moment in train_times:
        train_fit[f"{moment:g}"] = score_against_reference(fields[moment], shown["fields"][moment], view["x"], view["y"])
    masks = region_masks(view["phi0"], view["fields"][view["times"][-1]])
    stats = rate_accuracy(rate, reference["rate_true_eval_only"], masks)
    record = _record(name, study, seed, forecast, train_fit, stats, train_seconds, elapsed, extra)
    return record, fields


def speed_diagnostic(view: dict, x0: float, y0: float, n_bins: int) -> dict:
    segments = segment_speeds(view, x0, y0, n_bins)
    if len(segments) < 2:
        return {"segments": len(segments)}
    early, late = segments[0], segments[-1]
    both = np.isfinite(early["speed_m_s"]) & np.isfinite(late["speed_m_s"])
    discrepancy = late["speed_m_s"][both] - early["speed_m_s"][both]
    average = 0.5 * (early["speed_m_s"][both] + late["speed_m_s"][both])
    angle = early["angle_rad"][both]
    radius = late["radius_mid_m"][both]
    truth_mid = spatial_rate_np(x0 + radius * np.cos(angle), y0 + radius * np.sin(angle))
    outside = radius + 150.0
    truth_out = spatial_rate_np(x0 + outside * np.cos(angle), y0 + outside * np.sin(angle))

    def corr(left, right):
        if len(left) < 8 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
            return None
        return float(np.corrcoef(left, right)[0, 1])

    return {
        "early_interval_s": [early["t0"], early["t1"]],
        "late_interval_s": [late["t0"], late["t1"]],
        "speed_change_rms_m_s": float(np.sqrt(np.mean(discrepancy ** 2))),
        "corr_early_late": corr(early["speed_m_s"][both], late["speed_m_s"][both]),
        "corr_average_with_true_R_on_late_midradius": corr(average, truth_mid),
        "corr_average_with_true_R_150m_farther_out": corr(average, truth_out),
        "angles_rad": angle,
        "average_speed_m_s": average,
        "true_mid_m_s": truth_mid,
        "true_outside_m_s": truth_out,
        "early_speed_m_s": early["speed_m_s"][both],
        "late_speed_m_s": late["speed_m_s"][both],
    }


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _forecast_cell(row: dict, moment: str, key: str, digits=3):
    return _fmt((row.get("forecast") or {}).get(moment, {}).get(key), digits)


def write_report(payload: dict):
    rows = payload["runs"]
    lines = [
        "# Experiment 3",
        "",
        "A hybrid inverse problem on the Experiment 2A spatial rate. "
        "Training sees perimeters at 0, 1000, and 2000 s. "
        "Forecasts at 3000 and 4000 s are held out. "
        "The true rate field is used only to build those perimeters and to score R after training. "
        "The level-set equation is phi_t + R|grad phi| = 0, integrated by the Godunov solver.",
        "",
        "The two-parameter diagnostic is told the 1500 m product form sin(kx) cos(ky). "
        "The ray model and the MLP are not. "
        "The MLP is trained by the discrete adjoint of the Godunov update, one step at a time, and matched to observed perimeters. "
        "Its forecast is then recomputed with the NumPy solver. A single deep autograd graph of the full rollout overflows the process stack, so that form is not used.",
        "",
        "Commands:",
        "",
        "```",
        "python -m unittest experiments.wildfirepinn_v3.test_experiment3",
        "python -m experiments.wildfirepinn_v3.experiment3 --phase all",
        "```",
        "",
        "| run | study | fit m | IoU 3000 | sym 3000 | IoU 4000 | sym 4000 | p95 4000 | area km² | gap error m | sign | R rmse annulus | R corr annulus | R rmse exterior | train s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        fit_values = list((row.get("train_fit") or {}).values())
        fit = None if not fit_values else float(np.mean([item["symmetric_distance_m"] for item in fit_values if item["symmetric_distance_m"] is not None]))
        rate = row.get("rate") or {}
        lines.append(
            "| {name} | {study} | {fit} | {i3} | {s3} | {i4} | {s4} | {p4} | {area} | {gap} | {sign} | {ra} | {ca} | {re} | {train} |".format(
                name=f"{row['name']} s{row['seed']}",
                study=row["study"],
                fit=_fmt(fit, 2),
                i3=_forecast_cell(row, "3000", "iou"),
                s3=_forecast_cell(row, "3000", "symmetric_distance_m", 2),
                i4=_forecast_cell(row, "4000", "iou"),
                s4=_forecast_cell(row, "4000", "symmetric_distance_m", 2),
                p4=_forecast_cell(row, "4000", "p95_distance_m", 2),
                area=_forecast_cell(row, "4000", "area_error_km2"),
                gap=_forecast_cell(row, "4000", "gap_error_m", 1),
                sign="yes" if (row.get("forecast") or {}).get("4000", {}).get("correct_sign") else "no",
                ra=_fmt((rate.get("annulus") or {}).get("rmse_m_s"), 4),
                ca=_fmt((rate.get("annulus") or {}).get("corr"), 3),
                re=_fmt((rate.get("exterior") or {}).get("rmse_m_s"), 4),
                train=_fmt(row.get("train_seconds"), 1),
            )
        )
    lines.extend([
        "",
        f"Reference solver time for the hidden field on the 20 m grid: {_fmt(payload.get('reference_solver_seconds'), 3)} s.",
        "",
    ])
    path = HERE / "EXPERIMENT_3_RESULTS.md"
    previous = path.read_text(encoding="utf-8") if path.is_file() else ""
    if "\n## Reading\n" in previous:
        reading = previous.split("\n## Reading\n", 1)[1].strip()
        lines.extend(["## Reading", "", reading, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _draw(ax, field, x, y, color, width):
    for line in zero_contours(field, x, y):
        if len(line) >= 2:
            ax.plot(line[:, 0], line[:, 1], color=color, lw=width)


def plot_all(payload: dict, stored: dict, reference: dict):
    figures = HERE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    x, y = reference["x"], reference["y"]
    truth = reference["fields"][4000.0]
    extent = [x[0], x[-1], y[0], y[-1]]
    window = 1100.0
    origin = reference["problem"]
    chosen = [item for item in stored.get("panels", []) if item["study"] == "main"]
    columns = max(1, len(chosen))
    fig, axes = plt.subplots(1, columns, figsize=(3.4 * columns, 3.6), squeeze=False)
    for ax, panel in zip(axes[0], chosen):
        ax.imshow(reference["rate_true_eval_only"], origin="lower", extent=extent, cmap="viridis", vmin=0.025, vmax=0.075)
        _draw(ax, truth, x, y, "white", 1.3)
        _draw(ax, panel["phi"], x, y, "crimson", 1.0)
        ax.set_xlim(origin.x0 - window, origin.x0 + window)
        ax.set_ylim(origin.y0 - window, origin.y0 + window)
        ax.set_aspect("equal")
        ax.set_title(panel["title"])
    fig.suptitle("t = 4000 s. White: hidden truth. Red: forecast.")
    fig.tight_layout()
    fig.savefig(figures / "forecast_contours.png", dpi=140)
    plt.close(fig)

    rate_panels = stored.get("rates", [])
    if rate_panels:
        fig, axes = plt.subplots(1, len(rate_panels), figsize=(3.4 * len(rate_panels), 3.6), squeeze=False)
        for ax, panel in zip(axes[0], rate_panels):
            image = ax.imshow(panel["rate"], origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=0.1)
            ax.set_title(panel["title"])
            ax.set_xlim(origin.x0 - window, origin.x0 + window)
            ax.set_ylim(origin.y0 - window, origin.y0 + window)
        fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figures / "rate_fields.png", dpi=140)
        plt.close(fig)

    diagnostic = stored.get("speed")
    if diagnostic and "angles_rad" in diagnostic:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        degrees = np.degrees(diagnostic["angles_rad"])
        ax.plot(degrees, diagnostic["early_speed_m_s"], label="0–1000 s observed", lw=1.2)
        ax.plot(degrees, diagnostic["late_speed_m_s"], label="1000–2000 s observed", lw=1.2)
        ax.plot(degrees, diagnostic["true_mid_m_s"], label="true R on the late mid-radius", lw=1.0)
        ax.plot(degrees, diagnostic["true_outside_m_s"], label="true R 150 m farther out", lw=1.0, ls="--")
        ax.set_xlabel("angle from ignition (deg)")
        ax.set_ylabel("speed (m/s)")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "identifiability_speeds.png", dpi=140)
        plt.close(fig)


def _save(payload: dict):
    clean = _jsonable(payload)
    # Drop bulky diagnostic arrays from the JSON; the figure uses the in-memory copy.
    if "speed_diagnostic" in clean:
        for key in ("angles_rad", "average_speed_m_s", "true_mid_m_s", "true_outside_m_s", "early_speed_m_s", "late_speed_m_s"):
            clean["speed_diagnostic"].pop(key, None)
    (HERE / "metrics.json").write_text(json.dumps(clean, indent=2), encoding="utf-8")
    write_report(clean)


def run(phase: str):
    config = load_config()
    def mark(message: str):
        print(message, flush=True)

    mark("start")
    times = sorted(set(config["train_times_s"]) | set(config["forecast_times_s"]) | set(config["dense_times_s"]) | set(config["sparse_times_s"]))
    problem = circle_problem(int(config["grid_n"]))
    reference = make_reference(problem, times)
    mark(f"reference solver {reference['solver_seconds']:.3f} s")
    xx, yy = np.meshgrid(reference["x"], reference["y"])
    runs: list[dict] = []
    stored = {"panels": [], "rates": [{"title": "true R", "rate": reference["rate_true_eval_only"]}]}

    def add_learned(name, study, seed, rate, view, train_seconds, extra=None, panel=False):
        record, fields = evaluate_learned(name, study, seed, rate, view, reference if study != "second_ignition" else stored["second_reference"], train_seconds, extra)
        # second ignition carries its own reference on the view via closure below.
        runs.append(record)
        if panel:
            stored["panels"].append({"study": study, "title": f"{name} s{seed}", "phi": fields[4000.0]})
        return record

    main_view = training_view(reference["fields"], config["train_times_s"], reference["x"], reference["y"])
    x0, y0 = contour_center(main_view["phi0"], main_view["x"], main_view["y"])
    r_const = constant_rate(main_view, x0, y0)
    mark(f"constant rate {r_const:.5f}")
    diagnostic = speed_diagnostic(main_view, problem.x0, problem.y0, int(config["ray_bins"]))
    mark("speed diagnostic")
    stored["speed"] = diagnostic

    if phase in {"all", "lowdim"}:
        # Persistence of the last training perimeter.
        persisted = {moment: main_view["fields"][2000.0] for moment in FORECAST}
        persistence_forecast = score_fields(persisted, reference, list(FORECAST), problem.x0, problem.y0, problem.r0)
        runs.append(_record("persistence", "main", 0, persistence_forecast, None, None, 0.0, 0.0))
        stored["panels"].append({"study": "main", "title": "persistence", "phi": main_view["fields"][2000.0]})

        constant_grid = np.full_like(reference["rate_true_eval_only"], r_const)
        add_learned("constant", "main", 0, constant_grid, main_view, 0.0, {"r0": r_const}, panel=True)

        oracle_forecast = score_fields(reference["fields"], reference, list(FORECAST), problem.x0, problem.y0, problem.r0)
        oracle_rate = rate_accuracy(reference["rate_true_eval_only"], reference["rate_true_eval_only"], region_masks(main_view["phi0"], main_view["fields"][2000.0]))
        runs.append(_record("oracle", "main", 0, oracle_forecast, None, oracle_rate, 0.0, reference["solver_seconds"]))
        stored["panels"].append({"study": "main", "title": "oracle", "phi": reference["fields"][4000.0]})

        started = time.perf_counter()
        ray = ray_rate(main_view, x0, y0, int(config["ray_bins"]))
        ray["train_seconds"] = time.perf_counter() - started
        add_learned("ray", "main", 0, ray["rate"], main_view, ray["train_seconds"], {"clipped_cells": ray["clipped_cells"]}, panel=True)
        stored["rates"].append({"title": "ray", "rate": ray["rate"]})

        fourier_rows = []
        for index, start in enumerate(config["fourier_starts"]):
            r0_init = r_const if start[0] is None else float(start[0])
            fitted = fit_parametric(main_view, xx, yy, config, "product", r0_init, float(start[1]))
            mark(f"fourier start {index} R0 {fitted['r0']:.4f} a {fitted['amplitude']:.3f} fit {fitted['train_symmetric_m']:.2f} m")
            record, fields = evaluate_learned(
                "fourier", "main", index, fitted["rate"], main_view, reference, fitted["train_seconds"],
                {"r0": fitted["r0"], "amplitude": fitted["amplitude"], "optimizer_success": fitted["success"], "evaluations": fitted["evaluations"]},
            )
            runs.append(record)
            fourier_rows.append((record, fields, fitted["rate"]))
        best = min(fourier_rows, key=lambda item: item[0]["train_fit"] and np.mean([row["symmetric_distance_m"] for row in item[0]["train_fit"].values()]))
        stored["panels"].append({"study": "main", "title": f"fourier s{best[0]['seed']}", "phi": best[1][4000.0]})
        stored["rates"].append({"title": f"fourier s{best[0]['seed']}", "rate": best[2]})

        miss_rows = []
        for r0_init, amp_init in ((r_const, 0.0), (0.02, 0.3)):
            miss = fit_parametric(main_view, xx, yy, config, "x-only", r0_init, amp_init)
            mark(f"x-only R0 {miss['r0']:.4f} b {miss['amplitude']:.3f} fit {miss['train_symmetric_m']:.2f} m")
            record, _fields = evaluate_learned(
                "x-only", "main", len(miss_rows), miss["rate"], main_view, reference, miss["train_seconds"],
                {"r0": miss["r0"], "amplitude": miss["amplitude"]},
            )
            runs.append(record)
            miss_rows.append(record)

        for study, study_times in (("sparse", config["sparse_times_s"]), ("dense", config["dense_times_s"])):
            view = training_view(reference["fields"], study_times, reference["x"], reference["y"])
            center = contour_center(view["phi0"], view["x"], view["y"])
            baseline = constant_rate(view, center[0], center[1])
            for start_index, (r0_init, amp_init) in enumerate(((baseline, 0.0), (0.02, 0.3))):
                fitted = fit_parametric(view, xx, yy, config, "product", r0_init, amp_init)
                mark(f"{study} fourier s{start_index} R0 {fitted['r0']:.4f} a {fitted['amplitude']:.3f} fit {fitted['train_symmetric_m']:.2f} m")
                add_learned("fourier", study, start_index, fitted["rate"], view, fitted["train_seconds"], {"r0": fitted["r0"], "amplitude": fitted["amplitude"]})
            painted = ray_rate(view, center[0], center[1], int(config["ray_bins"]))
            add_learned("ray", study, 0, painted["rate"], view, 0.0, {"clipped_cells": painted["clipped_cells"]})

        for amplitude in config["perturbation_m"]:
            noisy = perturb_view(main_view, float(amplitude), int(config["perturbation_harmonic"]), float(config["perturbation_phase"]))
            center = contour_center(noisy["phi0"], noisy["x"], noisy["y"])
            baseline = constant_rate(noisy, center[0], center[1])
            fitted = fit_parametric(noisy, xx, yy, config, "product", baseline, 0.0)
            study = f"perturb{int(amplitude)}"
            print(f"{study} fourier R0 {fitted['r0']:.4f} a {fitted['amplitude']:.3f} fit {fitted['train_symmetric_m']:.2f} m", flush=True)
            add_learned("fourier", study, 0, fitted["rate"], noisy, fitted["train_seconds"], {"r0": fitted["r0"], "amplitude": fitted["amplitude"], "perturbation_m": amplitude})

        second_problem = circle_problem(int(config["grid_n"]), float(config["second_ignition_m"][0]), float(config["second_ignition_m"][1]))
        second = make_reference(second_problem, times)
        stored["second_reference"] = second
        second_view = training_view(second["fields"], config["train_times_s"], second["x"], second["y"])
        sxx, syy = np.meshgrid(second["x"], second["y"])
        center = contour_center(second_view["phi0"], second_view["x"], second_view["y"])
        baseline = constant_rate(second_view, center[0], center[1])
        fitted = fit_parametric(second_view, sxx, syy, config, "product", baseline, 0.0)
        print(f"second ignition fourier R0 {fitted['r0']:.4f} a {fitted['amplitude']:.3f} fit {fitted['train_symmetric_m']:.2f} m", flush=True)
        record, _fields = evaluate_learned(
            "fourier", "second_ignition", 0, fitted["rate"], second_view, second, fitted["train_seconds"],
            {"r0": fitted["r0"], "amplitude": fitted["amplitude"]},
        )
        runs.append(record)

    if phase in {"all", "mlp"}:
        runs_dir = HERE / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        for seed in config["seeds"]:
            trained = train_rate_network(main_view, config, r_const, int(seed))
            torch.save({"state_dict": trained["model"].state_dict(), "seed": int(seed), "r_init": r_const}, runs_dir / f"rate_mlp_seed{seed}.pt")
            np.savez_compressed(runs_dir / f"rate_mlp_seed{seed}.npz", rate=trained["rate"], history=np.asarray(trained["history"]))
            record, fields = evaluate_learned("mlp", "main", int(seed), trained["rate"], main_view, reference, trained["train_seconds"], {"dt_s": trained["dt_s"]})
            runs.append(record)
            stored["panels"].append({"study": "main", "title": f"mlp s{seed}", "phi": fields[4000.0]})
            if seed == config["seeds"][0]:
                stored["rates"].append({"title": "mlp s0", "rate": trained["rate"]})
            print(
                f"mlp seed {seed} forecast IoU {record['forecast']['4000']['iou']:.3f} "
                f"sym {record['forecast']['4000']['symmetric_distance_m']:.2f}",
                flush=True,
            )
        noisy = perturb_view(main_view, 40.0, int(config["perturbation_harmonic"]), float(config["perturbation_phase"]))
        center = contour_center(noisy["phi0"], noisy["x"], noisy["y"])
        trained = train_rate_network(noisy, config, constant_rate(noisy, center[0], center[1]), 0)
        record, _fields = evaluate_learned("mlp", "perturb40", 0, trained["rate"], noisy, reference, trained["train_seconds"])
        runs.append(record)
        if "second_reference" not in stored:
            second_problem = circle_problem(int(config["grid_n"]), float(config["second_ignition_m"][0]), float(config["second_ignition_m"][1]))
            stored["second_reference"] = make_reference(second_problem, times)
        second = stored["second_reference"]
        second_view = training_view(second["fields"], config["train_times_s"], second["x"], second["y"])
        center = contour_center(second_view["phi0"], second_view["x"], second_view["y"])
        trained = train_rate_network(second_view, config, constant_rate(second_view, center[0], center[1]), 0)
        record, _fields = evaluate_learned("mlp", "second_ignition", 0, trained["rate"], second_view, second, trained["train_seconds"])
        runs.append(record)

    payload = {
        "reference_solver_seconds": reference["solver_seconds"],
        "constant_rate_m_s": r_const,
        "speed_diagnostic": diagnostic,
        "coverage": {name: int(mask.sum()) for name, mask in region_masks(main_view["phi0"], main_view["fields"][2000.0]).items()},
        "runs": runs,
    }
    plot_all(payload, stored, reference)
    _save(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "lowdim", "mlp"])
    args = parser.parse_args()
    run(args.phase)


if __name__ == "__main__":
    main()
