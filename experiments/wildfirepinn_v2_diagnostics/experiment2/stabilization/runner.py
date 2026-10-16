"""Stabilize the Fourier hard-IC model on the Experiment 2A spatial rate.

    python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase diagnose
    python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1

F3 is the saved 2A.2 Fourier seed-0 checkpoint and is not retrained.
Training does not read the numerical field or its future front.
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

from experiments.wildfirepinn_v2_diagnostics.circle import analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment1 import HIDDEN, LAYERS, N_PDE, _sample_pde
from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.models import FourierPINN
from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.runner import (
    _reference,
    _residual_map,
)
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import circle_phi0, coded_residual, hard_ic
from experiments.wildfirepinn_v2_diagnostics.experiment2.runner import (
    EPOCHS,
    LR,
    SEED,
    _growth_by_rate,
    _probe,
    _problem,
    _times,
    predict_phi,
)
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import score_against_reference
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.banks import BANKS
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.net import BankFourierPINN
from experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization.sampling import (
    FRONT_JITTER_M,
    PredictedFrontCache,
    angular_comparison,
    component_areas,
    longest_contour,
    sample_mixture,
)
from wildfire_pinn.pinn import load_checkpoint, save_checkpoint, set_seed

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent
FOURIER_BASELINE = EXP2 / "cause_isolation_2" / "runs" / "B_seed0" / "levelset_pinn.pt"
REFRESH_EVERY = 50


def _phi0(problem, x, y):
    xx, yy = np.meshgrid(x, y)
    phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
    return phi0, (lambda xn, yn, problem=problem: circle_phi0(xn, yn, problem))


def _batch_terms(model, problem, phi0_fn, x, y, t, create_graph: bool):
    length = problem.length
    duration = problem.t_final
    tau = (t / duration).detach().requires_grad_(True)
    x_norm = ((x - problem.center_x) / length).detach().requires_grad_(True)
    y_norm = ((y - problem.center_y) / length).detach().requires_grad_(True)
    phi = hard_ic(model(tau, x_norm, y_norm), tau, phi0_fn(x_norm, y_norm))
    phi_tau, phi_x, phi_y = torch.autograd.grad(
        phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=create_graph
    )
    t_over_l = duration / length
    from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import spatial_rate_torch

    rate = spatial_rate_torch(x.detach(), y.detach())
    coded, physical, magnitude = coded_residual(phi_tau, phi_x, phi_y, rate, t_over_l)
    return {
        "coded": coded,
        "physical": physical,
        "magnitude": magnitude,
        "phi_t": phi_tau / t_over_l,
        "phi_x": phi_x,
        "phi_y": phi_y,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "rate": rate,
    }


def _hessian_stats(model, problem, phi0_fn, x, y, t) -> dict:
    """Physical second derivatives, in 1/m. |dPhi/dX| is already |d phi_meters / dx|."""
    terms = _batch_terms(model, problem, phi0_fn, x, y, t, create_graph=True)
    ones = torch.ones_like(terms["phi_x"])
    phi_xx, phi_xy = torch.autograd.grad(
        terms["phi_x"], (terms["x_norm"], terms["y_norm"]), ones, retain_graph=True
    )
    phi_yx, phi_yy = torch.autograd.grad(terms["phi_y"], (terms["x_norm"], terms["y_norm"]), ones)
    scale = 1.0 / problem.length
    xx = (phi_xx * scale).detach()
    yy = (phi_yy * scale).detach()
    xy = (phi_xy * scale).detach()
    frobenius = torch.sqrt(xx ** 2 + yy ** 2 + 2.0 * xy ** 2)
    magnitude = terms["magnitude"].detach()
    return {
        "mean_abs_grad": float(magnitude.mean()),
        "std_abs_grad": float(magnitude.std(unbiased=False)),
        "p95_abs_grad": float(torch.quantile(magnitude, 0.95)),
        "mean_hessian_per_m": float(frobenius.mean()),
        "p95_hessian_per_m": float(torch.quantile(frobenius, 0.95)),
        "mean_abs_residual_m_s": float(terms["physical"].detach().abs().mean()),
    }


def _band_points(contour: np.ndarray, problem, n: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    base = torch.tensor(contour, dtype=torch.float32)
    choice = torch.randint(0, len(base), (n,), generator=generator)
    jitter = (torch.rand(n, 2, generator=generator) * 2.0 - 1.0) * FRONT_JITTER_M
    xy = base[choice] + jitter
    return xy[:, 0], xy[:, 1]


def _numerical_hessian(field, x, y, problem) -> dict:
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    xx = (field[:, 2:] - 2.0 * field[:, 1:-1] + field[:, :-2]) / dx ** 2
    yy = (field[2:, :] - 2.0 * field[1:-1, :] + field[:-2, :]) / dy ** 2
    # Overlap interior.
    xx = xx[1:-1, :]
    yy = yy[:, 1:-1]
    band = np.abs(field[1:-1, 1:-1]) <= FRONT_JITTER_M
    if not np.any(band):
        return {"mean_hessian_per_m": None}
    frobenius = np.sqrt(xx[band] ** 2 + yy[band] ** 2)
    return {"mean_hessian_per_m": float(np.mean(frobenius)), "count": int(band.sum())}


def diagnose_model(model, problem, phi0_fn, fields, grid_x, grid_y) -> dict:
    final = predict_phi(model, problem, phi0_fn, problem.t_final, grid_x, grid_y)
    reference = fields[problem.t_final]
    pred_line = longest_contour(final, grid_x, grid_y)
    ref_line = longest_contour(reference, grid_x, grid_y)
    model.eval()
    uniform_x, uniform_y, uniform_t = _sample_pde(4096, problem, torch.device("cpu"))
    volume = _hessian_stats(model, problem, phi0_fn, uniform_x, uniform_y, uniform_t)
    predicted_band = None
    numerical_band = None
    if pred_line is not None:
        x, y = _band_points(pred_line, problem, 1500, seed=11)
        t = torch.full((x.numel(),), problem.t_final)
        predicted_band = _hessian_stats(model, problem, phi0_fn, x, y, t)
    if ref_line is not None:
        x, y = _band_points(ref_line, problem, 1500, seed=12)
        t = torch.full((x.numel(),), problem.t_final)
        numerical_band = _hessian_stats(model, problem, phi0_fn, x, y, t)
    angular = angular_comparison(
        final, reference, grid_x, grid_y, problem.x0, problem.y0, problem.r0, problem.t_final
    )
    storable = {key: value for key, value in angular.items() if key not in {"angles_rad", "signed_displacement_m", "predicted_radius_m", "reference_radius_m"}}
    near_pred = float(np.mean(np.abs(final - reference)[np.abs(final) <= FRONT_JITTER_M])) if pred_line is not None else None
    return {
        "components": component_areas(final, grid_x, grid_y),
        "volume_derivatives": volume,
        "predicted_front_derivatives": predicted_band,
        "numerical_front_derivatives_of_pinn": numerical_band,
        "numerical_field_hessian": _numerical_hessian(reference, grid_x, grid_y, problem),
        "field_error_within_100m_of_predicted_zero_m": near_pred,
        "angular": storable,
        "angles_rad": angular.get("angles_rad"),
        "signed_displacement_m": angular.get("signed_displacement_m"),
        "predicted_radius_m": angular.get("predicted_radius_m"),
        "reference_radius_m": angular.get("reference_radius_m"),
    }


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items() if not str(key).endswith("_m") or not isinstance(item, np.ndarray)}
    if isinstance(value, np.ndarray):
        return None
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _gap(growth, reference) -> dict:
    truth = float(reference["high_r_growth_m"] - reference["low_r_growth_m"])
    pred = float(growth["high_r_growth_m"] - growth["low_r_growth_m"])
    return {
        "growth_gap_m": pred,
        "reference_gap_m": truth,
        "gap_error_m": pred - truth,
        "correct_sign": bool(pred > 0.0 and truth > 0.0),
    }


def evaluate(model, problem, phi0_fn, fields, rate_grid, grid_x, grid_y) -> dict:
    spec = {"rate": "spatial", "wind": None}
    rows = []
    final = None
    for moment in _times(problem):
        pred = predict_phi(model, problem, phi0_fn, moment, grid_x, grid_y)
        row = score_against_reference(pred, fields[moment], grid_x, grid_y)
        row["t_s"] = moment
        rows.append(row)
        final = pred
    growth = _growth_by_rate(final, grid_x, grid_y, problem, rate_grid)
    reference_growth = _growth_by_rate(fields[problem.t_final], grid_x, grid_y, problem, rate_grid)
    diagnosis = diagnose_model(model, problem, phi0_fn, fields, grid_x, grid_y)
    record = {
        "initial_iou": rows[0]["iou"],
        "final_iou": rows[-1]["iou"],
        "final_symmetric_m": rows[-1]["symmetric_distance_m"],
        "final_p95_m": rows[-1]["p95_distance_m"],
        "final_area_error_km2": rows[-1]["area_error_km2"],
        "volume_residual_m_s": _probe(model, problem, phi0_fn, spec, grid_x, grid_y, fields, "uniform"),
        "predicted_front_residual_m_s": diagnosis["predicted_front_derivatives"]["mean_abs_residual_m_s"] if diagnosis["predicted_front_derivatives"] else None,
        "numerical_front_residual_m_s": diagnosis["numerical_front_derivatives_of_pinn"]["mean_abs_residual_m_s"] if diagnosis["numerical_front_derivatives_of_pinn"] else None,
        "pinn_growth": growth,
        "numerical_growth": reference_growth,
        "times": rows,
        "parameter_count": sum(int(param.numel()) for param in model.parameters()),
        "gradient_stats": {
            "volume": diagnosis["volume_derivatives"],
            "predicted_front": diagnosis["predicted_front_derivatives"],
        },
        "roughness": diagnosis["angular"],
        "components": diagnosis["components"],
        "numerical_field_hessian": diagnosis["numerical_field_hessian"],
    }
    record.update(_gap(growth, reference_growth))
    record["_arrays"] = diagnosis
    return record


def _bank_of(name: str) -> str:
    for key in ("F1", "F2", "F3", "F4"):
        if name.startswith(key):
            return key
    raise ValueError(name)


def _new_model(name: str, problem, seed: int):
    set_seed(seed)
    bank = _bank_of(name)
    if bank == "F3":
        return FourierPINN(HIDDEN, LAYERS, problem.length)
    return BankFourierPINN(BANKS[bank](problem.length), name, HIDDEN, LAYERS)


def _load_f3(problem):
    model = FourierPINN(HIDDEN, LAYERS, problem.length)
    load_checkpoint(FOURIER_BASELINE, model, torch.device("cpu"))
    model.eval()
    return model


def _finished(run_dir: Path, epochs: int, seed: int):
    history_path = run_dir / "history.json"
    checkpoint = run_dir / "levelset_pinn.pt"
    if not checkpoint.is_file() or not history_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history.get("loss", [])) != epochs or history.get("seed") != seed:
        return None
    return history


def _eikonal_loss(model, problem, phi0_fn, cache: PredictedFrontCache):
    if cache.xy is None or len(cache.xy) < 8:
        return None
    choice = torch.randint(0, len(cache.xy), (1024,))
    base = cache.xy[choice]
    jitter = (torch.rand(1024, 2) * 2.0 - 1.0) * FRONT_JITTER_M
    xy = (base + jitter).detach()
    x, y = xy[:, 0], xy[:, 1]
    t = cache.time[choice].detach()
    inside = (
        (x >= -problem.half_extent) & (x <= problem.half_extent)
        & (y >= -problem.half_extent) & (y <= problem.half_extent)
        & (torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m)
    )
    if int(inside.sum()) < 64:
        return None
    terms = _batch_terms(model, problem, phi0_fn, x[inside], y[inside], t[inside], create_graph=True)
    return torch.mean((terms["magnitude"] - 1.0) ** 2)


def train_config(name: str, problem, phi0_fn, grid_x, grid_y, epochs: int, seed: int, sampling: str, eikonal_weight: float):
    run_dir = HERE / "runs" / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = _finished(run_dir, epochs, seed)
    if existing is not None:
        model = _new_model(name, problem, seed)
        load_checkpoint(run_dir / "levelset_pinn.pt", model, torch.device("cpu"))
        model.eval()
        print(f"resume finished {name} seed {seed}", flush=True)
        return model, existing, float(existing.get("train_seconds") or 0.0)
    variant = _bank_of(name)
    model = _new_model(name, problem, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {
        "loss": [],
        "eikonal": [],
        "seed": seed,
        "variant": name,
        "sampling": sampling,
        "eikonal_weight": eikonal_weight,
        "growth": [],
    }
    cache = PredictedFrontCache()
    device = torch.device("cpu")
    model.train()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        if sampling == "front" and (epoch == 1 or epoch % REFRESH_EVERY == 0):
            cache.refresh(model, problem, phi0_fn, grid_x, grid_y, _times(problem), predict_phi)
        if sampling == "front":
            x, y, t = sample_mixture(N_PDE, problem, cache, device)
        else:
            x, y, t = _sample_pde(N_PDE, problem, device)
        terms = _batch_terms(model, problem, phi0_fn, x, y, t, create_graph=True)
        loss = torch.mean(terms["coded"] ** 2)
        eikonal_value = 0.0
        if eikonal_weight > 0.0:
            if epoch == 1 or epoch % REFRESH_EVERY == 0:
                cache.refresh(model, problem, phi0_fn, grid_x, grid_y, _times(problem), predict_phi)
            penalty = _eikonal_loss(model, problem, phi0_fn, cache)
            if penalty is not None:
                loss = loss + eikonal_weight * penalty
                eikonal_value = float(penalty.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(torch.mean(terms["coded"] ** 2).detach()))
        history["eikonal"].append(eikonal_value)
        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            pred = predict_phi(model, problem, phi0_fn, problem.t_final, grid_x, grid_y)
            growth = _growth_by_rate(pred, grid_x, grid_y, problem, None)
            high_m, low_m = growth["high_r_growth_m"], growth["low_r_growth_m"]
            gap_m = float("nan") if high_m is None or low_m is None else high_m - low_m
            history["growth"].append({"epoch": epoch, **growth})
            print(
                f"{name} seed {seed} epoch {epoch:4d} | pde {history['loss'][-1]:.3e} | eik {eikonal_value:.3e} | gap {gap_m:.1f} m",
                flush=True,
            )
            save_checkpoint(run_dir / "levelset_pinn.pt", model, {"variant": name, "epochs": epochs, "seed": seed, "sampling": sampling, "eikonal_weight": eikonal_weight})
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    elapsed = time.perf_counter() - started
    history["train_seconds"] = elapsed
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    model.eval()
    return model, history, elapsed


def _public(record: dict) -> dict:
    clean = {key: value for key, value in record.items() if key != "_arrays"}
    return json.loads(json.dumps(clean, default=lambda item: item.item() if isinstance(item, np.generic) else None))


def _store(record: dict):
    path = HERE / "metrics.json"
    rows = []
    if path.is_file():
        rows = [row for row in json.loads(path.read_text(encoding="utf-8")) if not (row.get("name") == record["name"] and row.get("seed") == record["seed"])]
    rows.append(_public(record))
    rows.sort(key=lambda row: (row["name"], row["seed"]))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def write_report(rows: list[dict]):
    lines = [
        "# Experiment 2A.3",
        "",
        "Fourier hard-IC models on the Experiment 2A spatial rate. "
        "F3 is the saved 2A.2 Fourier seed-0 checkpoint. "
        "Training does not use the numerical field or its future front. "
        "The level-set equation and the hard initial condition are unchanged.",
        "",
        "Commands:",
        "",
        "```",
        "python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2a3",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase diagnose",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F2",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F4",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase G --base F1",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase H --base F1 --eikonal-weight 1e-3",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase H --base F1 --eikonal-weight 1e-2",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1 --seed 1",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.stabilization --phase F --config F1 --seed 2",
        "```",
        "",
        "F1 is the two diagonal modes of sin(kx) cos(ky). F2 adds the axis-aligned fundamentals. "
        "F4 is the axis-aligned fundamental and first harmonic, with no diagonals. "
        "High-frequency fraction is angular Fourier power in modes 6 and above, divided by all nonzero modes.",
        "",
        "| run | init IoU | final IoU | symmetric m | p95 m | area km² | high m | low m | gap error m | sign | volume residual | front residual | HF fraction | components | other m² | params | train s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rough = row.get("roughness") or {}
        spectrum = (rough.get("predicted_spectrum") or {}) if isinstance(rough, dict) else {}
        components = row.get("components") or {}
        lines.append(
            "| {name} | {ii} | {fi} | {sym} | {p95} | {area} | {high} | {low} | {err} | {sign} | {vol} | {front} | {hf} | {comp} | {other} | {params} | {train} |".format(
                name=f"{row['name']} s{row['seed']}",
                ii=_fmt(row["initial_iou"], 3),
                fi=_fmt(row["final_iou"], 3),
                sym=_fmt(row["final_symmetric_m"], 2),
                p95=_fmt(row["final_p95_m"], 2),
                area=_fmt(row["final_area_error_km2"], 3),
                high=_fmt(row["pinn_growth"]["high_r_growth_m"], 1),
                low=_fmt(row["pinn_growth"]["low_r_growth_m"], 1),
                err=_fmt(row["gap_error_m"], 1),
                sign="yes" if row["correct_sign"] else "no",
                vol=_fmt(row["volume_residual_m_s"], 4),
                front=_fmt(row["predicted_front_residual_m_s"], 4),
                hf=_fmt(spectrum.get("high_frequency_fraction"), 3),
                comp=components.get("burned_components", "—"),
                other=_fmt(components.get("other_components_m2"), 1),
                params=row.get("parameter_count"),
                train=_fmt(row.get("train_seconds"), 1),
            )
        )
    lines.extend([
        "",
        "| run | IoU t=0 | IoU T/3 | IoU 2T/3 | IoU T | curvature 1/m | front grad | front Hessian 1/m |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        times = row.get("times") or []
        ious = [item.get("iou") for item in times]
        while len(ious) < 4:
            ious.append(None)
        front = ((row.get("gradient_stats") or {}).get("predicted_front") or {})
        rough = row.get("roughness") or {}
        lines.append(
            "| {name} | {a} | {b} | {c} | {d} | {curv} | {grad} | {hess} |".format(
                name=f"{row['name']} s{row['seed']}",
                a=_fmt(ious[0], 3),
                b=_fmt(ious[1], 3),
                c=_fmt(ious[2], 3),
                d=_fmt(ious[3], 3),
                curv=_fmt(rough.get("predicted_curvature_rms_per_m"), 4),
                grad=_fmt(front.get("mean_abs_grad"), 3),
                hess=_fmt(front.get("mean_hessian_per_m"), 4),
            )
        )
    lines.append("")
    if rows:
        ref = rows[0]["numerical_growth"]
        lines.append(
            f"Numerical growth: high {_fmt(ref['high_r_growth_m'], 1)} m, low {_fmt(ref['low_r_growth_m'], 1)} m, "
            f"gap {_fmt(ref['high_r_growth_m'] - ref['low_r_growth_m'], 1)} m. "
            "Numerical contour high-frequency fraction is 0.00117 and curvature RMS is 0.00191 1/m. "
            "Numerical-field Hessian near the front is 0.00151 1/m. Solver time on the 20 m grid is 0.076 s."
        )
        lines.append("")
    path = HERE / "EXPERIMENT_2A3_RESULTS.md"
    if path.is_file() and "\n## Reading\n" in path.read_text(encoding="utf-8"):
        reading = path.read_text(encoding="utf-8").split("\n## Reading\n", 1)[1].strip()
        lines.extend(["## Reading", "", reading, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def rebuild_figures():
    problem = _problem()
    grid_x, grid_y = problem.grid()
    phi0_field, phi0_fn = _phi0(problem, grid_x, grid_y)
    fields, rate_grid = _reference(problem, phi0_field, grid_x, grid_y)
    rows = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    models = {(row["name"], row["seed"]): _load_for_row(row, problem) for row in rows}
    plot_rows(rows, models, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
    write_report(rows)


def _draw(ax, field, grid_x, grid_y, color, width):
    from wildfire_pinn.geometry import zero_contours

    for line in zero_contours(field, grid_x, grid_y):
        if len(line) >= 2:
            ax.plot(line[:, 0], line[:, 1], color=color, lw=width)


def plot_rows(rows, models, problem, phi0_fn, fields, rate_grid, grid_x, grid_y):
    if not rows:
        return
    figures = HERE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    moment = problem.t_final
    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    window = 1100.0
    columns = len(rows)
    curves = []
    fig, axes = plt.subplots(1, columns, figsize=(3.6 * columns, 3.8), squeeze=False)
    for ax, row in zip(axes[0], rows):
        ax.imshow(rate_grid, origin="lower", extent=extent, cmap="viridis", vmin=0.02, vmax=0.08)
        _draw(ax, fields[moment], grid_x, grid_y, "white", 1.4)
        prediction = predict_phi(models[(row["name"], row["seed"])], problem, phi0_fn, moment, grid_x, grid_y)
        _draw(ax, prediction, grid_x, grid_y, "crimson", 1.1)
        angular = angular_comparison(
            prediction, fields[moment], grid_x, grid_y, problem.x0, problem.y0, problem.r0, problem.t_final
        )
        curves.append((row, angular))
        ax.set_xlim(problem.x0 - window, problem.x0 + window)
        ax.set_ylim(problem.y0 - window, problem.y0 + window)
        ax.set_aspect("equal")
        ax.set_title(f"{row['name']} s{row['seed']}")
    fig.suptitle("White: numerical. Red: PINN.")
    fig.tight_layout()
    fig.savefig(figures / "contours.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, columns, figsize=(3.6 * columns, 3.6), squeeze=False)
    image = None
    for ax, row in zip(axes[0], rows):
        grid = _residual_map(models[(row["name"], row["seed"])], problem, phi0_fn, grid_x, grid_y)
        image = ax.imshow(grid, origin="lower", extent=extent, cmap="coolwarm", vmin=-0.04, vmax=0.04)
        ax.set_xlim(problem.x0 - window, problem.x0 + window)
        ax.set_ylim(problem.y0 - window, problem.y0 + window)
        ax.set_aspect("equal")
        ax.set_title(row["name"])
    if image is not None:
        fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    fig.savefig(figures / "residual_maps.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for row, angular in curves:
        signed = angular.get("signed_displacement_m")
        angles = angular.get("angles_rad")
        if signed is None or angles is None:
            continue
        ax.plot(np.degrees(angles), signed, label=f"{row['name']} s{row['seed']}", lw=1.0)
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel("angle from ignition (deg)")
    ax.set_ylabel("predicted radius minus numerical (m)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "angular_displacement.png", dpi=140)
    plt.close(fig)

    labels = [f"{row['name']} s{row['seed']}" for row in rows]
    hf = [((row.get("roughness") or {}).get("predicted_spectrum") or {}).get("high_frequency_fraction") or 0.0 for row in rows]
    curvature = [((row.get("roughness") or {}).get("predicted_curvature_rms_per_m") or 0.0) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(labels, hf, color="0.35")
    axes[0].axhline(0.001172, color="crimson", lw=1.0, label="numerical contour")
    axes[0].set_ylabel("high-frequency radius fraction")
    axes[0].tick_params(axis="x", labelrotation=60)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(labels, curvature, color="0.35")
    axes[1].axhline(0.001911, color="crimson", lw=1.0, label="numerical contour")
    axes[1].set_ylabel("contour curvature RMS (1/m)")
    axes[1].tick_params(axis="x", labelrotation=60)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "roughness.png", dpi=140)
    plt.close(fig)


def _load_for_row(row, problem):
    if row["name"] == "F3" and row["seed"] == 0 and not (HERE / "runs" / "F3_seed0" / "levelset_pinn.pt").is_file():
        return _load_f3(problem)
    model = _new_model(row["name"], problem, row["seed"])
    load_checkpoint(HERE / "runs" / f"{row['name']}_seed{row['seed']}" / "levelset_pinn.pt", model, torch.device("cpu"))
    model.eval()
    return model


def run_diagnose():
    problem = _problem()
    grid_x, grid_y = problem.grid()
    _phi0_field, phi0_fn = _phi0(problem, grid_x, grid_y)
    fields, rate_grid = _reference(problem, _phi0_field, grid_x, grid_y)
    model = _load_f3(problem)
    record = evaluate(model, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
    record.update({"name": "F3", "seed": 0, "train_seconds": 43.0, "retrained": False, "label": "saved Fourier seed 0"})
    arrays = record.pop("_arrays")
    rows = _store(record)
    record["_arrays"] = arrays
    plot_rows([record], {("F3", 0): model}, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
    diagnosis = {
        "volume_derivatives": record["gradient_stats"]["volume"],
        "predicted_front_derivatives": record["gradient_stats"]["predicted_front"],
        "numerical_front_residual_m_s": record["numerical_front_residual_m_s"],
        "predicted_front_residual_m_s": record["predicted_front_residual_m_s"],
        "roughness": record["roughness"],
        "components": record["components"],
        "numerical_field_hessian": record["numerical_field_hessian"],
        "gap_error_m": record["gap_error_m"],
        "final_symmetric_m": record["final_symmetric_m"],
        "final_iou": record["final_iou"],
    }
    (HERE / "DIAGNOSIS.json").write_text(json.dumps(diagnosis, indent=2, default=float), encoding="utf-8")
    write_report(rows)
    rough = record["roughness"]
    print(
        f"diagnose F3 symmetric {record['final_symmetric_m']:.2f} gap error {record['gap_error_m']:.1f} "
        f"HF {rough.get('predicted_spectrum', {}).get('high_frequency_fraction')} "
        f"ref HF {rough.get('reference_spectrum', {}).get('high_frequency_fraction')} "
        f"curvature {rough.get('predicted_curvature_rms_per_m')} "
        f"front |grad| {record['gradient_stats']['predicted_front']}",
        flush=True,
    )
    return record


def run_train(name: str, epochs: int, seed: int, sampling: str, eikonal_weight: float):
    problem = _problem()
    grid_x, grid_y = problem.grid()
    phi0_field, phi0_fn = _phi0(problem, grid_x, grid_y)
    fields, rate_grid = _reference(problem, phi0_field, grid_x, grid_y)
    if name == "F3" and sampling == "uniform" and eikonal_weight == 0.0 and seed == 0:
        return run_diagnose()
    model, _history, seconds = train_config(name, problem, phi0_fn, grid_x, grid_y, epochs, seed, sampling, eikonal_weight)
    record = evaluate(model, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
    record.update({
        "name": name,
        "seed": seed,
        "train_seconds": seconds,
        "sampling": sampling,
        "eikonal_weight": eikonal_weight,
        "retrained": True,
    })
    arrays = record.get("_arrays")
    rows = _store(record)
    # Rebuild plot inputs from stored rows plus this model's arrays.
    models = {}
    plotted = []
    for row in rows:
        if row["name"] == name and row["seed"] == seed:
            row = _public(record)
            row["_arrays"] = arrays
            models[(name, seed)] = model
        else:
            models[(row["name"], row["seed"])] = _load_for_row(row, problem)
        plotted.append(row if row.get("_arrays") is not None else {**row, "_arrays": {}})
    # The stored row lost arrays. Attach them only for the current run.
    for item in plotted:
        if item["name"] == name and item["seed"] == seed:
            item["_arrays"] = arrays
    plot_rows(plotted, models, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
    write_report(rows)
    print(
        f"{name} IoU {record['final_iou']:.3f} symmetric {record['final_symmetric_m']:.2f} "
        f"gap error {record['gap_error_m']:.1f} sign {'yes' if record['correct_sign'] else 'no'} "
        f"HF {(record['roughness'].get('predicted_spectrum') or {}).get('high_frequency_fraction')}",
        flush=True,
    )
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["diagnose", "F", "G", "H", "combo", "figures"])
    parser.add_argument("--config", default="F1", choices=["F1", "F2", "F3", "F4"])
    parser.add_argument("--base", default="F1", help="Fourier bank used by G, H, and combo")
    parser.add_argument("--eikonal-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.phase == "diagnose":
        run_diagnose()
        return
    if args.phase == "figures":
        rebuild_figures()
        return
    if args.phase == "F":
        run_train(args.config, args.epochs, args.seed, "uniform", 0.0)
        return
    if args.phase == "G":
        run_train(f"{args.base}G", args.epochs, args.seed, "front", 0.0)
        return
    if args.phase == "H":
        if args.eikonal_weight <= 0.0:
            raise SystemExit("H requires a positive --eikonal-weight")
        tag = f"{args.base}H{args.eikonal_weight:.0e}".replace("+", "").replace("-", "m")
        run_train(tag, args.epochs, args.seed, "uniform", args.eikonal_weight)
        return
    if args.eikonal_weight <= 0.0:
        raise SystemExit("combo requires a positive --eikonal-weight")
    tag = f"{args.base}C{args.eikonal_weight:.0e}".replace("+", "").replace("-", "m")
    run_train(tag, args.epochs, args.seed, "front", args.eikonal_weight)


if __name__ == "__main__":
    main()
