"""Train and compare 2A.2 variants against the saved Experiment 2A checkpoint.

    python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment A
    python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B

A reloads the original 2A run and does not train. B, C, D, and E write only under
cause_isolation_2/runs/. The physics residual is unchanged. Future fronts and the
numerical field are evaluation targets only.
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
from experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2.optim import (
    lambda_at,
    pcgrad,
    rate_lambda,
    region_mask,
    uniform_and_balanced,
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
from wildfire_pinn.geometry import zero_contours
from wildfire_pinn.pinn import LevelSetPINN, load_checkpoint, save_checkpoint, set_seed

HERE = Path(__file__).resolve().parent
EXP2 = HERE.parent
FRONT_JITTER_M = 100.0
DIAGNOSTIC_EVERY = 50
VARIANTS = {
    "B": {"architecture": "fourier", "schedule": "full", "update": "adam", "label": "Fourier features, full spatial R"},
    "C": {"architecture": "mlp", "schedule": "staged", "update": "adam", "label": "staged lambda from constant R to spatial R"},
    "D": {"architecture": "mlp", "schedule": "full", "update": "pcgrad", "label": "PCGrad on the high-R and low-R losses"},
    "E": {"architecture": "fourier", "schedule": "staged", "update": "adam", "label": "Fourier features plus staged lambda"},
}


def _phi0(problem, x, y):
    xx, yy = np.meshgrid(x, y)
    phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
    return phi0, (lambda xn, yn, problem=problem: circle_phi0(xn, yn, problem))


def _reference(problem, phi0, x, y):
    cache = EXP2 / "cache" / "2a_reference.npz"
    times = _times(problem)
    if cache.is_file():
        data = np.load(cache)
        if data["x"].shape == x.shape and np.allclose(data["x"], x) and np.allclose(data["y"], y):
            fields = {times[index]: data[f"t{index}"] for index in range(len(times))}
            return fields, data["rate"]
    from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import spatial_rate_np
    from experiments.wildfirepinn_v2_diagnostics.solver import propagate

    xx, yy = np.meshgrid(x, y)
    rate = spatial_rate_np(xx, yy)
    return propagate(phi0, x, y, rate, times), rate


def _baseline_timing() -> dict:
    path = EXP2 / "metrics.json"
    if not path.is_file():
        return {"train_seconds": None, "solver_seconds": None}
    rows = json.loads(path.read_text(encoding="utf-8"))
    match = [row for row in rows if row.get("scenario") == "2a"]
    if not match:
        return {"train_seconds": None, "solver_seconds": None}
    return {"train_seconds": match[-1].get("train_seconds"), "solver_seconds": match[-1].get("solver_seconds")}


def _new_model(architecture: str, problem, seed: int):
    set_seed(seed)
    if architecture == "fourier":
        return FourierPINN(HIDDEN, LAYERS, problem.length)
    if architecture == "mlp":
        return LevelSetPINN(HIDDEN, LAYERS)
    raise ValueError(architecture)


def _batch_terms(model, problem, phi0_fn, x, y, t, lam: float, create_graph: bool):
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
    rate = rate_lambda(x.detach(), y.detach(), lam)
    coded, physical, magnitude = coded_residual(phi_tau, phi_x, phi_y, rate, t_over_l)
    return {
        "coded": coded,
        "physical": physical,
        "magnitude": magnitude,
        "phi_t": phi_tau / t_over_l,
        "rate": rate,
    }


def _flat(grads) -> torch.Tensor:
    return torch.cat([grad.reshape(-1) for grad in grads])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denom = float(left.norm() * right.norm())
    if denom < 1e-18:
        return None
    return float(torch.dot(left, right) / denom)


def _params(model):
    return [param for param in model.parameters() if param.requires_grad]


def _assign_grad(params, flat: torch.Tensor):
    offset = 0
    for param in params:
        count = param.numel()
        param.grad = flat[offset : offset + count].view_as(param).clone()
        offset += count


def _fixed_batch(problem, n: int, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    xs, ys, ts = [], [], []
    got = 0
    while got < n:
        count = max(n, 2048)
        x = torch.rand(count, generator=generator) * problem.length - problem.half_extent
        y = torch.rand(count, generator=generator) * problem.length - problem.half_extent
        t = torch.rand(count, generator=generator) * problem.t_final
        keep = torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m
        xs.append(x[keep])
        ys.append(y[keep])
        ts.append(t[keep])
        got += int(keep.sum())
    return torch.cat(xs)[:n], torch.cat(ys)[:n], torch.cat(ts)[:n]


def _corr(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() < 8:
        return None
    a = left - left.mean()
    b = right - right.mean()
    denom = float(torch.sqrt((a * a).sum() * (b * b).sum()))
    if denom < 1e-18:
        return None
    return float((a * b).sum() / denom)


def gradient_snapshot(model, problem, phi0_fn, lam: float, seed: int = 2024) -> dict:
    """High-R versus low-R gradients on one fixed uniform batch. Regions use full R."""
    x, y, t = _fixed_batch(problem, N_PDE, seed)
    was = model.training
    model.train()
    terms = _batch_terms(model, problem, phi0_fn, x, y, t, lam, create_graph=True)
    high = region_mask(x, y)
    params = _params(model)
    uniform_loss = torch.mean(terms["coded"] ** 2)
    high_loss = torch.mean(terms["coded"][high] ** 2)
    low_loss = torch.mean(terms["coded"][~high] ** 2)
    g_uniform = _flat(torch.autograd.grad(uniform_loss, params, retain_graph=True)).detach()
    g_high = _flat(torch.autograd.grad(high_loss, params, retain_graph=True)).detach()
    g_low = _flat(torch.autograd.grad(low_loss, params)).detach()
    merged = pcgrad([g_high, g_low])
    uniform_value, balanced_value = uniform_and_balanced(terms["coded"].detach() ** 2, high)
    phi_t = terms["phi_t"].detach()
    rate = terms["rate"].detach()
    model.zero_grad(set_to_none=True)
    if not was:
        model.eval()
    uniform_norm = float(g_uniform.norm())
    return {
        "lambda": lam,
        "cosine_high_low": _cosine(g_high, g_low),
        "grad_norm_uniform": uniform_norm,
        "grad_norm_high": float(g_high.norm()),
        "grad_norm_low": float(g_low.norm()),
        "pcgrad_norm": float(merged.norm()),
        "pcgrad_over_uniform": float(merged.norm()) / uniform_norm if uniform_norm > 1e-18 else None,
        "balanced_relative_difference": float(abs(balanced_value - uniform_value) / uniform_value.clamp(min=1e-12)),
        "phi_t_high": float(phi_t[high].mean()),
        "phi_t_low": float(phi_t[~high].mean()),
        "corr_phi_t_R": _corr(phi_t, rate),
        "std_phi_t": float(phi_t.std(unbiased=False)),
        "mean_abs_grad": float(terms["magnitude"].detach().mean()),
        "n_high": int(high.sum()),
        "n_low": int((~high).sum()),
    }


def _predicted_front_residual(model, problem, phi0_fn, grid_x, grid_y) -> float:
    """Mean |residual| within 100 m of this model's own final contour, at time T."""
    field = predict_phi(model, problem, phi0_fn, problem.t_final, grid_x, grid_y)
    lines = [line for line in zero_contours(field, grid_x, grid_y) if len(line) >= 2]
    if not lines:
        return float("nan")
    base = torch.tensor(np.concatenate(lines), dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(99)
    choice = torch.randint(0, len(base), (3000,), generator=generator)
    jitter = (torch.rand(3000, 2, generator=generator) * 2.0 - 1.0) * FRONT_JITTER_M
    xy = base[choice] + jitter
    x, y = xy[:, 0], xy[:, 1]
    inside = (
        (x >= -problem.half_extent) & (x <= problem.half_extent)
        & (y >= -problem.half_extent) & (y <= problem.half_extent)
        & (torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m)
    )
    x, y = x[inside], y[inside]
    if x.numel() < 32:
        return float("nan")
    t = torch.full((x.numel(),), problem.t_final)
    was = model.training
    model.eval()
    physical = _batch_terms(model, problem, phi0_fn, x, y, t, 1.0, create_graph=False)["physical"]
    if was:
        model.train()
    return float(physical.detach().abs().mean())


def _residual_map(model, problem, phi0_fn, grid_x, grid_y) -> np.ndarray:
    yy, xx = np.meshgrid(grid_y, grid_x, indexing="ij")
    flat_x = torch.tensor(xx.ravel(), dtype=torch.float32)
    flat_y = torch.tensor(yy.ravel(), dtype=torch.float32)
    values = []
    was = model.training
    model.eval()
    for start in range(0, flat_x.numel(), 5000):
        stop = start + 5000
        x = flat_x[start:stop]
        y = flat_y[start:stop]
        t = torch.full((x.numel(),), problem.t_final)
        physical = _batch_terms(model, problem, phi0_fn, x, y, t, 1.0, create_graph=False)["physical"]
        values.append(physical.detach().cpu())
    if was:
        model.train()
    grid = torch.cat(values).numpy().reshape(xx.shape)
    ignition = np.hypot(xx - problem.x0, yy - problem.y0) <= problem.pde_exclusion_m
    grid = grid.astype(np.float64)
    grid[ignition] = np.nan
    return grid


def _gap(growth: dict, reference: dict) -> dict:
    truth = float(reference["high_r_growth_m"] - reference["low_r_growth_m"])
    if growth["high_r_growth_m"] is None or growth["low_r_growth_m"] is None:
        return {"growth_gap_m": None, "reference_gap_m": truth, "gap_error_m": None, "correct_sign": False}
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
    reference = _growth_by_rate(fields[problem.t_final], grid_x, grid_y, problem, rate_grid)
    record = {
        "initial_iou": rows[0]["iou"],
        "final_iou": rows[-1]["iou"],
        "final_symmetric_m": rows[-1]["symmetric_distance_m"],
        "final_p95_m": rows[-1]["p95_distance_m"],
        "final_area_error_km2": rows[-1]["area_error_km2"],
        "volume_residual_m_s": _probe(model, problem, phi0_fn, spec, grid_x, grid_y, fields, "uniform"),
        "predicted_front_residual_m_s": _predicted_front_residual(model, problem, phi0_fn, grid_x, grid_y),
        "pinn_growth": growth,
        "numerical_growth": reference,
        "times": rows,
        "parameter_count": sum(int(param.numel()) for param in model.parameters()),
    }
    record.update(_gap(growth, reference))
    return record


def _finished(run_dir: Path, epochs: int, seed: int):
    history_path = run_dir / "history.json"
    checkpoint = run_dir / "levelset_pinn.pt"
    if not checkpoint.is_file() or not history_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history.get("loss", [])) != epochs or history.get("seed") != seed:
        return None
    return history


def train_variant(name: str, problem, phi0_fn, grid_x, grid_y, epochs: int, seed: int):
    spec = VARIANTS[name]
    run_dir = HERE / "runs" / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = _finished(run_dir, epochs, seed)
    if existing is not None:
        model = _new_model(spec["architecture"], problem, seed)
        load_checkpoint(run_dir / "levelset_pinn.pt", model, torch.device("cpu"))
        model.eval()
        print(f"resume finished {name} seed {seed}", flush=True)
        return model, existing, float(existing.get("train_seconds") or 0.0)
    model = _new_model(spec["architecture"], problem, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"loss": [], "seed": seed, "variant": name, "diagnostics": [], "growth": []}
    device = torch.device("cpu")
    model.train()
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        lam = 1.0 if spec["schedule"] == "full" else lambda_at(epoch, epochs)
        x, y, t = _sample_pde(N_PDE, problem, device)
        terms = _batch_terms(model, problem, phi0_fn, x, y, t, lam, create_graph=True)
        coded = terms["coded"]
        uniform_loss = torch.mean(coded ** 2)
        params = _params(model)
        optimizer.zero_grad(set_to_none=True)
        if spec["update"] == "adam":
            uniform_loss.backward()
        else:
            high = region_mask(x, y)
            g_high = torch.autograd.grad(torch.mean(coded[high] ** 2), params, retain_graph=True)
            g_low = torch.autograd.grad(torch.mean(coded[~high] ** 2), params)
            _assign_grad(params, pcgrad([_flat(g_high), _flat(g_low)]))
        optimizer.step()
        history["loss"].append(float(uniform_loss.detach()))
        if epoch == 1 or epoch % DIAGNOSTIC_EVERY == 0 or epoch == epochs:
            snap = gradient_snapshot(model, problem, phi0_fn, lam)
            snap["epoch"] = epoch
            history["diagnostics"].append(snap)
            model.train()
        if epoch == 1 or epoch % 250 == 0 or epoch == epochs:
            pred = predict_phi(model, problem, phi0_fn, problem.t_final, grid_x, grid_y)
            growth = _growth_by_rate(pred, grid_x, grid_y, problem, None)
            history["growth"].append({"epoch": epoch, "lambda": lam, **growth})
            high_m = growth["high_r_growth_m"]
            low_m = growth["low_r_growth_m"]
            gap_m = float("nan") if high_m is None or low_m is None else high_m - low_m
            print(
                f"{name} seed {seed} epoch {epoch:4d} | pde {history['loss'][-1]:.3e} | lambda {lam:.2f} | "
                f"gap {gap_m:.1f} m",
                flush=True,
            )
            save_checkpoint(
                run_dir / "levelset_pinn.pt",
                model,
                {"variant": name, "epochs": epochs, "seed": seed, **spec},
            )
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
            torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch}, run_dir / "trainer_state.pt")
    elapsed = time.perf_counter() - started
    history["train_seconds"] = elapsed
    (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    model.eval()
    return model, history, elapsed


def _load_baseline(problem):
    model = LevelSetPINN(HIDDEN, LAYERS)
    load_checkpoint(EXP2 / "runs" / "2a" / "levelset_pinn.pt", model, torch.device("cpu"))
    model.eval()
    history = json.loads((EXP2 / "runs" / "2a" / "history.json").read_text(encoding="utf-8"))
    return model, history


def _store(record: dict):
    path = HERE / "metrics.json"
    rows = []
    if path.is_file():
        rows = [row for row in json.loads(path.read_text(encoding="utf-8")) if row.get("name") != record["name"] or row.get("seed") != record["seed"]]
    rows.append(record)
    rows.sort(key=lambda row: (row["name"], row["seed"]))
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def _fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def write_report(rows: list[dict], timing: dict):
    lines = [
        "# Experiment 2A.2",
        "",
        "Hard-IC models on the Experiment 2A spatial rate. The baseline is the saved 2A checkpoint. "
        "Training does not use future fronts or the numerical field as a target. "
        "The level-set equation and the R-scaled ansatz are unchanged. "
        "High and low regions are the fixed partition R ≥ R0 and R < R0.",
        "",
        "Commands:",
        "",
        "```",
        "python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment2a2",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment A",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment C",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment E",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B --seed 1",
        "python -m experiments.wildfirepinn_v2_diagnostics.experiment2.cause_isolation_2 --experiment B --seed 2",
        "```",
        "",
        f"Numerical solver time on the 20 m grid: {_fmt(timing.get('solver_seconds'), 3)} s. "
        "That reference is shared. Fourier features use the R wavelength, its first harmonic, "
        "and the axis and diagonal directions (8 frequencies, 16 features).",
        "",
        "| run | init IoU | final IoU | symmetric m | p95 m | area error km² | volume residual m/s | front residual m/s | high growth m | low growth m | gap m | gap error m | sign | train s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        growth = row["pinn_growth"]
        lines.append(
            "| {name} | {ii} | {fi} | {sym} | {p95} | {area} | {vol} | {front} | {high} | {low} | {gap} | {err} | {sign} | {train} |".format(
                name=f"{row['name']} seed {row['seed']}",
                ii=_fmt(row["initial_iou"], 3),
                fi=_fmt(row["final_iou"], 3),
                sym=_fmt(row["final_symmetric_m"], 2),
                p95=_fmt(row["final_p95_m"], 2),
                area=_fmt(row["final_area_error_km2"], 3),
                vol=_fmt(row["volume_residual_m_s"], 4),
                front=_fmt(row["predicted_front_residual_m_s"], 4),
                high=_fmt(growth["high_r_growth_m"], 1),
                low=_fmt(growth["low_r_growth_m"], 1),
                gap=_fmt(row["growth_gap_m"], 1),
                err=_fmt(row["gap_error_m"], 1),
                sign="yes" if row["correct_sign"] else "no",
                train=_fmt(row["train_seconds"], 1),
            )
        )
    lines.append("")
    reference = rows[0]["numerical_growth"] if rows else None
    if reference:
        lines.append(
            f"Numerical growth: high {_fmt(reference['high_r_growth_m'], 1)} m, "
            f"low {_fmt(reference['low_r_growth_m'], 1)} m, "
            f"gap {_fmt(reference['high_r_growth_m'] - reference['low_r_growth_m'], 1)} m."
        )
        lines.append("")
    for row in rows:
        lines.append(f"## {row['name']} seed {row['seed']}")
        lines.append("")
        lines.append(row["label"] + ".")
        lines.append("")
        lines.append("| t s | IoU | symmetric m | p95 m | area error km² |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for item in row["times"]:
            lines.append(
                f"| {item['t_s']:.0f} | {_fmt(item['iou'], 3)} | {_fmt(item['symmetric_distance_m'], 2)} | "
                f"{_fmt(item['p95_distance_m'], 2)} | {_fmt(item['area_error_km2'], 3)} |"
            )
        trained = row.get("trained_gradients") or {}
        if trained:
            lines.append("")
            lines.append(
                f"Trained uniform gradient norm {_fmt(trained.get('grad_norm_uniform'), 4)}, "
                f"high/low cosine {_fmt(trained.get('cosine_high_low'), 3)}, "
                f"PCGrad norm / uniform norm {_fmt(trained.get('pcgrad_over_uniform'), 3)}, "
                f"balanced-loss relative difference {_fmt(trained.get('balanced_relative_difference'), 4)}, "
                f"φ_t high {_fmt(trained.get('phi_t_high'), 4)} m/s, low {_fmt(trained.get('phi_t_low'), 4)} m/s, "
                f"corr(φ_t, R) {_fmt(trained.get('corr_phi_t_R'), 3)}."
            )
        lines.append("")
    lines.append("Figures are in `cause_isolation_2/figures/`.")
    lines.append("")
    path = HERE / "EXPERIMENT_2A2_RESULTS.md"
    if path.is_file() and "\n## Reading\n" in path.read_text(encoding="utf-8"):
        reading = path.read_text(encoding="utf-8").split("\n## Reading\n", 1)[1].strip()
        lines.extend(["## Reading", "", reading, ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _histories(rows: list[dict]) -> dict:
    found = {}
    baseline = EXP2 / "runs" / "2a" / "history.json"
    if baseline.is_file():
        found["A"] = json.loads(baseline.read_text(encoding="utf-8"))
    for row in rows:
        if row["name"] == "A":
            continue
        path = HERE / "runs" / f"{row['name']}_seed{row['seed']}" / "history.json"
        if path.is_file():
            found[f"{row['name']}{row['seed']}"] = json.loads(path.read_text(encoding="utf-8"))
    return found


def _draw_contours(ax, field, grid_x, grid_y, color, width):
    for line in zero_contours(field, grid_x, grid_y):
        if len(line) >= 2:
            ax.plot(line[:, 0], line[:, 1], color=color, lw=width)


def plot_all(rows, models, problem, phi0_fn, fields, rate_grid, grid_x, grid_y):
    figures = HERE / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    moment = problem.t_final
    extent = [grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]]
    window = 1100.0
    columns = max(len(rows), 1)
    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.2), squeeze=False)
    for ax, row in zip(axes[0], rows):
        ax.imshow(rate_grid, origin="lower", extent=extent, cmap="viridis", vmin=0.02, vmax=0.08)
        _draw_contours(ax, fields[moment], grid_x, grid_y, "white", 1.5)
        prediction = predict_phi(models[row["name"]][row["seed"]], problem, phi0_fn, moment, grid_x, grid_y)
        _draw_contours(ax, prediction, grid_x, grid_y, "crimson", 1.2)
        ax.set_xlim(problem.x0 - window, problem.x0 + window)
        ax.set_ylim(problem.y0 - window, problem.y0 + window)
        ax.set_aspect("equal")
        ax.set_title(f"{row['name']} seed {row['seed']}")
    fig.suptitle("White: numerical front. Red: PINN. Color is R (m/s).")
    fig.tight_layout()
    fig.savefig(figures / "contours.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["numerical"] + [f"{row['name']} s{row['seed']}" for row in rows]
    high = [rows[0]["numerical_growth"]["high_r_growth_m"]] + [row["pinn_growth"]["high_r_growth_m"] for row in rows]
    low = [rows[0]["numerical_growth"]["low_r_growth_m"]] + [row["pinn_growth"]["low_r_growth_m"] for row in rows]
    index = np.arange(len(labels))
    ax.bar(index - 0.16, high, width=0.32, label="R ≥ R0", color="#d95f02")
    ax.bar(index + 0.16, low, width=0.32, label="R < R0", color="#7570b3")
    ax.set_xticks(index, labels)
    ax.set_ylabel("radial growth (m)")
    ax.legend(frameon=False)
    ax.set_title("Growth in the fixed R partition")
    fig.tight_layout()
    fig.savefig(figures / "growth.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.0), squeeze=False)
    for ax, row in zip(axes[0], rows):
        grid = _residual_map(models[row["name"]][row["seed"]], problem, phi0_fn, grid_x, grid_y)
        image = ax.imshow(grid, origin="lower", extent=extent, cmap="coolwarm", vmin=-0.04, vmax=0.04)
        _draw_contours(ax, fields[moment], grid_x, grid_y, "black", 0.8)
        ax.set_xlim(problem.x0 - window, problem.x0 + window)
        ax.set_ylim(problem.y0 - window, problem.y0 + window)
        ax.set_aspect("equal")
        ax.set_title(f"{row['name']} residual (m/s)")
    fig.colorbar(image, ax=axes[0].tolist(), fraction=0.046, pad=0.04)
    fig.savefig(figures / "residual_maps.png", dpi=140)
    plt.close(fig)

    histories = _histories(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if "A" in histories and histories["A"].get("pde"):
        ax.plot(np.arange(1, len(histories["A"]["pde"]) + 1), histories["A"]["pde"], label="A", color="0.3")
    for row in rows:
        key = f"{row['name']}{row['seed']}"
        loss = histories.get(key, {}).get("loss")
        if loss:
            ax.plot(np.arange(1, len(loss) + 1), loss, label=f"{row['name']} s{row['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared PDE residual")
    ax.legend(frameon=False)
    ax.set_title("Training loss")
    fig.tight_layout()
    fig.savefig(figures / "loss.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for row in rows:
        key = f"{row['name']}{row['seed']}"
        diagnostics = histories.get(key, {}).get("diagnostics", [])
        if not diagnostics:
            continue
        epochs = [item["epoch"] for item in diagnostics]
        tag = f"{row['name']} s{row['seed']}"
        axes[0].plot(epochs, [item["cosine_high_low"] for item in diagnostics], label=tag)
        axes[1].plot(epochs, [item["grad_norm_high"] for item in diagnostics], label=f"{tag} high")
        axes[1].plot(epochs, [item["grad_norm_low"] for item in diagnostics], linestyle="--", label=f"{tag} low")
    trained = [row.get("trained_gradients") for row in rows if row["name"] == "A" and row.get("trained_gradients")]
    if trained:
        axes[0].scatter([EPOCHS], [trained[0]["cosine_high_low"]], color="0.2", label="A at 1500", zorder=3)
    axes[0].axhline(0.0, color="0.7", lw=0.8)
    axes[0].set_ylim(-1.05, 1.05)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cosine of high-R and low-R gradients")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("gradient norm")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "gradients.png", dpi=140)
    plt.close(fig)


def run_experiment(name: str, epochs: int, seed: int):
    problem = _problem()
    grid_x, grid_y = problem.grid()
    phi0, phi0_fn = _phi0(problem, grid_x, grid_y)
    fields, rate_grid = _reference(problem, phi0, grid_x, grid_y)
    timing = _baseline_timing()
    if name == "A":
        model, _history = _load_baseline(problem)
        record = evaluate(model, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
        record.update(
            {
                "name": "A",
                "seed": 0,
                "label": "saved Experiment 2A hard-IC checkpoint",
                "train_seconds": timing["train_seconds"],
                "epochs": epochs,
                "retrained": False,
                "initialization_gradients": gradient_snapshot(_new_model("mlp", problem, SEED), problem, phi0_fn, 1.0),
                "trained_gradients": gradient_snapshot(model, problem, phi0_fn, 1.0),
            }
        )
        models = {"A": {0: model}}
    else:
        model, _history, seconds = train_variant(name, problem, phi0_fn, grid_x, grid_y, epochs, seed)
        record = evaluate(model, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
        record.update(
            {
                "name": name,
                "seed": seed,
                "label": VARIANTS[name]["label"],
                "train_seconds": seconds,
                "epochs": epochs,
                "retrained": True,
                "trained_gradients": gradient_snapshot(model, problem, phi0_fn, 1.0),
                "future_front_in_training": False,
                "numerical_field_as_target": False,
            }
        )
    rows = _store(record)
    # Models for every stored row are needed for the figures.
    models = _load_saved_models(problem)
    if name == "A":
        models.setdefault("A", {})[0] = model
    plot_all(rows, models, problem, phi0_fn, fields, rate_grid, grid_x, grid_y)
    write_report(rows, timing)
    print(
        f"{name} IoU {record['final_iou']:.3f} symmetric {record['final_symmetric_m']:.2f} m "
        f"gap {record['growth_gap_m']:.1f} m error {record['gap_error_m']:.1f} m "
        f"sign {'yes' if record['correct_sign'] else 'no'}",
        flush=True,
    )
    trained = record["trained_gradients"]
    print(
        f"gradients cosine {trained['cosine_high_low']:.3f} "
        f"pcgrad/uniform {trained['pcgrad_over_uniform']:.3f} "
        f"balanced rel {trained['balanced_relative_difference']:.4f} "
        f"phi_t {trained['phi_t_high']:.4f} / {trained['phi_t_low']:.4f} "
        f"corr {trained['corr_phi_t_R']}",
        flush=True,
    )
    return record


def _load_saved_models(problem, skip=None):
    found = {}
    path = HERE / "metrics.json"
    if not path.is_file():
        return found
    for row in json.loads(path.read_text(encoding="utf-8")):
        key = (row["name"], row["seed"])
        if skip is not None and key == skip:
            continue
        if row["name"] == "A":
            model, _history = _load_baseline(problem)
            found.setdefault("A", {})[0] = model
            continue
        run_dir = HERE / "runs" / f"{row['name']}_seed{row['seed']}"
        architecture = VARIANTS[row["name"]]["architecture"]
        model = _new_model(architecture, problem, row["seed"])
        load_checkpoint(run_dir / "levelset_pinn.pt", model, torch.device("cpu"))
        model.eval()
        found.setdefault(row["name"], {})[row["seed"]] = model
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.experiment == "A" and args.seed != 0:
        raise SystemExit("Experiment A is the saved seed-0 checkpoint")
    run_experiment(args.experiment, args.epochs, args.seed)


if __name__ == "__main__":
    main()
