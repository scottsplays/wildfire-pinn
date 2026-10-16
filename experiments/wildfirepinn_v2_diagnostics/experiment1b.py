"""Experiment 1B: hard initial conditions on the expanding circle.

Configuration A is the Experiment 1 checkpoint and is not retrained.
B enforces phi(x, y, 0) = phi0. C adds oracle samples near the analytical front.
E multiplies the learned correction by R, so R = 0 is exact by construction.
D raises the PDE weight; with a pure PDE loss that mostly rescales the Adam step,
and it is run only when requested.

From the repository root:

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment1b
    python -m experiments.wildfirepinn_v2_diagnostics.experiment1b --configs B
    python -m experiments.wildfirepinn_v2_diagnostics.experiment1b --configs B C E
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.ansatz import normalized_phi
from experiments.wildfirepinn_v2_diagnostics.circle import (
    CircleProblem,
    analytical_phi,
    front_radius_stats,
    mask_iou,
    true_radius,
)
from experiments.wildfirepinn_v2_diagnostics.experiment1 import (
    HIDDEN,
    LAYERS,
    N_PDE,
    _sample_pde,
    _tag,
    predict_phi,
)
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from wildfire_pinn.geometry import zero_contours
from wildfire_pinn.pinn import LevelSetPINN, isotropic_residual, load_checkpoint, save_checkpoint, set_seed

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / "experiments" / "wildfirepinn_v2_diagnostics"
RATES = (0.0, 0.01, 0.1)
SEED = 0
FULL_EPOCHS = 1500
LR = 1e-3
FRONT_BAND_M = 100.0

CONFIGS = {
    "B": {"kind": "hard_ic", "oracle": False, "w_pde": 1.0, "label": "hard IC, uniform PDE"},
    "C": {"kind": "hard_ic", "oracle": True, "w_pde": 1.0, "label": "hard IC, oracle front sampling"},
    "D": {"kind": "hard_ic", "oracle": False, "w_pde": 10.0, "label": "hard IC, PDE weight 10"},
    "E": {"kind": "r_scaled", "oracle": False, "w_pde": 1.0, "label": "R-scaled structural ansatz"},
}


def _times(problem: CircleProblem) -> list[float]:
    return [0.0, problem.t_final / 3.0, 2.0 * problem.t_final / 3.0, problem.t_final]


def _sample_oracle(n: int, problem: CircleProblem, rate: float, device: torch.device):
    """Half the batch is uniform. Half lies within 100 m of the analytical front.

    The front locations use the known solution, so this is an oracle diagnostic,
    not a method that can be deployed on an unobserved fire.
    """
    n_front = n // 2
    n_uniform = n - n_front
    x_u, y_u, t_u = _sample_pde(n_uniform, problem, device)
    xs, ys, ts = [], [], []
    got = 0
    while got < n_front:
        count = max(n_front, 1024)
        t = torch.rand(count, device=device) * problem.t_final
        theta = torch.rand(count, device=device) * (2.0 * math.pi)
        offset = (torch.rand(count, device=device) * 2.0 - 1.0) * FRONT_BAND_M
        radius = problem.r0 + rate * t + offset
        x = problem.x0 + radius * torch.cos(theta)
        y = problem.y0 + radius * torch.sin(theta)
        inside = (
            (x >= -problem.half_extent) & (x <= problem.half_extent)
            & (y >= -problem.half_extent) & (y <= problem.half_extent)
            & (radius > problem.pde_exclusion_m)
        )
        xs.append(x[inside])
        ys.append(y[inside])
        ts.append(t[inside])
        got += int(inside.sum())
    return (
        torch.cat([x_u, torch.cat(xs)[:n_front]]),
        torch.cat([y_u, torch.cat(ys)[:n_front]]),
        torch.cat([t_u, torch.cat(ts)[:n_front]]),
    )


def _residual_on(model, problem, rate, kind, x, y, t):
    length = problem.length
    duration = problem.t_final
    tau = (t / duration).detach().requires_grad_(True)
    x_norm = ((x - problem.center_x) / length).detach().requires_grad_(True)
    y_norm = ((y - problem.center_y) / length).detach().requires_grad_(True)
    raw = model(tau, x_norm, y_norm)
    phi = normalized_phi(raw, tau, x_norm, y_norm, problem, kind, rate)
    phi_t, phi_x, phi_y = torch.autograd.grad(
        phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=phi.requires_grad and model.training
    )
    coded, magnitude = isotropic_residual(phi_t, phi_x, phi_y, rate * duration / length)
    physical = coded * (length / duration)
    return physical, magnitude, coded


def _probe(model, problem, rate, kind, mode: str, n: int = 4096) -> dict:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345 if mode == "uniform" else 99)
    xs, ys, ts = [], [], []
    got = 0
    while got < n:
        if mode == "uniform":
            x = torch.rand(n, generator=generator) * problem.length - problem.half_extent
            y = torch.rand(n, generator=generator) * problem.length - problem.half_extent
            t = torch.rand(n, generator=generator) * problem.t_final
            keep = torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m
        else:
            t = torch.rand(n, generator=generator) * problem.t_final
            theta = torch.rand(n, generator=generator) * (2.0 * math.pi)
            offset = (torch.rand(n, generator=generator) * 2.0 - 1.0) * FRONT_BAND_M
            radius = problem.r0 + float(rate) * t + offset
            x = problem.x0 + radius * torch.cos(theta)
            y = problem.y0 + radius * torch.sin(theta)
            keep = (
                (x.abs() <= problem.half_extent) & (y.abs() <= problem.half_extent)
                & (radius > problem.pde_exclusion_m)
            )
        xs.append(x[keep])
        ys.append(y[keep])
        ts.append(t[keep])
        got += int(keep.sum())
    device = next(model.parameters()).device
    physical, magnitude, _coded = _residual_on(
        model,
        problem,
        rate,
        kind,
        torch.cat(xs)[:n].to(device),
        torch.cat(ys)[:n].to(device),
        torch.cat(ts)[:n].to(device),
    )
    return {
        "residual_m_s": float(physical.detach().abs().mean()),
        "grad_magnitude": float(magnitude.detach().mean()),
    }


def predict_field(model, problem, rate, kind, t, x, y) -> np.ndarray:
    device = next(model.parameters()).device
    yy, xx = np.meshgrid(y, x, indexing="ij")
    tau = torch.tensor(np.full(xx.shape, t / problem.t_final).ravel(), dtype=torch.float32, device=device)
    x_norm = torch.tensor(((xx - problem.center_x) / problem.length).ravel(), dtype=torch.float32, device=device)
    y_norm = torch.tensor(((yy - problem.center_y) / problem.length).ravel(), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        raw = model(tau, x_norm, y_norm)
        phi = normalized_phi(raw, tau, x_norm, y_norm, problem, kind, rate)
    if was_training:
        model.train()
    return phi.detach().cpu().numpy().reshape(xx.shape) * problem.length


def train_config(problem, rate, epochs, run_dir: Path, spec: dict, seed: int = SEED):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "levelset_pinn.pt"
    history_path = run_dir / "history.json"
    state_path = run_dir / "trainer_state.pt"
    device = torch.device("cpu")
    kind = spec["kind"]
    config = {
        "experiment": "1b",
        "kind": kind,
        "oracle_front_sampling": spec["oracle"],
        "w_pde": spec["w_pde"],
        "r_m_s": rate,
        "epochs": epochs,
        "hidden": HIDDEN,
        "layers": LAYERS,
        "seed": seed,
        "structural_r_factor": kind == "r_scaled",
    }
    start_epoch = 1
    history = {"total": [], "pde": []}
    if checkpoint.is_file() and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        done = len(history.get("total", []))
        if done == epochs:
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            load_checkpoint(checkpoint, model, device)
            model.eval()
            print(f"resume finished {run_dir.name}", flush=True)
            return model, history, 0.0
        if done > 0 and state_path.is_file():
            set_seed(seed)
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            load_checkpoint(checkpoint, model, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            np.random.set_state(state["numpy"])
            start_epoch = done + 1
            print(f"resume {run_dir.name} at epoch {start_epoch}", flush=True)
        else:
            history = {"total": [], "pde": []}
            set_seed(seed)
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        set_seed(seed)
        model = LevelSetPINN(HIDDEN, LAYERS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    started = time.perf_counter()
    log_every = 1 if epochs <= 5 else 250
    for epoch in range(start_epoch, epochs + 1):
        if spec["oracle"]:
            x_pde, y_pde, t_pde = _sample_oracle(N_PDE, problem, rate, device)
        else:
            x_pde, y_pde, t_pde = _sample_pde(N_PDE, problem, device)
        _physical, _magnitude, coded = _residual_on(model, problem, rate, kind, x_pde, y_pde, t_pde)
        loss_pde = torch.mean(coded ** 2)
        loss = spec["w_pde"] * loss_pde
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch} in {run_dir.name}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["total"].append(float(loss.detach()))
        history["pde"].append(float(loss_pde.detach()))
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"{run_dir.name} epoch {epoch:4d} | pde {history['pde'][-1]:.3e}",
                flush=True,
            )
            if epoch % log_every == 0 or epoch == epochs:
                save_checkpoint(checkpoint, model, config)
                history_path.write_text(json.dumps(history), encoding="utf-8")
                torch.save(
                    {"optimizer": optimizer.state_dict(), "epoch": epoch, "rng": torch.get_rng_state(), "numpy": np.random.get_state()},
                    state_path,
                )
    elapsed = time.perf_counter() - started
    model.eval()
    return model, history, elapsed


def _score_field(phi, truth, x, y, problem, radius) -> dict:
    stats = front_radius_stats(phi, x, y, problem.x0, problem.y0, radius)
    stats["iou"] = mask_iou(phi, truth)
    return stats


def evaluate_model(model, problem, rate, kind, train_seconds, history) -> dict:
    x, y = problem.grid()
    xx, yy = np.meshgrid(x, y)
    phi0 = analytical_phi(xx, yy, 0.0, problem, rate)
    numerical = propagate(phi0, x, y, rate, _times(problem))
    rows = []
    for t in _times(problem):
        truth = analytical_phi(xx, yy, t, problem, rate)
        pred = predict_field(model, problem, rate, kind, t, x, y)
        radius = true_radius(problem, rate, t)
        rows.append(
            {
                "t_s": t,
                "true_radius_m": radius,
                "pinn": _score_field(pred, truth, x, y, problem, radius),
                "numerical": _score_field(numerical[t], truth, x, y, problem, radius),
            }
        )
    ic = predict_field(model, problem, rate, kind, 0.0, x, y)
    initial = rows[0]["pinn"]
    final = rows[-1]["pinn"]
    inferred = None
    if initial["radius_m"] is not None and final["radius_m"] is not None and problem.t_final > 0:
        inferred = (final["radius_m"] - initial["radius_m"]) / problem.t_final
    uniform = _probe(model, problem, rate, kind, "uniform")
    front = _probe(model, problem, rate, kind, "front")
    return {
        "r_m_s": rate,
        "kind": kind,
        "train_seconds": train_seconds,
        "final_train_pde_loss": history["pde"][-1] if history.get("pde") else None,
        "initial_iou": initial["iou"],
        "final_iou": final["iou"],
        "initial_radius_error_m": initial["radius_error_m"],
        "final_radius_error_m": final["radius_error_m"],
        "initial_radial_miss_m": initial["mean_radial_displacement_m"],
        "final_radial_miss_m": final["mean_radial_displacement_m"],
        "ic_rmse_m": float(np.sqrt(np.mean((ic - phi0) ** 2))),
        "pde_residual_m_s": uniform["residual_m_s"],
        "front_residual_m_s": front["residual_m_s"],
        "front_grad_magnitude": front["grad_magnitude"],
        "inferred_speed_m_s": inferred,
        "speed_error_m_s": None if inferred is None else inferred - rate,
        "numerical_final_iou": rows[-1]["numerical"]["iou"],
        "numerical_final_radius_error_m": rows[-1]["numerical"]["radius_error_m"],
        "times": rows,
    }


def _baseline_records(problem: CircleProblem) -> list[dict]:
    path = DIAG / "experiment1" / "metrics.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    records = []
    device = torch.device("cpu")
    for item in saved:
        rate = float(item["r_m_s"])
        model = LevelSetPINN(HIDDEN, LAYERS).to(device)
        load_checkpoint(
            DIAG / "experiment1" / "runs" / f"{_tag(rate)}_seed0" / "levelset_pinn.pt",
            model,
            device,
        )
        model.eval()
        uniform = _probe(model, problem, rate, "soft", "uniform")
        front = _probe(model, problem, rate, "soft", "front")
        initial = item["times"][0]["pinn"]
        final = item["times"][-1]["pinn"]
        inferred = (final["radius_m"] - initial["radius_m"]) / problem.t_final
        records.append(
            {
                "config": "A",
                "label": "soft IC, uniform PDE",
                "r_m_s": rate,
                "kind": "soft",
                "oracle_front_sampling": False,
                "train_seconds": item["train_seconds"],
                "initial_iou": initial["iou"],
                "final_iou": final["iou"],
                "initial_radius_error_m": initial["radius_error_m"],
                "final_radius_error_m": final["radius_error_m"],
                "initial_radial_miss_m": initial["mean_radial_displacement_m"],
                "final_radial_miss_m": final["mean_radial_displacement_m"],
                "ic_rmse_m": item["ic_rmse_m"],
                "pde_residual_m_s": uniform["residual_m_s"],
                "front_residual_m_s": front["residual_m_s"],
                "front_grad_magnitude": front["grad_magnitude"],
                "inferred_speed_m_s": inferred,
                "speed_error_m_s": inferred - rate,
                "numerical_final_iou": item["times"][-1]["numerical"]["iou"],
                "numerical_final_radius_error_m": item["times"][-1]["numerical"]["radius_error_m"],
                "retrained": False,
            }
        )
    return records


def _draw(ax, phi, x, y, color, label, lw=1.3):
    for line in zero_contours(phi, x, y):
        ax.plot(line[:, 0], line[:, 1], color=color, lw=lw, label=label)
        label = None


def plot_perimeters(bundles, problem, path: Path):
    """bundles: list of (label, color, kind, rate_to_model). Analytical and numerical are always drawn."""
    x, y = problem.grid()
    xx, yy = np.meshgrid(x, y)
    times = _times(problem)
    fig, axes = plt.subplots(len(RATES), len(times), figsize=(12.8, 9.4), sharex=True, sharey=True)
    window = 1100.0
    for row, rate in enumerate(RATES):
        phi0 = analytical_phi(xx, yy, 0.0, problem, rate)
        numerical = propagate(phi0, x, y, rate, times)
        for col, t in enumerate(times):
            ax = axes[row, col]
            radius = true_radius(problem, rate, t)
            ax.add_patch(plt.Circle((problem.x0, problem.y0), radius, fill=False, color="black", lw=1.5, label="analytical"))
            _draw(ax, numerical[t], x, y, "tab:blue", "numerical")
            for label, color, kind, models in bundles:
                pred = predict_field(models[rate], problem, rate, kind, t, x, y)
                _draw(ax, pred, x, y, color, label)
            ax.set_xlim(problem.x0 - window, problem.x0 + window)
            ax.set_ylim(problem.y0 - window, problem.y0 + window)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"t = {t:.0f} s")
            if col == 0:
                ax.set_ylabel(f"R = {rate:g} m/s")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Analytical black, numerical blue. Other curves are PINN ablations.")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_radius_errors(records, path: Path):
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    configs = []
    for record in records:
        if record["config"] not in configs:
            configs.append(record["config"])
    width = 0.8 / max(len(configs), 1)
    for index, config in enumerate(configs):
        subset = [record for record in records if record["config"] == config]
        positions = np.arange(len(RATES)) + (index - 0.5 * (len(configs) - 1)) * width
        values = [
            next(item["final_radius_error_m"] for item in subset if item["r_m_s"] == rate)
            for rate in RATES
        ]
        values = [np.nan if value is None else value for value in values]
        ax.bar(positions, values, width=width, label=config)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.set_xticks(np.arange(len(RATES)), [f"R = {rate:g}" for rate in RATES])
    ax.set_ylabel("final radius error (m)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def write_reports(records, problem, smoke: bool):
    lines = [
        "# Experiment 1B",
        "",
        "Hard initial-condition ablations on the same circle as Experiment 1. "
        "Experiment 1 checkpoints were not retrained. Experiments 2–5 were not run.",
        "",
        "A 2-epoch smoke run is not evidence." if smoke else "Runs below use 1,500 epochs and seed 0.",
        "",
        "Audit of Experiment 1, measured on its checkpoints: the exact-field contour extractor "
        "is off by 0.008 m, so the reported radius error is not a contour-extraction bug. "
        "Normalization, the sign of phi, and the conversion from the coded residual to m/s match "
        "the chain rule. On the trained R = 0 network the gradient magnitude on the true front is "
        "about 0.62 rather than 1, and phi is biased about +33 m in the 50 m band around that front. "
        "A positive bias moves the zero contour inward. About 3% of uniform collocation points fall "
        "within 100 m of the moving front. The volume-averaged residual stays near 0.006 m/s while "
        "the front residual is larger, and at R = 0 the front time derivative changes from about "
        "−0.010 m/s to +0.028 m/s, which continues the inward drift.",
        "",
        "B forces phi(x, y, 0) = phi0 and trains only the PDE. "
        "C uses the same ansatz with half of the PDE samples inside a 100 m band of the analytical front. "
        "That band is an oracle: the true future front is known. "
        "E multiplies the learned correction by R. Its R = 0 result is exact by algebra, before any learning, "
        "and is not evidence that the network solved the PDE.",
        "",
        "| config | R | init IoU | final IoU | init radius m | final radius m | final radial miss m | IC RMSE m | volume residual m/s | front residual m/s | inferred speed m/s | train s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            "| {config} | {rate:g} | {ii} | {fi} | {ir} | {fr} | {miss} | {ic} | {vol} | {front} | {speed} | {seconds} |".format(
                config=record["config"],
                rate=record["r_m_s"],
                ii=_fmt(record["initial_iou"], 3),
                fi=_fmt(record["final_iou"], 3),
                ir=_fmt(record["initial_radius_error_m"], 2),
                fr=_fmt(record["final_radius_error_m"], 2),
                miss=_fmt(record["final_radial_miss_m"], 2),
                ic=_fmt(record["ic_rmse_m"], 2),
                vol=_fmt(record["pde_residual_m_s"], 4),
                front=_fmt(record["front_residual_m_s"], 4),
                speed=_fmt(record["inferred_speed_m_s"], 4),
                seconds=_fmt(record["train_seconds"], 1),
            )
        )
    lines.extend(
        [
            "",
            "Prescribed speeds are 0, 0.01, and 0.1 m/s. Inferred speed is the change in predicted radius divided by T. "
            "The numerical solver's final radius error on this grid is about 0.01 m, 0.22 m, and 1.57 m for those three speeds.",
            "",
            "Targets were initial IoU at least 0.99, final IoU at least 0.99, final mean radial error under 5 m, "
            "and no drift at R = 0. A configuration that misses them is reported as a miss.",
            "",
            "Figures: `experiment1b/figures/perimeters.png` and `experiment1b/figures/final_radius_error.png`.",
            "",
            "Reproduce with `python -m experiments.wildfirepinn_v2_diagnostics.experiment1b --configs B` "
            "from the repository root. Add `C` or `E` to train those ablations. Finished histories are skipped.",
            "",
        ]
    )
    report = DIAG / ("EXPERIMENT_1B_SMOKE.md" if smoke else "EXPERIMENT_1B_RESULTS.md")
    if report.is_file() and "\n## Reading\n" in report.read_text(encoding="utf-8"):
        reading = report.read_text(encoding="utf-8").split("\n## Reading\n", 1)[1].strip()
        lines.extend(["", "## Reading", "", reading, ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "experiment": "1b",
        "status": "smoke" if smoke else "completed",
        "epochs": 2 if smoke else FULL_EPOCHS,
        "seed": SEED,
        "targets": {
            "initial_iou": 0.99,
            "final_iou": 0.99,
            "final_radial_miss_m": 5.0,
            "stationary_preserved": True,
        },
        "runs": records,
        "experiments_2_to_5": "not_started",
    }
    summary_path = DIAG / ("EXPERIMENT_1B_SMOKE_SUMMARY.json" if smoke else "EXPERIMENT_1B_SUMMARY.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {report}", flush=True)


def _load_finished(problem, out_dir: Path, epochs: int) -> tuple[list[dict], dict]:
    """Load every completed ablation in this directory so a later run does not drop it."""
    records = []
    models = {}
    device = torch.device("cpu")
    for name, spec in CONFIGS.items():
        loaded = {}
        complete = True
        for rate in RATES:
            run_dir = out_dir / "runs" / f"{name}_{_tag(rate)}_seed{SEED}"
            history_path = run_dir / "history.json"
            checkpoint = run_dir / "levelset_pinn.pt"
            if not history_path.is_file() or not checkpoint.is_file():
                complete = False
                break
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if len(history.get("total", [])) != epochs:
                complete = False
                break
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            load_checkpoint(checkpoint, model, device)
            model.eval()
            loaded[rate] = (model, history)
        if not complete:
            continue
        models[name] = {rate: pair[0] for rate, pair in loaded.items()}
        for rate, (model, history) in loaded.items():
            record = evaluate_model(model, problem, rate, spec["kind"], 0.0, history)
            record["config"] = name
            record["label"] = spec["label"]
            record["oracle_front_sampling"] = spec["oracle"]
            record["retrained"] = True
            record["train_seconds"] = None
            records.append(record)
    return records, models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--configs", nargs="+", default=["B"], choices=sorted(CONFIGS))
    args = parser.parse_args()
    problem = CircleProblem()
    epochs = 2 if args.smoke else FULL_EPOCHS
    out_dir = DIAG / ("experiment1b_smoke" if args.smoke else "experiment1b")
    if "outputs" in out_dir.parts:
        raise RuntimeError("Refusing to write under outputs/.")
    print(f"experiment 1b epochs={epochs} configs={args.configs}", flush=True)
    fresh_seconds = {}
    for name in args.configs:
        spec = CONFIGS[name]
        for rate in RATES:
            run_dir = out_dir / "runs" / f"{name}_{_tag(rate)}_seed{SEED}"
            print(f"start {run_dir.name} ({spec['label']})", flush=True)
            model, history, seconds = train_config(problem, rate, epochs, run_dir, spec)
            fresh_seconds[(name, rate)] = seconds
            preview = evaluate_model(model, problem, rate, spec["kind"], seconds, history)
            print(
                f"{name} R={rate:g} init IoU {preview['initial_iou']:.3f} final IoU {preview['final_iou']:.3f} "
                f"radius {preview['initial_radius_error_m']} -> {preview['final_radius_error_m']} "
                f"front residual {preview['front_residual_m_s']:.4f} speed {preview['inferred_speed_m_s']}",
                flush=True,
            )
            del model
    prior_seconds = dict(fresh_seconds)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        for row in json.loads(metrics_path.read_text(encoding="utf-8")):
            key = (row.get("config"), row.get("r_m_s"))
            if row.get("train_seconds") not in (None, 0, 0.0):
                prior_seconds.setdefault(key, row["train_seconds"])
    records = []
    if not args.smoke:
        records.extend(_baseline_records(problem))
    finished, models = _load_finished(problem, out_dir, epochs)
    for record in finished:
        record["train_seconds"] = prior_seconds.get((record["config"], record["r_m_s"]))
    records.extend(finished)
    metrics_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    bundles = []
    if not args.smoke:
        device = torch.device("cpu")
        baseline_models = {}
        for rate in RATES:
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            load_checkpoint(DIAG / "experiment1" / "runs" / f"{_tag(rate)}_seed{SEED}" / "levelset_pinn.pt", model, device)
            model.eval()
            baseline_models[rate] = model
        bundles.append(("A soft IC", "0.45", "soft", baseline_models))
    colors = {"B": "tab:red", "C": "tab:orange", "D": "tab:purple", "E": "tab:green"}
    for name in CONFIGS:
        if name in models:
            bundles.append((f"{name} {CONFIGS[name]['kind']}", colors[name], CONFIGS[name]["kind"], models[name]))
    plot_perimeters(bundles, problem, out_dir / "figures" / "perimeters.png")
    plot_radius_errors(records, out_dir / "figures" / "final_radius_error.png")
    write_reports(records, problem, args.smoke)


if __name__ == "__main__":
    main()
