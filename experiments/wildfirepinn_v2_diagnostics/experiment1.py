"""Experiment 1: can the level-set PINN solve an expanding circle?

The exact solution is

    phi(x, y, t) = hypot(x - x0, y - y0) - (r0 + R t)

with R in {0, 0.01, 0.1} m/s. The network is the existing tanh MLP
(LevelSetPINN), narrower than the Kincade model so three runs stay small.
Inputs use the same normalization as the real trainer: tau = t / T and
X = (x - xc) / L. The network returns Phi = phi / L. There is no boundary
loss: the final circle stays more than 500 m inside the square.

Run from the repository root:

    python -m unittest experiments.wildfirepinn_v2_diagnostics.test_experiment1
    python -m experiments.wildfirepinn_v2_diagnostics.experiment1 --smoke
    python -m experiments.wildfirepinn_v2_diagnostics.experiment1
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

from experiments.wildfirepinn_v2_diagnostics.circle import (
    CircleProblem,
    analytical_phi,
    front_radius_stats,
    mask_iou,
    true_radius,
)
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from wildfire_pinn.geometry import zero_contours
from wildfire_pinn.pinn import LevelSetPINN, isotropic_residual, load_checkpoint, save_checkpoint, set_seed

ROOT = Path(__file__).resolve().parents[2]
RATES = (0.0, 0.01, 0.1)
HIDDEN = 64
LAYERS = 4
N_PDE = 4096
N_IC = 2048
W_IC = 10.0
W_PDE = 1.0
LR = 1e-3
SEED = 0
FULL_EPOCHS = 1500


def _tag(rate: float) -> str:
    return f"constant_r{rate:.3f}".replace(".", "p")


def _reject_center(x, y, problem: CircleProblem):
    return torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m


def _sample_pde(n: int, problem: CircleProblem, device: torch.device):
    xs, ys, ts = [], [], []
    got = 0
    while got < n:
        count = max(n, 1024)
        x = torch.rand(count, device=device) * problem.length - problem.half_extent
        y = torch.rand(count, device=device) * problem.length - problem.half_extent
        t = torch.rand(count, device=device) * problem.t_final
        keep = _reject_center(x, y, problem)
        xs.append(x[keep])
        ys.append(y[keep])
        ts.append(t[keep])
        got += int(keep.sum())
    return torch.cat(xs)[:n], torch.cat(ys)[:n], torch.cat(ts)[:n]


def _sample_ic(n: int, problem: CircleProblem, device: torch.device):
    x = torch.rand(n, device=device) * problem.length - problem.half_extent
    y = torch.rand(n, device=device) * problem.length - problem.half_extent
    target = (torch.hypot(x - problem.x0, y - problem.y0) - problem.r0) / problem.length
    return x, y, target


def train_circle(problem: CircleProblem, rate: float, epochs: int, run_dir: Path, seed: int = SEED):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "levelset_pinn.pt"
    history_path = run_dir / "history.json"
    state_path = run_dir / "trainer_state.pt"
    device = torch.device("cpu")
    config = {
        "experiment": "circle",
        "r_m_s": rate,
        "epochs": epochs,
        "hidden": HIDDEN,
        "layers": LAYERS,
        "n_pde": N_PDE,
        "n_ic": N_IC,
        "w_ic": W_IC,
        "w_pde": W_PDE,
        "w_eik": 0.0,
        "w_boundary": 0.0,
        "lr": LR,
        "seed": seed,
        "normalization": "Phi=phi/L, tau=t/T, X=(x-xc)/L",
        "boundary": "no boundary loss; final front stays inside the square",
        "pde_exclusion_m": problem.pde_exclusion_m,
        **problem.to_dict(),
    }
    start_epoch = 1
    history = {"total": [], "pde": [], "ic": []}
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
            history = {"total": [], "pde": [], "ic": []}
            set_seed(seed)
            model = LevelSetPINN(HIDDEN, LAYERS).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        set_seed(seed)
        model = LevelSetPINN(HIDDEN, LAYERS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    length = problem.length
    duration = problem.t_final
    coded_rate = rate * duration / length
    model.train()
    started = time.perf_counter()
    log_every = 1 if epochs <= 5 else 250
    for epoch in range(start_epoch, epochs + 1):
        x_pde, y_pde, t_pde = _sample_pde(N_PDE, problem, device)
        tau = (t_pde / duration).detach().requires_grad_(True)
        x_norm = ((x_pde - problem.center_x) / length).detach().requires_grad_(True)
        y_norm = ((y_pde - problem.center_y) / length).detach().requires_grad_(True)
        phi = model(tau, x_norm, y_norm)
        phi_t, phi_x, phi_y = torch.autograd.grad(
            phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=True
        )
        residual, _magnitude = isotropic_residual(phi_t, phi_x, phi_y, coded_rate)
        loss_pde = torch.mean(residual ** 2)
        x_ic, y_ic, target = _sample_ic(N_IC, problem, device)
        pred_ic = model(
            torch.zeros_like(x_ic),
            (x_ic - problem.center_x) / length,
            (y_ic - problem.center_y) / length,
        )
        loss_ic = torch.mean((pred_ic - target) ** 2)
        loss = W_IC * loss_ic + W_PDE * loss_pde
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch} for R={rate}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["total"].append(float(loss.detach()))
        history["pde"].append(float(loss_pde.detach()))
        history["ic"].append(float(loss_ic.detach()))
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"{run_dir.name} epoch {epoch:4d} | total {history['total'][-1]:.3e} | "
                f"ic {history['ic'][-1]:.3e} | pde {history['pde'][-1]:.3e}",
                flush=True,
            )
            if epoch % log_every == 0 or epoch == epochs:
                save_checkpoint(checkpoint, model, config)
                history_path.write_text(json.dumps(history), encoding="utf-8")
                torch.save(
                    {
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "rng": torch.get_rng_state(),
                        "numpy": np.random.get_state(),
                    },
                    state_path,
                )
    elapsed = time.perf_counter() - started
    model.eval()
    return model, history, elapsed


def predict_phi(model: LevelSetPINN, problem: CircleProblem, t: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    yy, xx = np.meshgrid(y, x, indexing="ij")
    tau = torch.full((xx.size,), float(t) / problem.t_final, dtype=torch.float32, device=device)
    x_norm = torch.tensor((xx.ravel() - problem.center_x) / problem.length, dtype=torch.float32, device=device)
    y_norm = torch.tensor((yy.ravel() - problem.center_y) / problem.length, dtype=torch.float32, device=device)
    with torch.no_grad():
        phi = model(tau, x_norm, y_norm).detach().cpu().numpy().reshape(xx.shape)
    return phi * problem.length


def probe_residual(model: LevelSetPINN, problem: CircleProblem, rate: float, n: int = 4096) -> float:
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    xs, ys, ts = [], [], []
    got = 0
    while got < n:
        x = torch.rand(n, generator=generator) * problem.length - problem.half_extent
        y = torch.rand(n, generator=generator) * problem.length - problem.half_extent
        t = torch.rand(n, generator=generator) * problem.t_final
        keep = torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m
        xs.append(x[keep])
        ys.append(y[keep])
        ts.append(t[keep])
        got += int(keep.sum())
    x = torch.cat(xs)[:n].to(device)
    y = torch.cat(ys)[:n].to(device)
    t = torch.cat(ts)[:n].to(device)
    tau = (t / problem.t_final).detach().requires_grad_(True)
    x_norm = ((x - problem.center_x) / problem.length).detach().requires_grad_(True)
    y_norm = ((y - problem.center_y) / problem.length).detach().requires_grad_(True)
    phi = model(tau, x_norm, y_norm)
    phi_t, phi_x, phi_y = torch.autograd.grad(phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=False)
    coded, _magnitude = isotropic_residual(phi_t, phi_x, phi_y, rate * problem.t_final / problem.length)
    physical = coded.detach() * (problem.length / problem.t_final)
    return float(physical.abs().mean())


def _times(problem: CircleProblem) -> list[float]:
    return [0.0, problem.t_final / 3.0, 2.0 * problem.t_final / 3.0, problem.t_final]


def evaluate_rate(model, problem: CircleProblem, rate: float, train_seconds: float, history: dict) -> dict:
    x, y = problem.grid()
    xx, yy = np.meshgrid(x, y)
    phi0 = analytical_phi(xx, yy, 0.0, problem, rate)
    numerical = propagate(phi0, x, y, rate, _times(problem))
    rows = []
    for t in _times(problem):
        truth = analytical_phi(xx, yy, t, problem, rate)
        pinn = predict_phi(model, problem, t, x, y)
        radius = true_radius(problem, rate, t)
        stationary = phi0
        rows.append(
            {
                "t_s": t,
                "true_radius_m": radius,
                "pinn": {
                    **front_radius_stats(pinn, x, y, problem.x0, problem.y0, radius),
                    "iou": mask_iou(pinn, truth),
                },
                "numerical": {
                    **front_radius_stats(numerical[t], x, y, problem.x0, problem.y0, radius),
                    "iou": mask_iou(numerical[t], truth),
                },
                "stationary": {
                    **front_radius_stats(stationary, x, y, problem.x0, problem.y0, radius),
                    "iou": mask_iou(stationary, truth),
                },
            }
        )
    ic = predict_phi(model, problem, 0.0, x, y)
    return {
        "r_m_s": rate,
        "train_seconds": train_seconds,
        "final_train_loss": history["total"][-1] if history["total"] else None,
        "final_train_pde_loss": history["pde"][-1] if history["pde"] else None,
        "final_train_ic_loss": history["ic"][-1] if history["ic"] else None,
        "pde_residual_m_s": probe_residual(model, problem, rate),
        "ic_rmse_m": float(np.sqrt(np.mean((ic - phi0) ** 2))),
        "times": rows,
    }


def _draw_front(ax, phi, x, y, color, label):
    for line in zero_contours(phi, x, y):
        ax.plot(line[:, 0], line[:, 1], color=color, lw=1.4, label=label)
        label = None


def plot_perimeters(model_by_rate, problem: CircleProblem, path: Path):
    x, y = problem.grid()
    xx, yy = np.meshgrid(x, y)
    times = _times(problem)
    fig, axes = plt.subplots(len(RATES), len(times), figsize=(12.5, 9.2), sharex=True, sharey=True)
    window = 1100.0
    for row, rate in enumerate(RATES):
        phi0 = analytical_phi(xx, yy, 0.0, problem, rate)
        numerical = propagate(phi0, x, y, rate, times)
        model = model_by_rate[rate]
        for col, t in enumerate(times):
            ax = axes[row, col]
            truth = analytical_phi(xx, yy, t, problem, rate)
            radius = true_radius(problem, rate, t)
            circle = plt.Circle((problem.x0, problem.y0), radius, fill=False, color="black", lw=1.6, label="analytical")
            ax.add_patch(circle)
            _draw_front(ax, numerical[t], x, y, "tab:blue", "numerical")
            _draw_front(ax, predict_phi(model, problem, t, x, y), x, y, "tab:red", "PINN")
            ax.set_xlim(problem.x0 - window, problem.x0 + window)
            ax.set_ylim(problem.y0 - window, problem.y0 + window)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"t = {t:.0f} s")
            if col == 0:
                ax.set_ylabel(f"R = {rate:g} m/s")
            if row == 0 and col == 0:
                ax.legend(loc="upper right", fontsize=8)
            del truth
    fig.suptitle("Black analytical circle, blue numerical front, red PINN front")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_radius_errors(metrics: list[dict], path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True)
    for ax, record in zip(axes, metrics):
        times = [row["t_s"] / 3600.0 for row in record["times"]]
        for name, color in (("pinn", "tab:red"), ("numerical", "tab:blue"), ("stationary", "0.4")):
            values = [row[name]["radius_error_m"] for row in record["times"]]
            ax.plot(times, values, marker="o", color=color, label=name)
        ax.axhline(0.0, color="black", lw=0.6)
        ax.set_title(f"R = {record['r_m_s']:g} m/s")
        ax.set_xlabel("hours")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("radius error (m)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def write_reports(metrics: list[dict], problem: CircleProblem, out_dir: Path, smoke: bool):
    lines = [
        "# WildfirePINN v2 diagnostics",
        "",
        "Experiment 1 is the expanding circle with a known solution. "
        "Experiments 2–5 have not been run.",
        "",
        "Smoke runs are pipeline checks. The table below is the full 1,500-epoch comparison."
        if not smoke
        else "This file was produced by `--smoke` (2 epochs). It is not evidence about whether the PINN solves the level-set equation.",
        "",
        f"Domain side {problem.length:.0f} m, ignition ({problem.x0:.0f}, {problem.y0:.0f}) m, "
        f"r0 {problem.r0:.0f} m, T {problem.t_final:.0f} s, grid {problem.n} nodes "
        f"({problem.length / (problem.n - 1):.1f} m). "
        f"Network: LevelSetPINN hidden {HIDDEN}, layers {LAYERS}. "
        f"Loss: {W_IC:g} L_ic + {W_PDE:g} L_pde. No eikonal term and no boundary loss. "
        f"PDE collocation excludes a {problem.pde_exclusion_m:.0f} m disk around the ignition point, "
        "where the signed-distance gradient is undefined.",
        "",
        "Radius error is the mean contour radius minus r0 + R t. "
        "Displacement is the mean absolute radial miss of the zero contour. "
        "Stationary repeats the initial circle. "
        "PDE residual is the mean absolute physical residual in m/s, away from the ignition disk. "
        "A small residual is not by itself a correct front.",
        "",
        "## Final time",
        "",
        "| R m/s | method | radius error m | radial miss m | IoU | PDE residual m/s | IC RMSE m | train s |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in metrics:
        final = record["times"][-1]
        for method in ("pinn", "numerical", "stationary"):
            block = final[method]
            lines.append(
                "| {rate:g} | {method} | {err} | {miss} | {iou} | {resid} | {ic} | {seconds} |".format(
                    rate=record["r_m_s"],
                    method=method,
                    err=_fmt(block["radius_error_m"], 2),
                    miss=_fmt(block["mean_radial_displacement_m"], 2),
                    iou=_fmt(block["iou"], 3),
                    resid=_fmt(record["pde_residual_m_s"], 4) if method == "pinn" else "—",
                    ic=_fmt(record["ic_rmse_m"], 2) if method == "pinn" else "—",
                    seconds=_fmt(record["train_seconds"], 1) if method == "pinn" else "—",
                )
            )
    lines.extend(["", "## Radius error by time", ""])
    lines.append("| R m/s | t s | true radius m | PINN error m | numerical error m | stationary error m | PINN IoU | numerical IoU |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for record in metrics:
        for row in record["times"]:
            lines.append(
                "| {rate:g} | {t:.0f} | {radius:.1f} | {pinn} | {num} | {stay} | {piou} | {niou} |".format(
                    rate=record["r_m_s"],
                    t=row["t_s"],
                    radius=row["true_radius_m"],
                    pinn=_fmt(row["pinn"]["radius_error_m"], 2),
                    num=_fmt(row["numerical"]["radius_error_m"], 2),
                    stay=_fmt(row["stationary"]["radius_error_m"], 2),
                    piou=_fmt(row["pinn"]["iou"], 3),
                    niou=_fmt(row["numerical"]["iou"], 3),
                )
            )
    lines.extend(
        [
            "",
            "Perimeters: `experiment1/figures/perimeters.png`.",
            "Radius errors: `experiment1/figures/radius_error.png`.",
            "",
            "Reproduce with `python -m experiments.wildfirepinn_v2_diagnostics.experiment1` "
            "from the repository root. A finished run is skipped when its history already has the requested epoch count.",
            "",
        ]
    )
    report = ROOT / "experiments" / "wildfirepinn_v2_diagnostics" / ("SMOKE_RESULTS.md" if smoke else "RESULTS.md")
    if report.is_file() and "\n## Reading\n" in report.read_text(encoding="utf-8"):
        reading = report.read_text(encoding="utf-8").split("\n## Reading\n", 1)[1].strip()
        lines.extend(["", "## Reading", "", reading, ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "experiment1": {
            "status": "smoke" if smoke else "completed",
            "epochs": 2 if smoke else FULL_EPOCHS,
            "problem": problem.to_dict(),
            "architecture": {"hidden": HIDDEN, "layers": LAYERS, "class": "LevelSetPINN"},
            "runs": metrics,
            "experiments_2_to_5": "not_started",
        }
    }
    summary_path = ROOT / "experiments" / "wildfirepinn_v2_diagnostics" / ("SMOKE_SUMMARY.json" if smoke else "SUMMARY.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {report}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    problem = CircleProblem()
    for rate in RATES:
        clearance = problem.clearance_m(rate)
        if clearance < 500.0:
            raise RuntimeError(f"R={rate} ends {clearance:.1f} m from the boundary; no boundary loss would be invalid")
    epochs = 2 if args.smoke else FULL_EPOCHS
    out_dir = ROOT / "experiments" / "wildfirepinn_v2_diagnostics" / ("smoke" if args.smoke else "experiment1")
    if "outputs" in out_dir.parts:
        raise RuntimeError("Refusing to write under outputs/.")
    (out_dir / "cache").mkdir(parents=True, exist_ok=True)
    (out_dir / "cache" / "circle_problem.json").write_text(json.dumps(problem.to_dict(), indent=2), encoding="utf-8")
    print(
        f"circle L={problem.length:.0f} m T={problem.t_final:.0f} s epochs={epochs} "
        f"arch={HIDDEN}x{LAYERS} device=cpu",
        flush=True,
    )
    metrics = []
    models = {}
    for rate in RATES:
        run_dir = out_dir / "runs" / f"{_tag(rate)}_seed{SEED}"
        print(f"start {run_dir.name}", flush=True)
        model, history, seconds = train_circle(problem, rate, epochs, run_dir)
        record = evaluate_rate(model, problem, rate, seconds, history)
        metrics.append(record)
        models[rate] = model
        final = record["times"][-1]
        print(
            f"R={rate:g} residual {record['pde_residual_m_s']:.4f} m/s | "
            f"IC RMSE {record['ic_rmse_m']:.2f} m | "
            f"PINN radius error {final['pinn']['radius_error_m']} | "
            f"numerical radius error {final['numerical']['radius_error_m']} | "
            f"PINN IoU {final['pinn']['iou']:.3f}",
            flush=True,
        )
    plot_perimeters(models, problem, out_dir / "figures" / "perimeters.png")
    plot_radius_errors(metrics, out_dir / "figures" / "radius_error.png")
    write_reports(metrics, problem, out_dir, args.smoke)


if __name__ == "__main__":
    main()
