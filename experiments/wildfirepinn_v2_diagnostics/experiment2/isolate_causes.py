"""Isolate sampling, representation, and optimization for Experiment 2A.

The 2A checkpoint is loaded and not overwritten. Experiment 1, Experiment 1B,
and the Kincade model are not modified. No wind or irregular-fire training.

    python -m experiments.wildfirepinn_v2_diagnostics.experiment2.isolate_causes
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.wildfirepinn_v2_diagnostics.circle import CircleProblem, analytical_phi
from experiments.wildfirepinn_v2_diagnostics.experiment1 import HIDDEN, LAYERS, N_PDE, _sample_pde
from experiments.wildfirepinn_v2_diagnostics.experiment2.fields import R0_M_S, spatial_rate_np, spatial_rate_torch
from experiments.wildfirepinn_v2_diagnostics.experiment2.metrics import score_against_reference
from experiments.wildfirepinn_v2_diagnostics.experiment2.physics import circle_phi0, hard_ic
from experiments.wildfirepinn_v2_diagnostics.experiment2.runner import (
    EPOCHS,
    LR,
    SEED,
    _growth_by_rate,
    _problem,
    _times,
    predict_phi,
)
from experiments.wildfirepinn_v2_diagnostics.solver import propagate
from wildfire_pinn.pinn import LevelSetPINN, load_checkpoint, save_checkpoint, set_seed

HERE = Path(__file__).resolve().parent
OUT = HERE / "cause_isolation"
BAND_M = 100.0
SNAPSHOT_DT_S = 50.0
N_STAT = 8192
N_REP = 4


def _spec_2a():
    return {
        "name": "2a",
        "label": "spatial R, circular initial fire, uniform samples",
        "rate": "spatial",
        "geometry": "circle",
        "wind": None,
        "sampling": "uniform",
    }


def _phi0(problem, x, y):
    xx, yy = np.meshgrid(x, y)
    phi0 = analytical_phi(xx, yy, 0.0, problem, 0.0)
    return phi0, lambda xn, yn: circle_phi0(xn, yn, problem)


def _rate_gradient(x, y):
    kx = 2.0 * np.pi / 1500.0
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    drdx = R0_M_S * 0.5 * kx * np.cos(kx * xx) * np.cos(kx * yy)
    drdy = -R0_M_S * 0.5 * kx * np.sin(kx * xx) * np.sin(kx * yy)
    return np.sqrt(drdx ** 2 + drdy ** 2)


def _rate_gradient_torch(x, y):
    kx = 2.0 * torch.pi / 1500.0
    drdx = R0_M_S * 0.5 * kx * torch.cos(kx * x) * torch.cos(kx * y)
    drdy = -R0_M_S * 0.5 * kx * torch.sin(kx * x) * torch.sin(kx * y)
    return torch.sqrt(drdx ** 2 + drdy ** 2)


class ReferenceVolume:
    """Linear-in-time, bilinear-in-space samples of the numerical phi field."""

    def __init__(self, times: np.ndarray, stack: np.ndarray, x: np.ndarray, y: np.ndarray):
        self.times = np.asarray(times, dtype=np.float64)
        self.stack = torch.tensor(np.asarray(stack, dtype=np.float32))
        self.x0 = float(x[0])
        self.y0 = float(y[0])
        self.dx = float(x[1] - x[0])
        self.dy = float(y[1] - y[0])
        self.dt = float(self.times[1] - self.times[0])
        self.nx = int(stack.shape[2])
        self.ny = int(stack.shape[1])
        self.nt = int(stack.shape[0])

    def at(self, xq, yq, tq) -> torch.Tensor:
        fx = (xq.detach().reshape(-1).float() - self.x0) / self.dx
        fy = (yq.detach().reshape(-1).float() - self.y0) / self.dy
        ft = tq.detach().reshape(-1).float() / self.dt
        ix, wx = _fractional(fx, self.nx)
        iy, wy = _fractional(fy, self.ny)
        it, wt = _fractional(ft, self.nt)
        volume = self.stack

        def take(t_index, y_index, x_index):
            return volume[t_index, y_index, x_index]

        c00 = (1.0 - wx) * take(it, iy, ix) + wx * take(it, iy, ix + 1)
        c01 = (1.0 - wx) * take(it, iy + 1, ix) + wx * take(it, iy + 1, ix + 1)
        c10 = (1.0 - wx) * take(it + 1, iy, ix) + wx * take(it + 1, iy, ix + 1)
        c11 = (1.0 - wx) * take(it + 1, iy + 1, ix) + wx * take(it + 1, iy + 1, ix + 1)
        c0 = (1.0 - wy) * c00 + wy * c01
        c1 = (1.0 - wy) * c10 + wy * c11
        return (1.0 - wt) * c0 + wt * c1


def _fractional(coord: torch.Tensor, count: int):
    i0 = torch.floor(coord).long()
    at_end = i0 >= count - 1
    i0 = i0.clamp(0, count - 2)
    weight = (coord - i0.float()).clamp(0.0, 1.0)
    weight = torch.where(at_end, torch.ones_like(weight), weight)
    return i0, weight


def build_reference(problem, phi0, x, y):
    times = np.arange(0.0, problem.t_final + 1e-9, SNAPSHOT_DT_S, dtype=np.float64)
    if abs(times[-1] - problem.t_final) > 1e-6:
        times = np.append(times, problem.t_final)
    xx, yy = np.meshgrid(x, y)
    rate = spatial_rate_np(xx, yy)
    started = time.perf_counter()
    fields = propagate(phi0, x, y, rate, times.tolist())
    elapsed = time.perf_counter() - started
    stack = np.stack([fields[float(moment)] for moment in times], axis=0)
    volume = ReferenceVolume(times, stack, x, y)
    return volume, rate, elapsed


def interpolation_error(problem, phi0, x, y, volume: ReferenceVolume) -> dict:
    """Compare a mid-interval blend against an independent solver time."""
    moment = SNAPSHOT_DT_S * 0.5
    xx, yy = np.meshgrid(x, y)
    rate = spatial_rate_np(xx, yy)
    truth = propagate(phi0, x, y, rate, [moment])[moment]
    query_x = torch.tensor(xx.ravel(), dtype=torch.float32)
    query_y = torch.tensor(yy.ravel(), dtype=torch.float32)
    query_t = torch.full((xx.size,), moment, dtype=torch.float32)
    blended = volume.at(query_x, query_y, query_t).numpy().reshape(xx.shape)
    row = score_against_reference(blended, truth, x, y)
    row["rmse_m"] = float(np.sqrt(np.mean((blended - truth) ** 2)))
    row["t_s"] = moment
    return row


def forward_terms(model, problem, phi0_fn, x, y, t):
    length = problem.length
    duration = problem.t_final
    tau = (t / duration).detach().requires_grad_(True)
    x_norm = ((x - problem.center_x) / length).detach().requires_grad_(True)
    y_norm = ((y - problem.center_y) / length).detach().requires_grad_(True)
    phi = hard_ic(model(tau, x_norm, y_norm), tau, phi0_fn(x_norm, y_norm))
    phi_tau, phi_x, phi_y = torch.autograd.grad(
        phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=True
    )
    t_over_l = duration / length
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-8)
    rate = spatial_rate_torch(x.detach(), y.detach())
    phi_t = phi_tau / t_over_l
    physical = phi_t + rate * magnitude
    coded = physical * t_over_l
    return {
        "phi_norm": phi,
        "phi_t": phi_t,
        "magnitude": magnitude,
        "rate": rate,
        "physical": physical,
        "coded": coded,
    }


def _flat_grad(scalar, params):
    grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=False)
    return torch.cat([grad.reshape(-1) for grad in grads])


def _cosine(left, right) -> float | None:
    denom = float(left.norm() * right.norm())
    if denom < 1e-18:
        return None
    return float(torch.dot(left, right) / denom)


def _params(model):
    return [p for p in model.parameters() if p.requires_grad]


def _uniform_batch(n, problem, seed):
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


def _band_batch(n, problem, volume, seed, band):
    generator = torch.Generator()
    generator.manual_seed(seed)
    xs, ys, ts = [], [], []
    got = 0
    while got < n:
        x = torch.rand(8192, generator=generator) * problem.length - problem.half_extent
        y = torch.rand(8192, generator=generator) * problem.length - problem.half_extent
        t = torch.rand(8192, generator=generator) * problem.t_final
        away = torch.hypot(x - problem.x0, y - problem.y0) > problem.pde_exclusion_m
        near = volume.at(x, y, t).abs() <= band
        keep = away & near
        xs.append(x[keep])
        ys.append(y[keep])
        ts.append(t[keep])
        got += int(keep.sum())
    return torch.cat(xs)[:n], torch.cat(ys)[:n], torch.cat(ts)[:n]


def _region_row(name, coded, physical, phi_t, magnitude, rate, mask) -> dict:
    count = int(mask.sum())
    total = float((coded ** 2).sum())
    if count == 0 or total <= 0:
        return {"region": name, "count": count, "fraction": 0.0}
    chosen = mask
    share = float((coded[chosen] ** 2).sum() / (coded ** 2).sum())
    return {
        "region": name,
        "count": count,
        "fraction": float(count / coded.numel()),
        "loss_share": share,
        "mean_abs_residual_m_s": float(physical[chosen].abs().mean()),
        "mean_phi_t_m_s": float(phi_t[chosen].mean()),
        "mean_abs_grad": float(magnitude[chosen].mean()),
        "mean_R_m_s": float(rate[chosen].mean()),
        "std_phi_t_m_s": float(phi_t[chosen].std(unbiased=False)),
        "corr_phi_t_R": _corr(phi_t[chosen], rate[chosen]),
        "corr_residual_R": _corr(physical[chosen], rate[chosen]),
    }


def _corr(left, right) -> float | None:
    if left.numel() < 8:
        return None
    a = left - left.mean()
    b = right - right.mean()
    denom = float(torch.sqrt((a ** 2).sum() * (b ** 2).sum()))
    if denom < 1e-18:
        return None
    return float((a * b).sum() / denom)


def sampling_report(model, problem, phi0_fn, volume) -> dict:
    model.eval()
    rows = []
    x, y, t = _uniform_batch(N_STAT, problem, seed=1100)
    terms = forward_terms(model, problem, phi0_fn, x, y, t)
    coded = terms["coded"].detach()
    physical = terms["physical"].detach()
    phi_t = terms["phi_t"].detach()
    magnitude = terms["magnitude"].detach()
    rate = terms["rate"].detach()
    ref = volume.at(x, y, t)
    grad_r = _rate_gradient_torch(x, y)
    steep_cut = float(grad_r.max()) * 0.5
    masks = {
        "all_uniform": torch.ones_like(rate, dtype=torch.bool),
        "front_100m": ref.abs() <= BAND_M,
        "outside_front": ref.abs() > BAND_M,
        "high_R": rate >= R0_M_S,
        "low_R": rate < R0_M_S,
        "steep_R": grad_r >= steep_cut,
        "bulk_abs_grad_gt_0p5": magnitude > 0.5,
    }
    for name, mask in masks.items():
        rows.append(_region_row(name, coded, physical, phi_t, magnitude, rate, mask))
    floor_rate = float(rate.mean())
    constant_floor = float(((rate - floor_rate) ** 2).mean())
    optimal_constant = -float((rate * magnitude).mean())
    return {
        "n": int(rate.numel()),
        "mean_R_m_s": floor_rate,
        "std_R_m_s": float(rate.std(unbiased=False)),
        "constant_speed_loss_floor": constant_floor,
        "optimal_constant_phi_t_m_s": optimal_constant,
        "model_mean_squared_residual": float((physical ** 2).mean()),
        "steep_threshold_per_m": steep_cut,
        "regions": rows,
    }


def _batch_gradients(model, problem, phi0_fn, x, y, t) -> dict:
    model.train()
    terms = forward_terms(model, problem, phi0_fn, x, y, t)
    params = _params(model)
    rate = terms["rate"]
    high = rate >= R0_M_S
    low = ~high
    if int(high.sum()) < 32 or int(low.sum()) < 32:
        raise RuntimeError(f"R split is too small: high {int(high.sum())} low {int(low.sum())}")
    loss = torch.mean(terms["coded"] ** 2)
    g_loss = _flat_grad(loss, params).detach()
    g_high = _flat_grad(torch.mean(terms["coded"][high] ** 2), params).detach()
    g_low = _flat_grad(torch.mean(terms["coded"][low] ** 2), params).detach()
    gap = terms["phi_t"][high].mean() - terms["phi_t"][low].mean()
    g_gap = _flat_grad(gap, params).detach()
    phi_t = terms["phi_t"].detach()
    record = {
        "loss": float(loss.detach()),
        "grad_norm": float(g_loss.norm()),
        "high_grad_norm": float(g_high.norm()),
        "low_grad_norm": float(g_low.norm()),
        "contrast_grad_norm": float((g_high - g_low).norm()),
        "common_grad_norm": float((g_high + g_low).norm()),
        "cosine_high_low": _cosine(g_high, g_low),
        "cosine_loss_vs_speed_gap": _cosine(g_loss, g_gap),
        "predicted_gap_change_per_sgd_step": float(-LR * torch.dot(g_loss, g_gap)),
        "phi_t_high": float(phi_t[high].mean()),
        "phi_t_low": float(phi_t[low].mean()),
        "speed_gap_m_s": float(gap.detach()),
        "n_high": int(high.sum()),
        "n_low": int(low.sum()),
        "mean_abs_grad_high": float(terms["magnitude"].detach()[high].mean()),
        "mean_abs_grad_low": float(terms["magnitude"].detach()[low].mean()),
    }
    model.zero_grad(set_to_none=True)
    return record


def _mean_record(rows: list[dict]) -> dict:
    keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) or value is None]
    out = {"batches": len(rows)}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        if not values:
            out[key] = None
            continue
        out[key] = float(np.mean(values))
        if len(values) > 1:
            out[key + "_std"] = float(np.std(values))
    return out


def gradient_survey(model, problem, phi0_fn, volume, label: str) -> dict:
    uniform_rows = []
    front_rows = []
    for index in range(N_REP):
        x, y, t = _uniform_batch(N_PDE, problem, seed=2000 + index)
        uniform_rows.append(_batch_gradients(model, problem, phi0_fn, x, y, t))
        x, y, t = _band_batch(N_PDE, problem, volume, seed=3000 + index, band=BAND_M)
        front_rows.append(_batch_gradients(model, problem, phi0_fn, x, y, t))
    model.eval()
    return {"label": label, "uniform": _mean_record(uniform_rows), "front_100m": _mean_record(front_rows)}


def empirical_step(model, problem, phi0_fn, volume, optimizer_state) -> dict:
    """One Adam update from the saved 2A state, then restore the weights."""
    backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
    x, y, t = _uniform_batch(N_PDE, problem, seed=4040)
    front_x, front_y, front_t = _band_batch(2048, problem, volume, seed=4041, band=BAND_M)
    before = _speed_gap(model, problem, phi0_fn, x, y, t)
    front_before = _speed_gap(model, problem, phi0_fn, front_x, front_y, front_t)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    terms = forward_terms(model, problem, phi0_fn, x, y, t)
    loss = torch.mean(terms["coded"] ** 2)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in model.parameters())))
    weight_norm = float(torch.sqrt(sum(p.detach().pow(2).sum() for p in model.parameters())))
    optimizer.step()
    step_norm = float(torch.sqrt(sum((model.state_dict()[key] - backup[key]).pow(2).sum() for key in backup)))
    after = _speed_gap(model, problem, phi0_fn, x, y, t)
    front_after = _speed_gap(model, problem, phi0_fn, front_x, front_y, front_t)
    model.load_state_dict(backup)
    model.eval()
    return {
        "uniform_batch_loss": float(loss.detach()),
        "grad_norm": grad_norm,
        "weight_norm": weight_norm,
        "adam_step_norm": step_norm,
        "uniform_speed_gap_before": before,
        "uniform_speed_gap_after": after,
        "uniform_gap_change": after - before,
        "front_speed_gap_before": front_before,
        "front_speed_gap_after": front_after,
        "front_gap_change": front_after - front_before,
    }


def _speed_gap(model, problem, phi0_fn, x, y, t) -> float:
    was = model.training
    model.train()
    terms = forward_terms(model, problem, phi0_fn, x, y, t)
    high = terms["rate"] >= R0_M_S
    gap = float((terms["phi_t"][high].mean() - terms["phi_t"][~high].mean()).detach())
    model.zero_grad(set_to_none=True)
    if not was:
        model.eval()
    return gap


def _new_model():
    set_seed(SEED)
    return LevelSetPINN(HIDDEN, LAYERS)


def _finished(run_dir: Path, epochs: int):
    history_path = run_dir / "history.json"
    checkpoint = run_dir / "levelset_pinn.pt"
    if not history_path.is_file() or not checkpoint.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if len(history.get("loss", [])) != epochs:
        return None
    model = LevelSetPINN(HIDDEN, LAYERS)
    load_checkpoint(checkpoint, model, torch.device("cpu"))
    model.eval()
    return model, history


def train_supervised(name, problem, phi0_fn, target_meters, epochs: int):
    run_dir = OUT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    finished = _finished(run_dir, epochs)
    if finished is not None:
        print(f"resume finished {name}", flush=True)
        return finished[0], finished[1], 0.0
    model = _new_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"loss": []}
    length = problem.length
    duration = problem.t_final
    log_every = 250
    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        x, y, t = _sample_pde(N_PDE, problem, torch.device("cpu"))
        tau = t / duration
        x_norm = (x - problem.center_x) / length
        y_norm = (y - problem.center_y) / length
        pred = hard_ic(model(tau, x_norm, y_norm), tau, phi0_fn(x_norm, y_norm))
        target = target_meters(x, y, t).detach() / length
        loss = torch.mean((pred - target) ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite supervised loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))
        if epoch == 1 or epoch % log_every == 0:
            print(f"{name} epoch {epoch:4d} | mse {history['loss'][-1]:.3e}", flush=True)
            save_checkpoint(run_dir / "levelset_pinn.pt", model, {"name": name, "epochs": epochs, "hidden": HIDDEN, "layers": LAYERS})
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    model.eval()
    return model, history, time.perf_counter() - started


def train_pde_from(name, start: LevelSetPINN, problem, phi0_fn, epochs: int):
    run_dir = OUT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    finished = _finished(run_dir, epochs)
    if finished is not None:
        print(f"resume finished {name}", flush=True)
        return finished[0], finished[1], 0.0
    model = _new_model()
    model.load_state_dict(start.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = {"loss": []}
    spec = _spec_2a()
    from experiments.wildfirepinn_v2_diagnostics.experiment2.runner import _batch_residual

    log_every = 250
    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        x, y, t = _sample_pde(N_PDE, problem, torch.device("cpu"))
        coded, _physical, _mag = _batch_residual(model, problem, phi0_fn, spec, x, y, t)
        loss = torch.mean(coded ** 2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite PDE loss at epoch {epoch}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["loss"].append(float(loss.detach()))
        if epoch == 1 or epoch % log_every == 0:
            print(f"{name} epoch {epoch:4d} | pde {history['loss'][-1]:.3e}", flush=True)
            save_checkpoint(run_dir / "levelset_pinn.pt", model, {"name": name, "epochs": epochs, "hidden": HIDDEN, "layers": LAYERS})
            (run_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    model.eval()
    return model, history, time.perf_counter() - started


def score_model(model, problem, phi0_fn, fields, rate_grid, x, y) -> dict:
    rows = []
    final = None
    for moment in _times(problem):
        pred = predict_phi(model, problem, phi0_fn, moment, x, y)
        row = score_against_reference(pred, fields[moment], x, y)
        row["t_s"] = moment
        row["rmse_m"] = float(np.sqrt(np.mean((pred - fields[moment]) ** 2)))
        yy, xx = np.meshgrid(y, x, indexing="ij")
        away = np.hypot(xx - problem.x0, yy - problem.y0) > problem.pde_exclusion_m
        row["rmse_outside_kink_m"] = float(np.sqrt(np.mean((pred - fields[moment])[away] ** 2)))
        rows.append(row)
        final = pred
    growth = _growth_by_rate(final, x, y, problem, rate_grid)
    return {"times": rows, "growth": growth, "final_iou": rows[-1]["iou"], "final_symmetric_m": rows[-1]["symmetric_distance_m"]}


def constant_fields(problem, x, y):
    fields = {}
    for moment in _times(problem):
        yy, xx = np.meshgrid(y, x, indexing="ij")
        fields[moment] = analytical_phi(xx, yy, moment, problem, R0_M_S)
    return fields


def scatter_phi_t(path: Path, series: list[tuple[str, np.ndarray, np.ndarray]]):
    fig, axes = plt.subplots(1, len(series), figsize=(4.2 * len(series), 4.0), sharey=True)
    if len(series) == 1:
        axes = [axes]
    for ax, (title, rate, phi_t) in zip(axes, series):
        ax.scatter(rate, phi_t, s=6, alpha=0.35, c="0.2", linewidths=0)
        grid = np.linspace(float(rate.min()), float(rate.max()), 40)
        ax.plot(grid, -grid, color="crimson", lw=1.4, label="phi_t = -R")
        ax.set_xlabel("R (m/s)")
        ax.set_title(title)
        ax.set_xlim(0.02, 0.08)
    axes[0].set_ylabel("phi_t (m/s)")
    axes[0].legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def front_cloud(model, problem, phi0_fn, volume, n=2500, seed=77):
    """Points within 100 m of the final numerical front."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    kept_x, kept_y = [], []
    got = 0
    while got < n:
        xs = torch.rand(8192, generator=generator) * problem.length - problem.half_extent
        ys = torch.rand(8192, generator=generator) * problem.length - problem.half_extent
        ts = torch.full((8192,), problem.t_final)
        away = torch.hypot(xs - problem.x0, ys - problem.y0) > problem.pde_exclusion_m
        near = volume.at(xs, ys, ts).abs() <= BAND_M
        keep = away & near
        kept_x.append(xs[keep])
        kept_y.append(ys[keep])
        got += int(keep.sum())
    x = torch.cat(kept_x)[:n]
    y = torch.cat(kept_y)[:n]
    t = torch.full((n,), problem.t_final)
    terms = forward_terms(model, problem, phi0_fn, x, y, t)
    return terms["rate"].detach().numpy(), terms["phi_t"].detach().numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problem = _problem()
    x, y = problem.grid()
    phi0, phi0_fn = _phi0(problem, x, y)
    print("numerical reference", flush=True)
    volume, rate_grid, solver_seconds = build_reference(problem, phi0, x, y)
    interp = interpolation_error(problem, phi0, x, y, volume)
    print(
        f"interpolation at {interp['t_s']:.0f}s symmetric {interp['symmetric_distance_m']:.3f} m rmse {interp['rmse_m']:.3f} m",
        flush=True,
    )
    node_x = torch.tensor(x[::5], dtype=torch.float32)
    node_y = torch.tensor(y[::5], dtype=torch.float32)
    node_yy, node_xx = torch.meshgrid(node_y, node_x, indexing="ij")
    sampled0 = volume.at(node_xx.reshape(-1), node_yy.reshape(-1), torch.zeros(node_xx.numel()))
    node_gap = float(np.max(np.abs(sampled0.numpy().reshape(node_yy.shape) - phi0[::5, ::5])))
    if node_gap > 1e-3:
        raise RuntimeError(f"reference interpolation misses the grid by {node_gap:.3e} m")
    print(f"node interpolation error {node_gap:.3e} m", flush=True)
    fields = propagate(phi0, x, y, rate_grid, _times(problem))

    device = torch.device("cpu")
    trained = LevelSetPINN(HIDDEN, LAYERS).to(device)
    load_checkpoint(HERE / "runs" / "2a" / "levelset_pinn.pt", trained, device)
    trained.eval()
    state_path = HERE / "runs" / "2a" / "trainer_state.pt"
    optimizer_state = torch.load(state_path, map_location=device, weights_only=False)["optimizer"]

    print("sampling", flush=True)
    samples = sampling_report(trained, problem, phi0_fn, volume)
    print("gradients at initialization and at 2A", flush=True)
    initial = _new_model()
    gradients = {
        "initialization": gradient_survey(initial, problem, phi0_fn, volume, "initialization"),
        "trained_2a": gradient_survey(trained, problem, phi0_fn, volume, "trained_2a"),
    }
    print("one Adam step from the 2A state", flush=True)
    step = empirical_step(trained, problem, phi0_fn, volume, optimizer_state)

    print("supervised fit to the numerical field", flush=True)
    spatial_model, spatial_history, spatial_seconds = train_supervised(
        "supervised_spatial",
        problem,
        phi0_fn,
        lambda xq, yq, tq: volume.at(xq, yq, tq),
        EPOCHS,
    )
    print("supervised fit to a constant-R circle", flush=True)

    def constant_target(xq, yq, tq):
        return torch.hypot(xq - problem.x0, yq - problem.y0) - (problem.r0 + R0_M_S * tq)

    constant_model, constant_history, constant_seconds = train_supervised(
        "supervised_constant",
        problem,
        phi0_fn,
        constant_target,
        EPOCHS,
    )
    print("PDE training from the supervised spatial weights", flush=True)
    continued, continued_history, continued_seconds = train_pde_from(
        "pde_from_supervised", spatial_model, problem, phi0_fn, EPOCHS
    )

    print("scoring", flush=True)
    scores = {
        "trained_2a": score_model(trained, problem, phi0_fn, fields, rate_grid, x, y),
        "supervised_spatial": score_model(spatial_model, problem, phi0_fn, fields, rate_grid, x, y),
        "supervised_constant_vs_spatial_reference": score_model(constant_model, problem, phi0_fn, fields, rate_grid, x, y),
        "supervised_constant_vs_constant_circle": score_model(
            constant_model, problem, phi0_fn, constant_fields(problem, x, y), rate_grid, x, y
        ),
        "pde_from_supervised": score_model(continued, problem, phi0_fn, fields, rate_grid, x, y),
    }
    for name, model in (
        ("supervised_spatial", spatial_model),
        ("pde_from_supervised", continued),
    ):
        gradients[name] = gradient_survey(model, problem, phi0_fn, volume, name)

    rate_s, speed_s = front_cloud(spatial_model, problem, phi0_fn, volume)
    rate_a, speed_a = front_cloud(trained, problem, phi0_fn, volume)
    rate_c, speed_c = front_cloud(continued, problem, phi0_fn, volume)
    scatter_phi_t(
        OUT / "figures" / "phi_t_vs_r.png",
        [
            ("2A PDE, uniform samples", rate_a, speed_a),
            ("supervised fit to numerical phi", rate_s, speed_s),
            ("PDE loss after that fit", rate_c, speed_c),
        ],
    )
    payload = {
        "experiment": "2a-cause-isolation",
        "epochs": EPOCHS,
        "seed": SEED,
        "architecture": {"hidden": HIDDEN, "layers": LAYERS},
        "solver_seconds": solver_seconds,
        "interpolation_check": interp,
        "sampling": samples,
        "gradients": gradients,
        "adam_step_from_2a": step,
        "scores": scores,
        "train_seconds": {
            "supervised_spatial": spatial_seconds,
            "supervised_constant": constant_seconds,
            "pde_from_supervised": continued_seconds,
        },
        "final_losses": {
            "supervised_spatial_mse": spatial_history["loss"][-1],
            "supervised_constant_mse": constant_history["loss"][-1],
            "pde_from_supervised": continued_history["loss"][-1],
            "supervised_spatial_epoch1": spatial_history["loss"][0],
            "pde_from_supervised_epoch1": continued_history["loss"][0],
        },
    }
    (OUT / "CAUSE_ISOLATION.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'CAUSE_ISOLATION.json'}", flush=True)
    for name, score in scores.items():
        growth = score["growth"]
        print(
            f"{name} IoU {score['final_iou']:.3f} symmetric {score['final_symmetric_m']:.2f} m "
            f"high {growth['high_r_growth_m']:.1f} low {growth['low_r_growth_m']:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
