"""Learned positive correction to R_eff on the three-pass window.

The phi network, observation loss, and physical R_eff formula are unchanged.
A separate 3-32-32-1 network supplies

    C = 0.1 + 1.9 * sigmoid(raw)

so 0.1 <= C <= 2, and the residual is the usual nondimensional level-set
residual with R_eff replaced by R_eff * C. PDE times stay inside pass 0
through pass 2. The production trainer is not modified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from run_front_pde_diagnostic import _target_meters, _uniform
from run_three_pass_diagnostic import PASS_INDEXES, _draw_rings, _evaluate, _prepare
from wildfire_pinn.pinn import (
    LevelSetPINN,
    _perimeter_batch,
    day_index_from_tau,
    directed_residual,
    downwind_unit_vectors,
    predict_phi_meters,
    sample_volume,
    set_seed,
)
from wildfire_pinn.plots import _increasing_axes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_learned_rate_diagnostic"
BASELINE_DIR = ROOT / "outputs" / "kincade_front_pde_window"
EPOCHS = 1500
EVAL_EVERY = 100
W_PDE = 0.1
LAMBDAS = (0.01, 0.1, 1.0)
NEAR_M = 2000.0


class SpreadCorrection(nn.Module):
    """Small map from normalized (tau, X, Y) to a positive rate factor."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        gain = nn.init.calculate_gain("tanh")
        for module in self.net:
            if isinstance(module, nn.Linear) and module.out_features != 1:
                nn.init.xavier_normal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        # sigmoid(log(0.9)) = 0.9/1.9, so C starts at 1.
        nn.init.constant_(last.bias, math.log(0.9))

    def forward(self, tau, x, y) -> torch.Tensor:
        columns = [value.reshape(-1, 1) for value in (tau, x, y)]
        raw = self.net(torch.cat(columns, dim=1)).squeeze(-1)
        return 0.1 + 1.9 * torch.sigmoid(raw)


def _run_dir(lambda_corr: float) -> Path:
    label = f"lambda_{lambda_corr:g}".replace(".", "p")
    return OUT / label


def _distribution(values: np.ndarray) -> dict:
    chosen = np.asarray(values, dtype=np.float64)
    chosen = chosen[np.isfinite(chosen)]
    if chosen.size == 0:
        return {"n": 0}
    return {
        "n": int(chosen.size),
        "mean": float(chosen.mean()),
        "median": float(np.median(chosen)),
        "std": float(chosen.std()),
        "p05": float(np.percentile(chosen, 5)),
        "p95": float(np.percentile(chosen, 95)),
        "min": float(chosen.min()),
        "max": float(chosen.max()),
    }


def _pde_and_correction(phi_model, correction, prepared, generator):
    case = prepared["case"]
    hp = prepared["hp"]
    device = prepared["device"]
    length = prepared["length"]
    duration = prepared["duration"]
    tau_hi = float(prepared["tau"][2])
    x_pde, y_pde, tau_pde = _uniform(hp.n_pde, prepared, generator, 0.0, tau_hi)
    tau_leaf = tau_pde.detach().requires_grad_(True)
    x_norm = ((x_pde - prepared["center_x"]) / length).detach().requires_grad_(True)
    y_norm = ((y_pde - prepared["center_y"]) / length).detach().requires_grad_(True)
    phi = phi_model(tau_leaf, x_norm, y_norm)
    phi_t, phi_x, phi_y = torch.autograd.grad(
        phi, (tau_leaf, x_norm, y_norm), torch.ones_like(phi), create_graph=True
    )
    factor = correction(tau_pde.detach(), x_norm.detach(), y_norm.detach())
    n_days = int(prepared["spread"].shape[0])
    days = day_index_from_tau(tau_leaf, prepared["t0_offset"], duration, n_days)
    rate = sample_volume(prepared["spread"], days, case.x, case.y, x_pde, y_pde).detach()
    wind_now = prepared["wind"][days]
    residual = directed_residual(
        phi_t, phi_x, phi_y, rate * factor,
        wind_now[:, 0], wind_now[:, 1], hp.c_wdir, duration / length,
    )
    loss_pde = torch.mean(residual ** 2)
    loss_corr = torch.mean((factor - 1.0) ** 2)
    return loss_pde, loss_corr, factor.detach()


def _factor_on_grid(correction, prepared, tau_value: float) -> np.ndarray:
    case = prepared["case"]
    device = prepared["device"]
    length = prepared["length"]
    yy, xx = np.meshgrid(case.y, case.x, indexing="ij")
    flat_x = torch.tensor(xx.reshape(-1), dtype=torch.float32, device=device)
    flat_y = torch.tensor(yy.reshape(-1), dtype=torch.float32, device=device)
    tau = torch.full((flat_x.shape[0],), float(tau_value), device=device)
    pieces = []
    correction.eval()
    with torch.no_grad():
        for start in range(0, flat_x.shape[0], 40000):
            stop = start + 40000
            xn = (flat_x[start:stop] - prepared["center_x"]) / length
            yn = (flat_y[start:stop] - prepared["center_y"]) / length
            pieces.append(correction(tau[start:stop], xn, yn).cpu())
    return torch.cat(pieces).numpy().reshape(xx.shape)


def _sample_factor_stats(correction, prepared) -> dict:
    """C on a fixed uniform draw, and on the part of that draw near the target front."""
    device = prepared["device"]
    length = prepared["length"]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(19)
    x_query, y_query, tau = _uniform(8192, prepared, generator, 0.0, float(prepared["tau"][2]))
    xn = (x_query - prepared["center_x"]) / length
    yn = (y_query - prepared["center_y"]) / length
    correction.eval()
    with torch.no_grad():
        factor = correction(tau, xn, yn).cpu().numpy()
    target_m = _target_meters(x_query, y_query, tau, prepared).detach().cpu().numpy()
    near = np.abs(target_m) <= NEAR_M
    return {
        "all_collocation": _distribution(factor),
        "within_2km": _distribution(factor[near]),
    }


def _physical_rates(prepared):
    """Target phi_t and physical R_eff on the two observation intervals."""
    case = prepared["case"]
    length = prepared["length"]
    duration = prepared["duration"]
    hp = prepared["hp"]
    wind = downwind_unit_vectors(case.wind_from_deg)
    n_days = int(case.spread.shape[0])
    fields = [np.asarray(prepared["sdfs"][i], dtype=np.float64) * length for i in range(3)]
    intervals = []
    for left, right in ((0, 1), (1, 2)):
        dt = (float(prepared["tau"][right]) - float(prepared["tau"][left])) * duration
        tau_mid = 0.5 * (float(prepared["tau"][left]) + float(prepared["tau"][right]))
        seconds = prepared["t0_offset"] + tau_mid * duration
        day = int(np.clip(np.floor(seconds / 86400.0), 0, n_days - 1))
        phi_t = (fields[right] - fields[left]) / dt
        mid = 0.5 * (fields[left] + fields[right])
        dpy, dpx = np.gradient(mid, case.y, case.x)
        magnitude = np.sqrt(dpx ** 2 + dpy ** 2)
        safe = np.maximum(magnitude, 1e-12)
        east, north = wind[day]
        alignment = (east * dpx + north * dpy) / safe
        spread = np.asarray(case.spread[day], dtype=np.float64)
        r_eff = np.clip(spread * (1.0 + hp.c_wdir * alignment), 0.0, None)
        near = (np.abs(fields[left]) <= NEAR_M) | (np.abs(fields[right]) <= NEAR_M)
        usable = near & (magnitude > 0.5) & (magnitude < 1.5)
        implied = np.full(phi_t.shape, np.nan)
        implied[usable] = -phi_t[usable] / magnitude[usable]
        intervals.append({
            "passes": [left, right],
            "dt_hours": dt / 3600.0,
            "tau_mid": tau_mid,
            "day": day,
            "near": near,
            "phi_t": phi_t,
            "r_eff": r_eff,
            "implied_speed": implied,
            "phi_t_near": _distribution(phi_t[near]),
            "r_eff_near": _distribution(r_eff[near]),
            "implied_speed_where_grad_near_1": _distribution(implied[usable]),
        })
    return intervals


def _attach_corrections(intervals, factors_at_mid):
    reports = []
    for interval, factor in zip(intervals, factors_at_mid):
        near = interval["near"]
        corrected = interval["r_eff"] * factor
        reports.append({
            "passes": interval["passes"],
            "dt_hours": interval["dt_hours"],
            "C_theta_near": _distribution(factor[near]),
            "R_eff_near_m_s": interval["r_eff_near"],
            "R_corrected_near_m_s": _distribution(corrected[near]),
            "target_phi_t_near_m_s": interval["phi_t_near"],
            "target_implied_speed_m_s": interval["implied_speed_where_grad_near_1"],
        })
    return reports


def _train_one(lambda_corr: float, prepared, intervals) -> dict:
    out_dir = _run_dir(lambda_corr)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        print(f"reusing {out_dir.name}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    hp = prepared["hp"]
    case = prepared["case"]
    device = prepared["device"]
    set_seed(hp.seed)
    phi_model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    correction = SpreadCorrection().to(device)
    set_seed(hp.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(hp.seed + 17)
    optimizer = torch.optim.Adam(
        list(phi_model.parameters()) + list(correction.parameters()), lr=hp.lr
    )
    with torch.no_grad():
        probe = correction(
            torch.zeros(8, device=device),
            torch.zeros(8, device=device),
            torch.zeros(8, device=device),
        )
    print(
        f"\n=== {out_dir.name} | w_pde={W_PDE:g} | lambda={lambda_corr:g} | "
        f"initial C={float(probe.mean()):.4f} ===",
        flush=True,
    )
    history = {"obs": [], "pde": [], "corr": []}
    evals = []

    def record(epoch: int) -> None:
        phi_model.eval()
        correction.eval()
        rows, _fields = _evaluate(phi_model, prepared)
        factor_stats = _sample_factor_stats(correction, prepared)
        mids = [_factor_on_grid(correction, prepared, item["tau_mid"]) for item in intervals]
        interval_stats = _attach_corrections(intervals, mids)
        payload = {
            "epoch": epoch,
            "passes": rows,
            "C_theta": factor_stats,
            "intervals": interval_stats,
            "observation_loss": float(np.mean([row["observation_loss"] for row in rows])),
            "pde_loss": None if epoch == 0 else history["pde"][-1],
            "corr_loss": None if epoch == 0 else history["corr"][-1],
        }
        evals.append(payload)
        bits = []
        for row in rows:
            distance = row["distance_to_target_m"]
            distance_text = "none" if distance is None else f"{distance:.0f}m"
            bits.append(
                f"k{row['index']} {'Y' if row['zero_contour'] else 'N'} "
                f"{row['negative_km2']:.0f}km2 {distance_text}"
            )
        near = factor_stats["within_2km"]
        median = near.get("median")
        median_text = "n/a" if median is None else f"{median:.2f}"
        print(f"  epoch {epoch:4d} | " + " | ".join(bits) + f" | C_near {median_text}", flush=True)
        phi_model.train()
        correction.train()

    record(0)
    phi_model.train()
    correction.train()
    for epoch in range(1, EPOCHS + 1):
        losses = [
            _perimeter_batch(
                phi_model, float(prepared["tau"][index]), prepared["pools"][local],
                prepared["sdfs"][local], case, hp.n_ic,
                prepared["center_x"], prepared["center_y"], prepared["length"],
            )
            for local, index in enumerate(PASS_INDEXES)
        ]
        loss_obs = torch.stack(losses).mean()
        loss_pde, loss_corr, _factor = _pde_and_correction(phi_model, correction, prepared, generator)
        loss = loss_obs + W_PDE * loss_pde + float(lambda_corr) * loss_corr
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["obs"].append(float(loss_obs.detach()))
        history["pde"].append(float(loss_pde.detach()))
        history["corr"].append(float(loss_corr.detach()))
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            record(epoch)
    torch.save(
        {
            "phi": phi_model.state_dict(),
            "correction": correction.state_dict(),
            "arch": phi_model.arch,
            "lambda_corr": lambda_corr,
            "w_pde": W_PDE,
        },
        out_dir / "levelset_pinn.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (out_dir / "evals.json").write_text(json.dumps(evals, indent=2), encoding="utf-8")
    final = evals[-1]
    final["pde_loss"] = history["pde"][-1]
    final["corr_loss"] = history["corr"][-1]
    summary = {
        "strategy": f"learned_lambda_{lambda_corr:g}",
        "lambda_corr": lambda_corr,
        "w_pde": W_PDE,
        "epochs": EPOCHS,
        "reused": False,
        "passes": final["passes"],
        "observation_loss": final["observation_loss"],
        "pde_loss": final["pde_loss"],
        "corr_loss": final["corr_loss"],
        "C_theta": final["C_theta"],
        "intervals": final["intervals"],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _load_baseline(folder: str, strategy: str) -> dict:
    path = BASELINE_DIR / folder / "summary.json"
    if not path.exists():
        raise RuntimeError(f"Missing reused baseline {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["strategy"] = strategy
    raw["reused"] = True
    raw["epochs"] = 1000
    raw["source"] = str(path)
    return raw


def _load_phi(path: Path, prepared) -> LevelSetPINN:
    checkpoint = torch.load(path, map_location=prepared["device"], weights_only=False)
    model = LevelSetPINN(prepared["hp"].hidden, prepared["hp"].layers).to(prepared["device"])
    state = checkpoint["phi"] if "phi" in checkpoint else checkpoint["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def _load_correction(path: Path, prepared) -> SpreadCorrection:
    checkpoint = torch.load(path, map_location=prepared["device"], weights_only=False)
    correction = SpreadCorrection().to(prepared["device"])
    correction.load_state_dict(checkpoint["correction"])
    correction.eval()
    return correction


def _zoom(prepared, mask):
    case = prepared["case"]
    yy, xx = np.meshgrid(case.y, case.x, indexing="ij")
    pad = 1500.0
    return (
        float(xx[mask].min()) - pad,
        float(xx[mask].max()) + pad,
        float(yy[mask].min()) - pad,
        float(yy[mask].max()) + pad,
    )


def _write_plots(results, prepared, intervals) -> None:
    case = prepared["case"]
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    panels = [
        ("Observation only", BASELINE_DIR / "observation_only_w0" / "levelset_pinn.pt"),
        ("Fixed R, w=0.1", BASELINE_DIR / "current_w0p1" / "levelset_pinn.pt"),
    ]
    for lambda_corr in LAMBDAS:
        panels.append((f"Learned λ={lambda_corr:g}", _run_dir(lambda_corr) / "levelset_pinn.pt"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 4.6), layout="constrained")
    mesh = None
    for ax, (title, path) in zip(axes, panels):
        model = _load_phi(path, prepared)
        phi = predict_phi_meters(model, case, float(prepared["tau"][2]))
        shown, xd, yd = _increasing_axes(phi, x, y)
        mesh = ax.pcolormesh(xd, yd, shown, cmap="RdBu", vmin=-5000, vmax=5000, shading="auto")
        ax.contour(xd, yd, shown, levels=[0], colors="black", linewidths=0.9)
        _draw_rings(ax, prepared["targets"][2], "darkorange", 0.8, None)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        del model
    fig.colorbar(mesh, ax=axes, fraction=0.02, label="predicted φ (m), capped at ±5 km")
    fig.suptitle("Pass 2. Black is φ=0. Orange is the persistent target.")
    fig.savefig(OUT / "pass2_comparison.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(4.4 * len(LAMBDAS), 4.6), layout="constrained")
    zoom = None
    for ax, lambda_corr in zip(axes, LAMBDAS):
        correction = _load_correction(_run_dir(lambda_corr) / "levelset_pinn.pt", prepared)
        field = _factor_on_grid(correction, prepared, float(prepared["tau"][2]))
        target = np.asarray(prepared["sdfs"][2], dtype=np.float64) * prepared["length"]
        if zoom is None:
            zoom = _zoom(prepared, np.abs(target) <= 4000.0)
        shown, xd, yd = _increasing_axes(field, x, y)
        mesh = ax.pcolormesh(xd, yd, shown, cmap="RdBu_r", vmin=0.1, vmax=2.0, shading="auto")
        _draw_rings(ax, prepared["targets"][2], "black", 0.8, None)
        ax.set_title(f"λ={lambda_corr:g}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
        del correction
    fig.colorbar(mesh, ax=axes, fraction=0.03, label="C_theta at pass 2")
    fig.suptitle("Learned rate factor. 1 leaves physical R_eff unchanged.")
    fig.savefig(OUT / "correction_maps.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.2), layout="constrained")
    for row, interval in enumerate(intervals):
        near = interval["near"]
        columns = [
            (np.where(near, -interval["phi_t"], np.nan), "target −φ_t", -0.15, 0.15),
            (np.where(near, interval["r_eff"], np.nan), "R_eff", 0.0, 0.2),
        ]
        for lambda_corr in LAMBDAS:
            correction = _load_correction(_run_dir(lambda_corr) / "levelset_pinn.pt", prepared)
            factor = _factor_on_grid(correction, prepared, interval["tau_mid"])
            columns.append((
                np.where(near, interval["r_eff"] * factor, np.nan),
                f"R_eff C, λ={lambda_corr:g}",
                0.0,
                0.2,
            ))
            del correction
        x0, x1, y0, y1 = _zoom(prepared, near)
        for ax, (field, label, vmin, vmax) in zip(axes[row], columns):
            shown, xd, yd = _increasing_axes(field, x, y)
            mesh = ax.pcolormesh(xd, yd, shown, cmap="RdBu_r", vmin=vmin, vmax=vmax, shading="auto")
            fig.colorbar(mesh, ax=ax, fraction=0.046)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"pass {interval['passes'][0]}→{interval['passes'][1]}: {label}", fontsize=9)
    fig.suptitle("Near the observed front. Positive −φ_t means the target signed distance is decreasing.")
    fig.savefig(OUT / "rate_comparison.png", dpi=130)
    plt.close(fig)


def _print_table(results: list[dict]) -> None:
    print(
        "strategy\tw\tlambda\tk0\tk1\tk2\tarea0\tarea1\tarea2\tmin2\tdist2\t"
        "obs\tpde\tcorr\tC_near_median",
        flush=True,
    )
    for item in results:
        passes = {row["index"]: row for row in item["passes"]}
        distance = passes[2]["distance_to_target_m"]
        distance_text = "" if distance is None else f"{distance:.0f}"
        corr = item.get("corr_loss")
        corr_text = "" if corr is None else f"{corr:.3e}"
        pde = item.get("pde_loss")
        pde_text = "" if pde is None else f"{pde:.3e}"
        median = ""
        if item.get("C_theta"):
            value = item["C_theta"]["within_2km"].get("median")
            median = "" if value is None else f"{value:.3f}"
        print(
            "\t".join([
                item["strategy"],
                f"{item.get('w_pde', 0):g}",
                "" if item.get("lambda_corr") is None else f"{item['lambda_corr']:g}",
                "Y" if passes[0]["zero_contour"] else "N",
                "Y" if passes[1]["zero_contour"] else "N",
                "Y" if passes[2]["zero_contour"] else "N",
                f"{passes[0]['negative_km2']:.1f}",
                f"{passes[1]['negative_km2']:.1f}",
                f"{passes[2]['negative_km2']:.1f}",
                f"{passes[2]['min_phi_m']:.0f}",
                distance_text,
                f"{item['observation_loss']:.3e}",
                pde_text,
                corr_text,
                median,
            ]),
            flush=True,
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = _prepare()
    tau_hi = float(prepared["tau"][2])
    print(f"PDE tau restricted to [0, {tau_hi:.4f}]", flush=True)
    intervals = _physical_rates(prepared)
    learned = [_train_one(value, prepared, intervals) for value in LAMBDAS]
    results = [
        _load_baseline("observation_only_w0", "observation_only"),
        _load_baseline("current_w0p1", "fixed_R_w0p1"),
        *learned,
    ]
    (OUT / "comparison.json").write_text(json.dumps({
        "w_pde": W_PDE,
        "lambdas": list(LAMBDAS),
        "epochs": EPOCHS,
        "tau_hi": tau_hi,
        "baselines_reused_from": str(BASELINE_DIR),
        "baseline_epochs": 1000,
        "results": results,
        "target_intervals": [
            {
                "passes": item["passes"],
                "phi_t_near": item["phi_t_near"],
                "r_eff_near": item["r_eff_near"],
                "implied_speed": item["implied_speed_where_grad_near_1"],
            }
            for item in intervals
        ],
    }, indent=2), encoding="utf-8")
    _write_plots(results, prepared, intervals)
    _print_table(results)


if __name__ == "__main__":
    main()
