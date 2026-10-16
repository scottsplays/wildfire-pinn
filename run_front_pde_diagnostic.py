"""Front-local PDE loss on the three-pass persistent targets.

The production trainer is not modified. Observation loss, network, optimizer,
R, wind, and directed_residual match run_three_pass_diagnostic.py.

The front mask is built from the persistent signed-distance target, linearly
interpolated in time between passes 0, 1, and 2. After pass 2 the last target
is held. The network's own phi is never used to decide where the PDE applies.

Hard bands average the usual residual only on uniform collocation points that
fall inside |phi_target| < band, drawing until the batch is full. Soft weights
use the same uniform collocation set as the current loss and replace the mean
with a normalized weighted mean, w = exp(-(|phi_target|/sigma)^2).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_three_pass_diagnostic import (
    EPOCHS,
    PASS_INDEXES,
    _draw_rings,
    _evaluate,
    _prepare,
    _probe_loss,
)
from wildfire_pinn.pinn import (
    LevelSetPINN,
    _perimeter_batch,
    day_index_from_tau,
    directed_residual,
    predict_phi_meters,
    sample_grid,
    sample_volume,
    set_seed,
)
from wildfire_pinn.plots import _increasing_axes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_front_pde_diagnostic"
WEIGHTS = (0.01, 0.1, 0.5, 1.0)
NEAR_REPORT_M = 2000.0
STRATEGIES = (
    {"name": "current", "kind": "current"},
    {"name": "band_500m", "kind": "hard", "meters": 500.0},
    {"name": "band_1km", "kind": "hard", "meters": 1000.0},
    {"name": "band_2km", "kind": "hard", "meters": 2000.0},
    {"name": "soft_500m", "kind": "soft", "meters": 500.0},
    {"name": "soft_1km", "kind": "soft", "meters": 1000.0},
    {"name": "soft_2km", "kind": "soft", "meters": 2000.0},
)


def _run_dir(strategy: str, weight: float) -> Path:
    label = f"{strategy}_w{weight:g}".replace(".", "p")
    return OUT / label


def _target_meters(x_query, y_query, tau, prepared) -> torch.Tensor:
    """Persistent SDF in meters at collocation points. No dependence on the network."""
    case = prepared["case"]
    length = prepared["length"]
    t0, t1, t2 = (float(prepared["tau"][index]) for index in range(3))
    sampled = [
        sample_grid(prepared["sdfs"][index], case.x, case.y, x_query, y_query)
        for index in range(3)
    ]
    tau = tau.detach().reshape(-1)
    alpha_01 = ((tau - t0) / max(t1 - t0, 1e-12)).clamp(0.0, 1.0)
    alpha_12 = ((tau - t1) / max(t2 - t1, 1e-12)).clamp(0.0, 1.0)
    between_01 = (1.0 - alpha_01) * sampled[0] + alpha_01 * sampled[1]
    between_12 = (1.0 - alpha_12) * sampled[1] + alpha_12 * sampled[2]
    phi = torch.where(tau <= t1, between_01, torch.where(tau >= t2, sampled[2], between_12))
    return phi * length


def _uniform(count: int, prepared, generator: torch.Generator, tau_lo: float = 0.0, tau_hi: float = 1.0):
    case = prepared["case"]
    device = prepared["device"]
    x_span = float(case.x.max()) - float(case.x.min())
    y_span = float(case.y.max()) - float(case.y.min())
    x_query = torch.rand(count, generator=generator) * x_span + float(case.x.min())
    y_query = torch.rand(count, generator=generator) * y_span + float(case.y.min())
    span = float(tau_hi) - float(tau_lo)
    tau = torch.rand(count, generator=generator) * span + float(tau_lo)
    return x_query.to(device), y_query.to(device), tau.to(device)


def _collocation(strategy: dict, prepared, generator: torch.Generator, tau_lo: float = 0.0, tau_hi: float = 1.0):
    hp = prepared["hp"]
    kind = strategy["kind"]
    if kind != "hard":
        x_query, y_query, tau = _uniform(hp.n_pde, prepared, generator, tau_lo, tau_hi)
        target_m = _target_meters(x_query, y_query, tau, prepared)
        if kind == "soft":
            sigma = float(strategy["meters"])
            weight = torch.exp(-((target_m.abs() / sigma) ** 2))
        else:
            weight = torch.ones_like(target_m)
        return x_query, y_query, tau, weight

    band = float(strategy["meters"])
    chosen_x = []
    chosen_y = []
    chosen_t = []
    have = 0
    for _ in range(40):
        need = hp.n_pde - have
        draw = max(need * 8, 8192)
        x_query, y_query, tau = _uniform(draw, prepared, generator, tau_lo, tau_hi)
        target_m = _target_meters(x_query, y_query, tau, prepared)
        keep = target_m.abs() < band
        if int(keep.sum()) == 0:
            continue
        chosen_x.append(x_query[keep])
        chosen_y.append(y_query[keep])
        chosen_t.append(tau[keep])
        have += int(keep.sum())
        if have >= hp.n_pde:
            break
    if have < 32:
        raise RuntimeError(f"{strategy['name']} found only {have} collocation points inside the band")
    x_query = torch.cat(chosen_x)[: hp.n_pde]
    y_query = torch.cat(chosen_y)[: hp.n_pde]
    tau = torch.cat(chosen_t)[: hp.n_pde]
    return x_query, y_query, tau, torch.ones(x_query.shape[0], device=prepared["device"])


def _pde_loss(model, strategy: dict, prepared, generator: torch.Generator, tau_lo: float = 0.0, tau_hi: float = 1.0) -> torch.Tensor:
    case = prepared["case"]
    hp = prepared["hp"]
    device = prepared["device"]
    length = prepared["length"]
    duration = prepared["duration"]
    x_pde, y_pde, tau_pde, weight = _collocation(strategy, prepared, generator, tau_lo, tau_hi)
    tau_pde = tau_pde.detach().requires_grad_(True)
    x_norm = ((x_pde - prepared["center_x"]) / length).detach().requires_grad_(True)
    y_norm = ((y_pde - prepared["center_y"]) / length).detach().requires_grad_(True)
    phi = model(tau_pde, x_norm, y_norm)
    phi_t, phi_x, phi_y = torch.autograd.grad(
        phi, (tau_pde, x_norm, y_norm), torch.ones_like(phi), create_graph=True
    )
    n_days = int(prepared["spread"].shape[0])
    days = day_index_from_tau(tau_pde, prepared["t0_offset"], duration, n_days)
    rate = sample_volume(prepared["spread"], days, case.x, case.y, x_pde, y_pde).detach()
    wind_now = prepared["wind"][days]
    residual = directed_residual(
        phi_t, phi_x, phi_y, rate,
        wind_now[:, 0], wind_now[:, 1], hp.c_wdir, duration / length,
    )
    squared = residual ** 2
    if strategy["kind"] == "soft":
        return torch.sum(weight.detach() * squared) / torch.clamp(weight.detach().sum(), min=1e-12)
    return torch.mean(squared)


def _front_residual(model, prepared) -> list[dict]:
    """Coded and physical residual where the persistent target is within 2 km.

    The mask is the observed SDF at that pass, not the predicted field.
    """
    case = prepared["case"]
    device = prepared["device"]
    length = prepared["length"]
    duration = prepared["duration"]
    t_over_l = duration / length
    hp = prepared["hp"]
    model.eval()
    rows = []
    yy, xx = np.meshgrid(case.y, case.x, indexing="ij")
    for local, index in enumerate(PASS_INDEXES):
        target_m = np.asarray(prepared["sdfs"][local], dtype=np.float64) * length
        mask = np.abs(target_m) <= NEAR_REPORT_M
        if int(mask.sum()) > 12000:
            picker = np.flatnonzero(mask)
            rng = np.random.default_rng(1000 + index)
            picker = np.sort(rng.choice(picker, size=12000, replace=False))
            mask = np.zeros(mask.shape, dtype=bool)
            mask.ravel()[picker] = True
        x_query = torch.tensor(xx[mask], dtype=torch.float32, device=device)
        y_query = torch.tensor(yy[mask], dtype=torch.float32, device=device)
        tau = torch.full((x_query.shape[0],), float(prepared["tau"][index]), device=device).requires_grad_(True)
        x_norm = ((x_query - prepared["center_x"]) / length).detach().requires_grad_(True)
        y_norm = ((y_query - prepared["center_y"]) / length).detach().requires_grad_(True)
        phi = model(tau, x_norm, y_norm)
        phi_t, phi_x, phi_y = torch.autograd.grad(
            phi, (tau, x_norm, y_norm), torch.ones_like(phi), create_graph=False
        )
        n_days = int(prepared["spread"].shape[0])
        days = day_index_from_tau(tau, prepared["t0_offset"], duration, n_days)
        rate = sample_volume(prepared["spread"], days, case.x, case.y, x_query, y_query).detach()
        wind_now = prepared["wind"][days]
        residual = directed_residual(
            phi_t, phi_x, phi_y, rate,
            wind_now[:, 0], wind_now[:, 1], hp.c_wdir, t_over_l,
        ).detach()
        coded = residual.abs()
        rows.append({
            "index": index,
            "n": int(coded.numel()),
            "mean_abs_coded": float(coded.mean()),
            "rmse_coded": float(torch.sqrt(torch.mean(residual ** 2))),
            "mean_abs_m_s": float(coded.mean()) / t_over_l,
        })
    model.train()
    return rows


def _train_one(strategy: dict, weight: float, prepared, tau_lo: float = 0.0, tau_hi: float = 1.0) -> dict:
    out_dir = _run_dir(strategy["name"], weight)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        print(f"reusing {out_dir.name}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    hp = prepared["hp"]
    case = prepared["case"]
    device = prepared["device"]
    set_seed(hp.seed)
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    set_seed(hp.seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(hp.seed + 17)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    history = {"obs": [], "pde": []}
    print(f"\n=== {out_dir.name} ===", flush=True)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = [
            _perimeter_batch(
                model, float(prepared["tau"][index]), prepared["pools"][local],
                prepared["sdfs"][local], case, hp.n_ic,
                prepared["center_x"], prepared["center_y"], prepared["length"],
            )
            for local, index in enumerate(PASS_INDEXES)
        ]
        loss_obs = torch.stack(losses).mean()
        if weight == 0:
            loss = loss_obs
            pde_value = None
        else:
            loss_pde = _pde_loss(model, strategy, prepared, generator, tau_lo, tau_hi)
            loss = loss_obs + float(weight) * loss_pde
            pde_value = float(loss_pde.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["obs"].append(float(loss_obs.detach()))
        history["pde"].append(pde_value)
        if epoch == 1 or epoch % 250 == 0:
            pde_text = "n/a" if pde_value is None else f"{pde_value:.3e}"
            print(
                f"  epoch {epoch:4d} | obs {history['obs'][-1]:.3e} | pde {pde_text}",
                flush=True,
            )

    rows, fields = _evaluate(model, prepared)
    front = _front_residual(model, prepared)
    for row, extra in zip(rows, front):
        row["front_mean_abs_m_s"] = extra["mean_abs_m_s"]
        row["front_rmse_coded"] = extra["rmse_coded"]
    torch.save(
        {"state_dict": model.state_dict(), "arch": model.arch, "strategy": strategy, "w_pde": weight},
        out_dir / "levelset_pinn.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    summary = {
        "strategy": strategy["name"],
        "kind": strategy["kind"],
        "meters": strategy.get("meters"),
        "w_pde": weight,
        "passes": rows,
        "front_residual": front,
        "observation_loss": float(np.mean([row["observation_loss"] for row in rows])),
        "pde_loss": history["pde"][-1],
        "front_mean_abs_m_s": float(np.mean([item["mean_abs_m_s"] for item in front])),
        "tau_lo": tau_lo,
        "tau_hi": tau_hi,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bits = []
    for row in rows:
        distance = row["distance_to_target_m"]
        distance_text = "none" if distance is None else f"{distance:.0f} m"
        bits.append(
            f"k{row['index']} contour {row['zero_contour']} min {row['min_phi_m']:.0f} "
            f"area {row['negative_km2']:.1f} dist {distance_text}"
        )
    pde_text = "n/a" if summary["pde_loss"] is None else f"{summary['pde_loss']:.3e}"
    print(
        "  final | " + " | ".join(bits)
        + f" | obs {summary['observation_loss']:.3e} | pde {pde_text} "
        + f"| front {summary['front_mean_abs_m_s']:.3f} m/s",
        flush=True,
    )
    del fields
    return summary


def _by_pass(summary: dict) -> dict:
    return {row["index"]: row for row in summary["passes"]}


def _write_comparison(results: list[dict], prepared, title: str | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    names = [item["name"] for item in STRATEGIES]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), layout="constrained")
    for name in names:
        rows = [item for item in results if item["strategy"] == name]
        rows = sorted(rows, key=lambda item: item["w_pde"])
        weights = [item["w_pde"] for item in rows]
        areas = [_by_pass(item)[2]["negative_km2"] for item in rows]
        distances = [
            _by_pass(item)[2]["distance_to_target_m"]
            if _by_pass(item)[2]["distance_to_target_m"] is not None else np.nan
            for item in rows
        ]
        residual = [item["front_mean_abs_m_s"] for item in rows]
        axes[0].plot(weights, areas, marker="o", label=name)
        axes[1].plot(weights, distances, marker="o", label=name)
        axes[2].plot(weights, residual, marker="o", label=name)
    axes[0].axhline(72.5, color="0.5", linestyle="--", linewidth=0.8, label="pass 2 target")
    baseline = next((item for item in results if item["strategy"] == "observation_only"), None)
    if baseline is not None:
        axes[0].axhline(
            _by_pass(baseline)[2]["negative_km2"],
            color="black", linestyle=":", linewidth=1.0, label="observation only",
        )
    axes[0].set_title("Pass 2 burned area")
    axes[0].set_ylabel("km²")
    axes[1].set_title("Pass 2 distance to target")
    axes[1].set_ylabel("m")
    axes[2].set_title("Mean |residual| within 2 km")
    axes[2].set_ylabel("m/s")
    for ax in axes:
        ax.set_xlabel("PDE weight")
        ax.set_xscale("log")
        ax.set_xticks(list(WEIGHTS))
        ax.set_xticklabels([f"{weight:g}" for weight in WEIGHTS])
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title or "Front-local PDE compared with the full-domain PDE. Same three passes and observation loss.")
    fig.savefig(OUT / "comparison_metrics.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(14.5, 7.2), layout="constrained")
    flat = list(axes.ravel())
    shown = 0
    case = prepared["case"]
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    for strategy in STRATEGIES:
        match = [
            item for item in results
            if item["strategy"] == strategy["name"] and abs(item["w_pde"] - 0.1) < 1e-9
        ]
        if not match:
            continue
        ax = flat[shown]
        shown += 1
        checkpoint = torch.load(
            _run_dir(strategy["name"], 0.1) / "levelset_pinn.pt",
            map_location=prepared["device"],
            weights_only=False,
        )
        model = LevelSetPINN(prepared["hp"].hidden, prepared["hp"].layers).to(prepared["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        phi = predict_phi_meters(model, case, float(prepared["tau"][2]))
        shown_field, xd, yd = _increasing_axes(phi, x, y)
        mesh = ax.pcolormesh(xd, yd, shown_field, cmap="RdBu", vmin=-5000, vmax=5000, shading="auto")
        ax.contour(xd, yd, shown_field, levels=[0], colors="black", linewidths=0.8)
        _draw_rings(ax, prepared["targets"][2], "darkorange", 0.8, None)
        row = _by_pass(match[0])[2]
        contour = "contour" if row["zero_contour"] else "no contour"
        ax.set_title(f"{strategy['name']}\n{row['negative_km2']:.0f} km², {contour}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        del model
    for ax in flat[shown:]:
        ax.axis("off")
    fig.colorbar(mesh, ax=flat[:shown], fraction=0.02, label="predicted φ (m), capped at ±5 km")
    fig.suptitle("Pass 2 at PDE weight 0.1. Orange is the persistent target. Black is φ=0.")
    fig.savefig(OUT / "pass2_weight_0p1.png", dpi=130)
    plt.close(fig)


def _print_table(results: list[dict]) -> None:
    header = (
        "strategy", "w", "k0", "k1", "k2",
        "area0", "area1", "area2",
        "min2", "dist2_m", "obs", "pde", "front_m_s",
    )
    print("\t".join(header), flush=True)
    ordered = sorted(results, key=lambda item: (item["strategy"], item["w_pde"]))
    for item in ordered:
        passes = _by_pass(item)
        distance = passes[2]["distance_to_target_m"]
        distance_text = "" if distance is None else f"{distance:.0f}"
        values = [
            item["strategy"],
            f"{item['w_pde']:g}",
            "Y" if passes[0]["zero_contour"] else "N",
            "Y" if passes[1]["zero_contour"] else "N",
            "Y" if passes[2]["zero_contour"] else "N",
            f"{passes[0]['negative_km2']:.1f}",
            f"{passes[1]['negative_km2']:.1f}",
            f"{passes[2]['negative_km2']:.1f}",
            f"{passes[2]['min_phi_m']:.0f}",
            distance_text,
            f"{item['observation_loss']:.3e}",
            "n/a" if item["pde_loss"] is None else f"{item['pde_loss']:.3e}",
            f"{item['front_mean_abs_m_s']:.3f}",
        ]
        print("\t".join(values), flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepared = _prepare()
    results = []
    for strategy in STRATEGIES:
        for weight in WEIGHTS:
            results.append(_train_one(strategy, weight, prepared))
    _write_comparison(results, prepared)
    _print_table(results)


if __name__ == "__main__":
    main()
