"""Can the existing PINN fit Kincade passes 0-2, with and without the PDE?

This does not change the training pipeline. It reuses the 4x128 tanh network,
the same normalization, the persistent burned-region signed distance, the
three-region observation loss, and the existing level-set residual.

Observation loss on each pass is the mean of three MSEs (interior, boundary,
exterior), which is what _perimeter_batch already does, so the large exterior
cannot drown the interior. The three passes are then averaged with equal
weight. Pass 0 is not given the usual initial-condition multiplier, because
the question is whether one network can represent all three fields.

The PDE term, when its weight is positive, is the same residual as train_real:
uniform collocation over the full Kincade time span, with the stored spread
rate and wind. tau and T/L stay on that full span, including held-out days.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.cli import setup_logging
from wildfire_pinn.config import DATASETS, ProjectConfig
from wildfire_pinn.geometry import cell_size_m, mean_distance_m, signed_distance, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    RealConfig,
    _draw_pool,
    _perimeter_batch,
    _region_counts,
    _sample_box,
    _supervision_pools,
    day_index_from_tau,
    directed_residual,
    domain_scales,
    downwind_unit_vectors,
    get_device,
    load_fire_case,
    predict_phi_meters,
    sample_grid,
    sample_volume,
    set_seed,
)
from wildfire_pinn.plots import _increasing_axes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_three_pass_diagnostic"
PASS_INDEXES = (0, 1, 2)
EPOCHS = 1000
EVAL_EVERY = 100
PDE_WEIGHTS = (0.01, 0.1, 0.5, 1.0)
ALIGN_M = 2000.0
MIN_AREA_KM2 = 5.0
EXPECTED_TIMES = (
    "2019-10-24 09:05",
    "2019-10-24 10:46",
    "2019-10-24 20:27",
)


def _kincade_config() -> ProjectConfig:
    spec = DATASETS["kincade"]
    return ProjectConfig(
        name=spec.name,
        bbox=spec.bbox,
        date_start=spec.date_start,
        date_end=spec.date_end,
        split_scheme=spec.split_scheme,
        train_fraction=spec.train_fraction,
        val_fraction=spec.val_fraction,
    )


def _polygon_parts(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for piece in geom.geoms:
            parts.extend(_polygon_parts(piece))
        return parts
    return []


def _draw_rings(ax, geom, color, linewidth, label):
    drawn = False
    for part in _polygon_parts(geom):
        xs, ys = part.exterior.xy
        ax.plot(xs, ys, color=color, linewidth=linewidth, label=label if not drawn else None)
        drawn = True
        for ring in part.interiors:
            ax.plot(ring.xy[0], ring.xy[1], color=color, linewidth=linewidth, linestyle=":")
    if not drawn and label:
        ax.plot([], [], color=color, linewidth=linewidth, label=label)


def _front_distance_m(contours, boundary):
    if not contours:
        return None
    main = max(contours, key=len)
    stride = main[::5] if len(main) > 5 else main
    distance = mean_distance_m(stride, boundary)
    if not np.isfinite(distance):
        return None
    return float(distance)


def _capture_probe(pool: dict, count: int) -> dict:
    n_in, n_on, n_out = _region_counts(count)
    inside = _draw_pool(*pool["inside"], n_in)
    boundary = _draw_pool(*pool["boundary"], n_on)
    outside = _draw_pool(*pool["outside"], n_out)
    return {"inside": inside, "boundary": boundary, "outside": outside}


def _probe_loss(model, tau_value, probe, sdf, case, center_x, center_y, length) -> float:
    """Same three-region MSE as _perimeter_batch, on a probe fixed before training."""
    device = next(model.parameters()).device
    losses = []
    groups = (
        (probe["inside"], False),
        (probe["boundary"], True),
        (probe["outside"], False),
    )
    for (x_query, y_query), on_boundary in groups:
        if on_boundary:
            target = torch.zeros(x_query.shape[0], device=device)
        else:
            target = sample_grid(sdf, case.x, case.y, x_query, y_query)
        phi = model(
            torch.full((x_query.shape[0],), float(tau_value), device=device),
            (x_query - center_x) / length,
            (y_query - center_y) / length,
        )
        losses.append(torch.nn.functional.mse_loss(phi, target))
    return float(torch.stack(losses).mean().detach())


def _prepare():
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    for index, stamp in zip(PASS_INDEXES, EXPECTED_TIMES):
        actual = str(case.perimeters["dt"].iloc[index])[:16]
        if actual != stamp:
            raise RuntimeError(f"pass {index} is {actual}, expected {stamp}")
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    center_x, center_y, length, duration, t0_offset, tau = domain_scales(case)
    device = get_device()
    hp = RealConfig()
    sdfs = []
    pools = []
    probes = []
    targets = []
    set_seed(123)
    for index in PASS_INDEXES:
        geometry = regions[index]
        distance_m = signed_distance(geometry, case.x, case.y)
        field = (distance_m / length).astype(np.float32)
        pool = _supervision_pools(geometry, field, case.x, case.y, device)
        sdfs.append(field)
        pools.append(pool)
        probes.append(_capture_probe(pool, hp.n_ic))
        targets.append(geometry)
        print(
            f"pass {index} persistent target {float(geometry.area) / 1e6:.1f} km² | "
            f"inside {pool['n_inside']} cells | boundary probe {hp.n_ic // 3} | "
            f"outside {pool['n_outside']} cells | tau {float(tau[index]):.4f}",
            flush=True,
        )
    spread = torch.tensor(np.asarray(case.spread, dtype=np.float32), device=device)
    wind = torch.tensor(downwind_unit_vectors(case.wind_from_deg), device=device)
    prepared = {
        "case": case,
        "hp": hp,
        "device": device,
        "sdfs": sdfs,
        "pools": pools,
        "probes": probes,
        "targets": targets,
        "tau": tau,
        "center_x": center_x,
        "center_y": center_y,
        "length": length,
        "duration": duration,
        "t0_offset": t0_offset,
        "spread": spread,
        "wind": wind,
        "cell_km2": (cell_size_m(case.x, case.y) ** 2) / 1e6,
    }
    return prepared


def _pde_loss(model, prepared) -> torch.Tensor:
    case = prepared["case"]
    hp = prepared["hp"]
    device = prepared["device"]
    center_x = prepared["center_x"]
    center_y = prepared["center_y"]
    length = prepared["length"]
    duration = prepared["duration"]
    x_pde, y_pde = _sample_box(hp.n_pde, case.x, case.y, device)
    tau_pde = torch.rand(hp.n_pde, device=device).requires_grad_(True)
    x_norm = ((x_pde - center_x) / length).detach().requires_grad_(True)
    y_norm = ((y_pde - center_y) / length).detach().requires_grad_(True)
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
    return torch.mean(residual ** 2)


def _evaluate(model, prepared) -> list[dict]:
    case = prepared["case"]
    tau = prepared["tau"]
    rows = []
    fields = []
    model.eval()
    with torch.no_grad():
        for local, index in enumerate(PASS_INDEXES):
            phi_m = predict_phi_meters(model, case, float(tau[index]))
            contours = zero_contours(phi_m, case.x, case.y)
            negative = phi_m < 0
            obs = _probe_loss(
                model, float(tau[index]), prepared["probes"][local], prepared["sdfs"][local],
                case, prepared["center_x"], prepared["center_y"], prepared["length"],
            )
            rows.append({
                "index": index,
                "time": str(case.perimeters["dt"].iloc[index]),
                "min_phi_m": float(phi_m.min()),
                "max_phi_m": float(phi_m.max()),
                "fraction_negative": float(negative.mean()),
                "zero_contour": bool(contours),
                "n_pieces": len(contours),
                "negative_km2": float(negative.sum()) * prepared["cell_km2"],
                "target_km2": float(prepared["targets"][local].area) / 1e6,
                "distance_to_target_m": _front_distance_m(contours, prepared["targets"][local].boundary),
                "observation_loss": obs,
            })
            fields.append(phi_m)
    return rows, fields


def _pass_ok(row: dict) -> bool:
    distance = row["distance_to_target_m"]
    return bool(
        row["zero_contour"]
        and row["min_phi_m"] < 0.0
        and row["negative_km2"] >= MIN_AREA_KM2
        and distance is not None
        and distance <= ALIGN_M
    )


def _write_panels(path: Path, prepared, rows, fields, title: str) -> None:
    case = prepared["case"]
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), layout="constrained")
    mesh = None
    for ax, row, phi, target in zip(axes, rows, fields, prepared["targets"]):
        shown, xd, yd = _increasing_axes(phi, x, y)
        mesh = ax.pcolormesh(xd, yd, shown, cmap="RdBu", vmin=-5000, vmax=5000, shading="auto")
        ax.contour(xd, yd, shown, levels=[0], colors="black", linewidths=1.2)
        _draw_rings(ax, target, "darkorange", 1.0, "persistent target")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylim(float(y.min()), float(y.max()))
        distance = row["distance_to_target_m"]
        distance_text = "no φ=0" if distance is None else f"{distance / 1000:.2f} km to target"
        contour_text = "contour" if row["zero_contour"] else "no contour"
        ax.set_title(
            f"pass {row['index']}  {row['time'][:16]}\n"
            f"min {row['min_phi_m']:.0f} m | {row['negative_km2']:.1f} km² | {contour_text}\n"
            f"{distance_text}",
            fontsize=9,
        )
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(mesh, ax=axes, label="predicted φ (m), color capped at ±5 km")
    fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_loss_curve(path: Path, history: dict, w_pde: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2), layout="constrained")
    ax.plot(history["obs"], label="observation loss")
    if w_pde > 0:
        ax.plot(history["pde"], label="PDE loss")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Three-pass diagnostic, PDE weight {w_pde:g}")
    ax.legend()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _train(name: str, w_pde: float, prepared) -> dict:
    out_dir = OUT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    hp = prepared["hp"]
    case = prepared["case"]
    device = prepared["device"]
    set_seed(hp.seed)
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    set_seed(hp.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    history = {"obs": [], "pde": [], "total": []}
    evals = []
    print(f"\n=== {name} | w_pde={w_pde:g} | seed={hp.seed} | epochs={EPOCHS} ===", flush=True)

    def record(epoch: int) -> None:
        rows, fields = _evaluate(model, prepared)
        payload = {"epoch": epoch, "passes": rows, "all_ok": all(_pass_ok(row) for row in rows)}
        evals.append(payload)
        _write_panels(
            out_dir / f"epoch_{epoch:04d}.png", prepared, rows, fields,
            f"{name}  epoch {epoch}  PDE weight {w_pde:g}",
        )
        bits = []
        for row in rows:
            distance = row["distance_to_target_m"]
            distance_text = "none" if distance is None else f"{distance:.0f} m"
            bits.append(
                f"k{row['index']} min {row['min_phi_m']:.0f} neg {row['negative_km2']:.1f} km² "
                f"contour {row['zero_contour']} dist {distance_text} obs {row['observation_loss']:.3e}"
            )
        print(f"epoch {epoch:4d} | " + " | ".join(bits), flush=True)
        model.train()

    record(0)
    model.train()
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for local, index in enumerate(PASS_INDEXES):
            losses.append(_perimeter_batch(
                model, float(prepared["tau"][index]), prepared["pools"][local],
                prepared["sdfs"][local], case, hp.n_ic,
                prepared["center_x"], prepared["center_y"], prepared["length"],
            ))
        loss_obs = torch.stack(losses).mean()
        if w_pde > 0:
            loss_pde = _pde_loss(model, prepared)
            loss = loss_obs + w_pde * loss_pde
        else:
            loss_pde = None
            loss = loss_obs
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["obs"].append(float(loss_obs.detach()))
        history["pde"].append(None if loss_pde is None else float(loss_pde.detach()))
        history["total"].append(float(loss.detach()))
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            record(epoch)
    torch.save(
        {"state_dict": model.state_dict(), "arch": model.arch, "w_pde": w_pde, "seed": hp.seed},
        out_dir / "levelset_pinn.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (out_dir / "evals.json").write_text(json.dumps(evals, indent=2), encoding="utf-8")
    _write_loss_curve(out_dir / "loss_curve.png", history, w_pde)
    final = evals[-1]
    return {
        "name": name,
        "w_pde": w_pde,
        "final": final,
        "success": bool(final["all_ok"]),
        "evals": evals,
        "final_observation_loss": float(np.mean([row["observation_loss"] for row in final["passes"]])),
        "final_pde_loss": history["pde"][-1],
    }


def _table_row(result: dict) -> dict:
    by_index = {row["index"]: row for row in result["final"]["passes"]}
    return {
        "pde_weight": result["w_pde"],
        "pass_0_contour": by_index[0]["zero_contour"],
        "pass_1_contour": by_index[1]["zero_contour"],
        "pass_2_contour": by_index[2]["zero_contour"],
        "pass_2_min_phi_m": by_index[2]["min_phi_m"],
        "pass_2_burned_km2": by_index[2]["negative_km2"],
        "pass_2_distance_m": by_index[2]["distance_to_target_m"],
        "observation_loss": result["final_observation_loss"],
        "pde_loss": result["final_pde_loss"],
        "all_three_aligned": result["success"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / "diagnostic.log")
    prepared = _prepare()
    stage_a = _train("stage_a_wpde_0", 0.0, prepared)
    results = [stage_a]
    if stage_a["success"]:
        for weight in PDE_WEIGHTS:
            label = f"stage_b_wpde_{weight:g}".replace(".", "p")
            results.append(_train(label, weight, prepared))
    else:
        print("Stage A did not fit all three passes. Stage B was not run.", flush=True)
    table = [_table_row(result) for result in results]
    summary = {
        "passes": list(EXPECTED_TIMES),
        "epochs": EPOCHS,
        "seed": prepared["hp"].seed,
        "architecture": {"hidden": prepared["hp"].hidden, "layers": prepared["hp"].layers},
        "points_per_pass": prepared["hp"].n_ic,
        "region_split": "equal interior, boundary, and exterior means",
        "alignment_rule": {
            "min_phi_m": "< 0",
            "negative_km2": f">= {MIN_AREA_KM2}",
            "distance_to_target_m": f"<= {ALIGN_M}",
        },
        "stage_a_success": stage_a["success"],
        "table": table,
    }
    (OUT / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
