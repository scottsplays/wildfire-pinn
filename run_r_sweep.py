"""Sweep the propagation speed R in the level-set PDE. No change to frozen v1.

The frozen checkpoint and the files under outputs/ are not written. New runs
go to experiments/r_sweep/. The PDE residual is the original directed_residual:
dimensionless Phi_tau + (R_eff * T / L) |grad_XY Phi|, with R in m/s.

Mode A uses a constant pre-wind rate. Mode B multiplies the weather rate by a
scale. Mode C is the weather rate at scale 1, which is the class-default
physics configuration (w_pde=1), not the frozen w_pde=0 checkpoint.

Observation loss sees training passes 0-11 only. Collocation times cover the
whole window, as train_real does, and the rate at those times is weather or a
constant. Future fire perimeters are not inputs.

    python run_r_sweep.py --smoke
    python run_r_sweep.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from shapely import contains_xy

from run_final_candidate_audit import _overlap, _symmetric_distance_m
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.config import DATASETS, ProjectConfig
from wildfire_pinn.geometry import cell_size_m, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    RealConfig,
    _perimeter_batch,
    _sample_box,
    _supervision_pools,
    build_training_sdfs,
    day_index_from_tau,
    directed_residual,
    domain_scales,
    downwind_unit_vectors,
    get_device,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
    resolve_time_split,
    sample_volume,
    save_checkpoint,
    set_seed,
)

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT / "outputs" / "kincade_final_candidates" / "observation" / "levelset_pinn.pt"
FROZEN_SHA256 = "93de4791f6017664174b6dec0bc9434bd2bbe5fffd52dc63d795168f4890e996"
R_VALUES = [0.0, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
R_SCALES = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
SEEDS = [0, 1, 2]
GRAD_EPS = 1e-3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tag(mode: str, r_value, r_scale) -> str:
    if mode == "constant":
        return "constant_r" + f"{float(r_value):.3f}".replace(".", "p")
    if mode == "weather_scale":
        return "weather_x" + f"{float(r_scale):.2f}".replace(".", "p")
    if mode == "baseline":
        return "baseline_weather"
    if mode == "control":
        return "control_wpde0"
    raise ValueError(mode)


def _diagnostic_jobs():
    """Four full trainings. Does not include the rest of the R sweep."""
    return RealConfig().epochs, [0], [
        ("constant", 0.0, None, 1.0),
        ("constant", 0.01, None, 1.0),
        ("constant", 0.1, None, 1.0),
        ("control", 0.0, None, 0.0),
    ]


def _jobs(smoke: bool):
    if smoke:
        specs = [
            ("constant", 0.05, None),
            ("weather_scale", None, 0.5),
            ("baseline", None, 1.0),
        ]
        return 2, [0], specs
    specs = [("constant", value, None) for value in R_VALUES]
    specs += [("weather_scale", None, scale) for scale in R_SCALES]
    specs.append(("baseline", None, 1.0))
    return RealConfig().epochs, list(SEEDS), specs


def _unit_check(t_over_l: float) -> dict:
    """coded residual / (T/L) must equal dphi/dt + R_eff |grad phi| in m/s."""
    phi_t = torch.tensor([0.25], dtype=torch.float32)
    phi_x = torch.tensor([0.8], dtype=torch.float32)
    phi_y = torch.tensor([0.6], dtype=torch.float32)
    rate = torch.tensor([0.05], dtype=torch.float32)
    wind_east = torch.tensor([1.0], dtype=torch.float32)
    wind_north = torch.tensor([0.0], dtype=torch.float32)
    coded = directed_residual(phi_t, phi_x, phi_y, rate, wind_east, wind_north, 0.6, t_over_l)
    magnitude = float(torch.sqrt(phi_x ** 2 + phi_y ** 2))
    alignment = float(phi_x / magnitude)
    r_eff = 0.05 * (1.0 + 0.6 * alignment)
    physical = float(phi_t) / t_over_l + r_eff * magnitude
    got = float(coded) / t_over_l
    if abs(got - physical) > 1e-4:
        raise RuntimeError(f"PDE units do not match: coded/T/L={got} physical={physical}")
    flat = directed_residual(
        torch.zeros(1), torch.zeros(1), torch.zeros(1), rate, wind_east, wind_north, 0.6, t_over_l
    )
    return {
        "t_over_l_s_per_m": t_over_l,
        "example_physical_residual_m_s": physical,
        "example_coded_over_t_over_l_m_s": got,
        "flat_gradient_coded_residual": float(flat),
        "gradient_floor": "directed_residual uses sqrt(phi_x^2+phi_y^2+1e-12), so a zero gradient does not divide by zero",
        "pde_sample_region": "uniform in the model bounding box and in tau from 0 to 1, matching train_real",
        "wind": "R_eff = clip(R * (1 + 0.6 * alignment), min=0). Listed R values are the pre-wind rate.",
    }


def _kincade(root: Path) -> ProjectConfig:
    spec = DATASETS["kincade"]
    return ProjectConfig(
        root=root,
        name=spec.name,
        bbox=spec.bbox,
        date_start=spec.date_start,
        date_end=spec.date_end,
        split_scheme=spec.split_scheme,
        train_fraction=spec.train_fraction,
        val_fraction=spec.val_fraction,
    )


def _rate(mode, r_value, r_scale, spread, wind_days, tau_pde, x_pde, y_pde, case, t0_offset, duration):
    if mode in ("constant", "control"):
        value = 0.0 if r_value is None else float(r_value)
        return torch.full((tau_pde.shape[0],), value, device=tau_pde.device)
    scale = 1.0 if mode == "baseline" else float(r_scale)
    days = day_index_from_tau(tau_pde, t0_offset, duration, int(spread.shape[0]))
    weather = sample_volume(spread, days, case.x, case.y, x_pde, y_pde)
    return (weather * scale).detach()


def _train(case, shared, mode, r_value, r_scale, seed: int, epochs: int, run_dir: Path, w_pde: float = 1.0):
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "levelset_pinn.pt"
    history_path = run_dir / "history.json"
    hp = RealConfig(seed=seed, epochs=epochs, w_pde=w_pde)
    device = shared["device"]
    config = {
        "mode": mode,
        "r_m_s": r_value,
        "r_scale": 1.0 if mode == "baseline" else r_scale,
        "seed": seed,
        "epochs": epochs,
        "w_pde": hp.w_pde,
        "w_ic": hp.w_ic,
        "w_obs": hp.w_obs,
        "w_eik": 0.0,
        "c_wdir": hp.c_wdir,
        "hidden": hp.hidden,
        "layers": hp.layers,
        "lr": hp.lr,
        "n_pde": hp.n_pde,
        "n_ic": hp.n_ic,
        "n_obs": hp.n_obs,
        "observation_target": "persistent_burn",
        "split": "0-11 train, 12-15 validation, 16-19 test",
        "pde": "directed_residual on the full box",
    }
    state_path = run_dir / "trainer_state.pt"
    start_epoch = 1
    history = {"total": [], "pde": [], "ic": [], "obs": []}
    if checkpoint.is_file() and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        saved_epochs = len(history.get("total", []))
        if saved_epochs == epochs:
            model = LevelSetPINN(hp.hidden, hp.layers).to(device)
            load_checkpoint(checkpoint, model, device)
            model.eval()
            print(f"resume finished {run_dir.name}", flush=True)
            return model, history, 0.0
        if saved_epochs > 0 and state_path.is_file():
            set_seed(seed)
            model = LevelSetPINN(hp.hidden, hp.layers).to(device)
            load_checkpoint(checkpoint, model, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
            state = torch.load(state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["rng"])
            np.random.set_state(state["numpy"])
            start_epoch = saved_epochs + 1
            print(f"resume {run_dir.name} at epoch {start_epoch}", flush=True)
        else:
            history = {"total": [], "pde": [], "ic": [], "obs": []}
            set_seed(seed)
            model = LevelSetPINN(hp.hidden, hp.layers).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    else:
        set_seed(seed)
        model = LevelSetPINN(hp.hidden, hp.layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    pools = shared["pools"]
    sdfs = shared["sdfs"]
    center_x = shared["center_x"]
    center_y = shared["center_y"]
    length = shared["length"]
    t_over_l = shared["t_over_l"]
    tau = shared["tau"]
    train = shared["train"]
    spread = shared["spread"]
    wind = shared["wind"]
    model.train()
    started = time.perf_counter()
    log_every = 1 if epochs <= 5 else hp.log_every
    for epoch in range(start_epoch, epochs + 1):
        loss_ic = _perimeter_batch(
            model, 0.0, pools[0], sdfs[0], case, hp.n_ic, center_x, center_y, length
        )
        x_pde, y_pde = _sample_box(hp.n_pde, case.x, case.y, device)
        tau_pde = torch.rand(hp.n_pde, device=device).requires_grad_(True)
        x_norm = ((x_pde - center_x) / length).detach().requires_grad_(True)
        y_norm = ((y_pde - center_y) / length).detach().requires_grad_(True)
        phi = model(tau_pde, x_norm, y_norm)
        phi_t, phi_x, phi_y = torch.autograd.grad(
            phi, (tau_pde, x_norm, y_norm), torch.ones_like(phi), create_graph=True
        )
        rate = _rate(
            mode, r_value, r_scale, spread, wind, tau_pde, x_pde, y_pde,
            case, shared["t0_offset"], shared["duration"],
        )
        wind_now = wind[day_index_from_tau(tau_pde, shared["t0_offset"], shared["duration"], int(spread.shape[0]))]
        residual = directed_residual(
            phi_t, phi_x, phi_y, rate, wind_now[:, 0], wind_now[:, 1], hp.c_wdir, t_over_l
        )
        loss_pde = torch.mean(residual ** 2)
        observed = [
            _perimeter_batch(
                model, float(tau[index]), pools[local], sdfs[local],
                case, hp.n_obs, center_x, center_y, length,
            )
            for local, index in enumerate(train)
            if local > 0
        ]
        loss_obs = torch.stack(observed).mean()
        loss = hp.w_ic * loss_ic + hp.w_pde * loss_pde + hp.w_obs * loss_obs
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch} in {run_dir.name}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["total"].append(float(loss.detach()))
        history["pde"].append(float(loss_pde.detach()))
        history["ic"].append(float(loss_ic.detach()))
        history["obs"].append(float(loss_obs.detach()))
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(
                f"{run_dir.name} seed {seed} epoch {epoch:4d} | total {history['total'][-1]:.3e} | "
                f"ic {history['ic'][-1]:.3e} | pde {history['pde'][-1]:.3e} | obs {history['obs'][-1]:.3e}",
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
    save_checkpoint(checkpoint, model, config)
    history_path.write_text(json.dumps(history), encoding="utf-8")
    model.eval()
    return model, history, elapsed


def _probe_residual(model, shared, mode, r_value, r_scale) -> dict:
    device = shared["device"]
    count = 4096
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    x = shared["case_x"]
    y = shared["case_y"]
    x_cpu = torch.rand(count, generator=generator) * (float(x.max()) - float(x.min())) + float(x.min())
    y_cpu = torch.rand(count, generator=generator) * (float(y.max()) - float(y.min())) + float(y.min())
    tau_cpu = torch.rand(count, generator=generator)
    x_pde = x_cpu.to(device)
    y_pde = y_cpu.to(device)
    tau_pde = tau_cpu.to(device).requires_grad_(True)
    center_x = shared["center_x"]
    center_y = shared["center_y"]
    length = shared["length"]
    x_norm = ((x_pde - center_x) / length).detach().requires_grad_(True)
    y_norm = ((y_pde - center_y) / length).detach().requires_grad_(True)
    phi = model(tau_pde, x_norm, y_norm)
    phi_t, phi_x, phi_y = torch.autograd.grad(phi, (tau_pde, x_norm, y_norm), torch.ones_like(phi))
    rate = _rate(
        mode, r_value, r_scale, shared["spread"], shared["wind"], tau_pde, x_pde, y_pde,
        shared["case"], shared["t0_offset"], shared["duration"],
    )
    days = day_index_from_tau(tau_pde, shared["t0_offset"], shared["duration"], int(shared["spread"].shape[0]))
    wind_now = shared["wind"][days]
    residual = directed_residual(
        phi_t, phi_x, phi_y, rate, wind_now[:, 0], wind_now[:, 1], shared["c_wdir"], shared["t_over_l"]
    )
    magnitude = torch.sqrt(phi_x ** 2 + phi_y ** 2 + 1e-12)
    physical = residual.detach() / shared["t_over_l"]
    return {
        "pde_residual_m_s": float(physical.abs().mean()),
        "near_zero_grad_fraction": float((magnitude.detach() < GRAD_EPS).float().mean()),
    }


def _boundary_lines(geom):
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    lines = []
    for part in parts:
        if getattr(part, "geom_type", None) != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords, dtype=np.float64)
        if len(coords) >= 2:
            lines.append(coords)
    return lines


def _score_passes(model, shared, passes):
    rows = []
    spacing = shared["cell_m"]
    for index in passes:
        phi = predict_phi_meters(model, shared["case"], float(shared["tau"][index]))
        mask = phi < 0
        contours = zero_contours(phi, shared["case_x"], shared["case_y"])
        target = shared["targets"][index]
        iou, dice = _overlap(mask, target)
        cell_km2 = spacing ** 2 / 1e6
        pred_area = float(np.count_nonzero(mask) * cell_km2)
        true_area = float(np.count_nonzero(target) * cell_km2)
        distance = _symmetric_distance_m(contours, shared["regions"][index], spacing)
        last = shared["train"][-1]
        base = shared["targets"][last]
        new = target & ~base
        pred_new = mask & ~base
        new_count = int(np.count_nonzero(new))
        rows.append({
            "pass": index,
            "split": shared["split_name"][index],
            "hours_since_last_train": float(shared["hours_since_train"][index]),
            "iou": iou,
            "dice": dice,
            "pred_area_km2": pred_area,
            "target_area_km2": true_area,
            "area_error_km2": pred_area - true_area,
            "symmetric_distance_m": distance,
            "contour": bool(contours),
            "observed_new_km2": new_count * cell_km2,
            "new_area_recall": float(np.count_nonzero(pred_new & new) / new_count) if new_count else None,
            "predicted_growth_km2": float(np.count_nonzero(pred_new) * cell_km2),
            "observed_front_displacement_m": _symmetric_distance_m(
                _boundary_lines(shared["regions"][last]), shared["regions"][index], spacing
            ),
            "predicted_front_displacement_m": _symmetric_distance_m(
                contours, shared["regions"][last], spacing
            ) if contours else None,
        })
    return rows


def _summarize(rows):
    frame = pd.DataFrame(rows)
    distance = frame["symmetric_distance_m"].dropna()
    return {
        "iou": float(frame["iou"].mean()),
        "dice": float(frame["dice"].mean()),
        "area_mae_km2": float(frame["area_error_km2"].abs().mean()),
        "area_signed_km2": float(frame["area_error_km2"].mean()),
        "sym_dist_m": float(distance.mean()) if len(distance) else None,
        "contour_fraction": float(frame["contour"].mean()),
    }


def _persistence_rows(shared):
    last = shared["train"][-1]
    mask = shared["targets"][last]
    held = list(shared["validation"]) + list(shared["test"])
    rows = []
    spacing = shared["cell_m"]
    for index in held:
        target = shared["targets"][index]
        iou, dice = _overlap(mask, target)
        cell_km2 = spacing ** 2 / 1e6
        pred_area = float(np.count_nonzero(mask) * cell_km2)
        true_area = float(np.count_nonzero(target) * cell_km2)
        distance = _symmetric_distance_m(
            _boundary_lines(shared["regions"][last]),
            shared["regions"][index],
            spacing,
        )
        new = target & ~mask
        new_count = int(np.count_nonzero(new))
        rows.append({
            "pass": index,
            "split": shared["split_name"][index],
            "hours_since_last_train": float(shared["hours_since_train"][index]),
            "iou": iou,
            "dice": dice,
            "pred_area_km2": pred_area,
            "target_area_km2": true_area,
            "area_error_km2": pred_area - true_area,
            "symmetric_distance_m": distance,
            "contour": True,
            "observed_new_km2": new_count * cell_km2,
            "new_area_recall": 0.0 if new_count else None,
            "predicted_growth_km2": 0.0,
            "observed_front_displacement_m": distance,
            "predicted_front_displacement_m": 0.0,
        })
    return rows


def _row(mode, r_value, r_scale, seed, epochs, history, train_seconds, probe, by_split, smoke):
    return {
        "mode": mode,
        "r_m_s": r_value,
        "r_scale": 1.0 if mode == "baseline" else r_scale,
        "seed": seed,
        "epochs": epochs,
        "w_pde": RealConfig().w_pde,
        "smoke": smoke,
        "train_seconds": train_seconds,
        "train_loss": history["total"][-1],
        "train_pde_loss": history["pde"][-1],
        "train_ic_loss": history["ic"][-1],
        "train_obs_loss": history["obs"][-1],
        "val_iou": by_split["validation"]["iou"],
        "val_dice": by_split["validation"]["dice"],
        "val_area_mae_km2": by_split["validation"]["area_mae_km2"],
        "val_area_signed_km2": by_split["validation"]["area_signed_km2"],
        "val_sym_dist_m": by_split["validation"]["sym_dist_m"],
        "val_contour_fraction": by_split["validation"]["contour_fraction"],
        "test_iou": by_split["test"]["iou"],
        "test_dice": by_split["test"]["dice"],
        "test_area_mae_km2": by_split["test"]["area_mae_km2"],
        "test_area_signed_km2": by_split["test"]["area_signed_km2"],
        "test_sym_dist_m": by_split["test"]["sym_dist_m"],
        "test_contour_fraction": by_split["test"]["contour_fraction"],
        "pde_residual_m_s": probe["pde_residual_m_s"],
        "near_zero_grad_fraction": probe["near_zero_grad_fraction"],
        "selected_on_validation": False,
    }


def _select(frame: pd.DataFrame) -> pd.DataFrame:
    models = frame[frame["mode"] != "persistence"].copy()
    grouped = models.groupby(["mode", "r_m_s", "r_scale"], dropna=False)
    ranking = []
    for key, part in grouped:
        ranking.append({
            "mode": key[0],
            "r_m_s": key[1],
            "r_scale": key[2],
            "val_iou": float(part["val_iou"].mean()),
            "val_area_mae_km2": float(part["val_area_mae_km2"].mean()),
            "val_contour_fraction": float(part["val_contour_fraction"].mean()),
            "val_sym_dist_m": float(part["val_sym_dist_m"].mean()),
        })
    rank = pd.DataFrame(ranking)
    if rank.empty:
        return rank
    full_contour = rank[rank["val_contour_fraction"] >= 0.999]
    pool = full_contour if len(full_contour) else rank
    pool = pool.sort_values(
        ["val_iou", "val_area_mae_km2", "val_sym_dist_m"],
        ascending=[False, True, True],
    )
    return pool


def _mark_selected(frame: pd.DataFrame, best: pd.Series) -> pd.DataFrame:
    mode = frame["mode"] == best["mode"]
    if best["mode"] == "constant":
        same = frame["r_m_s"] == best["r_m_s"]
    else:
        same = frame["r_scale"] == best["r_scale"]
    frame = frame.copy()
    frame.loc[mode & same, "selected_on_validation"] = True
    return frame


def _figures(frame: pd.DataFrame, horizon: pd.DataFrame, shared, out_dir: Path, smoke: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    label = "smoke, not the full sweep" if smoke else "full sweep"
    models = frame[frame["mode"] != "persistence"]

    def _mean(mode, xcol):
        part = models[models["mode"] == mode]
        if part.empty:
            return None
        return part.groupby(xcol, dropna=False)[["val_iou", "val_area_mae_km2", "pde_residual_m_s"]].mean().reset_index()

    constant = _mean("constant", "r_m_s")
    scaled = _mean("weather_scale", "r_scale")
    if constant is not None:
        _xy(constant, "r_m_s", "val_iou", out_dir / "val_iou_vs_constant_r.png", "Constant R (m/s)", "Validation IoU", label)
        _xy(constant, "r_m_s", "pde_residual_m_s", out_dir / "pde_residual_vs_constant_r.png", "Constant R (m/s)", "Mean |PDE residual| (m/s)", label)
        _xy(constant, "r_m_s", "val_area_mae_km2", out_dir / "area_error_vs_constant_r.png", "Constant R (m/s)", "Validation area MAE (km²)", label)
    if scaled is not None:
        _xy(scaled, "r_scale", "val_iou", out_dir / "val_iou_vs_weather_scale.png", "Weather R scale", "Validation IoU", label)
        _xy(scaled, "r_scale", "pde_residual_m_s", out_dir / "pde_residual_vs_weather_scale.png", "Weather R scale", "Mean |PDE residual| (m/s)", label)
        _xy(scaled, "r_scale", "val_area_mae_km2", out_dir / "area_error_vs_weather_scale.png", "Weather R scale", "Validation area MAE (km²)", label)
    _comparison(frame, out_dir / "comparison_vs_baseline_persistence.png", label)
    _perimeters(frame, horizon, shared, out_dir / "val_perimeters.png", label)


def _xy(frame, xcol, ycol, path, xlabel, ylabel, label):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(frame[xcol], frame[ycol], marker="o", color="#9b2226")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _comparison(frame, path, label):
    chosen = frame[frame["selected_on_validation"] == True]
    base = frame[frame["mode"] == "baseline"]
    persist = frame[frame["mode"] == "persistence"]
    rows = []
    if len(chosen):
        rows.append(("selected", float(chosen["val_iou"].mean()), float(chosen["test_iou"].mean())))
    if len(base):
        rows.append(("weather baseline", float(base["val_iou"].mean()), float(base["test_iou"].mean())))
    if len(persist):
        rows.append(("persistence", float(persist["val_iou"].mean()), float(persist["test_iou"].mean())))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    if rows:
        x = np.arange(len(rows))
        ax.bar(x - 0.15, [item[1] for item in rows], width=0.3, label="validation IoU", color="#1d3557")
        ax.bar(x + 0.15, [item[2] for item in rows], width=0.3, label="test IoU", color="#9b2226")
        ax.set_xticks(x, [item[0] for item in rows])
    ax.set_ylabel("IoU")
    ax.set_title(label + "\nTest bars are reported, not used for selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _perimeters(frame, horizon, shared, path, label):
    selected = frame[frame["selected_on_validation"] == True]
    if selected.empty:
        return
    best = selected.iloc[0]
    run_name = _tag(best["mode"], best["r_m_s"], best["r_scale"]) + f"_seed{int(best['seed'])}"
    checkpoint = shared["run_root"] / run_name / "levelset_pinn.pt"
    if not checkpoint.is_file():
        return
    device = shared["device"]
    model = LevelSetPINN(RealConfig().hidden, RealConfig().layers).to(device)
    load_checkpoint(checkpoint, model, device)
    model.eval()
    passes = [shared["validation"][0], shared["validation"][-1]]
    fig, axes = plt.subplots(1, len(passes), figsize=(8.4, 4.2))
    if len(passes) == 1:
        axes = [axes]
    last = shared["train"][-1]
    for axis, index in zip(axes, passes):
        phi = predict_phi_meters(model, shared["case"], float(shared["tau"][index]))
        contours = zero_contours(phi, shared["case_x"], shared["case_y"])
        _draw_boundary(axis, shared["regions"][index], "#222222", "observed")
        _draw_boundary(axis, shared["regions"][last], "#1d3557", "persistence")
        for piece in contours:
            axis.plot(piece[:, 0], piece[:, 1], color="#9b2226", lw=1.4)
        axis.set_aspect("equal")
        axis.set_title(f"pass {index}  {shared['hours_since_train'][index]:.0f} h")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle("Validation fronts (" + label + ")\nred predicted, black observed, blue last training mask")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _draw_boundary(axis, geom, color, _label):
    parts = [geom] if geom.geom_type == "Polygon" else list(getattr(geom, "geoms", []))
    for part in parts:
        if part.geom_type != "Polygon":
            continue
        coords = np.asarray(part.exterior.coords)
        axis.plot(coords[:, 0], coords[:, 1], color=color, lw=1.0)


def _report(path: Path, frame: pd.DataFrame, notes: dict, smoke: bool, seconds_per_epoch: float):
    full_runs = len(R_VALUES) * len(SEEDS) + len(R_SCALES) * len(SEEDS) + len(SEEDS)
    hours = seconds_per_epoch * RealConfig().epochs * full_runs / 3600.0 if seconds_per_epoch else None
    selected = frame[frame["selected_on_validation"] == True]
    best = selected.iloc[0] if len(selected) else None
    lines = [
        "# Propagation-speed sweep",
        "",
        "## Status",
        "",
    ]
    if smoke:
        lines.append(
            "A two-epoch smoke test ran for one seed and one setting of each mode. "
            "The 4,000-epoch sweep did not run. The numbers in this file check that "
            "training, scoring, and figure code finish. They are not evidence about "
            "whether a different R improves the forecast."
        )
    else:
        lines.append(
            "The full sweep finished. Validation IoU, then area error and contour "
            "validity, selected the configuration. Test scores below are only for "
            "that configuration. The largest test IoU in the table was not used."
        )
    lines.extend([
        "",
        "## Experimental setup",
        "",
        "Kincade 2019, persistent burned-region targets, chronological passes "
        "0–11 train, 12–15 validation, 16–19 test. Network 4×128 tanh, Adam "
        f"learning rate {RealConfig().lr}, weights "
        f"w_ic={RealConfig().w_ic}, w_pde={RealConfig().w_pde}, w_obs={RealConfig().w_obs}, "
        "w_eik=0. Thirds sampling. Seeds control initialization and the batch stream. "
        "Future perimeters are not model inputs. Weather at later days is the same "
        "field the original residual samples.",
        "",
        "The frozen v1 checkpoint uses w_pde=0, so changing R cannot change that "
        "file. This sweep turns the PDE on at the class-default weight 1 and varies "
        "only the pre-wind rate. Mode C is weather R with scale 1.",
        "",
        "## Original R and units",
        "",
        "Weather R is the stored spread-rate volume in m/s. Wind direction is not "
        "inside that volume. `directed_residual` forms "
        "R_eff = clip(R * (1 + 0.6 * alignment), min=0), with alignment in [-1, 1], "
        "so R_eff is between 0.4 R and 1.6 R. The network outputs dimensionless Phi. "
        "The coded residual is Phi_tau + (R_eff * T/L) |grad_XY Phi|. Dividing by "
        f"T/L ({notes['t_over_l_s_per_m']:.4f} s/m) returns dphi/dt + R_eff |grad phi| "
        "in m/s. A unit check on a known gradient matched to "
        f"{notes['example_coded_over_t_over_l_m_s']:.6f} m/s.",
        "",
        "Collocation points are uniform in the model box and in tau over the whole "
        "fire, including times after the training passes. That matches `train_real`. "
        "The residual is not restricted to a front band. A zero spatial gradient is "
        "guarded by a 1e-12 floor inside the square root; the smoke and full runs "
        "record the fraction of probe points with |grad Phi| < 0.001.",
        "",
        "`train_real` builds the network before `set_seed`, so its seed does not "
        "control Xavier initialization. This sweep seeds before the network is built. "
        "Mode C is the same loss and the same weather R, not a bitwise replay of an "
        "old process.",
        "",
        f"Domain-median weather R is {notes['weather_r_median_m_s']:.4f} m/s.",
        "",
        "## Results",
        "",
        "```",
        frame.to_csv(index=False).strip(),
        "```",
        "",
        "## Best validation configuration",
        "",
    ])
    if best is None:
        lines.append("No configuration was scored.")
    else:
        if best["mode"] == "constant":
            chosen = f"constant R = {float(best['r_m_s']):.3f} m/s"
        elif best["mode"] == "weather_scale":
            chosen = f"weather-R scale {float(best['r_scale']):.2f}"
        else:
            chosen = "weather baseline, scale 1"
        lines.append(
            f"Selected on validation: {chosen}. "
            f"Mean validation IoU {float(selected['val_iou'].mean()):.4f}, "
            f"validation area MAE {float(selected['val_area_mae_km2'].mean()):.2f} km², "
            f"contour fraction {float(selected['val_contour_fraction'].mean()):.2f}."
        )
        lines.extend([
            "",
            "## Held-out test of that configuration",
            "",
            f"Test IoU {float(selected['test_iou'].mean()):.4f}, "
            f"test area MAE {float(selected['test_area_mae_km2'].mean()):.2f} km², "
            f"test contour fraction {float(selected['test_contour_fraction'].mean()):.2f}. "
            "This is the test score of the validation winner. It is not the best test "
            "score in the table.",
        ])
    persist = frame[frame["mode"] == "persistence"]
    lines.extend(["", "## Persistence", ""])
    if len(persist):
        lines.append(
            "Persistence repeats the pass-11 persistent mask with no motion. "
            f"Validation IoU {float(persist['val_iou'].mean()):.4f}, "
            f"test IoU {float(persist['test_iou'].mean()):.4f}."
        )
    lines.extend([
        "",
        "## What this can and cannot say",
        "",
        "A lower PDE residual at a new R shows that the magnitude was more compatible "
        "with the masks. A higher validation IoU, with contours still present and "
        "area error not worse, is evidence that the magnitude also helped the "
        "forecast. If the residual falls and IoU does not, the limitation is not "
        "fixed by rescaling R. That pattern points to the PDE form, the sparse "
        "VIIRS targets, or the network, which this sweep does not separate from "
        "each other.",
        "",
    ])
    if smoke:
        estimate = "unknown" if hours is None else f"about {hours:.1f} hours"
        lines.extend([
            "The smoke test does not answer the research question. "
            "Smoke figures are in `experiments/r_sweep/smoke/figures/`. "
            "The full command writes `experiments/r_sweep/results.csv` and "
            "`experiments/r_sweep/figures/`.",
            "",
            "## Next command",
            "",
            "```powershell",
            "python run_r_sweep.py",
            "```",
            "",
            f"Fifty-one trainings of 4,000 epochs. From the smoke timing, a rough "
            f"CPU estimate is {estimate}. Completed runs are skipped on a restart.",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _prepare(root: Path, out_root: Path):
    if FROZEN.is_file() and _sha256(FROZEN) != FROZEN_SHA256:
        raise RuntimeError("Frozen v1 checkpoint hash changed. Aborting before any training.")
    cfg = _kincade(root)
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _detections = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    hp = RealConfig()
    split = resolve_time_split(
        len(case.perimeters), hp.k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    if split.train != list(range(12)) or split.validation != list(range(12, 16)) or split.test != list(range(16, 20)):
        raise RuntimeError("Kincade split is not 12/4/4.")
    center_x, center_y, length, duration, t0_offset, tau = domain_scales(case)
    device = get_device()
    out_root.mkdir(parents=True, exist_ok=True)
    sdfs = build_training_sdfs(
        case, len(split.train), length, out_root / "sdf_train.npz",
        [regions[index] for index in split.train],
    )
    pools = [
        _supervision_pools(regions[index], sdfs[local], case.x, case.y, device)
        for local, index in enumerate(split.train)
    ]
    yy, xx = np.meshgrid(np.asarray(case.y), np.asarray(case.x), indexing="ij")
    targets = [np.asarray(contains_xy(geom, xx, yy), dtype=bool) for geom in regions]
    times = pd.to_datetime(case.perimeters["dt"])
    last = split.train[-1]
    hours = ((times - times.iloc[last]).dt.total_seconds() / 3600.0).to_numpy()
    names = ["train"] * len(case.perimeters)
    for index in split.validation:
        names[index] = "validation"
    for index in split.test:
        names[index] = "test"
    weather = np.asarray(case.spread, dtype=np.float64)
    shared = {
        "case": case,
        "case_x": np.asarray(case.x, dtype=np.float64),
        "case_y": np.asarray(case.y, dtype=np.float64),
        "regions": regions,
        "targets": targets,
        "pools": pools,
        "sdfs": sdfs,
        "center_x": center_x,
        "center_y": center_y,
        "length": length,
        "duration": duration,
        "t0_offset": t0_offset,
        "t_over_l": duration / length,
        "tau": tau,
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
        "split_name": names,
        "hours_since_train": hours,
        "device": device,
        "spread": torch.tensor(np.asarray(case.spread, dtype=np.float32), device=device),
        "wind": torch.tensor(downwind_unit_vectors(case.wind_from_deg), device=device),
        "c_wdir": hp.c_wdir,
        "cell_m": float(cell_size_m(case.x, case.y)),
        "run_root": out_root / "runs",
    }
    notes = _unit_check(shared["t_over_l"])
    notes["weather_r_median_m_s"] = float(np.nanmedian(weather))
    notes["weather_r_mean_m_s"] = float(np.nanmean(weather))
    return shared, notes


def _mean_key(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _diagnostic_record(record, ic_row, val_rows):
    record = dict(record)
    record["ic_iou"] = ic_row["iou"]
    record["ic_area_error_km2"] = ic_row["area_error_km2"]
    record["ic_contour"] = ic_row["contour"]
    record["val_new_area_recall"] = _mean_key(val_rows, "new_area_recall")
    record["val_predicted_growth_km2"] = _mean_key(val_rows, "predicted_growth_km2")
    record["val_observed_new_km2"] = _mean_key(val_rows, "observed_new_km2")
    record["val_observed_front_displacement_m"] = _mean_key(val_rows, "observed_front_displacement_m")
    record["val_predicted_front_displacement_m"] = _mean_key(val_rows, "predicted_front_displacement_m")
    return record


def _diagnostic_figure(shared, out_path: Path):
    specs = [
        ("Observed", None),
        ("Persistence", "persistence"),
        ("R = 0", shared["run_root"] / "constant_r0p000_seed0" / "levelset_pinn.pt"),
        ("R = 0.01", shared["run_root"] / "constant_r0p010_seed0" / "levelset_pinn.pt"),
        ("R = 0.1", shared["run_root"] / "constant_r0p100_seed0" / "levelset_pinn.pt"),
        ("w_pde = 0", shared["run_root"] / "control_wpde0_seed0" / "levelset_pinn.pt"),
    ]
    device = shared["device"]
    models = []
    for title, source in specs:
        if not isinstance(source, Path):
            models.append((title, source))
            continue
        model = LevelSetPINN(RealConfig().hidden, RealConfig().layers).to(device)
        load_checkpoint(source, model, device)
        model.eval()
        models.append((title, model))
    times = [0, shared["validation"][0], shared["validation"][-1]]
    fig, axes = plt.subplots(len(times), len(models), figsize=(14.5, 7.2))
    last = shared["train"][-1]
    for row, index in enumerate(times):
        for col, (title, model) in enumerate(models):
            axis = axes[row, col]
            _draw_boundary(axis, shared["regions"][index], "#222222", "observed")
            if model == "persistence":
                _draw_boundary(axis, shared["regions"][last], "#1d3557", "persistence")
            elif model is not None:
                phi = predict_phi_meters(model, shared["case"], float(shared["tau"][index]))
                for piece in zero_contours(phi, shared["case_x"], shared["case_y"]):
                    axis.plot(piece[:, 0], piece[:, 1], color="#9b2226", lw=1.2)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=9)
            if col == 0:
                when = "initial" if index == 0 else "future"
                axis.set_ylabel(f"pass {index}\n{when}\n{shared['hours_since_train'][index]:.0f} h")
    fig.suptitle("Black is the observed persistent perimeter. Red is the predicted perimeter. Blue is persistence.")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _diagnostic_report(path: Path, frame: pd.DataFrame):
    models = frame[frame["mode"] != "persistence"].copy()
    persist = frame[frame["mode"] == "persistence"].iloc[0]
    lines = [
        "# Constant-R diagnostic",
        "",
        "Four trainings, seed 0, 4,000 epochs, persistent Kincade targets, passes 0–11 / 12–15 / 16–19. "
        "Architecture and preprocessing match the existing level-set network. "
        "The other 47 sweep runs were not started.",
        "",
        "Constant R uses w_pde = 1. The control uses the same batches with w_pde = 0, so the PDE does not update the weights. "
        "Persistence repeats the pass-11 mask.",
        "",
        "## Results",
        "",
        "```",
        frame.to_csv(index=False).strip(),
        "```",
        "",
        "## Reading",
        "",
    ]
    control = models[models["mode"] == "control"]
    constants = models[models["mode"] == "constant"]
    if len(control) and len(constants):
        best = constants.sort_values("val_iou", ascending=False).iloc[0]
        control_iou = float(control["val_iou"].iloc[0])
        best_iou = float(best["val_iou"])
        persist_iou = float(persist["val_iou"])
        ic_values = models["ic_iou"].astype(float)
        lines.append(
            f"Validation IoU is {best_iou:.3f} at the best constant R "
            f"({float(best['r_m_s']):.3f} m/s), {control_iou:.3f} with w_pde = 0, "
            f"and {persist_iou:.3f} for persistence. "
            f"Initial-condition IoU across the four networks spans "
            f"{float(ic_values.min()):.3f} to {float(ic_values.max()):.3f}."
        )
        lines.append("")
        gap = best_iou - control_iou
        if abs(gap) < 0.03 and best_iou + 0.15 < persist_iou:
            lines.append(
                "Changing R does not meaningfully improve validation IoU over the no-physics control, "
                "and both remain well below persistence. The fire after pass 11 adds little area, so a "
                "motionless mask is already a strong forecast. The networks are not held back only by "
                "the scalar value of R."
            )
        elif gap >= 0.03:
            lines.append(
                "The best constant R improves validation IoU over the no-physics control. "
                "That is evidence that this speed helps the forecast under w_pde = 1, "
                "on this one seed."
            )
        else:
            lines.append(
                "Validation IoU stays near the no-physics control. "
                "A different constant R is not producing a clearly better forecast on this seed."
            )
        lines.append("")
        if float(ic_values.max()) >= 0.6 and best_iou < 0.45:
            lines.append(
                "Initial masks are reconstructed more successfully than the later validation masks. "
                "The drop from the initial condition to the forecast is larger than the gap between R values."
            )
        elif float(ic_values.max()) < 0.45:
            lines.append(
                "Initial-condition IoU is low, so the network is not accurately reconstructing the starting fire shape."
            )
        lines.append("")
        lines.append(
            "New-area recall is the fraction of cells that burn after pass 11 and are predicted burned. "
            "Persistence scores zero on that quantity because it does not move. "
            "Front displacement is the symmetric distance from the pass-11 boundary."
        )
    lines.extend([
        "",
        "Test metrics are stored and were not used to choose among these four runs.",
        "",
        "Side-by-side perimeters: `experiments/r_sweep/diagnostic/figures/perimeters.png`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.diagnostic:
        raise SystemExit("Use either --smoke or --diagnostic.")
    if args.diagnostic:
        out_root = ROOT / "experiments" / "r_sweep" / "diagnostic"
    elif args.smoke:
        out_root = ROOT / "experiments" / "r_sweep" / "smoke"
    else:
        out_root = ROOT / "experiments" / "r_sweep"
    if "outputs" in out_root.parts:
        raise RuntimeError("Refusing to write this sweep under outputs/.")
    shared, notes = _prepare(ROOT, out_root)
    (out_root / "implementation_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    epochs, seeds, specs = _diagnostic_jobs() if args.diagnostic else _jobs(args.smoke)
    print(
        f"device {shared['device']} epochs {epochs} jobs {len(specs)} "
        f"T/L {shared['t_over_l']:.4f} weather median R {notes['weather_r_median_m_s']:.4f} m/s",
        flush=True,
    )
    rows = []
    horizons = []
    epoch_seconds = []
    for spec in specs:
        mode, r_value, r_scale = spec[0], spec[1], spec[2]
        w_pde = float(spec[3]) if len(spec) > 3 else 1.0
        for seed in seeds:
            run_dir = shared["run_root"] / f"{_tag(mode, r_value, r_scale)}_seed{seed}"
            print(f"start {run_dir.name} w_pde {w_pde}", flush=True)
            model, history, train_seconds = _train(
                shared["case"], shared, mode, r_value, r_scale, seed, epochs, run_dir, w_pde
            )
            if train_seconds > 0:
                epoch_seconds.append(train_seconds / epochs)
            if mode == "control":
                probe = _probe_residual(model, shared, "baseline", None, 1.0)
            else:
                probe = _probe_residual(model, shared, mode, r_value, r_scale)
            ic_rows = _score_passes(model, shared, [0])
            held = _score_passes(model, shared, list(shared["validation"]) + list(shared["test"]))
            by_split = {
                "validation": _summarize([row for row in held if row["split"] == "validation"]),
                "test": _summarize([row for row in held if row["split"] == "test"]),
            }
            record = _row(mode, r_value, r_scale, seed, epochs, history, train_seconds, probe, by_split, args.smoke)
            record["w_pde"] = w_pde
            if args.diagnostic:
                record = _diagnostic_record(record, ic_rows[0], [row for row in held if row["split"] == "validation"])
            rows.append(record)
            (run_dir / "metrics.json").write_text(json.dumps({"summary": record, "initial": ic_rows, "passes": held}, indent=2), encoding="utf-8")
            for item in ic_rows + held:
                horizons.append({"mode": mode, "r_m_s": r_value, "r_scale": record["r_scale"], "seed": seed, "w_pde": w_pde, **item})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    persist_passes = _persistence_rows(shared)
    persist_summary = {
        "validation": _summarize([row for row in persist_passes if row["split"] == "validation"]),
        "test": _summarize([row for row in persist_passes if row["split"] == "test"]),
    }
    persist = _row(
        "persistence", None, None, None, 0,
        {"total": [None], "pde": [None], "ic": [None], "obs": [None]},
        0.0,
        {"pde_residual_m_s": None, "near_zero_grad_fraction": None},
        persist_summary,
        args.smoke,
    )
    persist["train_loss"] = None
    persist["train_pde_loss"] = None
    persist["train_ic_loss"] = None
    persist["train_obs_loss"] = None
    if args.diagnostic:
        val_persist = [row for row in persist_passes if row["split"] == "validation"]
        persist = _diagnostic_record(persist, {"iou": None, "area_error_km2": None, "contour": None}, val_persist)
    rows.append(persist)
    for item in persist_passes:
        horizons.append({"mode": "persistence", "r_m_s": None, "r_scale": None, "seed": None, **item})
    frame = pd.DataFrame(rows)
    ranking = _select(frame)
    if len(ranking):
        frame = _mark_selected(frame, ranking.iloc[0])
    result_path = out_root / "results.csv"
    frame.to_csv(result_path, index=False)
    pd.DataFrame(horizons).to_csv(out_root / "horizons.csv", index=False)
    if len(ranking):
        (out_root / "selection.json").write_text(
            json.dumps({"rule": "max validation IoU among configs with contours on every validation pass; then lower area MAE; then lower symmetric distance. Test is not used.", "ranking": ranking.to_dict(orient="records")}, indent=2),
            encoding="utf-8",
        )
    shared["run_root"] = out_root / "runs"
    _figures(frame, pd.DataFrame(horizons), shared, out_root / "figures", args.smoke or args.diagnostic)
    if args.diagnostic:
        _diagnostic_figure(shared, out_root / "figures" / "perimeters.png")
        report_path = out_root / "REPORT.md"
        _diagnostic_report(report_path, frame)
    else:
        seconds = float(np.mean(epoch_seconds)) if epoch_seconds else 0.0
        report_path = ROOT / "experiments" / "r_sweep" / "REPORT.md"
        _report(report_path, frame, notes, args.smoke, seconds)
    if FROZEN.is_file() and _sha256(FROZEN) != FROZEN_SHA256:
        raise RuntimeError("Frozen checkpoint hash changed during the sweep.")
    print(f"wrote {result_path}", flush=True)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
