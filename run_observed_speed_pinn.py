"""Observed-speed level-set experiment. Trains two models and does not sweep weights.

The persistent targets, 12/4/4 split, 4x128 tanh network, seed, normalization,
IC loss, observation loss, and residual sign stay as in train_real. The only
training change is the spread rate inside the PDE.

Observed normal speed replaces R_eff. It is estimated from consecutive
training persistent SDFs only. Validation and test fronts are not used.
For a collocation time after the last training pass, the last training
interval is held. The wind factor is not applied again: the measured normal
speed already includes the realized motion, and directed_residual then sees
a zero alignment, so its clamp and sign are unchanged.

Speed construction, for each training interval i -> i+1:

1. phi_t = (phi_{i+1} - phi_i) / dt, with phi the persistent signed distance
   in meters and dt in seconds. Gradients use the midpoint field.
2. A cell is valid only when |phi_mid| <= 500 m and 0.5 <= |grad phi| <= 1.5.
   Other near-front cells are rejected and later filled.
3. On valid cells, raw speed is -phi_t / |grad phi|. Negative values and
   values above that interval's 95th percentile are clipped.
4. Rejected near-front cells are filled with the median of the clipped
   valid speeds. Outside |phi_mid| > 1000 m the speed is 0.
5. The filled band is smoothed with a mask-normalized Gaussian of sigma
   2 grid cells. The result is piecewise constant in time on each
   training interval.

Training starts only after the near-front mean absolute residual with this
speed is less than half the residual with the weather-derived R_eff.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from shapely import contains_xy

from run_final_candidate_audit import _overlap, _symmetric_distance_m
from run_pde_residual_audit import _day_index, _grad, _term_maps
from run_persistent_target_experiment import _kincade_config
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.cli import setup_logging
from wildfire_pinn.geometry import cell_size_m, signed_distance, zero_contours
from wildfire_pinn.pinn import (
    LevelSetPINN,
    RealConfig,
    _perimeter_batch,
    _sample_box,
    _scalar,
    _supervision_pools,
    build_training_sdfs,
    directed_residual,
    domain_scales,
    downwind_unit_vectors,
    get_device,
    load_checkpoint,
    load_fire_case,
    predict_phi_meters,
    resolve_time_split,
    sample_grid,
    save_checkpoint,
    set_seed,
    train_real,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_observed_speed_experiment"
BASELINE_SOURCE = ROOT / "outputs" / "kincade_final_candidates" / "observation"
NEAR_M = 500.0
SUPPORT_M = 1000.0
GRAD_LO = 0.5
GRAD_HI = 1.5
SMOOTH_SIGMA = 2.0
W_PDE = 0.001


def _levels(values: np.ndarray) -> dict:
    chosen = np.asarray(values, dtype=np.float64)
    chosen = chosen[np.isfinite(chosen)]
    if chosen.size == 0:
        return {"n": 0}
    pct = np.percentile(chosen, [5, 25, 50, 75, 95])
    return {
        "n": int(chosen.size),
        "mean": float(chosen.mean()),
        "min": float(chosen.min()),
        "p05": float(pct[0]),
        "p25": float(pct[1]),
        "median": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
        "max": float(chosen.max()),
    }


def _estimate_speeds(fields, x, y, train, tau, duration):
    """Return (n_intervals, y, x) speeds and the per-interval accounting."""
    intervals = []
    speeds = []
    for local in range(len(train) - 1):
        left = train[local]
        right = train[local + 1]
        dt = (float(tau[right]) - float(tau[left])) * duration
        phi_a = fields[local]
        phi_b = fields[local + 1]
        phi_mid = 0.5 * (phi_a + phi_b)
        phi_t = (phi_b - phi_a) / dt
        _gx, _gy, grad_mag = _grad(phi_mid, x, y)
        near = np.abs(phi_mid) <= NEAR_M
        valid = near & (grad_mag >= GRAD_LO) & (grad_mag <= GRAD_HI) & np.isfinite(phi_t)
        raw = np.zeros(phi_mid.shape, dtype=np.float64)
        raw[valid] = -phi_t[valid] / grad_mag[valid]
        valid_values = raw[valid]
        if valid_values.size == 0:
            raise RuntimeError(f"No valid front cells for passes {left}->{right}.")
        upper = float(np.percentile(valid_values, 95))
        upper = max(upper, 0.0)
        clipped_mask = valid & ((raw < 0.0) | (raw > upper))
        clipped = np.clip(raw, 0.0, upper)
        rejected = near & ~valid
        fill = float(np.median(clipped[valid]))
        band = np.abs(phi_mid) <= SUPPORT_M
        filled = np.zeros(phi_mid.shape, dtype=np.float64)
        filled[valid] = clipped[valid]
        filled[rejected] = fill
        weight = near.astype(np.float64)
        numerator = gaussian_filter(filled * weight, SMOOTH_SIGMA, mode="nearest")
        denominator = gaussian_filter(weight, SMOOTH_SIGMA, mode="nearest")
        smooth = np.zeros(phi_mid.shape, dtype=np.float64)
        usable = denominator > 0.05
        smooth[usable] = numerator[usable] / denominator[usable]
        smooth[~band] = 0.0
        smooth = np.clip(smooth, 0.0, None)
        near_count = int(np.count_nonzero(near))
        rejected_count = int(np.count_nonzero(rejected))
        clipped_count = int(np.count_nonzero(clipped_mask))
        intervals.append({
            "passes": [left, right],
            "elapsed_hours": dt / 3600.0,
            "near_cells": near_count,
            "valid_cells": int(np.count_nonzero(valid)),
            "rejected_gradient_fraction": rejected_count / near_count,
            "clipped_fraction_of_valid": clipped_count / int(np.count_nonzero(valid)),
            "rejected_or_clipped_fraction_of_near": (rejected_count + clipped_count) / near_count,
            "clip_upper_m_s": upper,
            "fill_m_s": fill,
            "raw_valid": _levels(valid_values),
            "final_near": _levels(smooth[near]),
            "mask_near": near,
            "speed": smooth.astype(np.float32),
            "phi_t": phi_t,
            "grad_mag": grad_mag,
            "phi_mid": phi_mid,
        })
        speeds.append(smooth.astype(np.float32))
    return np.stack(speeds), intervals


def _verify(case, packed, scales) -> dict:
    speeds, intervals = packed
    rows = []
    for local, interval in enumerate(intervals):
        left, right = interval["passes"]
        dt = interval["elapsed_hours"] * 3600.0
        tau_mid = 0.5 * (float(scales["tau"][left]) + float(scales["tau"][right]))
        # Rebuild the weather residual from the stored finite difference.
        hp = RealConfig()
        wind = downwind_unit_vectors(case.wind_from_deg)
        day = _day_index(tau_mid, scales["t0_offset"], scales["duration"], int(case.spread.shape[0]))
        terms = _term_maps(
            interval["phi_mid"] - 0.5 * interval["phi_t"] * dt,
            interval["phi_mid"] + 0.5 * interval["phi_t"] * dt,
            dt,
            np.asarray(case.spread[day], dtype=np.float64),
            float(wind[day, 0]),
            float(wind[day, 1]),
            hp.c_wdir,
            scales["x"],
            scales["y"],
        )
        mask = interval["mask_near"]
        observed_residual = interval["phi_t"] + speeds[local] * interval["grad_mag"]
        weather_abs = float(np.mean(np.abs(terms["residual"][mask])))
        observed_abs = float(np.mean(np.abs(observed_residual[mask])))
        rows.append({
            "passes": interval["passes"],
            "elapsed_hours": interval["elapsed_hours"],
            "weather_mean_abs_residual_m_s": weather_abs,
            "observed_mean_abs_residual_m_s": observed_abs,
            "rejected_gradient_fraction": interval["rejected_gradient_fraction"],
            "clipped_fraction_of_valid": interval["clipped_fraction_of_valid"],
            "rejected_or_clipped_fraction_of_near": interval["rejected_or_clipped_fraction_of_near"],
            "final_near": interval["final_near"],
        })
    weather_mean = float(np.mean([row["weather_mean_abs_residual_m_s"] for row in rows]))
    observed_mean = float(np.mean([row["observed_mean_abs_residual_m_s"] for row in rows]))
    near_total = sum(item["near_cells"] for item in intervals)
    rejected = sum(item["rejected_gradient_fraction"] * item["near_cells"] for item in intervals) / near_total
    clipped = sum(
        item["clipped_fraction_of_valid"] * item["valid_cells"] for item in intervals
    ) / sum(item["valid_cells"] for item in intervals)
    combined = sum(
        item["rejected_or_clipped_fraction_of_near"] * item["near_cells"] for item in intervals
    ) / near_total
    final_values = np.concatenate([
        speeds[local][interval["mask_near"]] for local, interval in enumerate(intervals)
    ])
    report = {
        "weather_mean_abs_residual_m_s": weather_mean,
        "observed_mean_abs_residual_m_s": observed_mean,
        "residual_ratio_observed_over_weather": observed_mean / weather_mean,
        "rejected_gradient_fraction": float(rejected),
        "clipped_fraction_of_valid": float(clipped),
        "rejected_or_clipped_fraction_of_near": float(combined),
        "observed_speed_near_front": _levels(final_values),
        "intervals": rows,
        "dramatic_reduction": bool(observed_mean < 0.5 * weather_mean),
    }
    print(
        f"weather R_eff residual {weather_mean:.4f} m/s | "
        f"observed-speed residual {observed_mean:.4f} m/s | "
        f"ratio {report['residual_ratio_observed_over_weather']:.3f}",
        flush=True,
    )
    dist = report["observed_speed_near_front"]
    print(
        "observed speed m/s | "
        f"min {dist['min']:.4f} | p05 {dist['p05']:.4f} | median {dist['median']:.4f} | "
        f"mean {dist['mean']:.4f} | p95 {dist['p95']:.4f} | max {dist['max']:.4f}",
        flush=True,
    )
    print(
        f"near-front gradient rejected {100 * rejected:.1f}% | "
        f"valid cells clipped {100 * clipped:.1f}% | "
        f"rejected or clipped {100 * combined:.1f}%",
        flush=True,
    )
    return report


def _sample_speed(speed, edges, x_coords, y_coords, x_query, y_query, tau_query):
    tau_cpu = tau_query.detach().cpu().numpy()
    index = np.searchsorted(edges, tau_cpu, side="right") - 1
    index = np.clip(index, 0, speed.shape[0] - 1)
    rate = torch.empty(tau_cpu.shape[0], dtype=torch.float32, device=x_query.device)
    for choice in np.unique(index):
        chosen = torch.as_tensor(index == choice, device=x_query.device)
        rate[chosen] = sample_grid(speed[int(choice)], x_coords, y_coords, x_query[chosen], y_query[chosen])
    return rate.detach()


def _train_observed(case, regions, scales, speed, edges) -> None:
    out_dir = OUT / "observed_speed_w0p001"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "levelset_pinn.pt").exists():
        print("reusing observed-speed checkpoint", flush=True)
        return
    hp = RealConfig(w_pde=W_PDE)
    device = scales["device"]
    cfg = _kincade_config()
    time_split = resolve_time_split(
        len(case.perimeters), hp.k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction
    )
    train = time_split.train
    set_seed(hp.seed)
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    sdfs = build_training_sdfs(
        case, len(train), scales["length"], out_dir / "sdf_train.npz",
        [regions[index] for index in train],
    )
    pools = [
        _supervision_pools(regions[index], sdfs[local], case.x, case.y, device)
        for local, index in enumerate(train)
    ]
    set_seed(hp.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
    history = {"total": [], "pde": [], "ic": [], "obs": []}
    center_x = scales["center_x"]
    center_y = scales["center_y"]
    length = scales["length"]
    t_over_l = scales["t_over_l"]
    model.train()
    started = time.perf_counter()
    for epoch in range(1, hp.epochs + 1):
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
        rate = _sample_speed(speed, edges, case.x, case.y, x_pde, y_pde, tau_pde)
        zeros = torch.zeros_like(rate)
        residual = directed_residual(
            phi_t, phi_x, phi_y, rate, zeros, zeros, hp.c_wdir, t_over_l
        )
        loss_pde = torch.mean(residual ** 2)
        observed = [
            _perimeter_batch(
                model, float(scales["tau"][index]), pools[local], sdfs[local],
                case, hp.n_obs, center_x, center_y, length,
            )
            for local, index in enumerate(train)
            if local > 0
        ]
        loss_obs = torch.stack(observed).mean()
        loss = hp.w_ic * loss_ic + hp.w_pde * loss_pde + hp.w_obs * loss_obs
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss became non-finite at epoch {epoch}.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history["total"].append(_scalar(loss))
        history["pde"].append(_scalar(loss_pde))
        history["ic"].append(_scalar(loss_ic))
        history["obs"].append(_scalar(loss_obs))
        if epoch == 1:
            print(f"first observed-speed epoch took {time.perf_counter() - started:.2f} s", flush=True)
        if epoch % hp.log_every == 0 or epoch == hp.epochs:
            print(
                f"epoch {epoch:4d} | total {_scalar(loss):.3e} | ic {_scalar(loss_ic):.3e} | "
                f"pde {_scalar(loss_pde):.3e} | obs {_scalar(loss_obs):.3e}",
                flush=True,
            )
    config = {
        "epochs": hp.epochs,
        "seed": hp.seed,
        "hidden": hp.hidden,
        "layers": hp.layers,
        "w_ic": hp.w_ic,
        "w_obs": hp.w_obs,
        "w_pde": hp.w_pde,
        "c_wdir": hp.c_wdir,
        "observation_target": "persistent_burn",
        "pde_rate": "observed_normal_speed",
        "wind_applied_to_observed_speed": False,
    }
    save_checkpoint(out_dir / "levelset_pinn.pt", model, config)
    (out_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")


def _ensure_baseline(case, regions) -> None:
    dest = OUT / "observation"
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "levelset_pinn.pt").exists():
        print("observation checkpoint already present", flush=True)
        return
    source = BASELINE_SOURCE / "levelset_pinn.pt"
    if source.exists():
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        config = checkpoint.get("config") or {}
        same = (
            config.get("w_pde") == 0.0
            and config.get("observation_target") == "persistent_burn"
            and config.get("epochs") == RealConfig().epochs
            and config.get("seed") == RealConfig().seed
        )
        if same:
            shutil.copyfile(source, dest / "levelset_pinn.pt")
            history = BASELINE_SOURCE / "history.json"
            if history.exists():
                shutil.copyfile(history, dest / "history.json")
            print("reusing the identical observation-only checkpoint", flush=True)
            return
    cfg = _kincade_config()
    set_seed(RealConfig().seed)
    train_real(
        case, dest, RealConfig(w_pde=0.0), retrain=False,
        split_scheme=cfg.split_scheme, train_fraction=cfg.train_fraction,
        val_fraction=cfg.val_fraction, sdf_sampling="thirds", target_geometries=regions,
    )


def _load(name: str) -> LevelSetPINN:
    hp = RealConfig()
    device = get_device()
    model = LevelSetPINN(hp.hidden, hp.layers).to(device)
    load_checkpoint(OUT / name / "levelset_pinn.pt", model, device)
    model.eval()
    return model


def _score_split(name, label, weight, case, regions, masks, scales, split_name: str) -> dict:
    model = _load(name)
    indexes = scales["time_split"].validation if split_name == "validation" else scales["time_split"].test
    cell_km2 = scales["cell_km2"]
    spacing = scales["spacing_m"]
    rows = []
    for index in indexes:
        phi_m = predict_phi_meters(model, case, float(scales["tau"][index]))
        contours = zero_contours(phi_m, case.x, case.y)
        predicted = phi_m < 0
        target = masks[index]
        predicted_km2 = float(predicted.sum()) * cell_km2
        target_km2 = float(target.sum()) * cell_km2
        absolute = abs(predicted_km2 - target_km2)
        iou, dice = _overlap(predicted, target)
        rows.append({
            "index": index,
            "split": split_name,
            "zero_contour": bool(contours),
            "predicted_km2": predicted_km2,
            "persistent_target_km2": target_km2,
            "absolute_area_error_km2": absolute,
            "relative_area_error": absolute / target_km2 if target_km2 else None,
            "symmetric_distance_m": _symmetric_distance_m(contours, regions[index], spacing),
            "iou": iou,
            "dice": dice,
        })
    del model
    survivors = [row["symmetric_distance_m"] for row in rows if row["symmetric_distance_m"] is not None]
    summary = {
        "contour_survival": sum(1 for row in rows if row["zero_contour"]) / len(rows),
        "mean_symmetric_distance_m": float(np.mean(survivors)) if survivors else None,
        "median_symmetric_distance_m": float(np.median(survivors)) if survivors else None,
        "area_mae_km2": float(np.mean([row["absolute_area_error_km2"] for row in rows])),
        "mean_relative_area_error": float(np.mean([row["relative_area_error"] for row in rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "passes": rows,
    }
    print(
        f"{label} {split_name} | contours {100 * summary['contour_survival']:.0f}% | "
        f"IoU {summary['mean_iou']:.3f} | Dice {summary['mean_dice']:.3f} | "
        f"area MAE {summary['area_mae_km2']:.1f} | "
        f"dist {summary['mean_symmetric_distance_m']}",
        flush=True,
    )
    return {"name": name, "label": label, "w_pde": weight, split_name: summary}


def _select(results: list[dict]) -> dict:
    qualified = [item for item in results if item["validation"]["contour_survival"] == 1.0]
    if not qualified:
        raise RuntimeError("Neither model keeps a zero contour on every validation pass.")

    def key(item):
        validation = item["validation"]
        return (
            -validation["mean_iou"],
            -validation["mean_dice"],
            validation["area_mae_km2"],
            validation["mean_symmetric_distance_m"],
        )

    return min(qualified, key=key)


def _write_verification_plot(report) -> None:
    labels = [f"{row['passes'][0]}→{row['passes'][1]}" for row in report["intervals"]]
    positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    ax.bar(positions - width / 2, [row["weather_mean_abs_residual_m_s"] for row in report["intervals"]], width, label="weather R_eff")
    ax.bar(positions + width / 2, [row["observed_mean_abs_residual_m_s"] for row in report["intervals"]], width, label="observed speed")
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("Mean |residual| within 500 m (m/s)")
    ax.set_title("Training-target PDE residual before either model is trained")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "residual_before_training.png", dpi=140)
    plt.close(fig)


def _public(item, split_name: str) -> dict:
    summary = item[split_name]
    return {
        "name": item["name"],
        "label": item["label"],
        "w_pde": item["w_pde"],
        "contour_survival": summary["contour_survival"],
        "mean_symmetric_distance_m": summary["mean_symmetric_distance_m"],
        "median_symmetric_distance_m": summary["median_symmetric_distance_m"],
        "area_mae_km2": summary["area_mae_km2"],
        "mean_relative_area_error": summary["mean_relative_area_error"],
        "mean_iou": summary["mean_iou"],
        "mean_dice": summary["mean_dice"],
        "passes": summary["passes"],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / "train.log")
    cfg = _kincade_config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    center_x, center_y, length, duration, t0_offset, tau = domain_scales(case)
    hp = RealConfig()
    split = resolve_time_split(len(regions), hp.k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction)
    if split.train != list(range(12)):
        raise RuntimeError(f"Expected training passes 0–11, found {split.train}")
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    fields = [signed_distance(regions[index], x, y) for index in split.train]
    for index, distance_m in zip(split.train, fields):
        print(f"pass {index} persistent area {regions[index].area / 1e6:.1f} km2", flush=True)
    speed, intervals = _estimate_speeds(fields, x, y, split.train, tau, duration)
    scales = {
        "center_x": center_x,
        "center_y": center_y,
        "length": length,
        "duration": duration,
        "t0_offset": t0_offset,
        "t_over_l": duration / length,
        "tau": tau,
        "x": x,
        "y": y,
        "device": get_device(),
        "time_split": split,
        "cell_km2": (cell_size_m(x, y) ** 2) / 1e6,
        "spacing_m": cell_size_m(x, y),
    }
    report = _verify(case, (speed, intervals), scales)
    _write_verification_plot(report)
    stored_intervals = []
    for item in intervals:
        stored = {key: value for key, value in item.items() if key not in {"mask_near", "speed", "phi_t", "grad_mag", "phi_mid"}}
        stored_intervals.append(stored)
    (OUT / "speed_verification.json").write_text(json.dumps({
        "method": {
            "near_m": NEAR_M,
            "support_m": SUPPORT_M,
            "grad_window": [GRAD_LO, GRAD_HI],
            "clip": "[0, interval 95th percentile of valid raw speeds]",
            "fill": "median of clipped valid speeds inside the rejected near-front cells",
            "smooth": f"mask-normalized Gaussian, sigma {SMOOTH_SIGMA} cells",
            "time": "piecewise constant on each training interval; last interval held after pass 11",
            "validation_and_test_fronts_used": False,
            "wind_reapplied": False,
        },
        "report": report,
        "intervals": stored_intervals,
    }, indent=2), encoding="utf-8")
    np.save(OUT / "observed_speed.npy", speed)
    if not report["dramatic_reduction"]:
        raise RuntimeError(
            "Observed speed did not cut the near-front residual in half. Training was not started."
        )
    print("residual reduction is large enough to train the two candidates", flush=True)
    _ensure_baseline(case, regions)
    edges = np.asarray([float(tau[index]) for index in split.train], dtype=np.float64)
    _train_observed(case, regions, scales, speed, edges)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    masks = [np.asarray(contains_xy(geometry, xx, yy), dtype=bool) for geometry in regions]
    candidates = (
        ("observation", "Observation only", 0.0),
        ("observed_speed_w0p001", "Observed-speed PDE 0.001", W_PDE),
    )
    validation = [
        _score_split(name, label, weight, case, regions, masks, scales, "validation")
        for name, label, weight in candidates
    ]
    winner = _select(validation)
    print(f"VALIDATION WINNER {winner['label']}", flush=True)
    testing = [
        _score_split(name, label, weight, case, regions, masks, scales, "test")
        for name, label, weight in candidates
    ]
    (OUT / "comparison.json").write_text(json.dumps({
        "question": "Does observed-speed PDE improve validation spatial overlap over observation-only?",
        "selected_from_validation": winner["name"],
        "validation": [_public(item, "validation") for item in validation],
        "test_after_selection": [_public(item, "test") for item in testing],
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
