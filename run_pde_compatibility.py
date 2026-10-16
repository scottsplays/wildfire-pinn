"""Does the training persistent SDF obey the current level-set PDE?

No training and no checkpoint changes. For each consecutive training
overpass the observed fields supply

    phi_t + R_eff |grad phi| = 0

with the same R_eff, wind vector, sign, and T/L factor as directed_residual.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from run_pde_residual_audit import _analytic_circle_check, _day_index, _grad, _term_maps, _zoom_limits
from wildfire_pinn.burned_region import persistent_burned_regions
from wildfire_pinn.config import DATASETS, ProjectConfig
from wildfire_pinn.geometry import signed_distance
from wildfire_pinn.pinn import (
    RealConfig,
    directed_residual,
    domain_scales,
    downwind_unit_vectors,
    load_fire_case,
    resolve_time_split,
)
from wildfire_pinn.plots import _increasing_axes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_pde_compatibility"
BANDS_M = (250.0, 500.0, 1000.0)
PRIMARY_M = 500.0


def _config() -> ProjectConfig:
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


def _finite(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    chosen = np.asarray(values, dtype=np.float64)[mask]
    return chosen[np.isfinite(chosen)]


def _mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    chosen = _finite(values, mask)
    if chosen.size == 0:
        return None
    return float(chosen.mean())


def _median(values: np.ndarray, mask: np.ndarray) -> float | None:
    chosen = _finite(values, mask)
    if chosen.size == 0:
        return None
    return float(np.median(chosen))


def _band_report(terms: dict, mask: np.ndarray) -> dict:
    phi_t = terms["phi_t"]
    product = terms["R_eff_grad"]
    residual = terms["residual"]
    speed = terms["observed_speed"]
    usable = mask & np.isfinite(phi_t) & np.isfinite(product)
    opposed = -product
    comparable = usable & (np.abs(product) > 1e-6)
    compatible = comparable & ((phi_t * opposed) > 0)
    ratio_cells = usable & (terms["R_eff"] > 1e-4) & np.isfinite(speed)
    ratio = speed[ratio_cells] / terms["R_eff"][ratio_cells]
    model_r = _median(terms["R"], mask)
    model_eff = _median(terms["R_eff"], mask)
    observed = _median(speed, mask)
    return {
        "n_cells": int(np.count_nonzero(mask)),
        "phi_t_mean_m_s": _mean(phi_t, mask),
        "phi_t_median_m_s": _median(phi_t, mask),
        "R_grad_mean_m_s": _mean(product, mask),
        "R_grad_median_m_s": _median(product, mask),
        "mean_abs_residual_m_s": _mean(np.abs(residual), mask),
        "rmse_residual_m_s": (
            float(np.sqrt(np.mean(_finite(residual, mask) ** 2))) if np.any(mask) else None
        ),
        "fraction_compatible_signs": (
            float(np.count_nonzero(compatible) / np.count_nonzero(comparable))
            if np.any(comparable) else None
        ),
        "observed_normal_speed_median_m_s": observed,
        "observed_normal_speed_mean_m_s": _mean(speed, mask),
        "model_R_median_m_s": model_r,
        "model_R_eff_median_m_s": model_eff,
        "observed_over_model_R_eff": (
            float(observed / model_eff) if observed is not None and model_eff not in (None, 0.0) else None
        ),
        "cellwise_speed_ratio_median": float(np.median(ratio)) if ratio.size else None,
        "grad_magnitude_median": _median(terms["grad_mag"], mask),
    }


def _coded_check(terms: dict, mask: np.ndarray, wind_east: float, wind_north: float, c_wdir: float, t_over_l: float) -> dict:
    """directed_residual on normalized derivatives must equal (T/L) times the physical residual."""
    flat = np.flatnonzero(mask)
    if flat.size > 4096:
        flat = flat[:: max(1, flat.size // 4096)][:4096]
    phi_t = terms["phi_t"].reshape(-1)[flat]
    gx = terms["gx"].reshape(-1)[flat]
    gy = terms["gy"].reshape(-1)[flat]
    rate = terms["R"].reshape(-1)[flat]
    physical = terms["residual"].reshape(-1)[flat]
    coded = directed_residual(
        torch.tensor(t_over_l * phi_t, dtype=torch.float32),
        torch.tensor(gx, dtype=torch.float32),
        torch.tensor(gy, dtype=torch.float32),
        torch.tensor(rate, dtype=torch.float32),
        torch.full((flat.size,), wind_east),
        torch.full((flat.size,), wind_north),
        c_wdir,
        t_over_l,
    ).numpy()
    gap = coded - (t_over_l * physical)
    return {
        "n": int(flat.size),
        "max_abs_coded_minus_T_over_L_times_physical": float(np.max(np.abs(gap))),
        "mean_physical_m_s": float(np.mean(physical)),
        "mean_coded": float(np.mean(coded)),
        "mean_coded_over_T_over_L_m_s": float(np.mean(coded) / t_over_l),
    }


def _build_fields(case, regions, train, length):
    fields = []
    for index in train:
        distance_m = signed_distance(regions[index], case.x, case.y)
        fields.append(np.asarray(distance_m, dtype=np.float64))
        print(
            f"pass {index} phi [{distance_m.min():.0f}, {distance_m.max():.0f}] m "
            f"area {regions[index].area / 1e6:.1f} km2",
            flush=True,
        )
    return fields


def _intervals(case, fields, train, scales):
    hp = RealConfig()
    wind = downwind_unit_vectors(case.wind_from_deg)
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    rows = []
    maps = []
    for local in range(len(train) - 1):
        left = train[local]
        right = train[local + 1]
        dt = (float(scales["tau"][right]) - float(scales["tau"][left])) * scales["duration"]
        tau_mid = 0.5 * (float(scales["tau"][left]) + float(scales["tau"][right]))
        day = _day_index(tau_mid, scales["t0_offset"], scales["duration"], int(case.spread.shape[0]))
        east, north = (float(wind[day, 0]), float(wind[day, 1]))
        spread = np.asarray(case.spread[day], dtype=np.float64)
        terms = _term_maps(fields[local], fields[local + 1], dt, spread, east, north, hp.c_wdir, x, y)
        grad_x, grad_y, _magnitude = _grad(0.5 * (fields[local] + fields[local + 1]), x, y)
        terms["gx"] = grad_x
        terms["gy"] = grad_y
        magnitude = np.maximum(terms["grad_mag"], 1e-6)
        terms["observed_speed"] = -terms["phi_t"] / magnitude
        bands = {}
        for band in BANDS_M:
            mask = np.abs(0.5 * (fields[local] + fields[local + 1])) <= band
            bands[str(int(band))] = _band_report(terms, mask)
        primary_mask = np.abs(0.5 * (fields[local] + fields[local + 1])) <= PRIMARY_M
        check = _coded_check(terms, primary_mask, east, north, hp.c_wdir, scales["t_over_l"])
        hours = dt / 3600.0
        primary = bands[str(int(PRIMARY_M))]
        print(
            f"pass {left}->{right} {hours:.2f} h | n {primary['n_cells']} | "
            f"phi_t {primary['phi_t_median_m_s']:.4f} | "
            f"R|grad {primary['R_grad_median_m_s']:.4f} | "
            f"speed {primary['observed_normal_speed_median_m_s']:.4f} | "
            f"R_eff {primary['model_R_eff_median_m_s']:.4f} | "
            f"ratio {primary['observed_over_model_R_eff']:.2f}",
            flush=True,
        )
        rows.append({
            "passes": [left, right],
            "start": str(case.perimeters["dt"].iloc[left]),
            "end": str(case.perimeters["dt"].iloc[right]),
            "elapsed_hours": hours,
            "elapsed_seconds": dt,
            "mid_hours": scales["hours"][left] + 0.5 * hours,
            "day_index": day,
            "wind_toward": [east, north],
            "bands_m": bands,
            "nondim_check_500m": check,
        })
        maps.append({
            "passes": [left, right],
            "residual": terms["residual"],
            "phi_t": terms["phi_t"],
            "opposed": -terms["R_eff_grad"],
            "mask": primary_mask,
        })
    return rows, maps


def _write_plots(case, rows, maps) -> None:
    x = np.asarray(case.x, dtype=np.float64)
    y = np.asarray(case.y, dtype=np.float64)
    hours = [row["mid_hours"] for row in rows]
    labels = [f"{row['passes'][0]}→{row['passes'][1]}" for row in rows]
    observed = [row["bands_m"]["500"]["observed_normal_speed_median_m_s"] for row in rows]
    model_r = [row["bands_m"]["500"]["model_R_median_m_s"] for row in rows]
    model_eff = [row["bands_m"]["500"]["model_R_eff_median_m_s"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 4.6), layout="constrained")
    positions = np.arange(len(rows))
    width = 0.26
    ax.bar(positions - width, observed, width, label="observed normal speed")
    ax.bar(positions, model_r, width, label="model R")
    ax.bar(positions + width, model_eff, width, label="model R_eff")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("m/s")
    ax.set_title("Near-front median speed, |φ| ≤ 500 m")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "speed_vs_rate.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.6), layout="constrained")
    for band in BANDS_M:
        series = [row["bands_m"][str(int(band))]["mean_abs_residual_m_s"] for row in rows]
        ax.plot(hours, series, marker="o", label=f"|φ| ≤ {band:.0f} m")
    ax.set_xlabel("Hours since first training pass")
    ax.set_ylabel("Mean |φ_t + R_eff |∇φ|| (m/s)")
    ax.set_title("Physical PDE residual on the observed training fronts")
    ax.legend(fontsize=8)
    fig.savefig(OUT / "residual_vs_time.png", dpi=140)
    plt.close(fig)

    chosen = [maps[0], maps[len(maps) // 2], maps[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), layout="constrained")
    for ax, item in zip(axes, chosen):
        shown = np.where(item["mask"], item["residual"], np.nan)
        x0, x1, y0, y1 = _zoom_limits(x, y, item["mask"])
        image, xd, yd = _increasing_axes(shown, x, y)
        finite = image[np.isfinite(image)]
        limit = float(np.percentile(np.abs(finite), 95)) if finite.size else 0.1
        mesh = ax.pcolormesh(xd, yd, image, cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto")
        fig.colorbar(mesh, ax=ax, fraction=0.046)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"pass {item['passes'][0]}→{item['passes'][1]}")
    fig.suptitle("φ_t + R_eff |∇φ| within 500 m of the observed front (m/s)")
    fig.savefig(OUT / "residual_maps.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2), layout="constrained")
    bins = np.linspace(-0.4, 0.4, 81)
    for ax, item in zip(axes, chosen):
        ax.hist(item["phi_t"][item["mask"]], bins=bins, alpha=0.55, label="φ_t", density=True)
        ax.hist(item["opposed"][item["mask"]], bins=bins, alpha=0.55, label="−R_eff |∇φ|", density=True)
        ax.axvline(0.0, color="black", linewidth=0.6)
        ax.set_title(f"pass {item['passes'][0]}→{item['passes'][1]}")
        ax.set_xlabel("m/s")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.suptitle("Near-front terms, |φ| ≤ 500 m")
    fig.savefig(OUT / "term_distributions.png", dpi=140)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = _config()
    case = load_fire_case(cfg.stack_nc, cfg.rate_nc, cfg.perimeters_path)
    regions, _used = persistent_burned_regions(
        cfg.firms_csv, case.perimeters["dt"], crs=str(case.perimeters.crs)
    )
    center_x, center_y, length, duration, t0_offset, tau = domain_scales(case)
    hp = RealConfig()
    split = resolve_time_split(len(regions), hp.k_train, cfg.split_scheme, cfg.train_fraction, cfg.val_fraction)
    if split.train != list(range(12)):
        raise RuntimeError(f"Expected training passes 0–11, found {split.train}")
    hours = [
        float((stamp - case.perimeters["dt"].iloc[0]).total_seconds()) / 3600.0
        for stamp in case.perimeters["dt"]
    ]
    scales = {
        "tau": tau,
        "duration": duration,
        "t0_offset": t0_offset,
        "t_over_l": duration / length,
        "hours": hours,
    }
    print(
        f"T={duration:.1f} s ({duration / 86400:.3f} d) | L={length:.1f} m | "
        f"T/L={scales['t_over_l']:.6f} s/m | c_wdir={hp.c_wdir}",
        flush=True,
    )
    fields = _build_fields(case, regions, split.train, length)
    rows, maps = _intervals(case, fields, split.train, scales)
    circle = _analytic_circle_check(length, duration)
    payload = {
        "units": {
            "phi": "meters, negative inside the persistent burned region",
            "x_y": "meters",
            "t": "seconds",
            "phi_t": "m/s",
            "grad_phi": "dimensionless for a signed distance",
            "R_and_R_eff": "m/s",
            "physical_residual": "phi_t + R_eff |grad phi|, m/s",
            "T_seconds": duration,
            "L_meters": length,
            "T_over_L_s_per_m": duration / length,
            "coded_residual": "(T/L) * physical residual, dimensionless, as returned by directed_residual",
            "center_x_m": center_x,
            "center_y_m": center_y,
            "c_wdir": hp.c_wdir,
            "wind": "one domain-mean blowing-toward unit vector per day",
            "analytic_circle": circle,
        },
        "primary_band_m": PRIMARY_M,
        "sensitivity_bands_m": list(BANDS_M),
        "train_passes": split.train,
        "intervals": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_plots(case, rows, maps)
    header = "interval\thours\tn500\tphi_t\tRgrad\tabs\trmse\tsigns\tspeed\tR\tReff\tratio"
    print(header, flush=True)
    for row in rows:
        item = row["bands_m"]["500"]
        print(
            "\t".join([
                f"{row['passes'][0]}-{row['passes'][1]}",
                f"{row['elapsed_hours']:.2f}",
                str(item["n_cells"]),
                f"{item['phi_t_median_m_s']:.4f}",
                f"{item['R_grad_median_m_s']:.4f}",
                f"{item['mean_abs_residual_m_s']:.4f}",
                f"{item['rmse_residual_m_s']:.4f}",
                f"{item['fraction_compatible_signs']:.2f}",
                f"{item['observed_normal_speed_median_m_s']:.4f}",
                f"{item['model_R_median_m_s']:.4f}",
                f"{item['model_R_eff_median_m_s']:.4f}",
                f"{item['observed_over_model_R_eff']:.2f}",
            ]),
            flush=True,
        )


if __name__ == "__main__":
    main()
