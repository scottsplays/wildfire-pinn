"""Does the persistent signed distance satisfy the level-set PDE?

No training. The residual is computed from the target fields themselves.

Physical statement, with phi in meters, x and y in meters, t in seconds, R in m/s:

    phi_t + R_eff |grad_xy phi| = 0
    R_eff = max(R * (1 + c_wdir * wind_hat · n), 0)
    n = grad phi / |grad phi|

The network and directed_residual use

    tau = t / T,   X = (x - xc) / L,   Y = (y - yc) / L,   Phi = phi / L

Chain rule, because Phi = phi/L and X = (x-xc)/L:

    dPhi/dtau = (T/L) phi_t
    dPhi/dX   = phi_x
    dPhi/dY   = phi_y
    |grad_XY Phi| = |grad_xy phi|

directed_residual returns the dimensionless quantity

    dPhi/dtau + (R_eff * T/L) |grad_XY Phi|
  = (T/L) * (phi_t + R_eff |grad_xy phi|)

T/L is in s/m. Both phi_t and R_eff |grad phi| are in m/s, since |grad phi| is
dimensionless for a signed distance in meters. The factor T/L converts that
sum into the normalized residual the trainer squares. It does not change where
the residual is zero.

Wind in the trainer is one domain-mean blowing-toward vector per day, not the
local cell direction. This audit uses that same vector, and also records the
local-wind version so the two can be compared.
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
from wildfire_pinn.config import DATASETS, ProjectConfig
from wildfire_pinn.geometry import signed_distance
from wildfire_pinn.pinn import (
    RealConfig,
    directed_residual,
    domain_scales,
    downwind_unit_vectors,
    load_fire_case,
)
from wildfire_pinn.plots import _increasing_axes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "kincade_pde_audit"
PASS_INDEXES = (0, 1, 2)
NEAR_M = 2000.0
TIGHT_M = 500.0
FAR_M = 10000.0


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


def _stats(values: np.ndarray, mask: np.ndarray) -> dict:
    chosen = np.asarray(values)[mask & np.isfinite(values)]
    if chosen.size == 0:
        return {"n": 0}
    pct = np.percentile(chosen, [5, 25, 50, 75, 95])
    return {
        "n": int(chosen.size),
        "mean": float(chosen.mean()),
        "p05": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
    }


def _grad(phi: np.ndarray, x: np.ndarray, y: np.ndarray):
    dphi_dy, dphi_dx = np.gradient(phi, y, x)
    magnitude = np.sqrt(dphi_dx ** 2 + dphi_dy ** 2)
    return dphi_dx, dphi_dy, magnitude


def _day_index(tau: float, t0_offset: float, duration: float, n_days: int) -> int:
    seconds = t0_offset + float(tau) * duration
    return int(np.clip(np.floor(seconds / 86400.0), 0, n_days - 1))


def _local_downwind(wind_from_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radians = np.deg2rad(wind_from_deg.astype(np.float64))
    east = np.sin(radians + np.pi)
    north = np.cos(radians + np.pi)
    return east, north


def _analytic_circle_check(length: float, duration: float) -> dict:
    """Expanding circle whose physical residual is exactly zero, then not zero."""
    speed = 0.05
    radius0 = 500.0
    dt = 3600.0
    radius1 = radius0 + speed * dt
    samples = np.linspace(-2000.0, 2000.0, 9)
    xs, ys = np.meshgrid(samples, samples)
    dist = np.sqrt(xs ** 2 + ys ** 2)
    off_origin = dist > 50.0
    phi0 = dist - radius0
    phi1 = dist - radius1
    phi_t = (phi1 - phi0) / dt
    phi_x = xs / np.maximum(dist, 1e-12)
    phi_y = ys / np.maximum(dist, 1e-12)
    magnitude = np.sqrt(phi_x ** 2 + phi_y ** 2)
    t_over_l = duration / length
    rate = torch.full((int(off_origin.sum()),), speed)
    residual = directed_residual(
        torch.tensor(t_over_l * phi_t[off_origin], dtype=torch.float32),
        torch.tensor(phi_x[off_origin], dtype=torch.float32),
        torch.tensor(phi_y[off_origin], dtype=torch.float32),
        rate,
        torch.zeros_like(rate),
        torch.zeros_like(rate),
        0.6,
        t_over_l,
    )
    too_fast = directed_residual(
        torch.tensor(t_over_l * phi_t[off_origin], dtype=torch.float32),
        torch.tensor(phi_x[off_origin], dtype=torch.float32),
        torch.tensor(phi_y[off_origin], dtype=torch.float32),
        torch.full_like(rate, 0.177),
        torch.zeros_like(rate),
        torch.zeros_like(rate),
        0.6,
        t_over_l,
    )
    physical = phi_t + speed * magnitude
    stalled = phi_t + 0.177 * magnitude
    return {
        "speed_m_s": speed,
        "physical_residual_when_R_matches_speed_m_s": _stats(physical, off_origin),
        "coded_residual_when_R_matches_speed": _stats(residual.numpy(), np.ones(residual.shape, dtype=bool)),
        "coded_over_T_over_L": float(residual.mean()) / t_over_l,
        "physical_residual_when_R_is_0p177_m_s": _stats(stalled, off_origin),
        "coded_residual_when_R_is_0p177": _stats(too_fast.numpy(), np.ones(too_fast.shape, dtype=bool)),
        "T_over_L_s_per_m": t_over_l,
        "phi_t_m_s": float(phi_t[off_origin][0]),
        "grad_magnitude": float(magnitude[off_origin][0]),
    }


def _term_maps(phi_a, phi_b, dt, spread_day, wind_east, wind_north, c_wdir, x, y):
    phi_t = (phi_b - phi_a) / dt
    phi_mid = 0.5 * (phi_a + phi_b)
    gx, gy, mag = _grad(phi_mid, x, y)
    mag_safe = np.maximum(mag, 1e-12)
    alignment = (wind_east * gx + wind_north * gy) / mag_safe
    factor = np.clip(1.0 + c_wdir * alignment, 0.0, None)
    r_eff = spread_day * factor
    product = r_eff * mag
    residual = phi_t + product
    return {
        "phi_t": phi_t,
        "grad_mag": mag,
        "alignment": alignment,
        "R": spread_day,
        "R_eff": r_eff,
        "R_eff_grad": product,
        "residual": residual,
    }


def _region_report(terms: dict, mask: np.ndarray, t_over_l: float) -> dict:
    residual = terms["residual"]
    coded = t_over_l * residual
    phi_t = terms["phi_t"]
    product = terms["R_eff_grad"]
    denom = np.median(product[mask & np.isfinite(product)]) if np.any(mask) else np.nan
    numer = np.median(phi_t[mask & np.isfinite(phi_t)]) if np.any(mask) else np.nan
    ratio = float(numer / denom) if denom and np.isfinite(denom) and abs(denom) > 1e-8 else None
    return {
        "phi_t_m_s": _stats(phi_t, mask),
        "grad_magnitude": _stats(terms["grad_mag"], mask),
        "R_m_s": _stats(terms["R"], mask),
        "R_eff_m_s": _stats(terms["R_eff"], mask),
        "R_eff_grad_m_s": _stats(product, mask),
        "residual_m_s": _stats(residual, mask),
        "coded_residual": _stats(coded, mask),
        "coded_mse": float(np.mean(coded[mask] ** 2)) if np.any(mask) else None,
        "median_phi_t_over_median_R_eff_grad": ratio,
        "fraction_phi_t_negative": float(np.mean(phi_t[mask] < 0)) if np.any(mask) else None,
        "fraction_abs_residual_below_0p02": float(np.mean(np.abs(residual[mask]) < 0.02)) if np.any(mask) else None,
    }


def _zoom_limits(x, y, mask, pad=1500.0):
    yy, xx = np.meshgrid(y, x, indexing="ij")
    if not np.any(mask):
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())
    return (
        float(xx[mask].min()) - pad,
        float(xx[mask].max()) + pad,
        float(yy[mask].min()) - pad,
        float(yy[mask].max()) + pad,
    )


def _plot_interval(path, case_x, case_y, terms, near, title):
    x0, x1, y0, y1 = _zoom_limits(case_x, case_y, near)
    panels = (
        ("phi_t", "φ_t (m/s)", "RdBu_r"),
        ("grad_mag", "|∇φ|", "viridis"),
        ("R_eff", "R_eff (m/s)", "magma"),
        ("R_eff_grad", "R_eff |∇φ| (m/s)", "magma"),
        ("residual", "φ_t + R_eff|∇φ| (m/s)", "RdBu_r"),
    )
    fig, axes = plt.subplots(1, 5, figsize=(16.5, 4.2), layout="constrained")
    for ax, (key, label, cmap) in zip(axes, panels):
        field = np.array(terms[key], dtype=np.float64)
        shown = np.where(near, field, np.nan)
        shown, xd, yd = _increasing_axes(shown, case_x, case_y)
        finite = shown[np.isfinite(shown)]
        if key in ("phi_t", "residual"):
            limit = float(np.percentile(np.abs(finite), 95)) if finite.size else 0.1
            limit = max(limit, 0.02)
            mesh = ax.pcolormesh(xd, yd, shown, cmap=cmap, vmin=-limit, vmax=limit, shading="auto")
        elif key == "grad_mag":
            mesh = ax.pcolormesh(xd, yd, shown, cmap=cmap, vmin=0.0, vmax=2.0, shading="auto")
        else:
            vmax = float(np.percentile(finite, 95)) if finite.size else 0.2
            mesh = ax.pcolormesh(xd, yd, shown, cmap=cmap, vmin=0.0, vmax=max(vmax, 0.02), shading="auto")
        fig.colorbar(mesh, ax=ax, fraction=0.046)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    fig.suptitle(title + "\nCells farther than 2 km from either front are hidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_histograms(path, intervals):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), layout="constrained")
    for ax, item in zip(axes, intervals):
        residual = item["terms"]["residual"]
        near = residual[item["near"] & np.isfinite(residual)]
        far = residual[item["far"] & np.isfinite(residual)]
        bins = np.linspace(-0.4, 0.4, 81)
        if near.size:
            ax.hist(near, bins=bins, density=True, histtype="step", linewidth=1.6, label="within 2 km of a front")
        if far.size:
            ax.hist(far, bins=bins, density=True, histtype="step", linewidth=1.6, label="more than 10 km from both fronts")
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("φ_t + R_eff |∇φ|  (m/s)")
        ax.set_ylabel("density")
        ax.set_title(item["title"])
        ax.legend(fontsize=8)
    fig.suptitle("Physical residual of the persistent signed-distance targets")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
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
    fields = [signed_distance(regions[index], case.x, case.y) for index in PASS_INDEXES]
    wind_mean = downwind_unit_vectors(case.wind_from_deg)
    n_days = int(case.spread.shape[0])
    times = [pd_stamp(case, index) for index in PASS_INDEXES]
    analytic = _analytic_circle_check(length, duration)

    interval_payloads = []
    plotted = []
    for left, right in ((0, 1), (1, 2)):
        t_left = float(tau[left]) * duration
        t_right = float(tau[right]) * duration
        dt = t_right - t_left
        tau_mid = 0.5 * (float(tau[left]) + float(tau[right]))
        day_left = _day_index(float(tau[left]), t0_offset, duration, n_days)
        day_right = _day_index(float(tau[right]), t0_offset, duration, n_days)
        day_mid = _day_index(tau_mid, t0_offset, duration, n_days)
        spread = np.asarray(case.spread[day_mid], dtype=np.float64)
        east_hat, north_hat = wind_mean[day_mid]
        local_east, local_north = _local_downwind(case.wind_from_deg[day_mid])
        terms = _term_maps(
            fields[left], fields[right], dt, spread,
            float(east_hat), float(north_hat), hp.c_wdir, case.x, case.y,
        )
        local_terms = _term_maps(
            fields[left], fields[right], dt, spread,
            local_east, local_north, hp.c_wdir, case.x, case.y,
        )
        near = (np.abs(fields[left]) <= NEAR_M) | (np.abs(fields[right]) <= NEAR_M)
        tight = (np.abs(fields[left]) <= TIGHT_M) | (np.abs(fields[right]) <= TIGHT_M)
        far = (np.abs(fields[left]) >= FAR_M) & (np.abs(fields[right]) >= FAR_M)
        unit_grad = near & (terms["grad_mag"] > 0.5) & (terms["grad_mag"] < 1.5)
        t_over_l = duration / length
        title = f"Passes {left} to {right}: {times[left]} → {times[right]}"
        payload = {
            "passes": [left, right],
            "times": [times[left], times[right]],
            "dt_hours": dt / 3600.0,
            "dt_seconds": dt,
            "day_index_start": day_left,
            "day_index_end": day_right,
            "day_index_midpoint": day_mid,
            "domain_mean_wind_toward_east_north": [float(east_hat), float(north_hat)],
            "area_km2": [float(regions[left].area) / 1e6, float(regions[right].area) / 1e6],
            "near_2km_fraction_of_domain": float(near.mean()),
            "domain_mean_wind": {
                "within_2km": _region_report(terms, near, t_over_l),
                "within_500m": _region_report(terms, tight, t_over_l),
                "within_2km_and_grad_near_1": _region_report(terms, unit_grad, t_over_l),
                "beyond_10km": _region_report(terms, far, t_over_l),
                "full_domain": _region_report(terms, np.ones(near.shape, dtype=bool), t_over_l),
            },
            "local_wind_within_2km": _region_report(local_terms, near, t_over_l),
        }
        interval_payloads.append(payload)
        plotted.append({"title": title, "terms": terms, "near": near, "far": far})
        _plot_interval(OUT / f"interval_{left}_{right}.png", case.x, case.y, terms, near, title)

    _plot_histograms(OUT / "residual_histograms.png", plotted)
    summary = {
        "equation": {
            "physical": "phi_t + R_eff * |grad_xy phi| = 0",
            "R_eff": "max(R * (1 + c_wdir * wind_hat · n), 0)",
            "normalization": "tau=t/T, X=(x-xc)/L, Y=(y-yc)/L, Phi=phi/L",
            "chain_rule": "dPhi/dtau=(T/L)*phi_t, dPhi/dX=phi_x, dPhi/dY=phi_y",
            "coded": "dPhi/dtau + (R_eff * T/L) * |grad_XY Phi|",
            "coded_equals": "(T/L) * (phi_t + R_eff * |grad_xy phi|)",
            "units": {
                "phi": "m",
                "x_y": "m",
                "t": "s",
                "phi_t": "m/s",
                "grad_xy_phi": "dimensionless",
                "R": "m/s",
                "R_eff_grad": "m/s",
                "physical_residual": "m/s",
                "T_over_L": "s/m",
                "coded_residual": "dimensionless",
            },
            "T_s": duration,
            "L_m": length,
            "T_over_L_s_per_m": duration / length,
            "c_wdir": hp.c_wdir,
            "wind_used_by_trainer": "one domain-mean blowing-toward unit vector per day",
        },
        "analytic_expanding_circle": analytic,
        "intervals": interval_payloads,
    }
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def pd_stamp(case, index: int) -> str:
    import pandas as pd

    return pd.Timestamp(case.perimeters["dt"].iloc[index]).strftime("%Y-%m-%d %H:%M")


if __name__ == "__main__":
    main()
