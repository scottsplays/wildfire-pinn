"""Inverse PINN: learn WHERE the fire burns faster or slower than the physics, from two satellite
perimeters and the eikonal spread equation; then the solver forecasts with that corrected rate.

    python -m pinn_v2.pinn_inverse --checks              # checks A + B (~5-15 min CPU)
    python -m pinn_v2.pinn_inverse --realfire --quick    # 3 validation fires, 2 starts: checks it runs
    python -m pinn_v2.pinn_inverse --realfire            # 26 validation fires, 3 starts each (~1 h CPU)

The physics (arrival-time / eikonal form of the level-set spread equation). T(x, y) is the time the
front reaches (x, y); n = grad T / |grad T| is the front's outward normal; w is the wind direction:

        R(x, y) * m(x, y) * g(n . w) * |grad T| = 1

  R  spread rate from LANDFIRE fuel, slope, wind and moisture (pinn_service physics, round-2 settings),
     homogenised: the travel-time (harmonic) average over ~1.5 cells, the speed a front actually
     makes across patchy fuel
  g  front shape: elliptical, g = (1 - e) / (1 - e cos) with e from wind speed (Anderson 1983),
     or the service's (1 + 0.6 cos)+ in check A
  m  THE UNKNOWN: actual / predicted spread rate. m = 0.3 means the fire burned at 30 % of the
     physics' rate there. This is what the PINN learns, as a smooth map.

Two networks: T_theta(x, y) and m_psi(x, y) = exp(s_psi). The loss has three parts:
  physics   the eikonal residual above, on burnable cells around the fire
  data      the two observed perimeters: T <= 0 inside the first, 0 < T <= dt on the area that
            burned in between, T > dt outside the second (hinge losses: VIIRS pixels are coarse)
  prior     s stays near log(alpha), the single best-fit factor, where the data say nothing. m can
            only be identified where the fire grew between the two perimeters (that is where T is
            known), so the learned part fades to alpha over ~1.5 km outside that zone
Then the level-set solver forecasts forward with rate R * m (the PINN does not forecast itself:
Gate 1 showed a solver does that better).

Checks, fixed before running (plan doc, "Experiment 3"):
  A  exact physics: a fire growing in wind has an exact eikonal solution (circle drifting downwind).
     A1 forward (m known): arrival times within 2 % of exact.
     A2 inverse: from two exact perimeters, recover m = 1.5 within 10 % (prior deliberately at 1.0).
  B  recovers a known map: LANDFIRE fuel, a hidden smooth multiplier map; observe 0 h and 12 h, forecast
     to 24 h. Pass: PINN map error at least 20 % below the single-factor fit, and a better 24 h forecast.
  C  real fires (--realfire): the PINN map replaces round 2's single alpha ("diurnal" step), same
     starts, scored against VIIRS. Pass: 24 h change-IoU gain with a 95 % CI above zero; reported
     honestly either way.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from pinn_service import physics
from pinn_service.forecaster import signed_distance
from pinn_v2 import realfire_round2 as r2

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "pinn_inverse"


# --------------------------------------------------------------------------- the inverse PINN

def fit_inverse(rate_ms, res_m, wind_from, ws10, shape, mask_a, mask_b, dt_h, prior_alpha,
                steps=2000, learn_m=True, m_fixed=None, seed=0, verbose=False, margin_km=3.0):
    """Fit T(x, y) and m(x, y) on one grid. rate_ms: physics rate (m/s); mask_a / mask_b: burned at
    the start / end of the interval (mask_b may be None for a forward solve). Returns
    (T_hours, m, info) on the full grid."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 2)))
    f32 = dict(dtype=torch.float32)
    ny, nx = mask_a.shape
    res_km = res_m / 1000.0
    rate = rate_ms * 3.6                                            # km/h
    east, north = (float(v) for v in physics.wind_toward_from_met(wind_from))
    e = r2.ellipse_e(ws10) if shape == "ellipse" else 0.0

    # domain: around the fire, outside the starting fire
    sd_a = signed_distance(mask_a, res_m) / 1000.0                  # km, negative inside
    reach = 0.0
    if mask_b is not None and (mask_b & ~mask_a).any():
        reach = float(sd_a[mask_b & ~mask_a].max())
    dmax = reach + margin_km
    rows, cols = np.mgrid[0:ny, 0:nx]
    xk, yk = cols * res_km, rows * res_km
    cx, cy = float(xk[mask_a].mean()), float(yk[mask_a].mean())
    L = max(float(np.abs(xk - cx)[sd_a <= dmax].max()), float(np.abs(yk - cy)[sd_a <= dmax].max()), 1.0)

    burnable = rate > 0
    # homogenised rate: the travel-time (harmonic) average over ~1.5 cells. A front crossing patchy
    # fuel moves at the harmonic mean of the cell speeds; a smooth network cannot follow 10x jumps
    # between neighbouring 90 m cells, and the raw per-cell residual is dominated by the fast cells.
    if burnable.any():
        floor = 0.02 * float(np.median(rate[burnable]))
        slowness = ndimage.gaussian_filter(1.0 / np.maximum(rate, floor), 1.5)
        rate = np.where(burnable, 1.0 / slowness, 0.0)
    phys_cells = np.argwhere((sd_a > -res_km) & (sd_a <= dmax) & burnable)
    inside = np.argwhere(mask_a & (sd_a > -2.0))                    # near the first perimeter, inside
    ring = np.argwhere(~mask_a & (sd_a <= 2 * res_km))              # just outside it: T >= 0
    if mask_b is not None:
        new = np.argwhere(mask_b & ~mask_a)
        outside = np.argwhere(~mask_b & (sd_a <= dmax) & burnable)
    else:
        new = outside = np.empty((0, 2), dtype=int)
    if len(phys_cells) == 0:
        raise ValueError("no burnable cells around the fire")

    v_bar = float(np.median(rate[burnable])) * prior_alpha * (1.0 - e if shape == "ellipse" else 1.0) + 1e-6
    log_prior = math.log(prior_alpha)

    def tens(a):
        return torch.as_tensor(a, **f32)

    # wind-aware first guess: T0 = distance / local speed in the direction the front faces
    gy0, gx0 = np.gradient(sd_a, res_km)
    cos0 = (east * gx0 + north * gy0) / (np.hypot(gx0, gy0) + 1e-9)
    g0 = (1 - e) / (1 - e * cos0) if shape == "ellipse" else np.clip(1 + physics.C_WDIR * cos0, 0.05, None)
    v_loc = rate * prior_alpha * g0
    v_med = float(np.median(v_loc[burnable])) if burnable.any() else 1e-3
    v_loc = np.maximum(ndimage.gaussian_filter(v_loc, 2.0), 0.2 * v_med)
    T0 = np.where(sd_a > 0, sd_a / v_loc, sd_a / v_med)
    T0y, T0x = np.gradient(T0, res_km)
    # where m can be learned: only where the fire grew between the two perimeters (arrival times are
    # known there); fades to the prior over ~1.5 km. Beyond the second perimeter any m would fit.
    if mask_b is not None and (mask_b & ~mask_a).any():
        d_new = ndimage.distance_transform_edt(~(mask_b & ~mask_a)) * res_km
        w_m = np.exp(-(d_new / 1.5) ** 2)
    else:
        w_m = np.ones_like(sd_a)
    grids = {"rate": tens(rate), "T0": tens(T0), "T0x": tens(T0x), "T0y": tens(T0y), "w_m": tens(w_m)}

    def lookup(name, ij):
        ii, jj = torch.as_tensor(ij[:, 0], dtype=torch.long), torch.as_tensor(ij[:, 1], dtype=torch.long)
        return grids[name][ii, jj]

    def feats(bank, x, y):
        p = 2 * math.pi * torch.stack([x, y], 1) @ bank.T
        return torch.cat([x[:, None], y[:, None], torch.sin(p), torch.cos(p)], 1)

    def mlp(n_in, width=64, depth=3):
        layers, w = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.Tanh()]
            w = width
        layers.append(nn.Linear(w, 1))
        net = nn.Sequential(*layers)
        nn.init.normal_(net[-1].weight, std=1e-3)
        nn.init.zeros_(net[-1].bias)
        return net

    bank_t = torch.cat([torch.randn(16, 2) * s for s in (0.5, 2.0, max(4.0, L / (4 * res_km) / 4))])
    bank_m = torch.randn(16, 2) * max(0.5, L / 3.0)                 # smooth: wavelengths >~ 3 km
    net_t, net_s = mlp(2 + 2 * len(bank_t)), mlp(2 + 2 * len(bank_m), width=32, depth=2)
    params = list(net_t.parameters()) + (list(net_s.parameters()) if learn_m else [])
    opt = torch.optim.Adam(params, lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-4)
    t_scale = dt_h if mask_b is not None else max(1.0, dmax / v_bar)

    def T_and_s(ij, jitter=True, need_grad=False):
        x = tens(xk[ij[:, 0], ij[:, 1]] - cx)
        y = tens(yk[ij[:, 0], ij[:, 1]] - cy)
        if jitter:
            x = x + tens(rng.uniform(-0.5, 0.5, len(x)) * res_km)
            y = y + tens(rng.uniform(-0.5, 0.5, len(y)) * res_km)
        X, Y = x / L, y / L
        if need_grad:
            X.requires_grad_(True)
            Y.requires_grad_(True)
        corr = t_scale * net_t(feats(bank_t, X, Y)).squeeze(1)      # learned correction (hours)
        T = lookup("T0", ij) + corr                                   # first guess (distance / speed) + correction
        s = (log_prior + lookup("w_m", ij) * net_s(feats(bank_m, X, Y)).squeeze(1)) if learn_m else None
        return T, s, X, Y, corr

    def draw(cells, n):
        return cells[rng.integers(0, len(cells), n)] if len(cells) else cells

    history = []
    for step in range(1, steps + 1):
        ij = draw(phys_cells, 4096)
        T, s, X, Y, corr = T_and_s(ij, need_grad=True)
        cx_, cy_ = torch.autograd.grad(corr.sum(), (X, Y), create_graph=True)
        gx = lookup("T0x", ij) + cx_ / L                              # dT/dx in h/km: first guess + correction
        gy = lookup("T0y", ij) + cy_ / L
        mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
        cosang = (east * gx + north * gy) / mag
        m = torch.exp(s) if learn_m else tens(np.full(len(ij), m_fixed if m_fixed is not None else 1.0))
        R = lookup("rate", ij)
        if shape == "ellipse":
            speed = R * m * (1 - e) / (1 - e * cosang)
        else:
            speed = R * m * torch.clamp(1 + physics.C_WDIR * cosang, min=0.05)
        res_phys = speed * mag - 1.0
        loss_phys = torch.mean(torch.clamp(res_phys, -3, 3) ** 2)

        terms = []
        for cells, kind in ((inside, "le0"), (ring, "ge0"), (new, "between"), (outside, "gedt")):
            if not len(cells):
                continue
            Tc = T_and_s(draw(cells, 1024))[0]
            u = Tc / t_scale
            if kind == "le0":
                terms.append(torch.mean(torch.relu(u) ** 2))
            elif kind == "ge0":
                terms.append(torch.mean(torch.relu(-u) ** 2))
            elif kind == "between":
                terms.append(torch.mean(torch.relu(-u) ** 2 + torch.relu(u - 1.0) ** 2))
            else:
                terms.append(torch.mean(torch.relu(1.0 - u) ** 2))
        loss_data = sum(terms) if terms else torch.zeros(())
        loss_prior = torch.mean((s - log_prior) ** 2) if learn_m else torch.zeros(())
        loss = loss_phys + 10.0 * loss_data + 0.01 * loss_prior
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 5) == 0 or step == steps:
            history.append((step, float(loss_phys.detach()), float(loss_data.detach())))
            if verbose:
                print(f"    step {step:5d}: physics {float(loss_phys.detach()):.3e}, data {float(loss_data.detach()):.3e}", flush=True)

    # evaluate on the full grid
    all_ij = np.argwhere(np.ones((ny, nx), dtype=bool))
    T_out, m_out = np.empty(len(all_ij)), np.empty(len(all_ij))
    with torch.no_grad():
        for k in range(0, len(all_ij), 65536):
            ij = all_ij[k:k + 65536]
            T, s = T_and_s(ij, jitter=False)[:2]
            T_out[k:k + 65536] = T.numpy()
            m_out[k:k + 65536] = np.exp(s.numpy()) if learn_m else (m_fixed if m_fixed is not None else 1.0)
    info = {"steps": steps, "history": history, "ellipse_e": round(e, 3), "domain_km": round(dmax, 2)}
    return T_out.reshape(ny, nx), np.clip(m_out.reshape(ny, nx), 0.003, 30.0), info


# --------------------------------------------------------------------------- check A: exact physics

def exact_wind_circle(xk, yk, r0, R_kmh, c, wind=(1.0, 0.0)):
    """Arrival time (h) for a circle of radius r0 growing at R with front speed R(1 + c n.w):
    the exact solution is a circle of radius r0 + R t whose centre drifts c R t downwind."""
    xw = xk * wind[0] + yk * wind[1]
    rr = xk ** 2 + yk ** 2
    A, B, C = c * c - 1.0, -2.0 * (c * xw + r0), rr - r0 * r0
    s = (-B - np.sqrt(np.maximum(B * B - 4 * A * C, 0.0))) / (2 * A)
    return np.where(rr <= r0 * r0, 0.0, s / R_kmh)


def check_a(steps):
    print("\n== Check A: exact physics (circle in wind, exact eikonal solution)")
    res_m, half = 100.0, 6000.0
    n = int(2 * half / res_m) + 1
    xs = np.arange(n) * res_m / 1000.0 - half / 1000.0
    X, Y = np.meshgrid(xs, xs)
    r0, R_ms, c = 0.5, 0.05, physics.C_WDIR
    R_kmh = R_ms * 3.6
    rate = np.full((n, n), R_ms)
    wind_from = 270.0                                              # from the west -> blows east
    T_exact = exact_wind_circle(X, Y, r0, R_kmh, c)
    mask_a = np.hypot(X, Y) <= r0

    t0 = time.perf_counter()
    T, _, _ = fit_inverse(rate, res_m, wind_from, 5.0, "service", mask_a, None, 6.0, 1.0,
                          steps=steps, learn_m=False, m_fixed=1.0)
    zone = (T_exact > 0.3) & (T_exact < 6.0)
    rel = np.abs(T[zone] - T_exact[zone]) / T_exact[zone]
    a1 = float(np.median(rel))
    print(f"   A1 forward: median |T - exact| / exact = {a1:.2%} over 0.3-6 h ({time.perf_counter() - t0:.0f} s)  "
          f"-> {'PASS' if a1 <= 0.02 else 'FAIL'} (<= 2 %)")

    m_true, dt = 1.5, 6.0
    mask_b = T_exact / m_true <= dt
    t0 = time.perf_counter()
    _, m_map, _ = fit_inverse(rate, res_m, wind_from, 5.0, "service", mask_a, mask_b, dt, 1.0, steps=steps)
    zone = mask_b & ~mask_a
    m_med = float(np.median(m_map[zone]))
    a2 = abs(m_med / m_true - 1)
    print(f"   A2 inverse: recovered m = {m_med:.3f} (true {m_true}; prior 1.0), error {a2:.1%} ({time.perf_counter() - t0:.0f} s)  "
          f"-> {'PASS' if a2 <= 0.10 else 'FAIL'} (<= 10 %)")
    return {"A1_median_rel_error": round(a1, 4), "A1_pass": a1 <= 0.02,
            "A2_m_recovered": round(m_med, 3), "A2_pass": a2 <= 0.10}


# --------------------------------------------------------------------------- check B: known map

def check_b(steps, seeds=(0, 1)):
    from pinn_v2 import calib_check as cc

    print("\n== Check B: recover a hidden multiplier map on real fuel (observe 0 h + 12 h, forecast 24 h)")
    try:
        xs, res, rate8 = cc.fuel_terrain()
        terrain = "LANDFIRE fuel (Pittsburg hills)"
    except SystemExit:
        xs, res, rate8 = cc.synthetic_terrain()
        terrain = "synthetic fuel"
    # rebuild the rate at check-B conditions with the round-2 physics (dry, 7 m/s)
    ws, wd, moist = 4.0, 320.0, 6.0
    dt = 12.0
    rate = rate8 * 0.0
    if terrain.startswith("LANDFIRE"):
        from pinn_service.features import load_npz
        from pinn_service.forecaster import metres_per_degree

        path = Path(os.environ.get("STATIC_LAYERS", REPO / "pinn_service" / "static" / "bay_area_static.npz"))
        st = load_npz(path, str(path))
        m_lat, m_lon = metres_per_degree(38.004)
        Xm, Ym = np.meshgrid(xs, xs)
        fuel = st.sample("fuel_fbfm40", -121.935 + Xm / m_lon, 38.004 + Ym / m_lat, fill=0.0)
        slope = st.sample("slope_deg", -121.935 + Xm / m_lon, 38.004 + Ym / m_lat, fill=0.0)
        rate, _ = physics.spread_rate(fuel, slope, np.full(fuel.shape, ws), np.full(fuel.shape, moist))
    else:
        rate = rate8
    rate = rate * (0.2 / float(np.median(rate[rate > 0])))      # a dry, fast day: median 0.2 m/s (0.7 km/h)
    print(f"   terrain: {terrain}, grid {rate.shape}, {res:g} m")
    X, Y = np.meshgrid(xs, xs)
    rows = []
    for seed in seeds:
        m_true = cc.hidden_multiplier(rate.shape, res, seed) / 1.5           # ~1x overall, patches ~0.45-2x
        phi = np.hypot(X, Y) - 800.0
        rate_c = rate.copy()
        rate_c[np.hypot(X, Y) < 900] = np.maximum(rate_c[np.hypot(X, Y) < 900], 0.02)
        truth = r2.advance(phi, res, rate_c * m_true, wd, ws, "ellipse", [dt * 3600.0, 2 * dt * 3600.0])
        mask_a, mask_b, mask_c = phi < 0, truth[dt * 3600.0] < 0, truth[2 * dt * 3600.0] < 0

        # single factor (round 2's alpha)
        times = [a * dt * 3600.0 for a in r2.ALPHAS]
        fields = r2.advance(signed_distance(mask_a, res), res, rate_c, wd, ws, "ellipse", times)
        alpha = float(r2.ALPHAS[int(np.argmin([((fields[t] < 0) ^ mask_b).sum() for t in times]))])
        t0 = time.perf_counter()
        _, m_map, _ = fit_inverse(rate_c, res, wd, ws, "ellipse", mask_a, mask_b, dt, alpha, steps=steps, seed=seed)
        fit_s = time.perf_counter() - t0
        phi_b = signed_distance(mask_b, res)
        f_alpha = r2.advance(phi_b, res, rate_c * alpha, wd, ws, "ellipse", [dt * 3600.0])[dt * 3600.0] < 0
        f_pinn = r2.advance(phi_b, res, rate_c * m_map, wd, ws, "ellipse", [dt * 3600.0])[dt * 3600.0] < 0
        zone = mask_c & ~mask_b & (rate_c > 0)                               # where the next 12 h burned
        err_alpha = float(np.median(np.abs(np.log(alpha / m_true[zone]))))
        err_pinn = float(np.median(np.abs(np.log(m_map[zone] / m_true[zone]))))
        _, ch_alpha, _, _ = r2.score(f_alpha, mask_c, mask_b)
        _, ch_pinn, _, _ = r2.score(f_pinn, mask_c, mask_b)
        print(f"   seed {seed}: map error |log m| alpha {err_alpha:.3f} vs PINN {err_pinn:.3f}; "
              f"24 h change IoU alpha {ch_alpha:.3f} vs PINN {ch_pinn:.3f} (PINN fit {fit_s:.0f} s)")
        rows.append({"seed": seed, "alpha": alpha, "map_err_alpha": err_alpha, "map_err_pinn": err_pinn,
                     "change_iou_alpha": ch_alpha, "change_iou_pinn": ch_pinn})
        if seed == seeds[0]:
            _map_figure(rate_c, mask_a, mask_b, mask_c, m_true, m_map, alpha, OUT / "check_b_map.png")
    df = pd.DataFrame(rows)
    ok_map = df.map_err_pinn.mean() <= 0.8 * df.map_err_alpha.mean()
    ok_fc = df.change_iou_pinn.mean() > df.change_iou_alpha.mean()
    print(f"   mean map error: alpha {df.map_err_alpha.mean():.3f}, PINN {df.map_err_pinn.mean():.3f} -> "
          f"{'PASS' if ok_map else 'FAIL'} (PINN <= 80 % of alpha)")
    print(f"   mean 24 h change IoU: alpha {df.change_iou_alpha.mean():.3f}, PINN {df.change_iou_pinn.mean():.3f} -> "
          f"{'PASS' if ok_fc else 'FAIL'} (PINN better)")
    print(f"   figure: {OUT / 'check_b_map.png'}")
    return {"B_map_pass": bool(ok_map), "B_forecast_pass": bool(ok_fc), "B_table": rows}


def _map_figure(rate, mask_a, mask_b, mask_c, m_true, m_map, alpha, path, title="Check B"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    panels = [("physics rate R (m/s)", rate, "Greens", None)]
    if m_true is not None:
        panels.append(("hidden truth m", m_true, "RdBu_r", (0.2, 2.0)))
    panels.append((f"PINN m (single factor = {alpha:.2f})", m_map, "RdBu_r", (0.2, 2.0)))
    union = mask_a | mask_b | (mask_c if mask_c is not None else False)
    rr, cc_ = np.where(union)
    pad = max(10, int(0.6 * max(np.ptp(rr), np.ptp(cc_))))
    win = np.s_[max(rr.min() - pad, 0):rr.max() + pad + 1, max(cc_.min() - pad, 0):cc_.max() + pad + 1]
    panels = [(n, a[win], c, l) for n, a, c, l in panels]
    mask_a, mask_b = mask_a[win], mask_b[win]
    mask_c = mask_c[win] if mask_c is not None else None
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6))
    for ax, (name, arr, cmap, lim) in zip(np.atleast_1d(axes), panels):
        if lim:
            im = ax.imshow(arr, origin="lower", cmap=cmap, norm=__import__("matplotlib.colors", fromlist=["LogNorm"]).LogNorm(*lim))
        else:
            im = ax.imshow(arr, origin="lower", cmap=cmap)
        for mask, col in ((mask_a, "k"), (mask_b, "tab:blue"), (mask_c, "tab:orange")):
            if mask is not None and mask.any():
                ax.contour(mask.astype(float), [0.5], colors=col, linewidths=1)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{title}: black = first perimeter, blue = second, orange = target")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


# --------------------------------------------------------------------------- check C: real fires

def realfire(max_starts, steps, quick):
    print("\n== Check C: real validation fires (PINN map vs round 2's single factor, same starts)")
    folders = r2.load("validation", 3 if quick else None)
    rows, started = [], time.perf_counter()
    cfg = r2.STEPS["diurnal"]
    for i, d in enumerate(folders, 1):
        t = time.perf_counter()
        try:
            fire = r2.Fire(d)
            n_done = 0
            for h_prev, h_start in r2.start_times(fire, max_starts):
                tg = r2.targets_for(fire, h_start)
                if not tg:
                    continue
                crop = r2.crop_for(fire, h_start, max(tg.values()))
                start = fire.burned(h_start)[crop]
                act = r2.active_field(fire, crop, h_prev, h_start)
                alpha = r2.fit_alpha(fire, crop, h_prev, h_start, cfg, act)
                ws, wd, moist = fire.mean_cond(h_prev, h_start, "diurnal")
                rate = r2.base_rate(fire, crop, ws, moist)
                _, m_map, _ = fit_inverse(rate, fire.res, wd, ws, "ellipse", fire.burned(h_prev)[crop], start,
                                          h_start - h_prev, alpha, steps=steps)
                if n_done == 0:
                    _map_figure(rate, fire.burned(h_prev)[crop], start, None, None, m_map, alpha,
                                OUT / "maps" / f"{fire.id}.png", title=fire.meta["name"])
                phi0 = signed_distance(start, fire.res)
                phi_a, phi_p, h_m = phi0, phi0, h_start
                for lead, h_t in sorted(tg.items(), key=lambda kv: kv[1]):
                    phi_a = r2.roll(fire, crop, phi_a, h_m, h_t, cfg, alpha * act)
                    phi_p = r2.roll(fire, crop, phi_p, h_m, h_t, cfg, m_map * act)
                    h_m = h_t
                    obs = fire.burned(h_t)[crop]
                    for method, pred in (("persistence", start), ("diurnal", phi_a < 0), ("pinn", phi_p < 0)):
                        iou, change, ratio, new = r2.score(pred, obs, start)
                        rows.append({"fire_id": fire.id, "name": fire.meta["name"], "h_start": h_start, "lead": lead,
                                     "method": method, "alpha": alpha, "m_median": float(np.median(m_map[act > 0])),
                                     "iou": iou, "change_iou": change, "growth_ratio": ratio, "observed_new_cells": new})
                n_done += 1
            print(f"  [{i}/{len(folders)}] {d.name}: {n_done} starts ({time.perf_counter() - t:.0f} s)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(folders)}] {d.name}: FAILED {exc}", flush=True)
    if not rows:
        print("no forecasts made")
        return {}
    df = pd.DataFrame(rows)
    out = OUT / ("realfire_quick" if quick else "realfire")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "forecasts.csv", index=False)
    lines, result = [], {}
    grew = df[df.observed_new_cells > 0]
    for lead in r2.LEADS:
        d = grew[grew.lead == lead]
        if not len(d):
            continue
        pf = d.pivot_table(index="fire_id", columns="method", values="change_iou", aggfunc="mean")
        diff = (pf["pinn"] - pf["diurnal"]).dropna()
        lo, hi = r2.bootstrap_ci(diff)
        lines.append(f"\n--- lead {lead}: {d[d.method == 'pinn'].shape[0]} forecasts from {d.fire_id.nunique()} fires")
        for m in ("persistence", "diurnal", "pinn"):
            x = d[d.method == m]
            lines.append(f"{m:12s} change IoU {x.change_iou.mean():.3f} (median {x.change_iou.median():.3f}), "
                         f"IoU {x.iou.mean():.3f}, growth ratio {x.growth_ratio.median():.2f}")
        lines.append(f"PINN - single factor: {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (95% CI over fires); "
                     f"PINN better on {(diff > 0).mean():.0%} of fires")
        result[lead] = {"gain": round(float(diff.mean()), 4), "ci": [round(lo, 4), round(hi, 4)]}
    if "24h" in result:
        ok = result["24h"]["ci"][0] > 0
        lines.append(f"\nCheck C (24 h gain CI above 0): {'PASS' if ok else 'not passed'}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    print(text)
    print(f"\n({(time.perf_counter() - started) / 60:.0f} min; maps of m for each fire in {OUT / 'maps'})")
    return result


def main():
    p = argparse.ArgumentParser(description="Inverse PINN: learned spread-rate correction map")
    p.add_argument("--checks", action="store_true", help="checks A (exact physics) and B (known hidden map)")
    p.add_argument("--realfire", action="store_true", help="check C on the validation fires")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--steps", type=int, default=None, help="training steps per fit (default 3000 checks, 1500 real fires)")
    p.add_argument("--max-starts", type=int, default=3)
    a = p.parse_args()
    if not (a.checks or a.realfire):
        a.checks = True
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    if a.checks:
        steps = a.steps or (600 if a.quick else 3000)
        report["A"] = check_a(steps)
        report["B"] = check_b(steps, seeds=(0,) if a.quick else (0, 1))
        (OUT / "checks.json").write_text(json.dumps(report, indent=2, default=float))
    if a.realfire:
        report["C"] = realfire(2 if a.quick else a.max_starts, a.steps or (400 if a.quick else 1500), a.quick)
    print(f"\nsaved in {OUT}")


if __name__ == "__main__":
    main()
