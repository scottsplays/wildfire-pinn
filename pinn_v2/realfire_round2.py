"""Real-fire round 2: an improvement LADDER on the validation fires, one command.

    python -m pinn_v2.realfire_round2 --quick     # 3 train + 3 validation fires, 2 starts each: checks it runs (~5-10 min)
    python -m pinn_v2.realfire_round2             # 30 train + 26 validation fires (~1-2 h CPU)

Each step adds ONE change to the step before, so each change's effect is measured on its own:

  0 persistence   the observed fire at the start, unchanged (the baseline to beat)
  1 physics       pinn_service as deployed: rate x (1 + 0.6 n.w)+, daily FM100, no calibration
  2 active        + only ACTIVE edges spread (within 2 km of last-interval growth) + rate factor alpha
                    fitted on the last observed interval (Gate 2's winner)
  3 ellipse       + elliptical spread: head at the full rate, flanks and back much slower, shape from
                    wind speed (Anderson 1983 length-to-breadth; the FARSITE/Prometheus fire shape)
  4 diurnal       + day/night dead-fuel moisture from hourly temperature and humidity (Simard EMC)
                    instead of the daily 100-h value: fires slow down at night
  5 learned       + alpha PREDICTED by a ridge regression trained on the TRAIN fires (features: last
                    fitted alpha, wind, moisture, FM100, fuel mix and slope at the active edge, fire
                    size, recent growth, time of day; target: the alpha that would have been right)
  * oracle        step 4 with the alpha fitted on the FUTURE interval: the ceiling for any
                    single-factor calibration. Reference only, not a forecast.

Leads 12 h and 24 h (the VIIRS revisit). Score: change IoU (IoU of the area burned between the start
and the target; persistence = 0), plus total IoU and growth ratio. 95 % confidence intervals for the
paired differences come from bootstrapping over fires. Decision rule, fixed before running: a step
is kept if its 24 h change-IoU gain over the previous step has a CI above 0; the best step replaces
the service physics if it beats step 1 on at least 70 % of validation fires.
Weather over the forecast window is the reanalysis (a stand-in for a weather forecast), as in step 1.
Writes outputs/realfire/round2/{forecasts.csv, train_alpha.csv, summary.txt, ladder.png}.
Test fires are never touched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from pinn_service import physics
from pinn_service.forecaster import signed_distance
from pinn_service.native_ingestion import emc_percent

REPO = Path(__file__).resolve().parent.parent
FIRES = REPO / "data" / "fires"
SEEN_SLACK_H = 0.34
ALPHAS = np.exp(np.linspace(np.log(0.003), np.log(3.0), 43))
LEADS = {"12h": (9.0, 15.0), "24h": (20.0, 28.0)}
PAD_M = 12000.0
MIN_START_KM2 = 1.0
MIN_CAL_GAP_H = 3.0
ACTIVE_M, QUIET_M = 2000.0, 3000.0
WAF = 0.4                     # 10 m wind -> midflame wind (partly sheltered fuels)
STEPS = {                     # name: (shape, moisture, alpha source, active fronts)
    "physics": ("service", "fm100", "none", False),
    "active": ("service", "fm100", "last", True),
    "ellipse": ("ellipse", "fm100", "last", True),
    "diurnal": ("ellipse", "diurnal", "last", True),
    "learned": ("ellipse", "diurnal", "learned", True),
    "oracle": ("ellipse", "diurnal", "oracle", True),
}
LADDER = ["persistence", "physics", "active", "ellipse", "diurnal", "learned"]


# --------------------------------------------------------------------------- data

class Fire:
    def __init__(self, folder: Path, factor: int = 2):
        self.id = folder.name
        self.meta = json.loads((folder / "meta.json").read_text())
        a, s = np.load(folder / "arrival.npz"), np.load(folder / "static.npz")
        arr = a["arrival_hours"].astype(np.float64)
        arr[~np.isfinite(arr)] = np.inf
        ny, nx = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
        blocks = lambda g: g[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
        self.arrival = blocks(arr).min(axis=(1, 3))
        self.fuel = s["fuel_fbfm40"][:ny:factor, :nx:factor]
        self.slope = blocks(s["slope_deg"].astype(np.float64)).mean(axis=(1, 3))
        self.res = float(a["res_m"]) * factor
        self.t0 = pd.Timestamp(str(a["t0_utc"]))
        w = pd.read_csv(folder / "weather.csv", parse_dates=["time_utc"])
        w["h"] = (w.time_utc - self.t0).dt.total_seconds() / 3600.0
        w["emc"] = np.clip(emc_percent(w.temp_c.to_numpy(), w.rh_pct.to_numpy()), 2.0, 35.0)
        self.weather = w
        fm = pd.read_csv(folder / "fm100.csv") if (folder / "fm100.csv").exists() else pd.DataFrame(columns=["date", "fm100_pct"])
        self.fm100 = dict(zip(fm.date.astype(str), fm.fm100_pct))
        self.fm_default = float(np.median(list(self.fm100.values()))) if self.fm100 else 8.0
        self.passes = sorted(set(np.round(pd.read_csv(folder / "passes.csv").hours.to_numpy(), 3)))

    def burned(self, h):
        return self.arrival <= h + SEEN_SLACK_H

    def cond(self, h, moisture: str):
        row = self.weather.iloc[int(np.argmin(np.abs(self.weather.h.to_numpy() - h)))]
        if moisture == "diurnal":
            m = float(row.emc)
        else:
            m = float(self.fm100.get(str((self.t0 + pd.Timedelta(hours=h)).date()), self.fm_default))
        return float(row.wind_speed_m_s), float(row.wind_from_deg), m

    def mean_cond(self, h_a, h_b, moisture):
        hrs = np.arange(h_a + 0.5, h_b, 1.0) if h_b - h_a >= 1 else np.array([(h_a + h_b) / 2])
        c = np.array([self.cond(h, moisture) for h in hrs])
        ang = np.deg2rad(c[:, 1])
        wd = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
        return float(c[:, 0].mean()), wd, float(c[:, 2].mean())


# --------------------------------------------------------------------------- physics variants

def ellipse_e(ws10: float) -> float:
    """Eccentricity from Anderson (1983) length-to-breadth, midflame wind in mph."""
    u = WAF * ws10 * 2.237
    lb = 0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397
    lb = max(lb, 1.0)
    return float(np.sqrt(1.0 - 1.0 / lb ** 2))


def advance(phi, res, rate, wind_from, ws10, shape, seconds_list):
    """Level-set advance with a direction-dependent speed. Returns {seconds: phi}."""
    east, north = physics.wind_toward_from_met(wind_from)
    east, north = float(east), float(north)
    if shape == "service":
        fmax, e = float(rate.max()) * (1 + physics.C_WDIR), 0.0
    else:
        e = ellipse_e(ws10)
        fmax = float(rate.max())                     # head speed is the largest
    out, field, now = {}, phi.astype(np.float64).copy(), 0.0
    targets = sorted(set(float(s) for s in seconds_list))
    if fmax <= 0:
        return {t: field.copy() for t in targets}
    step = 0.45 * res / (np.sqrt(2.0) * fmax)
    for target in targets:
        while now < target - 1e-9:
            dt = min(step, target - now)
            gy, gx = np.gradient(field, res)
            cosang = (east * gx + north * gy) / (np.sqrt(gx * gx + gy * gy) + 1e-12)
            if shape == "service":
                speed = rate * np.maximum(1.0 + physics.C_WDIR * cosang, 0.0)
            else:
                speed = rate * (1.0 - e) / (1.0 - e * cosang)
            field -= dt * speed * physics._upwind_magnitude(field, res)
            now += dt
        out[target] = field.copy()
    return out


def base_rate(fire, crop, ws, moisture_value):
    fuel, slope = fire.fuel[crop], fire.slope[crop]
    rate, _ = physics.spread_rate(fuel, slope, np.full(fuel.shape, ws), np.full(fuel.shape, moisture_value))
    return rate


def roll(fire, crop, phi, h_from, h_to, step_cfg, mult):
    shape, moisture, _, _ = step_cfg
    h = h_from
    while h < h_to - 1e-6:
        dh = min(1.0 - (h % 1.0) if h % 1.0 > 1e-6 else 1.0, h_to - h)
        ws, wd, m = fire.cond(h + dh / 2, moisture)
        phi = advance(phi, fire.res, base_rate(fire, crop, ws, m) * mult, wd, ws, shape, [dh * 3600.0])[dh * 3600.0]
        h += dh
    return phi


def fit_alpha(fire, crop, h_a, h_b, step_cfg, field):
    """alpha that best turns the observed fire at h_a into the observed fire at h_b, using the
    interval's mean conditions (so alpha x time scaling is exact: one solver run)."""
    shape, moisture, _, _ = step_cfg
    ws, wd, m = fire.mean_cond(h_a, h_b, moisture)
    rate = base_rate(fire, crop, ws, m) * field
    span = h_b - h_a
    times = [a * span * 3600.0 for a in ALPHAS]
    fields = advance(signed_distance(fire.burned(h_a)[crop], fire.res), fire.res, rate, wd, ws, shape, times)
    target = fire.burned(h_b)[crop]
    losses = [int(((fields[t] < 0) ^ target).sum()) for t in times]
    return float(ALPHAS[int(np.argmin(losses))])


def active_field(fire, crop, h_prev, h_start):
    start = fire.burned(h_start)[crop]
    grew = start & ~fire.burned(h_prev)[crop]
    if not grew.any():
        return np.where(start, 1.0, 0.0)
    edge = start & ~ndimage.binary_erosion(start)
    near = ndimage.distance_transform_edt(~grew) * fire.res <= ACTIVE_M
    act, quiet = edge & near, edge & ~near
    if not quiet.any():
        return np.ones(start.shape)
    d_q = ndimage.distance_transform_edt(~quiet) * fire.res
    d_a = ndimage.distance_transform_edt(~act) * fire.res if act.any() else np.full(start.shape, np.inf)
    return np.where(~start & (d_q < d_a) & (d_q <= QUIET_M), 0.0, 1.0)


# --------------------------------------------------------------------------- forecast set-up

def crop_for(fire, h_start, h_last):
    rows, cols = np.where(fire.burned(h_last) | fire.burned(h_start))
    pad = int(PAD_M / fire.res)
    return np.s_[max(rows.min() - pad, 0):min(rows.max() + pad + 1, fire.arrival.shape[0]),
                 max(cols.min() - pad, 0):min(cols.max() + pad + 1, fire.arrival.shape[1])]


def start_times(fire, max_starts):
    cell_km2 = fire.res ** 2 / 1e6
    out = []
    for i, h in enumerate(fire.passes):
        if fire.burned(h).sum() * cell_km2 < MIN_START_KM2:
            continue
        prev = [p for p in fire.passes[:i] if h - p >= MIN_CAL_GAP_H]
        if not prev or fire.burned(h).sum() <= fire.burned(prev[-1]).sum():
            continue
        if not any(lo <= t - h <= hi for t in fire.passes for lo, hi in LEADS.values()):
            continue
        out.append((prev[-1], h))
    if len(out) > max_starts:
        out = [out[int(k)] for k in np.linspace(0, len(out) - 1, max_starts)]
    return out


def targets_for(fire, h_start):
    found = {}
    for name, (lo, hi) in LEADS.items():
        c = [t for t in fire.passes if lo <= t - h_start <= hi]
        if c:
            found[name] = min(c, key=lambda t: abs(t - h_start - (lo + hi) / 2))
    return found


def score(pred, obs, start):
    u = (pred | obs).sum()
    iou = float((pred & obs).sum() / u) if u else 1.0
    pg, og = pred & ~start, obs & ~start
    gu = (pg | og).sum()
    return round(iou, 4), round(float((pg & og).sum() / gu) if gu else 1.0, 4), \
        round(float(pg.sum() / og.sum()) if og.sum() else float("nan"), 3), int(og.sum())


def features(fire, crop, h_prev, h_start, alpha_last, active):
    """Inputs for the learned alpha: only what is known at the forecast start (plus the weather
    forecast for the next 12 h, here the reanalysis)."""
    cell_km2 = fire.res ** 2 / 1e6
    start, prev = fire.burned(h_start)[crop], fire.burned(h_prev)[crop]
    ws, _, emc = fire.mean_cond(h_start, h_start + 12, "diurnal")
    _, _, fm = fire.mean_cond(h_start, h_start + 12, "fm100")
    zone = (active > 0) & ~start
    zone = zone & (ndimage.distance_transform_edt(~start) * fire.res <= 3000)
    fuel = fire.fuel[crop][zone] if zone.any() else fire.fuel[crop].ravel()
    n = max(len(fuel), 1)
    hour_local = ((fire.t0 + pd.Timedelta(hours=h_start)).hour - 8) % 24
    return {
        "log_alpha_last": np.log(alpha_last),
        "log_wind": np.log(ws + 0.5), "emc_next12": emc, "fm100": fm,
        "grass": float(((fuel >= 101) & (fuel <= 124)).sum() / n),
        "shrub": float(((fuel >= 141) & (fuel <= 149)).sum() / n),
        "timber": float(((fuel >= 161) & (fuel <= 189)).sum() / n),
        "slope_zone": float(fire.slope[crop][zone].mean()) if zone.any() else float(fire.slope[crop].mean()),
        "log_area": np.log(start.sum() * cell_km2 + 0.1),
        "log_growth_rate": np.log((start & ~prev).sum() * cell_km2 / max(h_start - h_prev, 1.0) + 1e-3),
        "hour_sin": np.sin(2 * np.pi * hour_local / 24), "hour_cos": np.cos(2 * np.pi * hour_local / 24),
    }


# --------------------------------------------------------------------------- learned alpha (ridge)

class Ridge:
    def __init__(self, lam=3.0):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        reg = self.lam * np.eye(Z.shape[1])
        reg[0, 0] = 0
        self.w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, X):
        return np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd]) @ self.w


def train_alpha_model(train_fires, max_starts, out):
    rows = []
    cfg = STEPS["diurnal"]
    for i, fire in enumerate(train_fires, 1):
        t = time.perf_counter()
        n = 0
        for h_prev, h_start in start_times(fire, max_starts):
            tg = targets_for(fire, h_start)
            if "12h" not in tg:
                continue
            crop = crop_for(fire, h_start, tg["12h"])
            act = active_field(fire, crop, h_prev, h_start)
            a_last = fit_alpha(fire, crop, h_prev, h_start, cfg, act)
            a_next = fit_alpha(fire, crop, h_start, tg["12h"], cfg, act)
            if not (fire.burned(tg["12h"])[crop] & ~fire.burned(h_start)[crop]).any():
                continue
            rows.append({"fire_id": fire.id, **features(fire, crop, h_prev, h_start, a_last, act), "log_alpha_next": np.log(a_next)})
            n += 1
        print(f"  train [{i}/{len(train_fires)}] {fire.id}: {n} samples ({time.perf_counter() - t:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "train_alpha.csv", index=False)
    if len(df) < 10:
        print("  too few training samples: the learned step will fall back to the last-interval alpha")
        return None, None, df
    cols = [c for c in df.columns if c not in ("fire_id", "log_alpha_next")]
    X, y = df[cols].to_numpy(float), df.log_alpha_next.to_numpy()
    # honest skill estimate: leave-one-FIRE-out
    pred = np.empty_like(y)
    for f in df.fire_id.unique():
        m = df.fire_id.to_numpy() == f
        if (~m).sum() < 5:
            pred[m] = y[~m].mean() if (~m).any() else 0.0
            continue
        pred[m] = Ridge().fit(X[~m], y[~m]).predict(X[m])
    base = X[:, cols.index("log_alpha_last")]
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    info = {"samples": len(df), "fires": int(df.fire_id.nunique()),
            "rmse_log_alpha_ridge_LOFO": round(rmse(pred), 3), "rmse_log_alpha_last_alpha": round(rmse(base), 3),
            "rmse_log_alpha_constant": round(rmse(np.full_like(y, y.mean())), 3)}
    model = Ridge().fit(X, y)
    info["weights"] = {c: round(float(w), 3) for c, w in zip(cols, model.w[1:])}
    return (model, cols), info, df


# --------------------------------------------------------------------------- evaluation

def evaluate_fire(fire, max_starts, learned):
    rows = []
    for h_prev, h_start in start_times(fire, max_starts):
        tg = targets_for(fire, h_start)
        if not tg:
            continue
        crop = crop_for(fire, h_start, max(tg.values()))
        start = fire.burned(h_start)[crop]
        phi0 = signed_distance(start, fire.res)
        act = active_field(fire, crop, h_prev, h_start)
        alphas = {}
        for name, cfg in STEPS.items():
            src = cfg[2]
            field = act if cfg[3] else 1.0
            if src == "none":
                alphas[name] = 1.0
            elif src == "last":
                alphas[name] = fit_alpha(fire, crop, h_prev, h_start, cfg, field)
        if learned is not None:
            model, cols = learned
            f = features(fire, crop, h_prev, h_start, alphas["diurnal"], act)
            alphas["learned"] = float(np.clip(np.exp(model.predict(np.array([[f[c] for c in cols]]))[0]), ALPHAS[0], ALPHAS[-1]))
        else:
            alphas["learned"] = alphas["diurnal"]
        phis = {name: phi0 for name in STEPS}
        h_m = h_start
        for lead, h_t in sorted(tg.items(), key=lambda kv: kv[1]):
            obs = fire.burned(h_t)[crop]
            alphas["oracle"] = fit_alpha(fire, crop, h_start, h_t, STEPS["oracle"], act)
            preds = {"persistence": start}
            for name, cfg in STEPS.items():
                mult = alphas[name] * (act if cfg[3] else 1.0)
                if name == "oracle":                       # re-run from the start with this lead's alpha
                    preds[name] = roll(fire, crop, phi0, h_start, h_t, cfg, mult) < 0
                    continue
                phis[name] = roll(fire, crop, phis[name], h_m, h_t, cfg, mult)
                preds[name] = phis[name] < 0
            h_m = h_t
            for name, pred in preds.items():
                iou, change, ratio, new = score(pred, obs, start)
                rows.append({"fire_id": fire.id, "name": fire.meta["name"], "region": fire.meta["region"],
                             "h_start": h_start, "lead": lead, "method": name, "alpha": round(alphas.get(name, 0.0), 4),
                             "iou": iou, "change_iou": change, "growth_ratio": ratio, "observed_new_cells": new})
    return rows


def bootstrap_ci(per_fire_diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(per_fire_diff)
    if len(v) < 2:
        return float("nan"), float("nan")
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(df, info, out):
    L = []
    if info:
        L.append(f"learned alpha model: {info['samples']} samples from {info['fires']} TRAIN fires; leave-one-fire-out RMSE of log alpha "
                 f"{info['rmse_log_alpha_ridge_LOFO']} (vs {info['rmse_log_alpha_last_alpha']} using the last alpha as is, "
                 f"{info['rmse_log_alpha_constant']} for a constant)")
        L.append("  weights (standardised): " + ", ".join(f"{k} {v:+.2f}" for k, v in info["weights"].items()))
    grew = df[df.observed_new_cells > 0]
    order = LADDER + ["oracle"]
    per_fire_all = {}
    for lead in LEADS:
        d = grew[grew.lead == lead]
        if not len(d):
            continue
        pf = d.pivot_table(index="fire_id", columns="method", values="change_iou", aggfunc="mean")
        per_fire_all[lead] = pf
        L.append(f"\n--- lead {lead}: {d[d.method == 'physics'].shape[0]} forecasts from {d.fire_id.nunique()} fires (only where the fire grew)")
        L.append(f"{'step':12s} {'changeIoU':>9s} {'median':>7s} {'IoU':>6s} {'growth':>7s} | {'gain vs previous step (95% CI)':>32s} | beats physics")
        prev = None
        for m in order:
            x = d[d.method == m]
            if not len(x):
                continue
            gain = ""
            if prev is not None and m in pf and prev in pf and m != "oracle":
                diff = (pf[m] - pf[prev]).dropna()
                lo, hi = bootstrap_ci(diff)
                gain = f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" *" if lo > 0 else "")
            beats = f"{(pf[m] > pf['physics']).mean():.0%} of fires" if m not in ("physics", "persistence") and m in pf else ""
            L.append(f"{m:12s} {x.change_iou.mean():9.3f} {x.change_iou.median():7.3f} {x.iou.mean():6.3f} {x.growth_ratio.median():7.2f} | {gain:>32s} | {beats}")
            if m != "oracle":
                prev = m
    L.append("\n* = gain whose 95% CI is above zero (bootstrap over fires). growth = median predicted/observed new area.")
    a = df.drop_duplicates(["fire_id", "h_start", "method"])
    for m in ("active", "ellipse", "diurnal", "learned"):
        x = a[a.method == m].alpha
        if len(x):
            at_edge = ((x <= ALPHAS[0] * 1.001) | (x >= ALPHAS[-1] * 0.999)).mean()
            L.append(f"alpha {m:8s}: median {x.median():.3f}, middle 50% {x.quantile(.25):.3f}-{x.quantile(.75):.3f}"
                     + (f"  (WARNING: {at_edge:.0%} at the search limit {ALPHAS[0]:g}-{ALPHAS[-1]:g})" if at_edge > 0.2 else ""))
    text = "\n".join(L)
    (out / "summary.txt").write_text(text)
    return text, per_fire_all


def figure(per_fire_all, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(per_fire_all), figsize=(6 * len(per_fire_all), 4.2), squeeze=False)
    for ax, (lead, pf) in zip(axes[0], per_fire_all.items()):
        names = [m for m in LADDER + ["oracle"] if m in pf]
        means = [pf[m].mean() for m in names]
        cis = [bootstrap_ci(pf[m].dropna()) for m in names]
        err = np.array([[mu - lo, hi - mu] for mu, (lo, hi) in zip(means, cis)]).T
        colours = {"persistence": "#9aa0a6", "oracle": "#d0d4d9"}
        colours = [colours.get(m, "#2a6fdb") for m in names]
        ax.bar(range(len(names)), means, yerr=err, color=colours, capsize=4)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("change IoU (mean over fires)")
        ax.set_title(f"lead {lead}")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Real-fire round 2: each step adds one change (95% CI over validation fires)")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=110)
    plt.close(fig)


def load(split, limit=None):
    fires = []
    for d in sorted(FIRES.iterdir()) if FIRES.exists() else []:
        if not ((d / "static.npz").exists() and (d / "arrival.npz").exists() and (d / "meta.json").exists()):
            continue
        if json.loads((d / "meta.json").read_text())["split"] == split:
            fires.append(d)
    return fires[:limit] if limit else fires


def main():
    p = argparse.ArgumentParser(description="Real-fire round 2: improvement ladder")
    p.add_argument("--max-starts", type=int, default=6, help="forecast starts per validation fire")
    p.add_argument("--train-starts", type=int, default=8, help="starts per training fire for the learned alpha")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(REPO / "outputs" / "realfire"))
    a = p.parse_args()
    lim = 3 if a.quick else None
    if a.quick:
        a.max_starts, a.train_starts = 2, 2
    out = Path(a.out) / ("round2_quick" if a.quick else "round2")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train = [Fire(d) for d in load("train", lim)]
    print(f"step 1/2: learning alpha from {len(train)} TRAIN fires", flush=True)
    learned, info, _ = train_alpha_model(train, a.train_starts, out)
    if info:
        print(f"  leave-one-fire-out RMSE(log alpha): ridge {info['rmse_log_alpha_ridge_LOFO']}, "
              f"last alpha {info['rmse_log_alpha_last_alpha']}, constant {info['rmse_log_alpha_constant']}", flush=True)
        (out / "learned_alpha.json").write_text(json.dumps(info, indent=2))

    val = load("validation", lim)
    print(f"step 2/2: ladder on {len(val)} VALIDATION fires, up to {a.max_starts} starts each", flush=True)
    rows = []
    for i, d in enumerate(val, 1):
        t = time.perf_counter()
        try:
            fire = Fire(d)
            fr = evaluate_fire(fire, a.max_starts, learned)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(val)}] {d.name}: FAILED {exc}", flush=True)
            continue
        rows += fr
        if fr:
            g = pd.DataFrame(fr).query("observed_new_cells > 0").groupby("method").change_iou.mean()
            print(f"  [{i}/{len(val)}] {d.name}: " + " ".join(f"{m} {g.get(m, float('nan')):.2f}" for m in LADDER[1:]) +
                  f" | oracle {g.get('oracle', float('nan')):.2f} ({time.perf_counter() - t:.0f} s)", flush=True)
    if not rows:
        print("no forecasts made")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecasts.csv", index=False)
    text, per_fire = summarise(df, info, out)
    print("\n" + text)
    figure(per_fire, out)
    print(f"\n({(time.perf_counter() - started) / 60:.0f} min; results, summary.txt and ladder.png in {out})")


if __name__ == "__main__":
    main()
