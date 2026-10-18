"""Real-fire evaluation, round 1: does calibrating on the last observation beat plain physics on REAL fires?

    python -m pinn_v2.realfire_eval --quick                    # 3 validation fires, 3 starts each: checks it runs (~5 min)
    python -m pinn_v2.realfire_eval                            # every validation fire with data (~30-90 min CPU)
    python -m pinn_v2.realfire_eval --fires 20191023_kincade 20200927_glass
    python -m pinn_v2.realfire_eval --split test --final       # ONCE, at the very end of the project

For each fire (data/fires/<id>/ from fetch_fire_data + fetch_fuel) and several forecast start times
(VIIRS passes), it forecasts the fire ~12 h and ~24 h ahead (the VIIRS revisit; shorter leads need GOES), and scores each
forecast against what VIIRS saw (arrival.npz):

  persistence  the observed fire at the start, unchanged
  physics      pinn_service physics: LANDFIRE fuel/slope, hourly wind, daily FM100, level-set solver
  calibrated   the same physics with one rate factor alpha, fitted so the physics reproduces the
               fire's most recent observed growth (previous pass -> start pass); Gate 2's winner

Scores: change IoU (IoU of the area burned between the start and the target; persistence = 0),
total IoU, and growth ratio (predicted new area / observed new area; 1 = right amount).
Runs on a 180 m grid (2x2 blocks of the 90 m arrival map; VIIRS pixels are 375 m) to stay fast.
Writes outputs/realfire/<split>/forecasts.csv, per_fire.csv, summary.txt.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pinn_service import physics
from pinn_service.forecaster import signed_distance

REPO = Path(__file__).resolve().parent.parent
FIRES = REPO / "data" / "fires"
SEEN_SLACK_H = 0.34             # a pass's detections are spread over ~20 min
ALPHAS = np.exp(np.linspace(np.log(0.2), np.log(5.0), 29))
LEADS = {"12h": (9.0, 15.0), "24h": (20.0, 28.0)}   # VIIRS passes come ~12 h apart (pairs ~50 min apart)
PAD_M = 12000.0                 # crop margin around the fire for each forecast
MIN_START_KM2 = 1.0             # skip starts where the fire is too small to see properly
MIN_CAL_GAP_H = 3.0             # calibration interval must be at least this long


# --------------------------------------------------------------------------- data

class Fire:
    def __init__(self, folder: Path, factor: int = 2):
        self.id = folder.name
        self.meta = json.loads((folder / "meta.json").read_text())
        a = np.load(folder / "arrival.npz")
        s = np.load(folder / "static.npz")
        arr = a["arrival_hours"].astype(np.float64)
        arr[~np.isfinite(arr)] = np.inf
        ny, nx = (arr.shape[0] // factor) * factor, (arr.shape[1] // factor) * factor
        blocks = lambda g: g[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
        self.arrival = blocks(arr).min(axis=(1, 3))                 # first seen anywhere in the block
        self.fuel = s["fuel_fbfm40"][:ny:factor, :nx:factor]
        self.slope = blocks(s["slope_deg"].astype(np.float64)).mean(axis=(1, 3))
        self.res = float(a["res_m"]) * factor
        self.t0 = pd.Timestamp(str(a["t0_utc"]))
        self.after_fire = bool(s["fuel_after_fire"])
        w = pd.read_csv(folder / "weather.csv", parse_dates=["time_utc"])
        w["h"] = (w.time_utc - self.t0).dt.total_seconds() / 3600.0
        self.weather = w
        fm = pd.read_csv(folder / "fm100.csv") if (folder / "fm100.csv").exists() else pd.DataFrame(columns=["date", "fm100_pct"])
        self.fm100 = dict(zip(fm.date.astype(str), fm.fm100_pct))
        self.passes = pd.read_csv(folder / "passes.csv")

    def burned(self, h: float) -> np.ndarray:
        return self.arrival <= h + SEEN_SLACK_H

    def conditions(self, h: float):
        i = int(np.argmin(np.abs(self.weather.h.to_numpy() - h)))
        row = self.weather.iloc[i]
        day = str((self.t0 + pd.Timedelta(hours=h)).date())
        fm = self.fm100.get(day, np.nan)
        if not np.isfinite(fm) and self.fm100:
            fm = float(np.median(list(self.fm100.values())))
        if not np.isfinite(fm):
            fm = 10.0
        return float(row.wind_speed_m_s), float(row.wind_from_deg), float(fm)


def rate_and_wind(fire: Fire, crop, h: float):
    ws, wd, fm = fire.conditions(h)
    fuel, slope = fire.fuel[crop], fire.slope[crop]
    rate, _ = physics.spread_rate(fuel, slope, np.full(fuel.shape, ws), np.full(fuel.shape, fm))
    east, north = physics.wind_toward_from_met(wd)
    return rate, (np.full(fuel.shape, float(east)), np.full(fuel.shape, float(north)))


def roll(fire: Fire, crop, phi, h_from: float, h_to: float, alpha: float = 1.0) -> np.ndarray:
    """Advance phi from h_from to h_to hour by hour with that hour's weather, rate * alpha."""
    h = h_from
    while h < h_to - 1e-6:
        step = min(1.0 - (h % 1.0) if h % 1.0 > 1e-6 else 1.0, h_to - h)
        rate, wind = rate_and_wind(fire, crop, h + step / 2)
        phi = physics.propagate(phi, fire.res, rate * alpha, wind, [step * 3600.0])[step * 3600.0]
        h += step
    return phi


def fit_alpha(fire: Fire, crop, h_prev: float, h_start: float) -> float:
    """Rate factor that best turns the observed fire at h_prev into the observed fire at h_start.
    Uses the interval's mean weather, so alpha * time scaling is exact: one solver run to
    alpha_max * interval, read off at each alpha."""
    prev, target = fire.burned(h_prev)[crop], fire.burned(h_start)[crop]
    span = h_start - h_prev
    hours = np.arange(h_prev + 0.5, h_start, 1.0) if span >= 1 else np.array([(h_prev + h_start) / 2])
    conds = [fire.conditions(h) for h in hours]
    ws = float(np.mean([c[0] for c in conds]))
    ang = np.deg2rad([c[1] for c in conds])
    wd = float(np.rad2deg(np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang)))) % 360)
    fm = float(np.mean([c[2] for c in conds]))
    fuel, slope = fire.fuel[crop], fire.slope[crop]
    rate, _ = physics.spread_rate(fuel, slope, np.full(fuel.shape, ws), np.full(fuel.shape, fm))
    east, north = physics.wind_toward_from_met(wd)
    wind = (np.full(fuel.shape, float(east)), np.full(fuel.shape, float(north)))
    times = [a * span * 3600.0 for a in ALPHAS]
    fields = physics.propagate(signed_distance(prev, fire.res), fire.res, rate, wind, times)
    losses = [int(((fields[t] < 0) ^ target).sum()) for t in times]
    return float(ALPHAS[int(np.argmin(losses))])


def crop_for(fire: Fire, h_start: float, h_last: float):
    seen = fire.burned(h_last) | fire.burned(h_start)
    rows, cols = np.where(seen)
    pad = int(PAD_M / fire.res)
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad + 1, fire.arrival.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad + 1, fire.arrival.shape[1])
    return np.s_[r0:r1, c0:c1]


def score(pred_mask, obs_mask, start_mask):
    union = (pred_mask | obs_mask).sum()
    iou = float((pred_mask & obs_mask).sum() / union) if union else 1.0
    pg, og = pred_mask & ~start_mask, obs_mask & ~start_mask
    gu = (pg | og).sum()
    change = float((pg & og).sum() / gu) if gu else 1.0
    ratio = float(pg.sum() / og.sum()) if og.sum() else float("nan")
    return round(iou, 4), round(change, 4), round(ratio, 3), int(og.sum())


# --------------------------------------------------------------------------- forecasts

def start_times(fire: Fire, max_starts: int) -> list[tuple[float, float]]:
    """(h_prev, h_start) pairs: VIIRS passes with a growing fire and an earlier pass to calibrate on."""
    hours = sorted(set(np.round(fire.passes.hours.to_numpy(), 3)))
    cell_km2 = fire.res ** 2 / 1e6
    out = []
    for i, h in enumerate(hours):
        if fire.burned(h).sum() * cell_km2 < MIN_START_KM2:
            continue
        prev = [p for p in hours[:i] if h - p >= MIN_CAL_GAP_H]
        if not prev:
            continue
        hp = prev[-1]
        if fire.burned(h).sum() <= fire.burned(hp).sum():
            continue                                      # nothing grew: nothing to calibrate on
        if not any(lo <= t - h <= hi for t in hours for lo, hi in LEADS.values()):
            continue
        out.append((hp, h))
    if len(out) > max_starts:                             # spread evenly over the fire's life
        out = [out[int(k)] for k in np.linspace(0, len(out) - 1, max_starts)]
    return out


def targets_for(fire: Fire, h_start: float) -> dict[str, float]:
    hours = sorted(set(np.round(fire.passes.hours.to_numpy(), 3)))
    found = {}
    for name, (lo, hi) in LEADS.items():
        mid = (lo + hi) / 2
        cands = [t for t in hours if lo <= t - h_start <= hi]
        if cands:
            found[name] = min(cands, key=lambda t: abs(t - h_start - mid))
    return found


def evaluate_fire(fire: Fire, max_starts: int) -> list[dict]:
    rows = []
    for h_prev, h_start in start_times(fire, max_starts):
        targets = targets_for(fire, h_start)
        if not targets:
            continue
        crop = crop_for(fire, h_start, max(targets.values()))
        start = fire.burned(h_start)[crop]
        phi0 = signed_distance(start, fire.res)
        alpha = fit_alpha(fire, crop, h_prev, h_start)
        phi_m, phi_c, h_m = phi0, phi0, h_start
        for name, h_t in sorted(targets.items(), key=lambda kv: kv[1]):
            phi_m = roll(fire, crop, phi_m, h_m, h_t)
            phi_c = roll(fire, crop, phi_c, h_m, h_t, alpha)
            h_m = h_t
            obs = fire.burned(h_t)[crop]
            for method, pred in (("persistence", start), ("physics", phi_m < 0), ("calibrated", phi_c < 0)):
                iou, change, ratio, obs_new = score(pred, obs, start)
                rows.append({"fire_id": fire.id, "name": fire.meta["name"], "split": fire.meta["split"],
                             "region": fire.meta["region"], "fuel_after_fire": fire.after_fire,
                             "h_prev": h_prev, "h_start": h_start, "lead": name, "lead_h": round(h_t - h_start, 2),
                             "alpha": round(alpha, 3), "method": method, "iou": iou, "change_iou": change,
                             "growth_ratio": ratio, "observed_new_cells": obs_new})
    return rows


def summarise(df: pd.DataFrame, out: Path) -> str:
    lines = []
    grew = df[df.observed_new_cells > 0]
    for lead in LEADS:
        d = grew[grew.lead == lead]
        if not len(d):
            continue
        n = d[d.method == "physics"].shape[0]
        lines.append(f"\n--- lead {lead} ({n} forecasts from {d.fire_id.nunique()} fires; only forecasts where the fire grew)")
        lines.append(f"{'method':12s} {'changeIoU mean':>14s} {'median':>7s} {'IoU mean':>9s} {'growth ratio (median)':>22s}")
        for m in ("persistence", "physics", "calibrated"):
            x = d[d.method == m]
            lines.append(f"{m:12s} {x.change_iou.mean():14.3f} {x.change_iou.median():7.3f} {x.iou.mean():9.3f} {x.growth_ratio.median():22.2f}")
        piv = d.pivot_table(index=["fire_id", "h_start"], columns="method", values="change_iou")
        if {"physics", "calibrated"} <= set(piv.columns):
            better = (piv.calibrated > piv.physics).mean()
            lines.append(f"calibrated beats physics in {better:.0%} of forecasts; "
                         f"per-fire mean: {(piv.groupby('fire_id').mean().pipe(lambda p: (p.calibrated > p.physics).mean())):.0%} of fires")
    a = df[df.method == "calibrated"].drop_duplicates(["fire_id", "h_start"]).alpha
    if len(a):
        lines.append(f"\nfitted alpha (calibration factor): median {a.median():.2f}, middle 50% {a.quantile(.25):.2f}-{a.quantile(.75):.2f} "
                     f"(1 = physics rate right; >1 = real fires faster than the physics)")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text)
    return text


def main() -> None:
    p = argparse.ArgumentParser(description="Real-fire evaluation, round 1")
    p.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    p.add_argument("--fires", nargs="*")
    p.add_argument("--max-starts", type=int, default=6, help="forecast start times per fire")
    p.add_argument("--quick", action="store_true", help="3 fires x 3 starts")
    p.add_argument("--final", action="store_true", help="required to score the test split")
    p.add_argument("--out", default=str(REPO / "outputs" / "realfire"))
    a = p.parse_args()
    if a.split == "test" and not a.final:
        raise SystemExit("the test split is scored ONCE at the end: add --final when that time comes")
    folders = []
    for d in sorted(FIRES.iterdir()) if FIRES.exists() else []:
        if not (d / "static.npz").exists() or not (d / "arrival.npz").exists():
            continue
        meta = json.loads((d / "meta.json").read_text())
        if a.fires and d.name not in a.fires:
            continue
        if not a.fires and meta["split"] != a.split:
            continue
        if meta["split"] == "test" and not a.final:
            continue
        folders.append(d)
    if a.quick:
        folders, a.max_starts = folders[:3], 3
    out = Path(a.out) / ("quick" if a.quick else (a.split if not a.fires else "selected"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(folders)} fires; up to {a.max_starts} forecast starts each; leads {list(LEADS)}")
    rows, started = [], time.perf_counter()
    for i, d in enumerate(folders, 1):
        t = time.perf_counter()
        try:
            fire = Fire(d)
            fr = evaluate_fire(fire, a.max_starts)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(folders)}] {d.name}: FAILED {exc}", flush=True)
            continue
        rows += fr
        if fr:
            f = pd.DataFrame(fr)
            g = f[f.observed_new_cells > 0].groupby("method").change_iou.mean()
            print(f"[{i}/{len(folders)}] {d.name}: {f.h_start.nunique()} starts, change IoU "
                  f"physics {g.get('physics', float('nan')):.2f} / calibrated {g.get('calibrated', float('nan')):.2f}, "
                  f"alpha {f.alpha.median():.2f} ({time.perf_counter() - t:.0f} s)", flush=True)
        else:
            print(f"[{i}/{len(folders)}] {d.name}: no usable start times", flush=True)
    if not rows:
        print("no forecasts made")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecasts.csv", index=False)
    per_fire = df[df.observed_new_cells > 0].pivot_table(index=["fire_id", "name"], columns="method", values="change_iou", aggfunc="mean").round(3)
    per_fire.to_csv(out / "per_fire.csv")
    print(summarise(df, out))
    print(f"\n({time.perf_counter() - started:.0f} s; results in {out})")


if __name__ == "__main__":
    main()