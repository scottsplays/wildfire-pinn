"""Real-fire round 3: fixes the two evaluation problems round 2 exposed, and asks the core question.

    python -m pinn_v2.realfire_round3 --quick     # 3 train + 3 validation fires, 2 starts: checks it runs (~5-10 min)
    python -m pinn_v2.realfire_round3             # 30 train + 26 validation fires (~1-1.5 h CPU)

Round 2 problems and fixes
  1. Calibration objective. Round 2 fitted alpha by minimising wrong cells. On real VIIRS data the
     predicted growth overlaps the observed growth less than half the time, so "grow nothing" always
     won (alpha hit its floor in 75 % of starts, even for the oracle). Round 3 fits alpha so the
     predicted NEW AREA equals the observed new area over the interval (area-matched rate of spread,
     how analysts calibrate). Where the growth lands is then left to the physics, and scored.
  2. Holes in the truth. The burned area was the union of satellite detection footprints: ground that
     burned between passes and was never seen hot stayed "unburned", so big fires looked like Swiss
     cheese and every hole the physics filled counted as a false positive. Round 3 closes gaps of
     about one VIIRS pixel and fills enclosed holes at every pass (standard perimeter mapping).

The core question: does fuel + wind physics put the growth in the right PLACE better than a plain
uniform buffer around the fire, when both grow by the same amount?

Methods (scored on new-area IoU against VIIRS, 12 h and 24 h ahead, as rounds 1-2)
  persistence      the fire at the start, unchanged
  physics          the service as deployed (alpha = 1)
  calibrated       + alpha area-matched on the last observed interval
  active           + only active edges spread (round 2's mask)
  ellipse          + elliptical fire shape from wind speed
  diurnal          + hourly day/night dead-fuel moisture
  learned          diurnal, alpha predicted by a ridge regression trained on the TRAIN fires
  buffer           uniform growth in all directions, ignoring fuel and wind, by the area the last
                   interval's growth rate implies (the "no physics" forecast)
 oracle versions (told the TRUE amount of future growth, so only placement is tested; not forecasts):
  buffer_oracle    uniform buffer with the true new area
  physics_oracle   service physics with the true new area
  diurnal_oracle   full stack (active + ellipse + diurnal) with the true new area

Decision rules, fixed before running:
  placement   physics adds placement skill if diurnal_oracle (or physics_oracle) beats buffer_oracle
              at 24 h with a 95 % CI above zero (bootstrap over fires)
  forecast    the service adopts the best non-oracle method if it beats BOTH physics and buffer at
              24 h with CIs above zero; otherwise it keeps the simplest method that is not worse.
Writes outputs/realfire/round3/{forecasts.csv, train_alpha.csv, summary.txt, ladder.png}. Test fires untouched.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from pinn_service.forecaster import signed_distance
from pinn_v2 import realfire_round2 as r2

REPO = Path(__file__).resolve().parent.parent
ALPHAS = np.exp(np.linspace(np.log(0.01), np.log(6.0), 44))
CFG = {                                     # (shape, moisture, alpha source, active mask)
    "physics": ("service", "fm100", "none", False),
    "calibrated": ("service", "fm100", "area", False),
    "active": ("service", "fm100", "area", True),
    "ellipse": ("ellipse", "fm100", "area", True),
    "diurnal": ("ellipse", "diurnal", "area", True),
}
LADDER = ["persistence", "physics", "calibrated", "active", "ellipse", "diurnal", "learned"]
ORACLES = ["buffer_oracle", "physics_oracle", "diurnal_oracle"]
CLOSE_CELLS = 2                              # 2 x 180 m ~ one VIIRS pixel


class Fire(r2.Fire):
    """Round-2 fire with gap-closed, hole-filled burned masks at every pass."""

    def __init__(self, folder: Path, factor: int = 2):
        super().__init__(folder, factor)
        raw = self.arrival
        filled = np.full(raw.shape, np.inf)
        disk = np.hypot(*np.mgrid[-CLOSE_CELLS:CLOSE_CELLS + 1, -CLOSE_CELLS:CLOSE_CELLS + 1]) <= CLOSE_CELLS
        burned = np.zeros(raw.shape, dtype=bool)
        for t in self.passes:
            seen = raw <= t + r2.SEEN_SLACK_H
            if not seen.any():
                continue
            m = ndimage.binary_fill_holes(ndimage.binary_closing(seen | burned, structure=disk, border_value=0) | seen | burned)
            new = m & ~burned
            filled[new] = t
            burned = m
        self.arrival = filled


# --------------------------------------------------------------------------- area-matched calibration

def new_area_curve(fire, crop, h_a, span_h, cfg, field):
    """Predicted new burned cells after span_h hours for every alpha (one solver run: with the
    interval's mean conditions alpha x time scaling is exact)."""
    shape, moisture, _, _ = cfg
    ws, wd, m = fire.mean_cond(h_a, h_a + span_h, moisture)
    rate = r2.base_rate(fire, crop, ws, m) * field
    start = fire.burned(h_a)[crop]
    times = [a * span_h * 3600.0 for a in ALPHAS]
    fields = r2.advance(signed_distance(start, fire.res), fire.res, rate, wd, ws, shape, times)
    return np.array([((fields[t] < 0) & ~start).sum() for t in times])


def pick_alpha(curve, target_cells):
    if target_cells <= 0:
        return float(ALPHAS[0])
    err = np.abs(np.log((curve + 0.5) / (target_cells + 0.5)))
    return float(ALPHAS[int(np.argmin(err))])


def buffer_growth(start, target_cells, burnable=None):
    """Uniform buffer around the fire with exactly target_cells new cells (ignores fuel and wind)."""
    if target_cells <= 0:
        return start.copy()
    d = ndimage.distance_transform_edt(~start)
    outside = d[~start]
    k = min(int(target_cells), len(outside)) - 1
    thresh = np.partition(outside, k)[k]
    return start | (d <= thresh)


# --------------------------------------------------------------------------- learned alpha

def train_alpha_model(train_fires, max_starts, out):
    rows, cfg = [], CFG["diurnal"]
    for i, fire in enumerate(train_fires, 1):
        t0, n = time.perf_counter(), 0
        for h_prev, h_start in r2.start_times(fire, max_starts):
            tg = r2.targets_for(fire, h_start)
            if "12h" not in tg:
                continue
            crop = r2.crop_for(fire, h_start, tg["12h"])
            act = r2.active_field(fire, crop, h_prev, h_start)
            obs_last = int((fire.burned(h_start)[crop] & ~fire.burned(h_prev)[crop]).sum())
            obs_next = int((fire.burned(tg["12h"])[crop] & ~fire.burned(h_start)[crop]).sum())
            if obs_next == 0 or obs_last == 0:
                continue
            a_last = pick_alpha(new_area_curve(fire, crop, h_prev, h_start - h_prev, cfg, act), obs_last)
            a_next = pick_alpha(new_area_curve(fire, crop, h_start, tg["12h"] - h_start, cfg, act), obs_next)
            rows.append({"fire_id": fire.id, **r2.features(fire, crop, h_prev, h_start, a_last, act),
                         "log_alpha_next": np.log(a_next)})
            n += 1
        print(f"  train [{i}/{len(train_fires)}] {fire.id}: {n} samples ({time.perf_counter() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "train_alpha.csv", index=False)
    if len(df) < 10:
        return None, None
    cols = [c for c in df.columns if c not in ("fire_id", "log_alpha_next")]
    X, y = df[cols].to_numpy(float), df.log_alpha_next.to_numpy()
    pred = np.empty_like(y)
    for f in df.fire_id.unique():
        m = df.fire_id.to_numpy() == f
        pred[m] = r2.Ridge().fit(X[~m], y[~m]).predict(X[m]) if (~m).sum() >= 5 else y[~m].mean()
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))
    info = {"samples": len(df), "fires": int(df.fire_id.nunique()),
            "rmse_ridge_LOFO": round(rmse(pred), 3),
            "rmse_last_alpha": round(rmse(X[:, cols.index("log_alpha_last")]), 3),
            "rmse_constant": round(rmse(np.full_like(y, y.mean())), 3),
            "median_alpha_next": round(float(np.exp(np.median(y))), 4)}
    model = r2.Ridge().fit(X, y)
    info["weights"] = {c: round(float(w), 3) for c, w in zip(cols, model.w[1:])}
    return (model, cols), info


# --------------------------------------------------------------------------- evaluation

def evaluate_fire(fire, max_starts, learned):
    rows = []
    for h_prev, h_start in r2.start_times(fire, max_starts):
        tg = r2.targets_for(fire, h_start)
        if not tg:
            continue
        crop = r2.crop_for(fire, h_start, max(tg.values()))
        start, prev = fire.burned(h_start)[crop], fire.burned(h_prev)[crop]
        obs_last = int((start & ~prev).sum())
        if obs_last == 0:
            continue
        rate_last = obs_last / (h_start - h_prev)                  # new cells per hour, last interval
        phi0 = signed_distance(start, fire.res)
        act = r2.active_field(fire, crop, h_prev, h_start)
        fields = {name: (act if c[3] else 1.0) for name, c in CFG.items()}
        alphas = {"physics": 1.0}
        for name, c in CFG.items():
            if c[2] == "area":
                alphas[name] = pick_alpha(new_area_curve(fire, crop, h_prev, h_start - h_prev, c, fields[name]), obs_last)
        if learned is not None:
            model, cols = learned
            f = r2.features(fire, crop, h_prev, h_start, alphas["diurnal"], act)
            alphas["learned"] = float(np.clip(np.exp(model.predict(np.array([[f[c] for c in cols]]))[0]), ALPHAS[0], ALPHAS[-1]))
        else:
            alphas["learned"] = alphas["diurnal"]
        run = dict(CFG, learned=CFG["diurnal"])
        fields["learned"] = act
        phis = {name: phi0 for name in run}
        h_m = h_start
        for lead, h_t in sorted(tg.items(), key=lambda kv: kv[1]):
            obs = fire.burned(h_t)[crop]
            obs_new = int((obs & ~start).sum())
            preds = {"persistence": start}
            for name, c in run.items():
                phis[name] = r2.roll(fire, crop, phis[name], h_m, h_t, c, alphas[name] * fields[name])
                preds[name] = phis[name] < 0
            h_m = h_t
            span = h_t - h_start
            preds["buffer"] = buffer_growth(start, rate_last * span)
            preds["buffer_oracle"] = buffer_growth(start, obs_new)
            a_po = pick_alpha(new_area_curve(fire, crop, h_start, span, CFG["physics"], 1.0), obs_new)
            preds["physics_oracle"] = r2.roll(fire, crop, phi0, h_start, h_t, CFG["physics"], a_po) < 0
            a_do = pick_alpha(new_area_curve(fire, crop, h_start, span, CFG["diurnal"], act), obs_new)
            preds["diurnal_oracle"] = r2.roll(fire, crop, phi0, h_start, h_t, CFG["diurnal"], a_do * act) < 0
            for name, pred in preds.items():
                iou, change, ratio, new = r2.score(pred, obs, start)
                rows.append({"fire_id": fire.id, "name": fire.meta["name"], "region": fire.meta["region"],
                             "h_start": h_start, "lead": lead, "method": name,
                             "alpha": round({"physics_oracle": a_po, "diurnal_oracle": a_do}.get(name, alphas.get(name, np.nan)), 4),
                             "iou": iou, "change_iou": change, "growth_ratio": ratio, "observed_new_cells": new})
    return rows


def paired(pf, a, b):
    if a not in pf or b not in pf:
        return None
    d = (pf[a] - pf[b]).dropna()
    lo, hi = r2.bootstrap_ci(d)
    return float(d.mean()), lo, hi, float((d > 0).mean())


def summarise(df, info, out):
    L = []
    if info:
        L.append(f"learned alpha: {info['samples']} samples, {info['fires']} TRAIN fires; leave-one-fire-out RMSE(log alpha) "
                 f"ridge {info['rmse_ridge_LOFO']} vs last alpha {info['rmse_last_alpha']} vs constant {info['rmse_constant']}; "
                 f"typical alpha {info['median_alpha_next']}")
    grew = df[df.observed_new_cells > 0]
    per_fire = {}
    for lead in r2.LEADS:
        d = grew[grew.lead == lead]
        if not len(d):
            continue
        pf = d.pivot_table(index="fire_id", columns="method", values="change_iou", aggfunc="mean")
        per_fire[lead] = pf
        L.append(f"\n--- lead {lead}: {d[d.method == 'physics'].shape[0]} forecasts from {d.fire_id.nunique()} fires (only where the fire grew)")
        L.append(f"{'method':15s} {'changeIoU':>9s} {'median':>7s} {'IoU':>6s} {'growth':>7s} | vs previous step (95% CI)        | vs buffer (95% CI)")
        prev = None
        for m in LADDER + ["buffer"] + ORACLES:
            x = d[d.method == m]
            if not len(x):
                continue
            g1 = paired(pf, m, prev) if prev and m in LADDER else None
            g2 = paired(pf, m, "buffer" if m not in ORACLES else "buffer_oracle") if m not in ("buffer", "buffer_oracle", "persistence") else None
            fmt = lambda g: "" if g is None else f"{g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]{' *' if g[1] > 0 else ''}"
            L.append(f"{m:15s} {x.change_iou.mean():9.3f} {x.change_iou.median():7.3f} {x.iou.mean():6.3f} {x.growth_ratio.median():7.2f} | "
                     f"{fmt(g1):32s} | {fmt(g2)}")
            if m in LADDER:
                prev = m
        L.append("  (oracle rows are compared with buffer_oracle: same true amount of growth, so only PLACEMENT differs)")
    L.append("\n* = 95% CI above zero (bootstrap over fires). growth = median predicted / observed new area.")
    if "24h" in per_fire:
        pf = per_fire["24h"]
        place = [paired(pf, m, "buffer_oracle") for m in ("physics_oracle", "diurnal_oracle")]
        ok_place = any(g and g[1] > 0 for g in place)
        L.append(f"\nPLACEMENT (24 h): physics places growth better than a uniform buffer: {'YES' if ok_place else 'NO'}")
        cands = [m for m in LADDER[2:] if m in pf]
        best = max(cands, key=lambda m: pf[m].mean()) if cands else None
        if best:
            gp, gb = paired(pf, best, "physics"), paired(pf, best, "buffer")
            ok = gp and gb and gp[1] > 0 and gb[1] > 0
            L.append(f"FORECAST (24 h): best method '{best}' beats physics {gp[0]:+.3f} [{gp[1]:+.3f},{gp[2]:+.3f}] and buffer "
                     f"{gb[0]:+.3f} [{gb[1]:+.3f},{gb[2]:+.3f}] -> {'ADOPT for the service' if ok else 'not clearly better than both'}")
    a = df.drop_duplicates(["fire_id", "h_start", "method"])
    for m in ("calibrated", "active", "ellipse", "diurnal", "learned"):
        x = a[a.method == m].alpha
        if len(x):
            edge = ((x <= ALPHAS[0] * 1.001) | (x >= ALPHAS[-1] * 0.999)).mean()
            L.append(f"alpha {m:10s}: median {x.median():.3f}, middle 50% {x.quantile(.25):.3f}-{x.quantile(.75):.3f}"
                     + (f"  (WARNING {edge:.0%} at the search limit)" if edge > 0.2 else ""))
    text = "\n".join(L)
    (out / "summary.txt").write_text(text)
    return text, per_fire


def figure(per_fire, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(per_fire), figsize=(7 * len(per_fire), 4.4), squeeze=False)
    for ax, (lead, pf) in zip(axes[0], per_fire.items()):
        names = [m for m in LADDER + ["buffer"] + ORACLES if m in pf]
        means = [pf[m].mean() for m in names]
        cis = [r2.bootstrap_ci(pf[m].dropna()) for m in names]
        err = np.array([[mu - lo, hi - mu] for mu, (lo, hi) in zip(means, cis)]).T
        col = {"persistence": "#9aa0a6", "buffer": "#e8a33d", "buffer_oracle": "#f3d19c"}
        colours = [col.get(m, "#c5d6f5" if m in ORACLES else "#2a6fdb") for m in names]
        ax.bar(range(len(names)), means, yerr=err, color=colours, capsize=3)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel("new-area IoU (mean over fires)")
        ax.set_title(f"lead {lead}  (light = oracle: true amount of growth)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Real-fire round 3: physics vs uniform buffer (95% CI over validation fires)")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Real-fire round 3")
    p.add_argument("--max-starts", type=int, default=5)
    p.add_argument("--train-starts", type=int, default=6)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out", default=str(REPO / "outputs" / "realfire"))
    a = p.parse_args()
    lim = 3 if a.quick else None
    if a.quick:
        a.max_starts, a.train_starts = 2, 2
    out = Path(a.out) / ("round3_quick" if a.quick else "round3")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = [Fire(d) for d in r2.load("train", lim)]
    print(f"step 1/2: learning alpha from {len(train)} TRAIN fires (area-matched)", flush=True)
    learned, info = train_alpha_model(train, a.train_starts, out)
    if info:
        print(f"  LOFO RMSE(log alpha): ridge {info['rmse_ridge_LOFO']}, last alpha {info['rmse_last_alpha']}, "
              f"constant {info['rmse_constant']}", flush=True)
        (out / "learned_alpha.json").write_text(json.dumps(info, indent=2))
    val = r2.load("validation", lim)
    print(f"step 2/2: {len(val)} VALIDATION fires, up to {a.max_starts} starts each", flush=True)
    rows = []
    for i, d in enumerate(val, 1):
        t0 = time.perf_counter()
        try:
            fr = evaluate_fire(Fire(d), a.max_starts, learned)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(val)}] {d.name}: FAILED {exc}", flush=True)
            continue
        rows += fr
        if fr:
            g = pd.DataFrame(fr).query("observed_new_cells > 0").groupby("method").change_iou.mean()
            print(f"  [{i}/{len(val)}] {d.name}: physics {g.get('physics', np.nan):.2f} calibrated {g.get('calibrated', np.nan):.2f} "
                  f"diurnal {g.get('diurnal', np.nan):.2f} buffer {g.get('buffer', np.nan):.2f} | oracles: buffer "
                  f"{g.get('buffer_oracle', np.nan):.2f} physics {g.get('physics_oracle', np.nan):.2f} "
                  f"diurnal {g.get('diurnal_oracle', np.nan):.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    if not rows:
        print("no forecasts made")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecasts.csv", index=False)
    text, per_fire = summarise(df, info, out)
    print("\n" + text)
    figure(per_fire, out)
    print(f"\n({(time.perf_counter() - started) / 60:.0f} min; summary.txt and ladder.png in {out})")


if __name__ == "__main__":
    main()
