"""FINAL held-out test: score the frozen method ONCE on the test fires, plus the placement control.

    python -m pinn_v2.realfire_final --fetch-only   # step 1 (~15 min): download data for the test fires
    python -m pinn_v2.realfire_final --quick        # check it runs on 2 VALIDATION fires (no test fire is scored)
    python -m pinn_v2.realfire_final --final        # THE run (~1.5-2 h). Run it once; do not tune afterwards.

Frozen before any test fire is scored (Sun 18 Oct 2026, after round 3 on the validation fires):
  primary method     learned   (area-calibrated, active fronts, elliptical, diurnal, alpha from the ridge model)
  secondary method   active    (area-calibrated, active fronts; the simplest method that beat the buffer)
  baselines          persistence, physics (as deployed), buffer (uniform, last-interval growth rate)
  leads              12 h and 24 h (VIIRS revisit); 1-6 h are not validated
  metric             new-area IoU against gap-filled VIIRS fire shapes, mean over fires, 95 % bootstrap CI
  speed rule         fires whose VIIRS burned area exceeds 2000 km2 run on a 360 m grid instead of 180 m
                     (August Complex size); fixed now, before seeing any test result

Placement control (also run here): active_buffer grows uniformly but only from the ACTIVE edges, by the
same amount as the forecast (active_buffer) or the true amount (active_buffer_oracle). Physics adds
placement skill beyond "where the fire is active" only if diurnal_oracle beats active_buffer_oracle.

The learned model is retrained on the 30 TRAIN fires (exactly as in round 3) and exported with
everything the service needs to use it: pinn_service/static/learned_alpha.json.
Writes outputs/realfire/final_test/{forecasts.csv, summary.txt, ladder.png}.
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
from pinn_v2 import realfire_round3 as r3

REPO = Path(__file__).resolve().parent.parent
BIG_FIRE_KM2 = 2000.0
METHODS_MAIN = ["persistence", "physics", "buffer", "active_buffer", "active", "learned"]
ORACLES = ["buffer_oracle", "active_buffer_oracle", "physics_oracle", "diurnal_oracle"]


# --------------------------------------------------------------------------- data for the test fires

def fetch_test_fires(min_detections=100):
    from pinn_v2 import fetch_fire_data as ffd
    from pinn_v2 import fetch_fuel as ff

    cat = pd.read_csv(REPO / "data" / "fire_catalogue.csv", dtype={"cont_date": str, "alarm_date": str})
    test = cat[(cat.split == "test")]
    if "viirs_detections" in test:
        test = test[test.viirs_detections >= min_detections]
    print(f"{len(test)} test fires with >= {min_detections} detections (data only; nothing is scored)")
    det = ffd.load_archive(REPO / "data" / "firms_archive")
    for i, (_, fire) in enumerate(test.sort_values("acres", ascending=False).iterrows(), 1):
        res = ffd.fetch_fire(fire, det, REPO / "data" / "fires")
        print(f"[{i}/{len(test)}] {fire.fire_id}: {res.get('status')}", flush=True)
    for fid in test.fire_id:
        folder = REPO / "data" / "fires" / fid
        if not (folder / "arrival.npz").exists():
            continue
        for attempt in range(4):                                   # LANDFIRE throttles bursts of requests
            try:
                res = ff.fetch_one(folder)
                break
            except Exception as exc:  # noqa: BLE001
                res = {"status": f"FAILED: {exc}"}
                time.sleep(10 * (attempt + 1))
        print(f"  fuel {fid}: {res.get('status')}", flush=True)


# --------------------------------------------------------------------------- learned model export

def export_model(learned, info, path: Path):
    model, cols = learned
    payload = {"features": cols, "mu": model.mu.tolist(), "sd": model.sd.tolist(), "w": model.w.tolist(),
               "lambda": model.lam, "alpha_min": float(r3.ALPHAS[0]), "alpha_max": float(r3.ALPHAS[-1]),
               "fallback_alpha": info["median_alpha_next"], "trained_on": f"{info['fires']} train fires, {info['samples']} samples",
               "rmse_log_alpha_LOFO": info["rmse_ridge_LOFO"], "rmse_log_alpha_constant": info["rmse_constant"],
               "config": "ellipse shape, diurnal (Simard EMC) moisture, active fronts, area-matched calibration"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"  exported learned model -> {path}")


# --------------------------------------------------------------------------- evaluation

def active_buffer(start, act, target_cells):
    """Uniform growth by target_cells, only into cells the active-front mask allows."""
    if target_cells <= 0:
        return start.copy()
    allowed = ~start & (act > 0)
    if not allowed.any():
        return start.copy()
    d = ndimage.distance_transform_edt(~start)
    vals = d[allowed]
    k = min(int(target_cells), len(vals)) - 1
    return start | (allowed & (d <= np.partition(vals, k)[k]))


def evaluate_fire(fire, max_starts, learned):
    rows = []
    cfg = r3.CFG
    for h_prev, h_start in r2.start_times(fire, max_starts):
        tg = r2.targets_for(fire, h_start)
        if not tg:
            continue
        crop = r2.crop_for(fire, h_start, max(tg.values()))
        start, prev = fire.burned(h_start)[crop], fire.burned(h_prev)[crop]
        obs_last = int((start & ~prev).sum())
        if obs_last == 0:
            continue
        rate_last = obs_last / (h_start - h_prev)
        phi0 = signed_distance(start, fire.res)
        act = r2.active_field(fire, crop, h_prev, h_start)
        a_active = r3.pick_alpha(r3.new_area_curve(fire, crop, h_prev, h_start - h_prev, cfg["active"], act), obs_last)
        a_diurnal = r3.pick_alpha(r3.new_area_curve(fire, crop, h_prev, h_start - h_prev, cfg["diurnal"], act), obs_last)
        model, cols = learned
        f = r2.features(fire, crop, h_prev, h_start, a_diurnal, act)
        a_learned = float(np.clip(np.exp(model.predict(np.array([[f[c] for c in cols]]))[0]), r3.ALPHAS[0], r3.ALPHAS[-1]))
        run = {"physics": (cfg["physics"], 1.0), "active": (cfg["active"], a_active * act),
               "learned": (cfg["diurnal"], a_learned * act)}
        alphas = {"physics": 1.0, "active": a_active, "learned": a_learned}
        phis = {k: phi0 for k in run}
        h_m = h_start
        for lead, h_t in sorted(tg.items(), key=lambda kv: kv[1]):
            obs = fire.burned(h_t)[crop]
            obs_new = int((obs & ~start).sum())
            span = h_t - h_start
            preds = {"persistence": start}
            for k, (c, mult) in run.items():
                phis[k] = r2.roll(fire, crop, phis[k], h_m, h_t, c, mult)
                preds[k] = phis[k] < 0
            h_m = h_t
            preds["buffer"] = r3.buffer_growth(start, rate_last * span)
            preds["active_buffer"] = active_buffer(start, act, rate_last * span)
            preds["buffer_oracle"] = r3.buffer_growth(start, obs_new)
            preds["active_buffer_oracle"] = active_buffer(start, act, obs_new)
            a_po = r3.pick_alpha(r3.new_area_curve(fire, crop, h_start, span, cfg["physics"], 1.0), obs_new)
            preds["physics_oracle"] = r2.roll(fire, crop, phi0, h_start, h_t, cfg["physics"], a_po) < 0
            a_do = r3.pick_alpha(r3.new_area_curve(fire, crop, h_start, span, cfg["diurnal"], act), obs_new)
            preds["diurnal_oracle"] = r2.roll(fire, crop, phi0, h_start, h_t, cfg["diurnal"], a_do * act) < 0
            for name, pred in preds.items():
                iou, change, ratio, new = r2.score(pred, obs, start)
                rows.append({"fire_id": fire.id, "name": fire.meta["name"], "region": fire.meta["region"],
                             "h_start": h_start, "lead": lead, "method": name,
                             "alpha": alphas.get(name, {"physics_oracle": a_po, "diurnal_oracle": a_do}.get(name, np.nan)),
                             "iou": iou, "change_iou": change, "growth_ratio": ratio, "observed_new_cells": new,
                             "grid_m": fire.res})
    return rows


def summarise(df, info, out, split):
    L = [f"FINAL {split.upper()} RESULTS (frozen method: learned; secondary: active)",
         f"learned model: {info['samples']} samples from {info['fires']} TRAIN fires; LOFO RMSE(log alpha) {info['rmse_ridge_LOFO']} "
         f"vs constant {info['rmse_constant']}"]
    grew = df[df.observed_new_cells > 0]
    per = {}
    fmt = lambda g: "" if g is None else f"{g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]{' *' if g[1] > 0 else ''} ({g[3]:.0%} of fires)"
    for lead in r2.LEADS:
        d = grew[grew.lead == lead]
        if not len(d):
            continue
        pf = d.pivot_table(index="fire_id", columns="method", values="change_iou", aggfunc="mean")
        per[lead] = pf
        L.append(f"\n--- lead {lead}: {d[d.method == 'physics'].shape[0]} forecasts from {d.fire_id.nunique()} fires (only where the fire grew)")
        L.append(f"{'method':22s} {'newIoU':>7s} {'median':>7s} {'IoU':>6s} {'growth':>7s}")
        for m in METHODS_MAIN + ORACLES:
            x = d[d.method == m]
            if len(x):
                L.append(f"{m:22s} {x.change_iou.mean():7.3f} {x.change_iou.median():7.3f} {x.iou.mean():6.3f} {x.growth_ratio.median():7.2f}")
        L.append("  key comparisons (paired over fires, 95% CI; * = CI above zero):")
        for a, b in (("learned", "physics"), ("learned", "buffer"), ("learned", "active_buffer"), ("active", "physics"),
                     ("active", "buffer"), ("diurnal_oracle", "buffer_oracle"), ("diurnal_oracle", "active_buffer_oracle")):
            L.append(f"    {a:16s} vs {b:22s} {fmt(r3.paired(pf, a, b))}")
    if "24h" in per:
        pf = per["24h"]
        g = r3.paired(pf, "diurnal_oracle", "active_buffer_oracle")
        L.append(f"\nPLACEMENT CONTROL (24 h): physics adds placement skill beyond the active-front mask: "
                 f"{'YES' if g and g[1] > 0 else 'NO'}")
        for m in ("learned", "active"):
            gp, gb = r3.paired(pf, m, "physics"), r3.paired(pf, m, "buffer")
            L.append(f"{m}: beats deployed physics {'YES' if gp and gp[1] > 0 else 'NO'}, beats uniform buffer {'YES' if gb and gb[1] > 0 else 'NO'}")
    big = df[df.grid_m > 200].fire_id.unique()
    if len(big):
        L.append(f"\nfires run on the 360 m grid (speed rule): {', '.join(big)}")
    per_fire = grew[grew.lead == "24h"].pivot_table(index=["fire_id", "name"], columns="method", values="change_iou", aggfunc="mean")
    if len(per_fire):
        cols = [c for c in ("physics", "buffer", "active", "learned") if c in per_fire]
        L.append("\nper fire, 24 h new-area IoU:\n" + per_fire[cols].round(3).to_string())
    text = "\n".join(L)
    (out / "summary.txt").write_text(text)
    return text, per


def figure(per, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(per), figsize=(7 * len(per), 4.4), squeeze=False)
    for ax, (lead, pf) in zip(axes[0], per.items()):
        names = [m for m in METHODS_MAIN + ORACLES if m in pf]
        means = [pf[m].mean() for m in names]
        cis = [r2.bootstrap_ci(pf[m].dropna()) for m in names]
        err = np.array([[mu - lo, hi - mu] for mu, (lo, hi) in zip(means, cis)]).T
        col = {"persistence": "#9aa0a6", "physics": "#7f8c9a", "buffer": "#e8a33d", "active_buffer": "#d98b2b",
               "learned": "#2a6fdb", "active": "#5b8fe6"}
        ax.bar(range(len(names)), means, yerr=err, capsize=3,
               color=[col.get(m, "#d6dceb") for m in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel("new-area IoU (mean over fires)")
        ax.set_title(f"lead {lead}  (pale = oracle, told the true amount of growth)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Held-out test fires: frozen method vs baselines (95% CI over fires)")
    fig.tight_layout()
    fig.savefig(out / "ladder.png", dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Final held-out test")
    p.add_argument("--fetch-only", action="store_true")
    p.add_argument("--quick", action="store_true", help="12 train + 2 validation fires, 2 starts (no test fire is touched)")
    p.add_argument("--final", action="store_true", help="score the TEST fires (once)")
    p.add_argument("--max-starts", type=int, default=5)
    a = p.parse_args()
    if a.fetch_only:
        fetch_test_fires()
        return
    if not (a.final or a.quick):
        raise SystemExit("choose --fetch-only, --quick or --final")
    split = "validation" if a.quick else "test"
    out = REPO / "outputs" / "realfire" / ("final_quick" if a.quick else "final_test")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = [r3.Fire(d) for d in r2.load("train", 12 if a.quick else None)]
    print(f"step 1/2: retraining the learned alpha on {len(train)} TRAIN fires (as in round 3)", flush=True)
    learned, info = r3.train_alpha_model(train, 2 if a.quick else 6, out)
    if learned is None:
        raise SystemExit("not enough training samples")
    export_model(learned, info, (out if a.quick else REPO / "pinn_service" / "static") / "learned_alpha.json")
    folders = r2.load(split, 2 if a.quick else None)
    if a.final and not folders:
        raise SystemExit("no test fires with data: run --fetch-only first")
    print(f"step 2/2: {len(folders)} {split.upper()} fires, up to {2 if a.quick else a.max_starts} starts each", flush=True)
    rows = []
    for i, d in enumerate(folders, 1):
        t0 = time.perf_counter()
        meta = json.loads((d / "meta.json").read_text())
        factor = 4 if meta.get("burned_km2_viirs", 0) > BIG_FIRE_KM2 else 2
        try:
            fr = evaluate_fire(r3.Fire(d, factor), 2 if a.quick else a.max_starts, learned)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(folders)}] {d.name}: FAILED {exc}", flush=True)
            continue
        rows += fr
        if fr:
            g = pd.DataFrame(fr).query("observed_new_cells > 0").groupby("method").change_iou.mean()
            print(f"  [{i}/{len(folders)}] {d.name} ({factor * 90} m): physics {g.get('physics', np.nan):.2f} buffer "
                  f"{g.get('buffer', np.nan):.2f} active {g.get('active', np.nan):.2f} learned {g.get('learned', np.nan):.2f} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    if not rows:
        print("no forecasts made")
        return
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecasts.csv", index=False)
    text, per = summarise(df, info, out, split)
    print("\n" + text)
    figure(per, out)
    print(f"\n({(time.perf_counter() - started) / 60:.0f} min; summary.txt and ladder.png in {out})")


if __name__ == "__main__":
    main()
