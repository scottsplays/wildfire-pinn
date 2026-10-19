"""v0.4: more training fires, hybrid physics/geometry forecasts, a FRESH 2026 test.

The 19 test fires used by realfire_final are spent (scored once on 18 Oct 2026). v0.4 is
developed on the TRAIN and VALIDATION fires only, frozen, then scored ONCE on 2026-season
California fires that no method has seen.

    python -m pinn_v2.realfire_v4 --fetch-more          # long (~2-4 h, resumable): data + fuel for every
                                                        #   train/validation fire with >= 100 VIIRS detections
    python -m pinn_v2.realfire_v4 --dev --quick         # minutes: checks the pipeline on a few fires
    python -m pinn_v2.realfire_v4 --dev                 # long (~3-5 h): train on all TRAIN fires, compare
                                                        #   candidates on VALIDATION, FREEZE the winner
    python -m pinn_v2.realfire_v4 --fetch-2026          # ~1 h: 2026 fires from data/firms_2026/*.csv
    python -m pinn_v2.realfire_v4 --final-2026          # ~1 h: score the frozen method ONCE on 2026 fires

Forecast = an AMOUNT of new burned area x a PLACEMENT rule (pre-registered here, 18 Oct 2026):

  amounts     last     observed growth rate of the last interval x span (no model)
              alpha    level set rolled with the learned alpha (v0.3 physics: fuel, slope, hourly wind,
                       equilibrium moisture, Anderson ellipse, active fronts); amount = its new area
              direct   ridge regression for log growth rate. Inputs: the v0.3 features, the lead, and
                       PHYS = the new area of the level set rolled with the last-interval alpha
                       (a physics-informed feature)
              true     the observed amount (oracle, a diagnostic only, never selectable)
  placements  levelset where the level set burns (v0.3; only with amount alpha)
              edge     nearest cells to the fire that the active-front mask allows (geometry)
              edgefuel edge, and burnable LANDFIRE fuel only
              soft25 / soft50   rank blend of distance (75 / 50 %) and physics arrival time (25 / 50 %)
                       on burnable allowed cells: the physics tilts the geometric growth

Selection rule (VALIDATION, fixed before any result): the candidate with the highest mean 24 h
new-area IoU over fires; candidates within 0.005 of the best count as tied and the simplest wins,
in the order  edge:last < edgefuel:last < edge:direct < edgefuel:direct < edge:alpha <
edgefuel:alpha < levelset:alpha < soft25:direct < soft25:alpha < soft50:direct < soft50:alpha.
The winner, its amount-model weights and its ensemble spread (std of log(predicted/observed)
amount on validation, per lead) are written to outputs/realfire/v4_dev/frozen.json.

Probabilities: the ensemble uses ten amount quantiles exp(sigma z_k); with a fixed placement order
a cell's probability is the share of members that reach it. The 2026 run reports reliability
(predicted vs observed frequency in 10 bins) and a Brier skill score vs the validation base rate.
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
from pinn_service.physics import NONBURNABLE
from pinn_v2 import realfire_round2 as r2
from pinn_v2 import realfire_round3 as r3

REPO = Path(__file__).resolve().parent.parent
BIG_FIRE_KM2 = 2000.0
AMOUNTS = ["last", "direct", "alpha"]
PLACES = ["edge", "edgefuel", "soft25", "soft50"]
SIMPLICITY = ["edge:last", "edgefuel:last", "edge:direct", "edgefuel:direct", "edge:alpha", "edgefuel:alpha",
              "levelset:alpha", "soft25:direct", "soft25:alpha", "soft50:direct", "soft50:alpha"]
ORACLES = ["edge:true", "edgefuel:true", "soft25:true", "soft50:true"]
BASELINES = ["persistence", "physics", "buffer", "v03"]
TIE = 0.005
ZONE_M = 6000.0
Z10 = np.array([-1.645, -1.036, -0.674, -0.385, -0.126, 0.126, 0.385, 0.674, 1.036, 1.645])  # N(0,1) deciles
LEAD_ORDER = ["12h", "24h"]


def nonburnable(fuel):
    return np.isin(np.rint(np.nan_to_num(fuel)).astype(int), list(NONBURNABLE))


def load_fire(d: Path) -> r3.Fire:
    meta = json.loads((d / "meta.json").read_text())
    factor = 4 if meta.get("burned_km2_viirs", 0) > BIG_FIRE_KM2 else 2
    return r3.Fire(d, factor)


# --------------------------------------------------------------------------- level-set roll with arrival times

def roll_arrival(fire, crop, start, h_start, spans, mult):
    """Roll the v0.3 level set (ellipse, diurnal moisture) from h_start; record the hour each cell burns.
    Stops at the longest span. Returns arrival (hours after h_start, inf = not reached)."""
    cfg = r3.CFG["diurnal"]
    stops = sorted(set([float(s) for s in spans] + [float(k) for k in range(1, int(np.ceil(max(spans))))]))
    phi = signed_distance(start, fire.res)
    arrival = np.where(start, 0.0, np.inf)
    t = 0.0
    for s in stops:
        if s <= t:
            continue
        phi = r2.roll(fire, crop, phi, h_start + t, h_start + s, cfg, mult)
        new = (phi < 0) & np.isinf(arrival)
        arrival[new] = s
        t = s
    return arrival


def rank_pct(values, mask):
    out = np.ones(values.shape)
    v = values[mask]
    if v.size:
        order = np.argsort(np.argsort(v, kind="stable"), kind="stable")
        out[mask] = order / max(v.size - 1, 1)
    return out


def placement_score(name, start, act, fuel, arrival, res):
    """(allowed mask, score): the forecast takes the n allowed cells with the lowest score."""
    allowed = ~start & (act > 0)
    if name != "edge":
        allowed &= ~nonburnable(fuel)
    d = ndimage.distance_transform_edt(~start) * res
    if name in ("edge", "edgefuel"):
        return allowed, d
    w = {"soft25": 0.25, "soft50": 0.5}[name]
    t = np.where(np.isfinite(arrival), arrival, np.nanmax(np.where(np.isfinite(arrival), arrival, 0)) + 1.0)
    return allowed, (1 - w) * rank_pct(d, allowed) + w * rank_pct(t, allowed)


def place(start, allowed, score, n):
    n = int(round(max(n, 0)))
    out = start.copy()
    idx = np.flatnonzero(allowed)
    if n <= 0 or not idx.size:
        return out
    n = min(n, idx.size)
    pick = idx[np.argpartition(score.ravel()[idx], n - 1)[:n]]
    out.ravel()[pick] = True
    return out


def cell_rank(allowed, score):
    """0-based rank of each allowed cell in placement order (inf elsewhere)."""
    rank = np.full(allowed.shape, np.inf)
    idx = np.flatnonzero(allowed)
    order = np.argsort(score.ravel()[idx], kind="stable")
    rank.ravel()[idx[order]] = np.arange(idx.size)
    return rank


# --------------------------------------------------------------------------- per-start quantities

def start_state(fire, h_prev, h_start, tg, alpha_model):
    """Everything known at the forecast start, plus the physics feature and the learned-alpha arrival map."""
    crop = r2.crop_for(fire, h_start, max(tg.values()))
    start, prev = fire.burned(h_start)[crop], fire.burned(h_prev)[crop]
    obs_last = int((start & ~prev).sum())
    if obs_last == 0:
        return None
    act = r2.active_field(fire, crop, h_prev, h_start)
    a_last = r3.pick_alpha(r3.new_area_curve(fire, crop, h_prev, h_start - h_prev, r3.CFG["diurnal"], act), obs_last)
    feats = r2.features(fire, crop, h_prev, h_start, a_last, act)
    spans = {lead: h_t - h_start for lead, h_t in tg.items()}
    arr_last = roll_arrival(fire, crop, start, h_start, list(spans.values()), a_last * act)
    phys = {lead: int(((arr_last <= s) & ~start).sum()) for lead, s in spans.items()}
    st = {"crop": crop, "start": start, "prev": prev, "act": act, "obs_last": obs_last, "a_last": a_last,
          "feats": feats, "spans": spans, "phys": phys, "rate_last": obs_last / (h_start - h_prev),
          "cell_km2": fire.res ** 2 / 1e6, "fuel": fire.fuel[crop]}
    if alpha_model is not None:
        model, cols = alpha_model
        a = float(np.clip(np.exp(model.predict(np.array([[feats[c] for c in cols]]))[0]), r3.ALPHAS[0], r3.ALPHAS[-1]))
        st["a_learned"] = a
        st["arrival"] = roll_arrival(fire, crop, start, h_start, list(spans.values()), a * act)
    return st


def amount_row(st, lead):
    span = st["spans"][lead]
    return {**st["feats"], "lead_h": span,
            "log_phys_rate": float(np.log(st["phys"][lead] * st["cell_km2"] / span + 1e-3))}


# --------------------------------------------------------------------------- training (cached per fire)

def train_rows(fire, max_starts, cache: Path):
    f = cache / f"{fire.id}.json"
    if f.exists():
        return json.loads(f.read_text())
    alpha_rows, amount_rows = [], []
    for h_prev, h_start in r2.start_times(fire, max_starts):
        tg = r2.targets_for(fire, h_start)
        if not tg:
            continue
        st = start_state(fire, h_prev, h_start, tg, None)
        if st is None:
            continue
        crop, start = st["crop"], st["start"]
        for lead, h_t in tg.items():
            obs_new = int((fire.burned(h_t)[crop] & ~start).sum())
            span = st["spans"][lead]
            amount_rows.append({"fire_id": fire.id, **amount_row(st, lead),
                                "target": float(np.log(obs_new * st["cell_km2"] / span + 1e-3))})
            if lead == "12h" and obs_new > 0:
                a_next = r3.pick_alpha(r3.new_area_curve(fire, crop, h_start, span, r3.CFG["diurnal"], st["act"]), obs_new)
                alpha_rows.append({"fire_id": fire.id, **st["feats"], "target": float(np.log(a_next))})
    out = {"alpha": alpha_rows, "amount": amount_rows}
    f.write_text(json.dumps(out))
    return out


def fit_lofo(df):
    cols = [c for c in df.columns if c not in ("fire_id", "target")]
    X, y = df[cols].to_numpy(float), df.target.to_numpy()
    pred = np.empty_like(y)
    for fid in df.fire_id.unique():
        m = df.fire_id.to_numpy() == fid
        pred[m] = r2.Ridge().fit(X[~m], y[~m]).predict(X[m]) if (~m).sum() >= 5 else y[~m].mean()
    rmse = lambda p: round(float(np.sqrt(np.mean((p - y) ** 2))), 3)  # noqa: E731
    info = {"samples": len(df), "fires": int(df.fire_id.nunique()), "rmse_LOFO": rmse(pred),
            "rmse_constant": rmse(np.full_like(y, y.mean())), "median_target": float(np.median(y))}
    model = r2.Ridge().fit(X, y)
    info["weights"] = {c: round(float(w), 3) for c, w in zip(cols, model.w[1:])}
    return (model, cols), info


def train_models(train_dirs, max_starts, cache):
    cache.mkdir(parents=True, exist_ok=True)
    A, M = [], []
    for i, d in enumerate(train_dirs, 1):
        t0 = time.perf_counter()
        try:
            rows = train_rows(load_fire(d), max_starts, cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  train [{i}/{len(train_dirs)}] {d.name}: FAILED {exc}", flush=True)
            continue
        A += rows["alpha"]
        M += rows["amount"]
        print(f"  train [{i}/{len(train_dirs)}] {d.name}: {len(rows['amount'])} amount / {len(rows['alpha'])} alpha samples "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    if len(A) < 10 or len(M) < 10:
        raise SystemExit("not enough training samples")
    alpha_model, alpha_info = fit_lofo(pd.DataFrame(A))
    amount_model, amount_info = fit_lofo(pd.DataFrame(M))
    return alpha_model, alpha_info, amount_model, amount_info


def model_json(model, info, kind):
    (m, cols) = model
    return {"kind": kind, "features": cols, "mu": m.mu.tolist(), "sd": m.sd.tolist(), "w": m.w.tolist(),
            "lambda": m.lam, "info": info}


def model_from_json(js):
    m = r2.Ridge(js["lambda"])
    m.mu, m.sd, m.w = np.array(js["mu"]), np.array(js["sd"]), np.array(js["w"])
    return m, js["features"]


# --------------------------------------------------------------------------- evaluation of one fire

def evaluate_fire(fire, max_starts, alpha_model, amount_model, v03_model=None, only=None):
    """One row per (start, lead, method). `only` restricts the candidate list (the 2026 run)."""
    rows, members = [], []
    for h_prev, h_start in r2.start_times(fire, max_starts):
        tg = r2.targets_for(fire, h_start)
        if not tg:
            continue
        st = start_state(fire, h_prev, h_start, tg, alpha_model)
        if st is None:
            continue
        crop, start, act, fuel = st["crop"], st["start"], st["act"], st["fuel"]
        v03_arrival = None
        if v03_model is not None:
            model, cols = v03_model
            a03 = float(np.clip(np.exp(model.predict(np.array([[st["feats"][c] for c in cols]]))[0]), r3.ALPHAS[0], r3.ALPHAS[-1]))
            v03_arrival = roll_arrival(fire, crop, start, h_start, list(st["spans"].values()), a03 * act)
        phi_phys, h_m = signed_distance(start, fire.res), h_start
        for lead, h_t in sorted(tg.items(), key=lambda kv: kv[1]):
            span = st["spans"][lead]
            obs = fire.burned(h_t)[crop]
            obs_new = int((obs & ~start).sum())
            m, cols = amount_model
            r = amount_row(st, lead)
            amounts = {"last": st["rate_last"] * span,
                       "direct": float(np.exp(m.predict(np.array([[r[c] for c in cols]]))[0]) - 1e-3) * span / st["cell_km2"],
                       "alpha": int(((st["arrival"] <= span) & ~start).sum()),
                       "true": obs_new}
            amounts["direct"] = max(amounts["direct"], 0.0)
            preds = {"persistence": start, "levelset:alpha": start | (st["arrival"] <= span),
                     "buffer": r3.buffer_growth(start, st["rate_last"] * span)}
            phi_phys = r2.roll(fire, crop, phi_phys, h_m, h_t, r3.CFG["physics"], 1.0)
            h_m = h_t
            preds["physics"] = phi_phys < 0
            if v03_arrival is not None:
                preds["v03"] = start | (v03_arrival <= span)
            scores = {p: placement_score(p, start, act, fuel, st["arrival"], fire.res) for p in PLACES}
            if lead == min(tg, key=lambda k: tg[k]):
                zone_full = ~start & (ndimage.distance_transform_edt(~start) * fire.res <= ZONE_M)
                rr, cc = np.where(zone_full | start)
                box = np.s_[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
                zone = zone_full[box]
                ranks = {p: cell_rank(*scores[p])[box][zone].astype(np.float32) for p in PLACES}
            for p in PLACES:
                for a in AMOUNTS + ["true"]:
                    name = f"{p}:{a}"
                    if only is None or name in only or name in ORACLES:
                        preds[name] = place(start, *scores[p], amounts[a])
            for name, pred in preds.items():
                if only is not None and name not in only and name not in BASELINES and name not in ORACLES:
                    continue
                iou, change, ratio, new = r2.score(pred, obs, start)
                amt = amounts.get(name.split(":")[-1]) if ":" in name else np.nan
                rows.append({"fire_id": fire.id, "name": fire.meta.get("name", fire.id), "region": fire.meta.get("region", ""),
                             "h_start": h_start, "lead": lead, "method": name, "iou": iou, "change_iou": change,
                             "growth_ratio": ratio, "observed_new_cells": new,
                             "log_amount_ratio": float(np.log((amt + 1) / (obs_new + 1))) if amt == amt else np.nan,
                             "grid_m": fire.res})
            members.append({"fire_id": fire.id, "lead": lead, "ranks": ranks,
                            "obs_new": (obs & ~start)[box][zone], "amounts": amounts})
    return rows, members


# --------------------------------------------------------------------------- probabilities

def zone_probability(mem, method, sigma):
    """Ensemble probability for the cells within ZONE_M of the fire (start excluded)."""
    place_name, amount_name = method.split(":")
    if place_name == "levelset":
        return None
    n = mem["amounts"][amount_name] * np.exp(sigma * Z10)
    return (mem["ranks"][place_name][:, None] < n[None, :]).mean(axis=1)


def reliability(members, method, sigmas, base_rate=None, calib=None):
    """Reliability table and Brier skill. `calib` = (x, y) points of a monotone map from raw ensemble
    share to observed frequency, fitted on VALIDATION (applied unchanged to the 2026 fires)."""
    P, O = [], []
    for mem in members:
        p = zone_probability(mem, method, sigmas.get(mem["lead"], 1.0))
        if p is None:
            return None
        P.append(p)
        O.append(mem["obs_new"])
    P, O = np.concatenate(P), np.concatenate(O).astype(float)
    if calib is not None:
        P = np.interp(P, calib[0], calib[1])
    bins = np.minimum((P * 10).astype(int), 9)
    table = []
    for b in range(10):
        m = bins == b
        if m.sum():
            table.append({"bin": f"{b / 10:.1f}-{(b + 1) / 10:.1f}", "cells": int(m.sum()),
                          "predicted": round(float(P[m].mean()), 3), "observed": round(float(O[m].mean()), 3)})
    clim = float(O.mean()) if base_rate is None else base_rate
    bs, bs_ref = float(np.mean((P - O) ** 2)), float(np.mean((clim - O) ** 2))
    return {"table": table, "brier": round(bs, 5), "brier_base_rate": round(bs_ref, 5),
            "brier_skill": round(1 - bs / bs_ref, 3) if bs_ref > 0 else None, "base_rate": round(clim, 5),
            "zone_cells": int(len(P))}


# --------------------------------------------------------------------------- summaries

def per_fire(df, lead):
    d = df[(df.lead == lead) & (df.observed_new_cells > 0)]
    return d.pivot_table(index="fire_id", columns="method", values="change_iou", aggfunc="mean")


def table(df, methods, title):
    L = [title]
    for lead in LEAD_ORDER:
        d = df[(df.lead == lead) & (df.observed_new_cells > 0)]
        if not len(d):
            continue
        pf = per_fire(df, lead)
        L.append(f"\n--- lead {lead}: {d.fire_id.nunique()} fires, {d[d.method == 'persistence'].shape[0]} forecasts")
        L.append(f"{'method':18s} {'newIoU':>7s} {'95% CI':>15s} {'IoU':>6s} {'growth':>7s}")
        for m in methods:
            x = d[d.method == m]
            if len(x) and m in pf:
                lo, hi = r2.bootstrap_ci(pf[m].dropna())
                L.append(f"{m:18s} {pf[m].mean():7.3f}  [{lo:.3f}, {hi:.3f}] {x.iou.mean():6.3f} {x.growth_ratio.median():7.2f}")
    return "\n".join(L)


def select(df):
    pf = per_fire(df, "24h")
    means = {m: float(pf[m].mean()) for m in SIMPLICITY if m in pf}
    best = max(means.values())
    tied = [m for m in SIMPLICITY if m in means and means[m] >= best - TIE]
    return tied[0], means


def sigmas_for(df, method):
    out = {}
    for lead in LEAD_ORDER:
        x = df[(df.method == method) & (df.lead == lead) & (df.observed_new_cells > 0)].log_amount_ratio.dropna()
        out[lead] = round(float(x.std()), 3) if len(x) > 2 else 1.0
    return out


# --------------------------------------------------------------------------- modes

def dirs(split, limit=None):
    return r2.load(split, limit)


def mode_fetch_more(limit):
    from pinn_v2 import fetch_fire_data as ffd
    from pinn_v2 import fetch_fuel as ff

    cat = pd.read_csv(REPO / "data" / "fire_catalogue.csv", dtype={"cont_date": str, "alarm_date": str})
    sel = cat[cat.split.isin(["train", "validation"]) & (cat.viirs_detections >= 100)].sort_values("acres", ascending=False)
    todo = [r for _, r in sel.iterrows() if not (REPO / "data" / "fires" / r.fire_id / "static.npz").exists()]
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} train/validation fires to fetch (data + fuel); already done fires are skipped", flush=True)
    det = ffd.load_archive(REPO / "data" / "firms_archive")
    fetch_list(todo, det, ffd, ff)


def fetch_list(todo, det, ffd, ff):
    started = time.perf_counter()
    for i, fire in enumerate(todo, 1):
        try:
            res = ffd.fetch_fire(fire, det, REPO / "data" / "fires")
        except Exception as exc:  # noqa: BLE001
            res = {"status": f"FAILED: {exc}"}
        folder = REPO / "data" / "fires" / fire.fire_id
        fuel = "skipped"
        if (folder / "arrival.npz").exists():
            for attempt in range(4):
                try:
                    fuel = ff.fetch_one(folder).get("status")
                    break
                except Exception as exc:  # noqa: BLE001
                    fuel = f"FAILED: {exc}"
                    time.sleep(10 * (attempt + 1))
        eta = (time.perf_counter() - started) / i * (len(todo) - i) / 60
        print(f"[{i}/{len(todo)}] {fire.fire_id}: {res.get('status')} | fuel {fuel} | ~{eta:.0f} min left", flush=True)


def mode_dev(quick, max_starts):
    out = REPO / "outputs" / "realfire" / ("v4_dev_quick" if quick else "v4_dev")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    train = dirs("train", 8 if quick else None)
    val = dirs("validation", 3 if quick else None)
    print(f"step 1/3: training on {len(train)} TRAIN fires (rows cached in outputs/realfire/v4_cache)", flush=True)
    alpha_model, alpha_info, amount_model, amount_info = train_models(train, 2 if quick else 6,
                                                                      REPO / "outputs" / "realfire" / "v4_cache")
    print(f"  alpha ridge: {alpha_info['samples']} samples / {alpha_info['fires']} fires, LOFO RMSE {alpha_info['rmse_LOFO']} "
          f"(constant {alpha_info['rmse_constant']})", flush=True)
    print(f"  amount ridge: {amount_info['samples']} samples, LOFO RMSE {amount_info['rmse_LOFO']} "
          f"(constant {amount_info['rmse_constant']})", flush=True)
    print(f"step 2/3: {len(val)} VALIDATION fires", flush=True)
    rows, members = [], []
    for i, d in enumerate(val, 1):
        t0 = time.perf_counter()
        try:
            r, mem = evaluate_fire(load_fire(d), 2 if quick else max_starts, alpha_model, amount_model)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(val)}] {d.name}: FAILED {exc}", flush=True)
            continue
        rows += r
        members += mem
        g = pd.DataFrame(r).query("observed_new_cells > 0 and lead == '24h'").groupby("method").change_iou.mean() if r else {}
        print(f"  [{i}/{len(val)}] {d.name}: edge:last {g.get('edge:last', np.nan):.2f} edge:alpha {g.get('edge:alpha', np.nan):.2f} "
              f"edge:direct {g.get('edge:direct', np.nan):.2f} soft25:alpha {g.get('soft25:alpha', np.nan):.2f} "
              f"levelset {g.get('levelset:alpha', np.nan):.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecasts.csv", index=False)
    print("step 3/3: selection", flush=True)
    winner, means = select(df)
    sig = sigmas_for(df, winner)
    rel = reliability(members, winner, sig)
    text = [table(df, BASELINES[:3] + SIMPLICITY + ORACLES, "v0.4 DEVELOPMENT (VALIDATION fires; selection uses 24 h)"),
            "\nselection (24 h mean new-area IoU, ties within 0.005 -> simplest):",
            *[f"  {m:18s} {v:.3f}{'   <- FROZEN' if m == winner else ''}" for m, v in sorted(means.items(), key=lambda kv: -kv[1])],
            f"\nensemble spread (std log amount ratio): {sig}"]
    if rel:
        text.append(f"reliability on validation (in-sample spread): Brier skill {rel['brier_skill']} vs base rate {rel['base_rate']}")
        text += [f"  {r['bin']}: predicted {r['predicted']:.3f} observed {r['observed']:.3f} ({r['cells']} cells)" for r in rel["table"]]
    text.append(f"\nalpha model: {alpha_info}\namount model: {amount_info}")
    text = "\n".join(text)
    print("\n" + text)
    (out / "summary.txt").write_text(text)
    if quick:
        print(f"\n(quick run: nothing frozen; {(time.perf_counter() - started) / 60:.0f} min)")
        return
    frozen = {"frozen_utc": pd.Timestamp.now(tz="UTC").isoformat(), "method": winner, "sigma_log_amount": sig,
              "base_rate": rel["base_rate"] if rel else None, "validation_means_24h": means,
              "alpha_model": model_json(alpha_model, alpha_info, "log alpha (next 12 h)"),
              "amount_model": model_json(amount_model, amount_info, "log new-area growth rate km2/h")}
    (out / "frozen.json").write_text(json.dumps(frozen, indent=2))
    print(f"\nFROZEN: {winner}  -> {out / 'frozen.json'}  ({(time.perf_counter() - started) / 60:.0f} min)")


# --------------------------------------------------------------------------- 2026 fires

def catalogue_2026(det: pd.DataFrame) -> pd.DataFrame:
    """2026 fire events from VIIRS detections alone: 1 km grid cells joined into areas, split where the
    fire went quiet for > 3 days. Kept: >= 300 detections, >= 8 km2 seen burning, 1-60 days."""
    v = det[det.viirs & ~det.confidence.astype(str).str.lower().isin(["l", "low"])].copy()
    v = v[(v.time_utc >= pd.Timestamp("2026-01-01", tz="UTC")) & v.longitude.between(-124.6, -114.0)
          & v.latitude.between(32.4, 42.1)]
    v = v.drop_duplicates(["latitude", "longitude", "time_utc"])
    if not len(v):
        return pd.DataFrame()
    x0, y0 = -124.6, 32.4
    ci = ((v.longitude - x0) / 0.01).astype(int).to_numpy()
    cj = ((v.latitude - y0) / 0.01).astype(int).to_numpy()
    grid = np.zeros((cj.max() + 2, ci.max() + 2), bool)
    grid[cj, ci] = True
    labels, _ = ndimage.label(ndimage.binary_dilation(grid, iterations=1), structure=np.ones((3, 3)))
    v["comp"] = labels[cj, ci]
    rows = []
    for comp, g in v.groupby("comp"):
        g = g.sort_values("time_utc")
        gaps = np.where(g.time_utc.diff().dt.total_seconds().fillna(0).to_numpy() > 3 * 86400)[0]
        for chunk in np.split(np.arange(len(g)), gaps):
            e = g.iloc[chunk]
            days = (e.time_utc.max() - e.time_utc.min()).total_seconds() / 86400
            km2 = len(set(zip((e.longitude / 0.004).astype(int), (e.latitude / 0.004).astype(int)))) * 0.14
            if len(e) < 300 or km2 < 8 or not (1 <= days <= 60):
                continue
            first = e.time_utc.min()
            lon, lat = float(e.longitude.median()), float(e.latitude.median())
            fid = f"{first:%Y%m%d}_c{abs(lat):.2f}n{abs(lon):.2f}w".replace(".", "p")
            region = "bay_area" if (-123.2 <= lon <= -121.2 and 36.8 <= lat <= 38.6) else ("socal" if lat < 35.5 else "norcal")
            rows.append({"fire_id": fid, "name": f"2026 fire near {lat:.2f}N {abs(lon):.2f}W", "inc_num": "", "year": 2026,
                         "alarm_date": f"{first:%Y-%m-%d}", "cont_date": f"{e.time_utc.max() + pd.Timedelta(days=1):%Y-%m-%d}",
                         "acres": km2 / 0.004046856, "cause": "", "complex_name": "", "lon": lon, "lat": lat,
                         "min_lon": float(e.longitude.min()), "min_lat": float(e.latitude.min()),
                         "max_lon": float(e.longitude.max()), "max_lat": float(e.latitude.max()),
                         "duration_days": round(days, 1), "region": region, "event": int(comp), "split": "test2026",
                         "viirs_detections": len(e)})
    return pd.DataFrame(rows).sort_values("acres", ascending=False)


def mode_fetch_2026():
    from pinn_v2 import fetch_fire_data as ffd
    from pinn_v2 import fetch_fuel as ff

    folder = REPO / "data" / "firms_2026"
    det = ffd.load_archive(folder)
    cat = catalogue_2026(det)
    if not len(cat):
        raise SystemExit(f"no 2026 fire events found in {folder}")
    cat.to_csv(REPO / "data" / "fire_catalogue_2026.csv", index=False)
    print(f"{len(cat)} 2026 fire events -> data/fire_catalogue_2026.csv")
    print(cat[["fire_id", "alarm_date", "duration_days", "acres", "viirs_detections", "region"]].to_string(index=False))
    fetch_list([r for _, r in cat.iterrows()], det, ffd, ff)
    dropped = 0
    for fid in cat.fire_id:
        d = REPO / "data" / "fires" / fid
        if not ((d / "static.npz").exists() and (d / "arrival.npz").exists()):
            continue
        arr = np.load(d / "arrival.npz")["arrival_hours"]
        fuel = np.load(d / "static.npz")["fuel_fbfm40"]
        seen = np.isfinite(arr)
        share = float((~nonburnable(fuel[seen])).mean()) if seen.any() else 0.0
        if share < 0.5:                                   # industrial heat or farmland, not a wildfire
            meta = json.loads((d / "meta.json").read_text())
            meta["split"] = "excluded2026"
            meta.setdefault("warnings", []).append(f"only {share:.0%} of the burned area is burnable fuel: excluded")
            (d / "meta.json").write_text(json.dumps(meta, indent=2))
            dropped += 1
    print(f"\n{len(dirs('test2026'))} 2026 fires ready; {dropped} excluded as non-wildland. Nothing has been scored.")


def mode_final_2026(max_starts):
    dev = REPO / "outputs" / "realfire" / "v4_dev" / "frozen.json"
    if not dev.exists():
        raise SystemExit("run --dev first: there is no frozen method")
    out = REPO / "outputs" / "realfire" / "v4_test2026"
    if (out / "summary.txt").exists():
        raise SystemExit(f"the 2026 test was already scored ({out / 'summary.txt'}). It is scored once; delete the folder "
                         "only if the run crashed before finishing.")
    out.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(dev.read_text())
    alpha_model, amount_model = model_from_json(frozen["alpha_model"]), model_from_json(frozen["amount_model"])
    v03 = json.loads((REPO / "pinn_service" / "static" / "learned_alpha.json").read_text())
    v03_model = model_from_json({"lambda": v03["lambda"], "mu": v03["mu"], "sd": v03["sd"], "w": v03["w"], "features": v03["features"]})
    winner = frozen["method"]
    fires = dirs("test2026")
    print(f"FINAL 2026 TEST: frozen method {winner} (frozen {frozen['frozen_utc'][:16]}), {len(fires)} fires", flush=True)
    rows, members = [], []
    for i, d in enumerate(fires, 1):
        t0 = time.perf_counter()
        try:
            r, mem = evaluate_fire(load_fire(d), max_starts, alpha_model, amount_model, v03_model,
                                   only={winner, "edge:last", "levelset:alpha"})
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(fires)}] {d.name}: FAILED {exc}", flush=True)
            continue
        rows += r
        members += mem
        g = pd.DataFrame(r).query("observed_new_cells > 0 and lead == '24h'").groupby("method").change_iou.mean() if r else {}
        print(f"  [{i}/{len(fires)}] {d.name}: physics {g.get('physics', np.nan):.2f} v0.3 {g.get('v03', np.nan):.2f} "
              f"buffer {g.get('buffer', np.nan):.2f} {winner} {g.get(winner, np.nan):.2f} ({time.perf_counter() - t0:.0f} s)", flush=True)
    if not rows:
        raise SystemExit("no forecasts made")
    df = pd.DataFrame(rows)
    df.to_csv(out / "forecasts.csv", index=False)
    methods = ["persistence", "physics", "buffer", "v03", "edge:last", "levelset:alpha", winner] + ORACLES
    L = [table(df, list(dict.fromkeys(methods)), f"v0.4 FINAL TEST on 2026 fires (frozen: {winner}; scored once)"),
         "\nkey comparisons, 24 h (paired over fires, 95% CI; * = CI above zero):"]
    pf = per_fire(df, "24h")
    for a, b in ((winner, "physics"), (winner, "v03"), (winner, "buffer"), (winner, "edge:last"), ("v03", "physics")):
        if a != b:
            g = r3.paired(pf, a, b)
            if g:
                L.append(f"  {a:16s} vs {b:12s} {g[0]:+.3f} [{g[1]:+.3f},{g[2]:+.3f}]{' *' if g[1] > 0 else ''} ({g[3]:.0%} of fires)")
    rel = reliability(members, winner, frozen["sigma_log_amount"], frozen.get("base_rate"))
    cal = frozen.get("prob_calibration")
    rel_cal = reliability(members, winner, frozen["sigma_log_amount"], frozen.get("base_rate"),
                          (cal["raw"], cal["calibrated"])) if cal else None
    for label, rr in (("raw ensemble share", rel), ("calibrated on validation", rel_cal)):
        if rr:
            L.append(f"\nprobabilities ({winner}, {label}): Brier skill {rr['brier_skill']} vs the validation base rate {rr['base_rate']}")
            L += [f"  {r['bin']}: predicted {r['predicted']:.3f} observed {r['observed']:.3f} ({r['cells']} cells)" for r in rr["table"]]
    rel = {"raw": rel, "calibrated": rel_cal}
    text = "\n".join(L)
    print("\n" + text)
    (out / "summary.txt").write_text(text)
    (out / "reliability.json").write_text(json.dumps(rel, indent=2))


def main():
    p = argparse.ArgumentParser(description="v0.4 development and 2026 test")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch-more", action="store_true")
    g.add_argument("--dev", action="store_true")
    g.add_argument("--fetch-2026", action="store_true")
    g.add_argument("--final-2026", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-starts", type=int, default=5)
    a = p.parse_args()
    if a.fetch_more:
        mode_fetch_more(a.limit)
    elif a.dev:
        mode_dev(a.quick, a.max_starts)
    elif a.fetch_2026:
        mode_fetch_2026()
    else:
        mode_final_2026(a.max_starts)


if __name__ == "__main__":
    main()
