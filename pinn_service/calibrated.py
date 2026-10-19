"""Calibrated level-set forecast (model 0.3.0): the method validated on real California fires.

This is the "learned" configuration from the real-fire experiments
(``pinn_v2/realfire_round3.py``, ``realfire_final.py``), ported line for line so the
service runs exactly what was tested:

1. Burned area at every VIIRS overpass in the last ``history_hours``: each detection
   paints a 212 m disk (one 375 m pixel), masks are cumulative, gaps up to one pixel are
   closed and holes filled (the satellite misses cells under smoke and between scans).
2. Latest pass t_b = the forecast start. An earlier pass t_a, at least 3 h before, with
   less burned area gives the fire's most recent growth.
3. Active fronts: edges within 2 km of the t_a -> t_b growth may spread; cells nearer to
   a quiet edge (no recent growth) than to an active one, within 3 km, may not.
4. Calibration: the spread-rate multiplier alpha that makes the level set reproduce the
   observed t_a -> t_b new area (area-matched, with that interval's mean weather).
5. Learned alpha: a ridge regression (``static/learned_alpha.json``, trained on the train
   split of 304 catalogued fires) maps alpha_last, wind, moisture, fuel mix, slope, size,
   growth rate and time of day to the alpha for the next 12 h.
6. Forecast: the level set phi_t + alpha * act * R (1 - e) / (1 - e n.w) |grad phi| = 0 hour
   by hour from t_b, with R from LANDFIRE fuel/slope and that hour's wind and Simard
   equilibrium moisture, and the Anderson (1983) ellipse eccentricity e from the wind.
7. Ensemble: alpha * exp(N(0, sigma)) with sigma = the model's out-of-fire RMSE in log
   alpha, and the wind direction turned by N(0, wind_dir_sigma_deg).

Validated at 12 h and 24 h leads (new-area IoU 0.205 vs 0.074 for the uncalibrated
physics and 0.152 for a uniform buffer, 26 validation fires). Shorter leads are not
separately validated.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import ndimage

from .physics import ellipse_eccentricity, propagate, spread_rate, wind_toward_from_met

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PACKAGE_DIR / "static" / "learned_alpha.json"

GRID_M = 180.0                 # validated grid spacing
MAX_CELLS = 220                # per axis; big fires coarsen (the experiments used 360 m above 2000 km2)
VIIRS_RADIUS_M = 212.0         # equal-area disk of a 375 m VIIRS pixel
SEEN_SLACK_S = 0.34 * 3600     # a pass sees detections up to 20 min after its first one
PASS_GAP_S = 20 * 60           # detections closer in time than this are one overpass
MIN_CAL_GAP_H = 3.0
CLOSE_M = 360.0                # gap closing radius (2 cells at 180 m)
ACTIVE_M, QUIET_M, ZONE_M = 2000.0, 3000.0, 3000.0
ALPHAS = np.exp(np.linspace(np.log(0.01), np.log(6.0), 44))
FALLBACK_ALPHA = 0.26          # median next-12 h alpha on the train fires (round 3)
FALLBACK_LOG_SIGMA = 1.14      # out-of-fire RMSE of log alpha (round 3)


# --------------------------------------------------------------------------- learned model

_MODEL: dict[str, tuple[float, dict | None]] = {}


def learned_model() -> dict | None:
    """The exported ridge model, or None if the file is missing or malformed."""
    path = Path(os.environ.get("LEARNED_ALPHA", DEFAULT_MODEL))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _MODEL.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        model = json.loads(path.read_text())
        n = len(model["features"])
        if len(model["mu"]) != n or len(model["sd"]) != n or len(model["w"]) != n + 1:
            raise ValueError("feature / weight length mismatch")
        model["_sha"] = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except (OSError, ValueError, KeyError, TypeError):
        model = None
    _MODEL[str(path)] = (mtime, model)
    return model


def predict_alpha(model: dict, feats: dict) -> float:
    x = np.array([feats[c] for c in model["features"]], dtype=float)
    z = (x - np.array(model["mu"])) / np.array(model["sd"])
    w = np.array(model["w"])
    log_alpha = float(w[0] + z @ w[1:])
    return float(np.clip(np.exp(log_alpha), model.get("alpha_min", ALPHAS[0]), model.get("alpha_max", ALPHAS[-1])))


# --------------------------------------------------------------------------- burned-area history

def pass_times(times_s: np.ndarray) -> list[float]:
    """One time per overpass: detections within 20 min of each other form one pass, timed at the first."""
    t = np.sort(np.asarray(times_s, dtype=float))
    if len(t) == 0:
        return []
    breaks = np.where(np.diff(t) > PASS_GAP_S)[0] + 1
    return [float(chunk[0]) for chunk in np.split(t, breaks)]


def arrival_seconds(xs, ys, fx, fy, times_s, res: float, radius_m: float = VIIRS_RADIUS_M) -> np.ndarray:
    """Earliest detection time covering each cell (inf where never seen)."""
    arrival = np.full((len(ys), len(xs)), np.inf)
    r = int(np.ceil((radius_m + res / 2) / res))
    di, dj = np.mgrid[-r:r + 1, -r:r + 1]
    inside = np.hypot(di, dj) * res <= radius_m + res / 2
    di, dj = di[inside], dj[inside]
    ci = np.rint((np.asarray(fy) - ys[0]) / res).astype(int)
    cj = np.rint((np.asarray(fx) - xs[0]) / res).astype(int)
    rows = np.clip(ci[:, None] + di[None, :], 0, len(ys) - 1).ravel()
    cols = np.clip(cj[:, None] + dj[None, :], 0, len(xs) - 1).ravel()
    np.minimum.at(arrival, (rows, cols), np.repeat(np.asarray(times_s, dtype=float), len(di)))
    return arrival


def history_masks(arrival: np.ndarray, passes: list[float], res: float) -> list[np.ndarray]:
    """Cumulative burned mask at each pass, gaps closed and holes filled (as realfire_round3.Fire)."""
    k = max(1, int(round(CLOSE_M / res)))
    disk = np.hypot(*np.mgrid[-k:k + 1, -k:k + 1]) <= k
    burned = np.zeros(arrival.shape, dtype=bool)
    out = []
    for t in passes:
        seen = arrival <= t + SEEN_SLACK_S
        if seen.any():
            closed = ndimage.binary_closing(seen | burned, structure=disk, border_value=0)
            burned = ndimage.binary_fill_holes(closed | seen | burned)
        out.append(burned.copy())
    return out


def active_field(start: np.ndarray, prev: np.ndarray, res: float) -> np.ndarray:
    """1 where the fire may spread, 0 in front of quiet edges (as realfire_round2.active_field)."""
    grew = start & ~prev
    if not grew.any():
        return np.where(start, 1.0, 0.0)
    edge = start & ~ndimage.binary_erosion(start)
    near = ndimage.distance_transform_edt(~grew) * res <= ACTIVE_M
    act, quiet = edge & near, edge & ~near
    if not quiet.any():
        return np.ones(start.shape)
    d_q = ndimage.distance_transform_edt(~quiet) * res
    d_a = ndimage.distance_transform_edt(~act) * res if act.any() else np.full(start.shape, np.inf)
    return np.where(~start & (d_q < d_a) & (d_q <= QUIET_M), 0.0, 1.0)


def signed_distance(burned: np.ndarray, res: float) -> np.ndarray:
    outside = ndimage.distance_transform_edt(~burned) * res
    inside = ndimage.distance_transform_edt(burned) * res
    return np.where(burned, -(inside - res / 2), outside - res / 2)


# --------------------------------------------------------------------------- weather along the forecast

def _when(t_s: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t_s, dt.timezone.utc)


def mean_conditions(env, t_a: float, t_b: float) -> tuple[float, float, float, float]:
    """Mean wind speed, circular-mean wind-from direction, mean EMC and FM100 over [t_a, t_b]."""
    span_h = (t_b - t_a) / 3600.0
    hours = np.arange(0.5, span_h, 1.0) if span_h >= 1 else np.array([span_h / 2])
    c = np.array([env.conditions(_when(t_a + h * 3600.0)) for h in hours])
    ang = np.deg2rad(c[:, 1])
    wd = float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360)
    return float(c[:, 0].mean()), wd, float(c[:, 2].mean()), float(c[:, 3].mean())


def _advance(phi, res, rate, wind_from, ws, seconds, turn=0.0):
    east, north = wind_toward_from_met(wind_from + np.rad2deg(turn))
    return propagate(phi, res, rate, (float(east), float(north)), [seconds], eccentricity=ellipse_eccentricity(ws))[float(seconds)]


def roll(phi, res, fuel, slope, env, t_from: float, t_to: float, mult, turn: float = 0.0):
    """Advance phi from t_from to t_to in steps that end on the clock hour, each with its own
    wind and EMC (as realfire_round2.roll with the 'diurnal' moisture)."""
    t = t_from
    while t < t_to - 1.0:
        into = t % 3600.0
        dt_s = min(3600.0 - into if into > 1.0 else 3600.0, t_to - t)
        ws, wd, emc, _ = env.conditions(_when(t + dt_s / 2))
        rate = spread_rate(fuel, slope, np.full(fuel.shape, ws), np.full(fuel.shape, emc))[0] * mult
        phi = _advance(phi, res, rate, wd, ws, dt_s, turn)
        t += dt_s
    return phi


def area_curve(start, res, rate, wind_from, ws, span_h, target_cells) -> np.ndarray:
    """New burned cells after span_h hours for each alpha in ALPHAS (constant conditions, so
    alpha x time scaling is exact). Stops once the area is far past the target."""
    counts = np.full(len(ALPHAS), np.iinfo(np.int64).max // 4, dtype=np.int64)
    phi, done = signed_distance(start, res), 0.0
    for i, a in enumerate(ALPHAS):
        t = a * span_h * 3600.0
        phi = _advance(phi, res, rate, wind_from, ws, t - done)
        done = t
        counts[i] = int(((phi < 0) & ~start).sum())
        if counts[i] > 4 * target_cells + 50:
            break
    return counts


def pick_alpha(curve: np.ndarray, target_cells: int) -> float:
    if target_cells <= 0:
        return float(ALPHAS[0])
    err = np.abs(np.log((curve + 0.5) / (target_cells + 0.5)))
    return float(ALPHAS[int(np.argmin(err))])


def features(fuel, slope, start, prev, act, res, env, t_a, t_b, alpha_last) -> dict:
    """Same inputs as realfire_round2.features (only what is known at t_b, plus the weather forecast)."""
    cell_km2 = res * res / 1e6
    ws, _, emc, fm = mean_conditions(env, t_b, t_b + 12 * 3600.0)
    zone = (act > 0) & ~start & (ndimage.distance_transform_edt(~start) * res <= ZONE_M)
    f = fuel[zone] if zone.any() else fuel.ravel()
    n = max(len(f), 1)
    hour_local = (_when(t_b).hour - 8) % 24
    return {
        "log_alpha_last": float(np.log(alpha_last)),
        "log_wind": float(np.log(ws + 0.5)), "emc_next12": emc, "fm100": fm,
        "grass": float(((f >= 101) & (f <= 124)).sum() / n),
        "shrub": float(((f >= 141) & (f <= 149)).sum() / n),
        "timber": float(((f >= 161) & (f <= 189)).sum() / n),
        "slope_zone": float(slope[zone].mean()) if zone.any() else float(slope.mean()),
        "log_area": float(np.log(start.sum() * cell_km2 + 0.1)),
        "log_growth_rate": float(np.log((start & ~prev).sum() * cell_km2 / max((t_b - t_a) / 3600.0, 1.0) + 1e-3)),
        "hour_sin": float(np.sin(2 * np.pi * hour_local / 24)), "hour_cos": float(np.cos(2 * np.pi * hour_local / 24)),
    }


# --------------------------------------------------------------------------- calibration at the forecast start

def calibrate(xs, ys, fx, fy, times_s, fuel, slope, env, res, ref_s: float, radius_m: float = VIIRS_RADIUS_M) -> dict:
    """Burned history, start mask, active field and alpha for one fire window.
    Detections without a time (offline replays) count as one pass at the reference time."""
    times = np.minimum(np.where(np.isfinite(times_s), times_s, ref_s), ref_s)
    passes = pass_times(times)
    arrival = arrival_seconds(xs, ys, fx, fy, times, res, radius_m)
    masks = history_masks(arrival, passes, res)
    b = len(passes) - 1
    start = masks[b]
    out = {"passes": passes, "masks": masks, "t_b": passes[b], "start": start, "t_a": None, "notes": []}
    a = next((i for i in range(b - 1, -1, -1)
              if passes[b] - passes[i] >= MIN_CAL_GAP_H * 3600 and masks[i].any() and masks[i].sum() < start.sum()), None)
    model = learned_model()
    out["model"] = model
    if a is None:
        out.update(act=np.ones(start.shape), alpha=float(model["fallback_alpha"]) if model else FALLBACK_ALPHA,
                   alpha_last=None, mode="fallback")
        out["notes"].append("no earlier overpass with less burned area (>= 3 h before the latest): "
                            "no growth to calibrate on, typical alpha used and all edges treated as active")
        return out
    prev = masks[a]
    act = active_field(start, prev, res)
    ws, wd, emc, _ = mean_conditions(env, passes[a], passes[b])
    rate = spread_rate(fuel, slope, np.full(fuel.shape, ws), np.full(fuel.shape, emc))[0] * act
    observed = int((start & ~prev).sum())
    alpha_last = pick_alpha(area_curve(prev, res, rate, wd, ws, (passes[b] - passes[a]) / 3600.0, observed), observed)
    out.update(t_a=passes[a], prev=prev, act=act, alpha_last=alpha_last, observed_new_cells=observed)
    if model is None:
        out.update(alpha=alpha_last, mode="last-interval")
        out["notes"].append("learned_alpha.json missing: alpha from the last interval only")
        return out
    feats = features(fuel, slope, start, prev, act, res, env, passes[a], passes[b], alpha_last)
    out.update(alpha=predict_alpha(model, feats), mode="learned", features={k: round(v, 4) for k, v in feats.items()})
    return out


# --------------------------------------------------------------------------- v0.4 nowcast (method="nowcast")

NOWCAST_MODEL = PACKAGE_DIR / "static" / "nowcast.json"
NOWCAST_DEFAULTS = {   # frozen on the validation fires (pinn_v2/realfire_v4.py --dev, 19 Oct 2026 05:36 UTC)
    "sigma_log_amount": {"12h": 1.423, "24h": 1.234},
    "prob_calibration": {"raw": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.946],
                         "calibrated": [0.006, 0.035, 0.074, 0.118, 0.163, 0.201, 0.242, 0.289, 0.341, 0.407]},
}
Z10 = np.array([-1.645, -1.036, -0.674, -0.385, -0.126, 0.126, 0.385, 0.674, 1.036, 1.645])


def nowcast_config() -> dict:
    try:
        return json.loads(NOWCAST_MODEL.read_text())
    except (OSError, ValueError):
        return NOWCAST_DEFAULTS


def nowcast_probability(start: np.ndarray, act: np.ndarray, res: float, rate_km2_h: float, span_h: float,
                        sigma: float, calib: dict) -> np.ndarray:
    """v0.4 'edge:last' (realfire_v4): the fire keeps growing at its last observed rate, outward from its
    ACTIVE edges, nearest cells first. Ten ensemble members scale the amount by exp(sigma z_k); a cell's
    raw probability is the share of members that reach it, then mapped through the validation calibration.
    Burned cells get 1."""
    allowed = ~start & (act > 0)
    out = np.where(start, 1.0, 0.0)
    idx = np.flatnonzero(allowed)
    if not idx.size or rate_km2_h <= 0 or span_h <= 0:
        return out
    d = ndimage.distance_transform_edt(~start).ravel()[idx]
    rank = np.empty(idx.size)
    rank[np.argsort(d, kind="stable")] = np.arange(idx.size)
    n = rate_km2_h * span_h / (res * res / 1e6) * np.exp(sigma * Z10)
    raw = (rank[:, None] < n[None, :]).mean(axis=1)
    out.ravel()[idx] = np.interp(raw, calib["raw"], calib["calibrated"]) * (raw > 0)
    return out
