"""Find the current fire and forecast its spread for 1, 3 and 6 hours.

Default method since 0.3.0: ``method="calibrated"`` (see calibrated.py). The fire's
recent growth between two VIIRS overpasses calibrates the spread-rate multiplier,
a learned model turns that into the rate for the next hours, quiet edges are held,
and the front is an Anderson wind ellipse driven by hourly wind and equilibrium
moisture. Validated on real California fires at 12 h and 24 h.

The v0.2 method below is still available with ``method="physics"``.

Method "physics" (model_version 0.2.x-physics-baseline):
  1. Horizon 0 = satellite fire detections (FIRMS, last ``lookback_hours``) inside the
     bbox grown by ``search_buffer_m``. The backend's bbox can be a single 1 km cell,
     so the whole surrounding fire must be found and modelled, then clipped back.
  2. Detections closer than 2 km are grouped into one fire. Each fire gets a local
     metre grid (default 90 m) covering it plus a margin; each detection paints a
     disk the size of its satellite pixel.
  3. Fuel and slope (LANDFIRE), 100-h fuel moisture and HOURLY wind give the physics
     spread rate R for each forecast hour (physics.spread_rate).
  4. An ensemble of N members advances the signed distance phi hour by hour with
     phi_t + R (1 + 0.6 n.w)_+ |grad phi| = 0. Member 0 is unperturbed; the others
     scale R by exp(N(0, rate_log_sigma)) and rotate the wind by N(0, wind_dir_sigma_deg).
  5. fire_probability of a 1 km cell at horizon h = share of members that burn any
     part of it.

This is the integration baseline. v2 replaces R with R_phys * exp(h_psi) and the
integrator with the hard-initial-condition PINN; the contract does not change.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import time

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from . import calibrated as cal
from .contract import PredictionRequest
from .native_ingestion import Detections, Environment
from .physics import propagate, spread_rate
from .postprocessing import GridSpec

MAX_WINDOW_CELLS = 400   # per axis; resolution coarsens beyond this
LINK_M = 2000.0          # detections closer than this belong to the same fire


def metres_per_degree(lat0: float) -> tuple[float, float]:
    phi = np.deg2rad(lat0)
    m_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    return float(m_lat), float(m_lon)


def search_bbox(bbox: list[float], buffer_m: float) -> list[float]:
    m_lat, m_lon = metres_per_degree((bbox[1] + bbox[3]) / 2)
    return [bbox[0] - buffer_m / m_lon, bbox[1] - buffer_m / m_lat, bbox[2] + buffer_m / m_lon, bbox[3] + buffer_m / m_lat]


def _seed(req: PredictionRequest, cluster_key: str) -> int:
    if req.additional_params.seed is not None:
        key = f"{req.additional_params.seed}|{cluster_key}"
    else:
        key = f"{req.reference_time.isoformat()}|{cluster_key}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")


def signed_distance(burned: np.ndarray, res: float) -> np.ndarray:
    outside = ndimage.distance_transform_edt(~burned) * res
    inside = ndimage.distance_transform_edt(burned) * res
    return np.where(burned, -(inside - res / 2), outside - res / 2)


def cluster_detections(dets: Detections) -> list[np.ndarray]:
    """Index arrays, one per fire (single linkage at LINK_M)."""
    if len(dets) == 0:
        return []
    m_lat, m_lon = metres_per_degree(float(np.mean(dets.lat)))
    xy = np.column_stack([(dets.lon - dets.lon.mean()) * m_lon, (dets.lat - dets.lat.mean()) * m_lat])
    pairs = cKDTree(xy).query_pairs(LINK_M, output_type="ndarray")
    from scipy.sparse import coo_matrix

    n = len(dets)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else coo_matrix((n, n))
    count, labels = connected_components(graph, directed=False)
    groups = [np.flatnonzero(labels == k) for k in range(count)]
    return sorted(groups, key=lambda g: -len(g))


NONBURNABLE_LABELS = {91: "urban", 92: "snow/ice", 93: "agriculture", 98: "water", 99: "barren", 0: "no data"}
RING_M = 500.0


def burnable_share(source, lon: np.ndarray, lat: np.ndarray) -> tuple[float, str]:
    """Share of burnable FBFM40 fuel within RING_M of the given detections, and the dominant non-burnable class."""
    m_lat, m_lon = metres_per_degree(float(np.mean(lat)))
    offsets = np.linspace(-RING_M, RING_M, 9)
    dx, dy = np.meshgrid(offsets, offsets)
    keep = np.hypot(dx, dy) <= RING_M
    px = (lon[:, None] + dx[keep][None, :] / m_lon).ravel()
    py = (lat[:, None] + dy[keep][None, :] / m_lat).ravel()
    codes = np.rint(np.nan_to_num(source.fuel(px, py), nan=0.0)).astype(int)
    burnable = ~np.isin(codes, list(NONBURNABLE_LABELS))
    values, counts = np.unique(codes[~burnable], return_counts=True) if (~burnable).any() else ([0], [0])
    dominant = NONBURNABLE_LABELS.get(int(values[int(np.argmax(counts))]), "non-burnable")
    return float(burnable.mean()), dominant


def wildland_only(fires: list[np.ndarray], dets: Detections, source, min_share: float) -> tuple[list[np.ndarray], list[dict]]:
    """Drop detection groups that sit on almost no burnable fuel.

    FIRMS flags anything hot: refinery flares (Richmond, Martinez), power plants and
    crop-residue burns show up every day. A level-set spread model has nothing to
    say about them, and the dashboard should not show them as wildfire.
    """
    if min_share <= 0:
        return fires, []
    kept, rejected = [], []
    for index in fires:
        share, dominant = burnable_share(source, dets.lon[index], dets.lat[index])
        if share >= min_share:
            kept.append(index)
        else:
            rejected.append({
                "centre_lon_lat": [round(float(dets.lon[index].mean()), 5), round(float(dets.lat[index].mean()), 5)],
                "detections": int(len(index)),
                "burnable_share": round(share, 2),
                "reason": f"{dominant} land, not wildland fuel (likely industrial heat source or {'crop burn' if dominant == 'agriculture' else 'non-vegetation fire'})",
            })
    return kept, rejected


METHOD_TEXT = {
    "calibrated": (
        "calibrated level-set ensemble: spread-rate multiplier fitted to the fire's growth between the last "
        "two VIIRS overpasses and adjusted by a learned model; quiet edges held; Anderson wind ellipse with hourly "
        "wind and equilibrium moisture (LANDFIRE fuel/slope). Validated on real California fires at 12-24 h"
    ),
    "nowcast": (
        "satellite nowcast (v0.4): the fire keeps growing at the rate seen between its last two VIIRS overpasses, "
        "outward from its active edges; ensemble over the amount, probabilities calibrated on validation fires. "
        "Fires seen by only one overpass use the calibrated level-set forecast"
    ),
    "physics": (
        "level-set ensemble, physics-only spread rate (LANDFIRE fuel/slope, hourly wind, FM100); "
        "uncalibrated v0.2 baseline"
    ),
}


def recent_mask(dets: Detections, req: PredictionRequest) -> np.ndarray:
    """Detections inside ``lookback_hours`` (or of unknown time) form the current fire."""
    cutoff = (req.reference_time.timestamp() - req.additional_params.lookback_hours * 3600.0)
    t = dets.time_s
    return ~np.isfinite(t) | (t >= cutoff - 1.0)


def forecast(req: PredictionRequest, source, grid: GridSpec) -> tuple[dict[int, dict[int, float]], dict]:
    started = time.perf_counter()
    params = req.additional_params
    horizons = req.horizons
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}
    area = search_bbox(req.bbox, params.search_buffer_m)
    dets = source.detections(req, area)
    recent = recent_mask(dets, req)
    if params.method == "physics":
        dets, recent = dets.subset(recent), recent[recent]
    groups = [g for g in cluster_detections(dets) if recent[g].any()]   # old scars with no current fire are dropped
    fires, rejected = wildland_only(groups, dets, source, params.min_burnable_share)
    meta: dict = {
        "forecast_method": METHOD_TEXT[params.method],
        "method": params.method,
        "research_only": True,
        "reference_time": req.reference_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizons_hours": horizons,
        "fires_detected": len(fires),
        "detections_rejected": rejected,
        "ensemble_size": params.members,
        "search_bbox": [round(v, 5) for v in area],
        "input_notes": list(dets.notes),
    } | source.describe()
    if not fires:
        meta["note"] = "no wildland fire detected in or near the bbox; nothing to forecast"
        meta["runtime_s"] = round(time.perf_counter() - started, 3)
        return probabilities, meta

    fires_meta, warnings = [], []
    for number, index in enumerate(fires, start=1):
        fire = dets.subset(np.isin(np.arange(len(dets)), index))
        fire.recent = recent[np.isin(np.arange(len(dets)), index)]
        lon0, lat0 = float(fire.lon.mean()), float(fire.lat.mean())
        env = source.environment(req, lon0, lat0)
        meta["input_notes"].extend(n for n in env.notes if n not in meta["input_notes"])
        margin = params.margin_m
        run = {"calibrated": forecast_fire_calibrated, "nowcast": forecast_fire_nowcast}.get(params.method, forecast_fire)
        for attempt in range(3):  # widen the window if the front reaches its edge
            result = run(number, fire, env, margin, req, grid)
            if not result["reached_edge"] or attempt == 2:
                break
            margin *= 2
        for h, per_cell in result["probabilities"].items():
            for cell, p in per_cell.items():
                probabilities[h][cell] = max(probabilities[h].get(cell, 0.0), p)
        warnings.extend(result["warnings"])
        if result["reached_edge"]:
            warnings.append(f"fire {number}: forecast front still reached the window edge at margin {margin:.0f} m")
        fires_meta.append(result["meta"] | {"margin_m": margin})

    meta["fires"] = fires_meta
    if warnings:
        meta["warnings"] = warnings
    meta["runtime_s"] = round(time.perf_counter() - started, 3)
    return probabilities, meta


def window(fire: Detections, margin: float, res_m: float, max_cells: int = MAX_WINDOW_CELLS) -> dict:
    """Local metre grid around one fire (rows = northing, cols = easting, both increasing)."""
    lon0, lat0 = float(fire.lon.mean()), float(fire.lat.mean())
    m_lat, m_lon = metres_per_degree(lat0)
    fx = (fire.lon - lon0) * m_lon
    fy = (fire.lat - lat0) * m_lat
    r_max = float(fire.radius_m.max())
    x_lo, x_hi = fx.min() - r_max - margin, fx.max() + r_max + margin
    y_lo, y_hi = fy.min() - r_max - margin, fy.max() + r_max + margin
    res = max(res_m, max(x_hi - x_lo, y_hi - y_lo) / max_cells)
    xs = np.arange(x_lo, x_hi + res / 2, res)
    ys = np.arange(y_lo, y_hi + res / 2, res)
    X, Y = np.meshgrid(xs, ys)
    return {"lon0": lon0, "lat0": lat0, "fx": fx, "fy": fy, "res": float(res), "xs": xs, "ys": ys, "X": X, "Y": Y,
            "lon": lon0 + X / m_lon, "lat": lat0 + Y / m_lat}


def paint_confidence(w: dict, fx, fy, radius, confidence) -> np.ndarray:
    """Union of pixel disks; value = best detection confidence covering the cell."""
    xs, ys, X, Y, res = w["xs"], w["ys"], w["X"], w["Y"], w["res"]
    fire_val = np.zeros(X.shape)
    for x, y, r, c in zip(fx, fy, radius, confidence):
        i0, i1 = np.searchsorted(ys, [y - r, y + r])
        j0, j1 = np.searchsorted(xs, [x - r, x + r])
        i1, j1 = max(i1, i0 + 1), max(j1, j0 + 1)
        i0, j0 = min(i0, len(ys) - 1), min(j0, len(xs) - 1)
        sub = (X[i0:i1, j0:j1] - x) ** 2 + (Y[i0:i1, j0:j1] - y) ** 2 <= max(r, res / 2) ** 2
        fire_val[i0:i1, j0:j1] = np.where(sub, np.maximum(fire_val[i0:i1, j0:j1], c), fire_val[i0:i1, j0:j1])
    return fire_val


def touches_edge(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def forecast_fire(number: int, fire: Detections, env: Environment, margin: float, req: PredictionRequest, grid: GridSpec) -> dict:
    """v0.2 uncalibrated physics ensemble (method="physics")."""
    params = req.additional_params
    horizons = req.horizons
    warnings: list[str] = []
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}

    w = window(fire, margin, params.resolution_m)
    lon0, lat0, res, xs, ys, X, lon, lat = (w[k] for k in ("lon0", "lat0", "res", "xs", "ys", "X", "lon", "lat"))
    fire_val = paint_confidence(w, w["fx"], w["fy"], fire.radius_m, fire.confidence)
    burned0 = fire_val > 0

    fuel = env.static("fuel_fbfm40", lon, lat)
    slope = env.static("slope_deg", lon, lat)
    fm100 = env.fm100(lon, lat)
    hours = max(horizons)
    rates, winds = [], []
    for hour in range(hours):
        speed, east, north = env.wind(hour, lon, lat)
        rate, unknown = spread_rate(fuel, slope, speed, fm100)
        if unknown and hour == 0:
            warnings.append(f"fire {number}: FBFM40 codes {unknown} have no spread parameters; treated as non-burnable")
        rates.append(rate)
        winds.append((east, north))
    if not np.any(fuel > 0):
        warnings.append(f"fire {number}: no fuel data under the fire (outside the static layers' coverage?)")

    phi0 = signed_distance(burned0, res)
    cells = grid.locate(lon, lat)

    if 0 in probabilities:
        sel = burned0 & (cells >= 0)
        for cell, value in zip(cells[sel], np.clip(fire_val[sel], 0, 1)):
            probabilities[0][int(cell)] = max(probabilities[0].get(int(cell), 0.0), float(value))

    counts: dict[int, dict[int, int]] = {h: {} for h in horizons if h > 0}
    rng = np.random.default_rng(_seed(req, f"{lon0:.4f},{lat0:.4f}"))
    reached_edge = False
    for member in range(params.members if hours else 0):
        if member == 0:
            mult, turn = 1.0, 0.0
        else:
            mult = float(np.clip(np.exp(rng.normal(0.0, params.rate_log_sigma)), 0.2, 5.0))
            turn = float(np.deg2rad(rng.normal(0.0, params.wind_dir_sigma_deg)))
        c, s = np.cos(turn), np.sin(turn)
        phi = phi0
        for hour in range(hours):
            east, north = winds[hour]
            toward = (c * east - s * north, s * east + c * north)
            phi = propagate(phi, res, rates[hour] * mult, toward, [3600.0])[3600.0]
            h = hour + 1
            if h in counts:
                burned = phi < 0
                reached_edge |= touches_edge(burned)
                for cell in np.unique(cells[burned & (cells >= 0)]):
                    counts[h][int(cell)] = counts[h].get(int(cell), 0) + 1
    for h, per_cell in counts.items():
        for cell, n in per_cell.items():
            probabilities[h][cell] = n / params.members

    burnable = rates[0] > 0 if rates else np.zeros(X.shape, bool)
    meta = {
        "fire": number,
        "detections": len(fire),
        "centre_lon_lat": [round(lon0, 5), round(lat0, 5)],
        "initial_area_km2": round(float(burned0.sum()) * res * res / 1e6, 3),
        "window_cells": [len(ys), len(xs)],
        "resolution_m": round(float(res), 1),
        "median_R_m_s_hour0": round(float(np.median(rates[0][burnable])), 4) if burnable.any() else 0.0,
        "p95_R_m_s_hour0": round(float(np.percentile(rates[0][burnable], 95)), 4) if burnable.any() else 0.0,
    }
    return {"probabilities": probabilities, "warnings": warnings, "reached_edge": reached_edge, "meta": meta}


def forecast_fire_calibrated(number: int, fire: Detections, env: Environment, margin: float, req: PredictionRequest,
                             grid: GridSpec) -> dict:
    """v0.3 calibrated ensemble (method="calibrated", see calibrated.py)."""
    params = req.additional_params
    horizons = req.horizons
    warnings: list[str] = []
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}
    ref_s = req.reference_time.timestamp()
    recent = getattr(fire, "recent", np.ones(len(fire), bool))

    res_m = params.resolution_m if "resolution_m" in params.model_fields_set else cal.GRID_M
    w = window(fire, margin, res_m, cal.MAX_CELLS)
    lon0, lat0, res, xs, ys, X, lon, lat = (w[k] for k in ("lon0", "lat0", "res", "xs", "ys", "X", "lon", "lat"))
    cells = grid.locate(lon, lat)

    # horizon 0: the current fire (detections in lookback_hours), as in v0.2
    fire_val = paint_confidence(w, w["fx"][recent], w["fy"][recent], fire.radius_m[recent], fire.confidence[recent])
    if 0 in probabilities:
        sel = (fire_val > 0) & (cells >= 0)
        for cell, value in zip(cells[sel], np.clip(fire_val[sel], 0, 1)):
            probabilities[0][int(cell)] = max(probabilities[0].get(int(cell), 0.0), float(value))

    fuel = env.static("fuel_fbfm40", lon, lat)
    slope = env.static("slope_deg", lon, lat)
    if not np.any(fuel > 0):
        warnings.append(f"fire {number}: no fuel data under the fire (outside the static layers' coverage?)")
    _, unknown = spread_rate(fuel, slope, np.zeros(fuel.shape), np.full(fuel.shape, 10.0))
    if unknown:
        warnings.append(f"fire {number}: FBFM40 codes {unknown} have no spread parameters; treated as non-burnable")

    radius = cal.VIIRS_RADIUS_M if fire.source == "firms" else float(np.median(fire.radius_m))
    c = cal.calibrate(xs, ys, w["fx"], w["fy"], fire.time_s, fuel, slope, env, res, ref_s, radius)
    start, t_b, alpha, act = c["start"], c["t_b"], c["alpha"], c["act"]
    warnings.extend(f"fire {number}: {n}" for n in c["notes"])

    # cells burned only before the lookback window (an old scar) are not reported as fire
    old_passes = [m for t, m in zip(c["passes"], c["masks"]) if t < ref_s - params.lookback_hours * 3600.0]
    old = (old_passes[-1] & ~(fire_val > 0)) if old_passes else np.zeros(start.shape, bool)
    report = start | (fire_val > 0)

    model = c["model"]
    if "rate_log_sigma" in params.model_fields_set:
        sigma = params.rate_log_sigma
    else:
        sigma = float(model.get("rmse_log_alpha_LOFO", cal.FALLBACK_LOG_SIGMA)) if model else cal.FALLBACK_LOG_SIGMA

    targets = sorted(h for h in horizons if h > 0)
    counts: dict[int, dict[int, int]] = {h: {} for h in targets}
    rng = np.random.default_rng(_seed(req, f"{lon0:.4f},{lat0:.4f}"))
    reached_edge = False
    phi0 = cal.signed_distance(start, res)
    for member in range(params.members if targets else 0):
        if member == 0:
            mult, turn = 1.0, 0.0
        else:
            mult = float(np.exp(rng.normal(0.0, sigma)))
            turn = float(np.deg2rad(rng.normal(0.0, params.wind_dir_sigma_deg)))
        a_m = float(np.clip(alpha * mult, cal.ALPHAS[0], cal.ALPHAS[-1]))
        phi, t_now = phi0, t_b
        for h in targets:
            t_h = ref_s + h * 3600.0
            phi = cal.roll(phi, res, fuel, slope, env, t_now, t_h, a_m * act, turn)
            t_now = max(t_now, t_h)
            burned = phi < 0
            reached_edge |= touches_edge(burned & ~start)
            shown = (burned | report) & ~old
            for cell in np.unique(cells[shown & (cells >= 0)]):
                counts[h][int(cell)] = counts[h].get(int(cell), 0) + 1
    for h, per_cell in counts.items():
        for cell, n in per_cell.items():
            probabilities[h][cell] = n / params.members

    cell_km2 = res * res / 1e6
    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    meta = {
        "fire": number,
        "detections": int(recent.sum()),
        "detections_history": len(fire),
        "centre_lon_lat": [round(lon0, 5), round(lat0, 5)],
        "initial_area_km2": round(float((fire_val > 0).sum()) * cell_km2, 3),
        "burned_area_at_start_km2": round(float(start.sum()) * cell_km2, 3),
        "window_cells": [len(ys), len(xs)],
        "resolution_m": round(float(res), 1),
        "overpasses_used": len(c["passes"]),
        "forecast_start": iso(t_b),
        "calibration_mode": c["mode"],
        "alpha": round(alpha, 4),
        "alpha_log_sigma": round(sigma, 3),
        "active_front_share": round(float((act[~start] > 0).mean()), 3) if (~start).any() else 1.0,
    }
    if c["t_a"] is not None:
        meta.update({
            "calibration_interval": [iso(c["t_a"]), iso(t_b)],
            "alpha_last_interval": round(c["alpha_last"], 4),
            "observed_growth_km2_per_h": round(c["observed_new_cells"] * cell_km2 / ((t_b - c["t_a"]) / 3600.0), 3),
        })
    if c.get("features"):
        meta["alpha_features"] = c["features"]
    if model:
        meta["learned_model"] = {"sha256_12": model["_sha"], "trained_on": model.get("trained_on", "")}
    return {"probabilities": probabilities, "warnings": warnings, "reached_edge": reached_edge, "meta": meta}


def forecast_fire_nowcast(number: int, fire: Detections, env: Environment, margin: float, req: PredictionRequest,
                          grid: GridSpec) -> dict:
    """v0.4 nowcast (method="nowcast", see calibrated.nowcast_probability). Falls back to the v0.3
    calibrated forecast when the fire has no earlier overpass to measure growth from."""
    params = req.additional_params
    horizons = req.horizons
    ref_s = req.reference_time.timestamp()
    recent = getattr(fire, "recent", np.ones(len(fire), bool))
    res_m = params.resolution_m if "resolution_m" in params.model_fields_set else cal.GRID_M
    w = window(fire, margin, res_m, cal.MAX_CELLS)
    lon0, lat0, res, xs, ys, lon, lat = (w[k] for k in ("lon0", "lat0", "res", "xs", "ys", "lon", "lat"))
    fuel = env.static("fuel_fbfm40", lon, lat)
    slope = env.static("slope_deg", lon, lat)
    radius = cal.VIIRS_RADIUS_M if fire.source == "firms" else float(np.median(fire.radius_m))
    times = np.minimum(np.where(np.isfinite(fire.time_s), fire.time_s, ref_s), ref_s)
    passes = cal.pass_times(times)
    masks = cal.history_masks(cal.arrival_seconds(xs, ys, w["fx"], w["fy"], times, res, radius), passes, res)
    b = len(passes) - 1
    start, t_b = masks[b], passes[b]
    a = next((i for i in range(b - 1, -1, -1) if t_b - passes[i] >= cal.MIN_CAL_GAP_H * 3600
              and masks[i].any() and masks[i].sum() < start.sum()), None)
    if a is None:
        result = forecast_fire_calibrated(number, fire, env, margin, req, grid)
        result["meta"]["nowcast"] = "fallback: no earlier overpass with less burned area, so the v0.3 calibrated forecast is used"
        return result

    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}
    warnings: list[str] = []
    cells = grid.locate(lon, lat)
    fire_val = paint_confidence(w, w["fx"][recent], w["fy"][recent], fire.radius_m[recent], fire.confidence[recent])
    if 0 in probabilities:
        sel = (fire_val > 0) & (cells >= 0)
        for cell, value in zip(cells[sel], np.clip(fire_val[sel], 0, 1)):
            probabilities[0][int(cell)] = max(probabilities[0].get(int(cell), 0.0), float(value))
    if not np.any(fuel > 0):
        warnings.append(f"fire {number}: no fuel data under the fire (outside the static layers' coverage?)")

    prev = masks[a]
    act = cal.active_field(start, prev, res)
    cell_km2 = res * res / 1e6
    rate = float((start & ~prev).sum()) * cell_km2 / ((t_b - passes[a]) / 3600.0)
    old_passes = [m for t, m in zip(passes, masks) if t < ref_s - params.lookback_hours * 3600.0]
    old = (old_passes[-1] & ~(fire_val > 0)) if old_passes else np.zeros(start.shape, bool)
    report = start | (fire_val > 0)
    cfg = cal.nowcast_config()
    reached_edge = False
    for h in sorted(x for x in horizons if x > 0):
        span = (ref_s + h * 3600.0 - t_b) / 3600.0
        sigma = cfg["sigma_log_amount"]["12h" if span <= 18 else "24h"]
        p = cal.nowcast_probability(start, act, res, rate, span, sigma, cfg["prob_calibration"])
        p = np.where(report, 1.0, p) * ~old
        reached_edge |= touches_edge((p > 0) & ~start)
        sel = (p > 0) & (cells >= 0)
        for cell, value in zip(cells[sel], p[sel]):
            probabilities[h][int(cell)] = max(probabilities[h].get(int(cell), 0.0), float(value))

    iso = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    meta = {
        "fire": number,
        "detections": int(recent.sum()),
        "detections_history": len(fire),
        "centre_lon_lat": [round(lon0, 5), round(lat0, 5)],
        "initial_area_km2": round(float((fire_val > 0).sum()) * cell_km2, 3),
        "burned_area_at_start_km2": round(float(start.sum()) * cell_km2, 3),
        "window_cells": [len(ys), len(xs)],
        "resolution_m": round(float(res), 1),
        "overpasses_used": len(passes),
        "forecast_start": iso(t_b),
        "calibration_interval": [iso(passes[a]), iso(t_b)],
        "observed_growth_km2_per_h": round(rate, 3),
        "active_front_share": round(float((act[~start] > 0).mean()), 3) if (~start).any() else 1.0,
        "nowcast": "active-edge persistence, calibrated probabilities",
    }
    return {"probabilities": probabilities, "warnings": warnings, "reached_edge": reached_edge, "meta": meta}
