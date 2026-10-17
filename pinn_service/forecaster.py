"""Find the current fire and forecast its spread for 1, 3 and 6 hours.

Method (model_version 0.2.0-physics-baseline):
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

import hashlib
import time

import numpy as np
from scipy import ndimage
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

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


def forecast(req: PredictionRequest, source, grid: GridSpec) -> tuple[dict[int, dict[int, float]], dict]:
    started = time.perf_counter()
    params = req.additional_params
    horizons = req.horizons
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}
    area = search_bbox(req.bbox, params.search_buffer_m)
    dets = source.detections(req, area)
    fires, rejected = wildland_only(cluster_detections(dets), dets, source, params.min_burnable_share)
    meta: dict = {
        "forecast_method": (
            "level-set ensemble, physics-only spread rate (LANDFIRE fuel/slope, hourly wind, FM100); "
            "learned PINN correction not yet deployed"
        ),
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
        lon0, lat0 = float(fire.lon.mean()), float(fire.lat.mean())
        env = source.environment(req, lon0, lat0)
        meta["input_notes"].extend(n for n in env.notes if n not in meta["input_notes"])
        margin = params.margin_m
        for attempt in range(3):  # widen the window if the front reaches its edge
            result = forecast_fire(number, fire, env, margin, req, grid)
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


def forecast_fire(number: int, fire: Detections, env: Environment, margin: float, req: PredictionRequest, grid: GridSpec) -> dict:
    params = req.additional_params
    horizons = req.horizons
    warnings: list[str] = []
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}

    lon0, lat0 = float(fire.lon.mean()), float(fire.lat.mean())
    m_lat, m_lon = metres_per_degree(lat0)
    fx = (fire.lon - lon0) * m_lon
    fy = (fire.lat - lat0) * m_lat
    r_max = float(fire.radius_m.max())
    x_lo, x_hi = fx.min() - r_max - margin, fx.max() + r_max + margin
    y_lo, y_hi = fy.min() - r_max - margin, fy.max() + r_max + margin
    res = max(params.resolution_m, max(x_hi - x_lo, y_hi - y_lo) / MAX_WINDOW_CELLS)
    xs = np.arange(x_lo, x_hi + res / 2, res)
    ys = np.arange(y_lo, y_hi + res / 2, res)
    X, Y = np.meshgrid(xs, ys)          # rows = northing (increasing), cols = easting
    lon = lon0 + X / m_lon
    lat = lat0 + Y / m_lat

    # horizon-0 fire: union of pixel disks; value = best detection confidence covering the cell
    fire_val = np.zeros(X.shape)
    for x, y, r, c in zip(fx, fy, fire.radius_m, fire.confidence):
        i0, i1 = np.searchsorted(ys, [y - r, y + r])
        j0, j1 = np.searchsorted(xs, [x - r, x + r])
        i1, j1 = max(i1, i0 + 1), max(j1, j0 + 1)
        i0, j0 = min(i0, len(ys) - 1), min(j0, len(xs) - 1)
        sub = (X[i0:i1, j0:j1] - x) ** 2 + (Y[i0:i1, j0:j1] - y) ** 2 <= max(r, res / 2) ** 2
        fire_val[i0:i1, j0:j1] = np.where(sub, np.maximum(fire_val[i0:i1, j0:j1], c), fire_val[i0:i1, j0:j1])
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
                reached_edge |= bool(burned[0, :].any() or burned[-1, :].any() or burned[:, 0].any() or burned[:, -1].any())
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
