"""Find the current fire in the daily features and forecast its spread.

Method (model_version 0.1.0-physics-baseline):
  1. Horizon 0 = feature-grid pixels with active_fire >= fire_threshold inside the bbox.
  2. Nearby pixels are grouped into fires; each fire gets a local metre grid
     (default 90 m) covering the fire plus a margin.
  3. Fuel, slope, 100-h fuel moisture and wind are sampled onto that grid and
     turned into the physics spread rate R (physics.spread_rate).
  4. An ensemble of N members advances the signed distance phi with
     phi_t + R (1 + 0.6 n.w)_+ |grad phi| = 0. Member 0 is unperturbed; the others
     scale R by exp(N(0, rate_log_sigma)) and rotate the wind by N(0, wind_dir_sigma_deg).
  5. fire_probability of a 1 km cell at horizon h = share of members that burn
     any part of it.

This is the integration baseline. v2 replaces step 4's rate with
R_phys * exp(h_psi) and the integrator with the hard-initial-condition PINN; the
request/response contract does not change.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
from scipy import ndimage

from .contract import PredictionRequest, ServiceError
from .features import Layers
from .physics import propagate, spread_rate, wind_toward_from_met
from .postprocessing import GridSpec

MAX_WINDOW_CELLS = 400  # per axis; resolution coarsens beyond this


def _metres_per_degree(lat0: float) -> tuple[float, float]:
    phi = np.deg2rad(lat0)
    m_lat = 111132.92 - 559.82 * np.cos(2 * phi) + 1.175 * np.cos(4 * phi)
    m_lon = 111412.84 * np.cos(phi) - 93.5 * np.cos(3 * phi)
    return float(m_lat), float(m_lon)


def _seed(req: PredictionRequest, cluster: int) -> int:
    if req.additional_params.seed is not None:
        return int(req.additional_params.seed) + cluster
    key = f"{req.date}|{req.bbox}|{cluster}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def _signed_distance(burned: np.ndarray, res: float) -> np.ndarray:
    outside = ndimage.distance_transform_edt(~burned) * res
    inside = ndimage.distance_transform_edt(burned) * res
    return np.where(burned, -(inside - res / 2), outside - res / 2)


def _fire_clusters(layers: Layers, req: PredictionRequest) -> tuple[np.ndarray, int, np.ndarray]:
    daily = layers.daily
    fire = np.nan_to_num(daily.channels["active_fire"], nan=0.0)
    lon2, lat2 = np.meshgrid(daily.lon, daily.lat)
    b = req.bbox
    in_box = (lon2 >= b[0]) & (lon2 <= b[2]) & (lat2 >= b[1]) & (lat2 <= b[3])
    mask = (fire >= req.additional_params.fire_threshold) & in_box
    if not mask.any():
        return mask, 0, np.zeros_like(mask, dtype=int)
    m_lat, _ = _metres_per_degree(float(np.mean(daily.lat)))
    pixel_m = daily.dlat * m_lat
    link = max(1, int(np.ceil(2000.0 / max(pixel_m, 1.0))))
    grown = ndimage.binary_dilation(mask, iterations=link)
    labels, count = ndimage.label(grown, structure=np.ones((3, 3)))
    return mask, count, np.where(mask, labels, 0)


def forecast(req: PredictionRequest, layers: Layers, grid: GridSpec) -> tuple[dict[int, dict[int, float]], dict]:
    started = time.perf_counter()
    params = req.additional_params
    horizons = req.horizons
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}

    if not layers.daily.has("active_fire"):
        raise ServiceError(422, "daily feature file has no active_fire channel; the PINN cannot find the current fire")
    mask, count, labels = _fire_clusters(layers, req)
    meta: dict = {
        "forecast_method": (
            "level-set ensemble, physics-only spread rate (fuel, slope, wind, FM100); "
            "learned PINN correction not yet deployed"
        ),
        "research_only": True,
        "issued_for_date": req.date,
        "horizons_hours": horizons,
        "fires_detected": int(count),
        "members": params.members,
        "input_notes": layers.notes,
    }
    if count == 0:
        meta["note"] = "no active fire above fire_threshold inside bbox; nothing to forecast"
        meta["runtime_s"] = round(time.perf_counter() - started, 3)
        return probabilities, meta

    missing = layers.missing_for_pinn()
    if missing:
        raise ServiceError(422, f"feature file lacks channels the PINN needs: {', '.join(missing)} (see feature_spec.md)")

    daily = layers.daily
    lon2, lat2 = np.meshgrid(daily.lon, daily.lat)
    fires_meta = []
    warnings: list[str] = []
    forecast_times = [h * 3600.0 for h in horizons if h > 0]

    for cluster in range(1, count + 1):
        pix = labels == cluster
        margin = params.margin_m
        for attempt in range(3):  # widen the window if the front reaches its edge
            result = _forecast_cluster(cluster, pix, lon2, lat2, margin, req, layers, grid, horizons, forecast_times)
            if result is None or not result["reached_edge"] or attempt == 2:
                break
            margin *= 2
        if result is None:
            continue
        for h, per_cell in result["probabilities"].items():
            for cell, p in per_cell.items():
                probabilities[h][cell] = max(probabilities[h].get(cell, 0.0), p)
        warnings.extend(result["warnings"])
        if result["reached_edge"]:
            warnings.append(f"fire {cluster}: forecast front still reached the window edge at margin {margin:.0f} m")
        fires_meta.append(result["meta"] | {"margin_m": margin})

    meta["fires"] = fires_meta
    if warnings:
        meta["warnings"] = warnings
    meta["runtime_s"] = round(time.perf_counter() - started, 3)
    return probabilities, meta


def _forecast_cluster(cluster, pix, lon2, lat2, margin, req, layers, grid, horizons, forecast_times):
    """Forecast one fire on its own metre grid. Returns None if the fire vanishes on resampling."""
    params = req.additional_params
    daily = layers.daily
    warnings: list[str] = []
    probabilities: dict[int, dict[int, float]] = {h: {} for h in horizons}

    lons, lats = lon2[pix], lat2[pix]
    lon0, lat0 = float(lons.mean()), float(lats.mean())
    m_lat, m_lon = _metres_per_degree(lat0)
    x_lo = (lons.min() - daily.dlon / 2 - lon0) * m_lon - margin
    x_hi = (lons.max() + daily.dlon / 2 - lon0) * m_lon + margin
    y_lo = (lats.min() - daily.dlat / 2 - lat0) * m_lat - margin
    y_hi = (lats.max() + daily.dlat / 2 - lat0) * m_lat + margin
    res = max(params.resolution_m, max(x_hi - x_lo, y_hi - y_lo) / MAX_WINDOW_CELLS)
    xs = np.arange(x_lo, x_hi + res / 2, res)
    ys = np.arange(y_lo, y_hi + res / 2, res)
    X, Y = np.meshgrid(xs, ys)          # rows = northing (increasing), cols = easting
    lon = lon0 + X / m_lon
    lat = lat0 + Y / m_lat

    fire_val = np.nan_to_num(daily.sample("active_fire", lon, lat, fill=0.0), nan=0.0)
    burned0 = fire_val >= params.fire_threshold
    if not burned0.any():
        return None
    fuel = layers.sample("fuel_fbfm40", lon, lat, fill=0.0)
    slope = layers.sample("slope_deg", lon, lat, fill=0.0)
    fm100 = layers.sample("fm100_pct", lon, lat)
    if layers.has("wind_u") and layers.has("wind_v"):
        u = np.nan_to_num(layers.sample("wind_u", lon, lat), nan=0.0)
        v = np.nan_to_num(layers.sample("wind_v", lon, lat), nan=0.0)
        speed = np.hypot(u, v)
        safe = np.maximum(speed, 1e-9)
        east = np.where(speed > 0, u / safe, 0.0)
        north = np.where(speed > 0, v / safe, 0.0)
    else:
        speed = np.nan_to_num(layers.sample("wind_speed", lon, lat), nan=0.0)
        east, north = wind_toward_from_met(np.nan_to_num(layers.sample("wind_dir", lon, lat), nan=0.0))
        east = np.where(speed > 0, east, 0.0)
        north = np.where(speed > 0, north, 0.0)

    rate, unknown = spread_rate(fuel, slope, speed, fm100)
    if unknown:
        warnings.append(f"fire {cluster}: FBFM40 codes {unknown} have no spread parameters; treated as non-burnable")
    phi0 = _signed_distance(burned0, res)
    cells = grid.locate(lon, lat)

    # horizon 0: the observed fire; probability = detection value (confidence) clipped to [0, 1]
    if 0 in probabilities:
        sel = burned0 & (cells >= 0)
        for cell, value in zip(cells[sel], np.clip(fire_val[sel], 0, 1)):
            probabilities[0][int(cell)] = max(probabilities[0].get(int(cell), 0.0), float(value))

    counts: dict[int, dict[int, int]] = {h: {} for h in horizons if h > 0}
    rng = np.random.default_rng(_seed(req, cluster))
    reached_edge = False
    for member in range(params.members):
        if member == 0:
            mult, turn = 1.0, 0.0
        else:
            mult = float(np.clip(np.exp(rng.normal(0.0, params.rate_log_sigma)), 0.2, 5.0))
            turn = float(np.deg2rad(rng.normal(0.0, params.wind_dir_sigma_deg)))
        c, s = np.cos(turn), np.sin(turn)
        toward = (c * east - s * north, s * east + c * north)
        fields = propagate(phi0, res, rate * mult, toward, forecast_times) if forecast_times else {}
        for h in counts:
            burned = fields[h * 3600.0] < 0
            edge = burned[0, :].any() or burned[-1, :].any() or burned[:, 0].any() or burned[:, -1].any()
            reached_edge |= bool(edge)
            for cell in np.unique(cells[burned & (cells >= 0)]):
                counts[h][int(cell)] = counts[h].get(int(cell), 0) + 1
    for h, per_cell in counts.items():
        for cell, n in per_cell.items():
            probabilities[h][cell] = n / params.members

    burnable = rate > 0
    meta = {
        "fire": cluster,
        "centre_lon_lat": [round(lon0, 5), round(lat0, 5)],
        "initial_area_km2": round(float(burned0.sum()) * res * res / 1e6, 3),
        "window_cells": [len(ys), len(xs)],
        "resolution_m": round(float(res), 1),
        "median_R_m_s": round(float(np.median(rate[burnable])), 4) if burnable.any() else 0.0,
        "p95_R_m_s": round(float(np.percentile(rate[burnable], 95)), 4) if burnable.any() else 0.0,
    }
    return {"probabilities": probabilities, "warnings": warnings, "reached_edge": reached_edge, "meta": meta}
