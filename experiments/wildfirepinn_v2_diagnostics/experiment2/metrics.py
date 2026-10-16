"""Front scores against a numerical reference. Distances are meters."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from shapely import distance, points
from shapely.geometry import LineString, MultiLineString

from experiments.wildfirepinn_v2_diagnostics.circle import mask_iou
from wildfire_pinn.geometry import zero_contours


def _lines(phi, x, y):
    raw = [line for line in zero_contours(phi, x, y) if len(line) >= 2]
    if not raw:
        return None, raw
    geom = MultiLineString([LineString(line) for line in raw])
    stacked = np.concatenate(raw, axis=0)
    return geom, stacked


def contour_distances(predicted, reference, x, y) -> dict:
    pred_geom, pred_pts = _lines(predicted, x, y)
    ref_geom, ref_pts = _lines(reference, x, y)
    n_pred = 0 if pred_geom is None else len(pred_geom.geoms)
    n_ref = 0 if ref_geom is None else len(ref_geom.geoms)
    if pred_geom is None or ref_geom is None:
        return {"symmetric_m": None, "p95_m": None, "n_pred_contours": n_pred, "n_ref_contours": n_ref}
    forward = np.asarray(distance(points(pred_pts), ref_geom), dtype=np.float64)
    backward = np.asarray(distance(points(ref_pts), pred_geom), dtype=np.float64)
    pooled = np.concatenate([forward, backward])
    symmetric = 0.5 * (float(forward.mean()) + float(backward.mean()))
    return {
        "symmetric_m": symmetric,
        "p95_m": float(np.percentile(pooled, 95)),
        "n_pred_contours": n_pred,
        "n_ref_contours": n_ref,
    }


def region_flags(phi, x, y) -> dict:
    burned = np.asarray(phi) < 0
    _labels, count = label(burned)
    cell = float(abs(x[1] - x[0]) * abs(y[1] - y[0]))
    return {
        "burned_components": int(count),
        "burned_area_km2": float(burned.sum() * cell / 1e6),
        "n_contours": len([line for line in zero_contours(phi, x, y) if len(line) >= 2]),
    }


def score_against_reference(predicted, reference, x, y) -> dict:
    distances = contour_distances(predicted, reference, x, y)
    pred_region = region_flags(predicted, x, y)
    ref_region = region_flags(reference, x, y)
    return {
        "iou": mask_iou(predicted, reference),
        "symmetric_distance_m": distances["symmetric_m"],
        "p95_distance_m": distances["p95_m"],
        "area_error_km2": pred_region["burned_area_km2"] - ref_region["burned_area_km2"],
        "pred_area_km2": pred_region["burned_area_km2"],
        "ref_area_km2": ref_region["burned_area_km2"],
        "burned_components": pred_region["burned_components"],
        "n_contours": pred_region["n_contours"],
    }


def support_growth(phi0, phi1, x, y, x0: float, y0: float, east: float, north: float) -> dict:
    """How far the burned set advanced along a unit direction and its opposite."""

    def reach(phi, sign: float) -> float:
        burned = np.asarray(phi) < 0
        if not burned.any():
            return float("nan")
        yy, xx = np.meshgrid(y, x, indexing="ij")
        projection = sign * ((xx - x0) * east + (yy - y0) * north)
        return float(projection[burned].max())

    initial_down = reach(phi0, 1.0)
    final_down = reach(phi1, 1.0)
    initial_up = reach(phi0, -1.0)
    final_up = reach(phi1, -1.0)
    return {
        "downwind_growth_m": final_down - initial_down,
        "upwind_growth_m": final_up - initial_up,
    }
