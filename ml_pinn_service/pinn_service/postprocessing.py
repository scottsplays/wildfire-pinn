"""Map forecasts onto the 1 km Bay Area grid and build the GeoJSON response.

Contract rules enforced here (sponsor guide, section C):
  * one FeatureCollection holding every horizon in {0, 1, 3, 6} <= time_horizon
  * geometry is always a closed Polygon, [lon, lat] order, clamped to the bbox
  * properties: grid_id (int > 0), fire_probability (0..1), horizon_hours,
    source_model, members

The grid. Sponsor FAQ Q1 (rev 2): any stable California 1 km lattice id is
accepted; the backend stubs unknown cells from the polygon we send. For the
ensemble to combine U-Net and PINN cell by cell, both services should use the
SAME lattice. Until the U-Net's lattice is shared, the service uses a
provisional California-wide 0.01 deg lattice (the cell geometry of the adapter's
U-Net fixtures); point ``GRID_SPEC`` at a
CSV (grid_id,min_lon,min_lat,max_lon,max_lat) or JSON spec to switch. The
response metadata says which lattice was used.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .contract import MODEL_TYPE, MODEL_VERSION

CALIFORNIA_EXTENT = (-124.5, 32.5, -114.0, 42.1)  # multiples of 0.01 deg


@dataclass(frozen=True)
class GridSpec:
    """A regular lon/lat grid. ``ids[row, col]`` is the grid_id (0 = no cell)."""

    min_lon: float
    min_lat: float
    dlon: float
    dlat: float
    ids: np.ndarray            # (nrows, ncols), row 0 = southernmost
    name: str
    provisional: bool

    @property
    def nrows(self) -> int:
        return self.ids.shape[0]

    @property
    def ncols(self) -> int:
        return self.ids.shape[1]

    @property
    def resolution_km(self) -> float:
        lat0 = self.min_lat + self.nrows * self.dlat / 2
        return round(float(np.sqrt(self.dlat * 111.0 * self.dlon * 111.0 * np.cos(np.deg2rad(lat0)))), 3)

    def locate(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Flat cell index (row * ncols + col) for each point, or -1 outside the grid / no id."""
        col = np.floor((np.asarray(lon) - self.min_lon) / self.dlon).astype(np.int64)
        row = np.floor((np.asarray(lat) - self.min_lat) / self.dlat).astype(np.int64)
        ok = (col >= 0) & (col < self.ncols) & (row >= 0) & (row < self.nrows)
        flat = np.where(ok, row * self.ncols + col, -1)
        has_id = np.zeros_like(ok)
        has_id[ok] = self.ids.ravel()[flat[ok]] > 0
        return np.where(has_id, flat, -1)

    def bounds(self, flat: int) -> tuple[float, float, float, float]:
        row, col = divmod(int(flat), self.ncols)
        x0 = self.min_lon + col * self.dlon
        y0 = self.min_lat + row * self.dlat
        return x0, y0, x0 + self.dlon, y0 + self.dlat

    def grid_id(self, flat: int) -> int:
        return int(self.ids.ravel()[int(flat)])


def provisional_grid(bbox=CALIFORNIA_EXTENT, step_deg: float = 0.01) -> GridSpec:
    """Regular 0.01 x 0.01 degree lattice over California, cell edges on round 0.01 degree lines.

    This matches the cell geometry in the team adapter's U-Net fixtures
    (ml_adapter/fixtures/frontend_team_geojson_files: e.g. [-122.15, 37.05]-[-122.14, 37.06]),
    so PINN and U-Net cells coincide. A cell is ~0.89 km (E-W) x 1.11 km (N-S) at 37 N.
    ids: row-major from the SW corner (-124.5, 32.5), starting at 1. The id formula is
    provisional until the U-Net's numbering is shared; switch with GRID_SPEC.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    nrows = int(round((max_lat - min_lat) / step_deg))
    ncols = int(round((max_lon - min_lon) / step_deg))
    ids = np.arange(1, nrows * ncols + 1, dtype=np.int64).reshape(nrows, ncols)
    return GridSpec(min_lon, min_lat, step_deg, step_deg, ids, "provisional California 0.01 deg lattice (same cells as the adapter fixtures)", True)


def grid_from_csv(path: Path) -> GridSpec:
    """Load the backend's BayAreaGrid export: grid_id,min_lon,min_lat,max_lon,max_lat (regular grid)."""
    with open(path, newline="") as handle:
        rows = [
            (int(r["grid_id"]), float(r["min_lon"]), float(r["min_lat"]), float(r["max_lon"]), float(r["max_lat"]))
            for r in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"{path} has no cells")
    arr = np.array([r[1:] for r in rows])
    dlon = float(np.median(arr[:, 2] - arr[:, 0]))
    dlat = float(np.median(arr[:, 3] - arr[:, 1]))
    min_lon = float(arr[:, 0].min())
    min_lat = float(arr[:, 1].min())
    cols = np.rint((arr[:, 0] - min_lon) / dlon).astype(int)
    rws = np.rint((arr[:, 1] - min_lat) / dlat).astype(int)
    if np.max(np.abs((arr[:, 0] - min_lon) / dlon - cols)) > 0.05 or np.max(np.abs((arr[:, 1] - min_lat) / dlat - rws)) > 0.05:
        raise ValueError(f"{path} is not a regular lon/lat grid; ask for a regular export or extend grid_from_csv")
    ids = np.zeros((rws.max() + 1, cols.max() + 1), dtype=np.int64)
    ids[rws, cols] = [r[0] for r in rows]
    return GridSpec(min_lon, min_lat, dlon, dlat, ids, f"BayAreaGrid from {Path(path).name}", False)


@lru_cache(maxsize=1)
def active_grid() -> GridSpec:
    spec = os.environ.get("GRID_SPEC")
    if spec:
        path = Path(spec)
        if path.suffix.lower() == ".csv":
            return grid_from_csv(path)
        data = json.loads(path.read_text())
        ids = np.arange(data.get("first_id", 1), data.get("first_id", 1) + data["nrows"] * data["ncols"])
        ids = ids.reshape(data["nrows"], data["ncols"])
        if data.get("row_order", "south_to_north") == "north_to_south":
            ids = ids[::-1]
        return GridSpec(data["min_lon"], data["min_lat"], data["dlon"], data["dlat"], ids, f"grid spec {path.name}", False)
    return provisional_grid()


def _ring(bounds, bbox) -> list[list[float]] | None:
    x0, y0, x1, y1 = bounds
    x0, y0 = max(x0, bbox[0]), max(y0, bbox[1])
    x1, y1 = min(x1, bbox[2]), min(y1, bbox[3])
    if x1 <= x0 or y1 <= y0:
        return None
    r = lambda v: round(float(v), 6)  # noqa: E731
    # NW, NE, SE, SW, NW: the order the sponsor fixture uses
    return [[r(x0), r(y1)], [r(x1), r(y1)], [r(x1), r(y0)], [r(x0), r(y0)], [r(x0), r(y1)]]


def risk_label(horizon: int, p: float, high: float, moderate: float) -> str:
    if horizon == 0:
        return "ACTIVE_FIRE"
    if p >= high:
        return "HIGH_RISK"
    if p >= moderate:
        return "MODERATE_RISK"
    return "LOW_RISK"


def build_feature_collection(
    probabilities: dict[int, dict[int, float]],
    ensemble_size: int,
    bbox: list[float],
    grid: GridSpec,
    threshold: float,
    extra_metadata: dict,
    high_risk: float = 0.6,
    moderate_risk: float = 0.3,
) -> dict:
    """``probabilities[horizon][flat_cell] = p``. Cells with p <= 0 or p < threshold are left out.

    Contract rev 2: ``members`` = 1 for a standalone model that predicts fire in the
    cell (the adapter counts models when it builds the ensemble). The PINN's own
    internal ensemble is reported as ``ensemble_votes`` / ``ensemble_size``.
    """
    features = []
    clipped = 0
    for horizon in sorted(probabilities):
        for flat, p in sorted(probabilities[horizon].items()):
            p = float(np.clip(p, 0.0, 1.0))
            if p <= 0.0 or p < threshold:
                continue
            ring = _ring(grid.bounds(flat), bbox)
            if ring is None:
                clipped += 1
                continue
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "grid_id": grid.grid_id(flat),
                    "fire_probability": round(p, 4),
                    "horizon_hours": int(horizon),
                    "source_model": MODEL_TYPE,
                    "members": 1,
                    "risk_label": risk_label(horizon, p, high_risk, moderate_risk),
                    "ensemble_votes": int(round(p * ensemble_size)) if horizon > 0 else ensemble_size,
                    "ensemble_size": int(ensemble_size),
                },
            })
    metadata = {
        "timestamp": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_model": MODEL_TYPE,
        "model_version": MODEL_VERSION,
        "grid_resolution_km": grid.resolution_km,
        "total_cells": len(features),
        "grid": grid.name,
        "grid_ids_provisional": grid.provisional,
        "cells_outside_bbox_dropped": clipped,
    }
    metadata.update(extra_metadata)
    return {"type": "FeatureCollection", "metadata": metadata, "features": features}
