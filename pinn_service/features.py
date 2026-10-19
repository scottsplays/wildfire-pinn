"""Gridded .npz inputs: the bundled static layers, and the offline replay mode.

Since sponsor guide rev 2 the backend writes no feature files; live requests use
``native_ingestion``. This module still (a) reads the static LANDFIRE layers and
(b) serves ``INGESTION=npz`` replays (tests, demos without network, historical
cases), where one file holds every input for a day.

The file layout is specified in ``feature_spec.md``. Two layouts are accepted:

* named arrays: ``lat``, ``lon`` (1-D cell centres, EPSG:4326) plus one 2-D
  array per channel, shape (len(lat), len(lon));
* stacked: ``lat``, ``lon``, ``features`` with shape (H, W, C) and ``channels``,
  a length-C array of channel names.

Replays read ``$FEATURE_DIR/YYYY-MM-DD.npz`` (or a GCS template with FEATURE_SOURCE=gcs).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .contract import ServiceError

# canonical name -> accepted aliases (first match wins)
ALIASES = {
    "active_fire": ("active_fire", "fire_confidence", "fire_mask", "fire"),
    "fuel_fbfm40": ("fuel_fbfm40", "fbfm40", "fuel_model"),
    "slope_deg": ("slope_deg", "slope"),
    "elevation_m": ("elevation_m", "elevation", "elev"),
    "fm100_pct": ("fm100_pct", "fm100", "fuel_moisture_100h", "fuel_moisture"),
    "wind_u": ("wind_u", "u10", "u"),
    "wind_v": ("wind_v", "v10", "v"),
    "wind_speed": ("wind_speed", "wind_speed_ms", "vs"),
    "wind_dir": ("wind_dir", "wind_from_deg", "th"),
}

REQUIRED_FOR_PINN = ("active_fire", "fuel_fbfm40", "slope_deg", "fm100_pct", "wind")


@dataclass
class FeatureGrid:
    lat: np.ndarray            # 1-D, cell centres
    lon: np.ndarray            # 1-D, cell centres
    channels: dict[str, np.ndarray]
    source: str
    notes: list[str] = field(default_factory=list)

    @property
    def dlat(self) -> float:
        return float(abs(self.lat[1] - self.lat[0])) if len(self.lat) > 1 else 0.01

    @property
    def dlon(self) -> float:
        return float(abs(self.lon[1] - self.lon[0])) if len(self.lon) > 1 else 0.01

    def has(self, name: str) -> bool:
        return name in self.channels

    def index(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest (row, col) for each point, plus a mask of points inside the grid."""
        rows = _nearest(self.lat, lat)
        cols = _nearest(self.lon, lon)
        inside = (
            (lat >= self.lat.min() - self.dlat / 2) & (lat <= self.lat.max() + self.dlat / 2)
            & (lon >= self.lon.min() - self.dlon / 2) & (lon <= self.lon.max() + self.dlon / 2)
        )
        return rows, cols, inside

    def sample(self, name: str, lon: np.ndarray, lat: np.ndarray, fill: float = np.nan) -> np.ndarray:
        rows, cols, inside = self.index(lon, lat)
        values = self.channels[name][rows, cols].astype(np.float64)
        return np.where(inside, values, fill)


def _nearest(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    if axis[0] <= axis[-1]:
        idx = np.searchsorted(axis, values)
        idx = np.clip(idx, 1, len(axis) - 1)
        left = axis[idx - 1]
        right = axis[idx]
        return np.where(np.abs(values - left) <= np.abs(values - right), idx - 1, idx)
    flipped = _nearest(axis[::-1], values)
    return len(axis) - 1 - flipped


def _canonicalise(raw: dict[str, np.ndarray], shape: tuple[int, int], notes: list[str]) -> dict[str, np.ndarray]:
    channels: dict[str, np.ndarray] = {}
    for canon, names in ALIASES.items():
        for name in names:
            if name in raw:
                array = np.asarray(raw[name], dtype=np.float64)
                if array.shape != shape:
                    raise ServiceError(422, f"channel '{name}' has shape {array.shape}, expected {shape}")
                channels[canon] = array
                break
    if "slope_deg" not in channels and "elevation_m" in channels:
        notes.append("slope_deg derived from elevation_m")
        channels["slope_deg"] = None  # filled by load_npz once spacing is known
    return channels


def _slope_from_elevation(elev: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat0 = float(np.mean(lat))
    m_lat = 111132.92 - 559.82 * np.cos(2 * np.deg2rad(lat0))
    m_lon = 111412.84 * np.cos(np.deg2rad(lat0))
    dy = abs(lat[1] - lat[0]) * m_lat
    dx = abs(lon[1] - lon[0]) * m_lon
    gy, gx = np.gradient(elev, dy, dx)
    return np.rad2deg(np.arctan(np.hypot(gx, gy)))


def load_npz(data: bytes | str | Path, source: str) -> FeatureGrid:
    with np.load(io.BytesIO(data) if isinstance(data, bytes) else data, allow_pickle=False) as npz:
        raw = {key: npz[key] for key in npz.files}
    if "lat" not in raw or "lon" not in raw:
        raise ServiceError(422, f"{source}: feature file needs 1-D 'lat' and 'lon' arrays")
    lat = np.asarray(raw.pop("lat"), dtype=np.float64).ravel()
    lon = np.asarray(raw.pop("lon"), dtype=np.float64).ravel()
    shape = (len(lat), len(lon))
    if "features" in raw:
        stack = np.asarray(raw.pop("features"))
        names = [str(n) for n in raw.pop("channels", [])]
        if stack.ndim != 3 or stack.shape[:2] != shape or len(names) != stack.shape[2]:
            raise ServiceError(422, f"{source}: 'features' must be (H, W, C) with C channel names")
        raw.update({name: stack[:, :, i] for i, name in enumerate(names)})
    notes: list[str] = []
    channels = _canonicalise(raw, shape, notes)
    if channels.get("slope_deg", 0) is None:
        channels["slope_deg"] = _slope_from_elevation(channels["elevation_m"], lat, lon)
    return FeatureGrid(lat=lat, lon=lon, channels=channels, source=source, notes=notes)


@lru_cache(maxsize=4)
def _load_cached(date: str) -> FeatureGrid:
    source_kind = os.environ.get("FEATURE_SOURCE", "local").lower()
    if source_kind == "gcs":
        template = os.environ.get("FEATURE_GCS_TEMPLATE", "gs://fightfirewai-features/daily/{date}.npz")
        uri = template.format(date=date)
        try:
            from google.cloud import storage  # imported lazily: only needed in production
        except ImportError as exc:  # pragma: no cover
            raise ServiceError(500, "FEATURE_SOURCE=gcs needs google-cloud-storage installed") from exc
        bucket_name, _, blob_name = uri.removeprefix("gs://").partition("/")
        blob = storage.Client().bucket(bucket_name).blob(blob_name)
        if not blob.exists():
            raise ServiceError(404, f"no feature file for {date} at {uri}")
        return load_npz(blob.download_as_bytes(), uri)
    folder = Path(os.environ.get("FEATURE_DIR", Path(__file__).resolve().parent / "test_features"))
    path = folder / f"{date}.npz"
    if not path.exists():
        raise ServiceError(404, f"no feature file for {date} at {path}")
    return load_npz(path, str(path))


@dataclass
class Layers:
    """Daily features plus optional static layers (fuel, slope at native resolution)."""

    daily: FeatureGrid
    static: FeatureGrid | None = None

    def grid_for(self, name: str) -> FeatureGrid | None:
        """Static layers win for fuel/slope (finer, e.g. 30-90 m LANDFIRE); daily otherwise."""
        if self.static is not None and name in ("fuel_fbfm40", "slope_deg") and self.static.has(name):
            return self.static
        if self.daily.has(name):
            return self.daily
        if self.static is not None and self.static.has(name):
            return self.static
        return None

    def has(self, name: str) -> bool:
        return self.grid_for(name) is not None

    def sample(self, name: str, lon: np.ndarray, lat: np.ndarray, fill: float = np.nan) -> np.ndarray:
        grid = self.grid_for(name)
        if grid is None:
            raise KeyError(name)
        return grid.sample(name, lon, lat, fill)

    def missing_for_pinn(self) -> list[str]:
        missing = [c for c in REQUIRED_FOR_PINN if c != "wind" and not self.has(c)]
        has_uv = self.has("wind_u") and self.has("wind_v")
        has_sd = self.has("wind_speed") and self.has("wind_dir")
        if not (has_uv or has_sd):
            missing.append("wind (wind_u + wind_v, or wind_speed + wind_dir)")
        return missing

    @property
    def notes(self) -> list[str]:
        notes = list(self.daily.notes)
        if self.static is not None:
            used = [n for n in ("fuel_fbfm40", "slope_deg") if self.grid_for(n) is self.static]
            if used:
                notes.append(f"{', '.join(used)} from static layers {self.static.source}")
        return notes


def load_features(date: str) -> Layers:
    """Daily replay file for ``date`` (YYYY-MM-DD) plus optional STATIC_LAYERS."""
    daily = _load_cached(date)
    static_path = os.environ.get("STATIC_LAYERS")
    return Layers(daily, _load_static(static_path) if static_path else None)


@lru_cache(maxsize=1)
def _load_static(path: str) -> FeatureGrid:
    if not Path(path).exists():
        raise ServiceError(500, f"STATIC_LAYERS points at a missing file: {path}")
    return load_npz(path, path)


# --------------------------------------------------------------------------- replay source

class NpzEnvironment:
    """Environment backed by a daily feature file (weather constant over the forecast)."""

    def __init__(self, layers: Layers, lon0: float | None = None, lat0: float | None = None):
        self.layers = layers
        self.notes = list(layers.notes) + [f"replay inputs from {layers.daily.source}"]
        self.lon0, self.lat0 = lon0, lat0

    def static(self, name, lon, lat):
        return self.layers.sample(name, lon, lat, fill=0.0)

    def wind(self, hour, lon, lat):
        from .physics import wind_toward_from_met

        if self.layers.has("wind_u") and self.layers.has("wind_v"):
            u = np.nan_to_num(self.layers.sample("wind_u", lon, lat), nan=0.0)
            v = np.nan_to_num(self.layers.sample("wind_v", lon, lat), nan=0.0)
            speed = np.hypot(u, v)
            safe = np.maximum(speed, 1e-9)
            return speed, np.where(speed > 0, u / safe, 0.0), np.where(speed > 0, v / safe, 0.0)
        speed = np.nan_to_num(self.layers.sample("wind_speed", lon, lat), nan=0.0)
        east, north = wind_toward_from_met(np.nan_to_num(self.layers.sample("wind_dir", lon, lat), nan=0.0))
        return speed, np.where(speed > 0, east, 0.0), np.where(speed > 0, north, 0.0)

    def fm100(self, lon, lat):
        return self.layers.sample("fm100_pct", lon, lat)

    def conditions(self, moment):
        """Replays hold one daily snapshot: the same conditions every hour. With no RH/T in the
        file, the fine-fuel moisture falls back to FM100."""
        point = (np.array([self.lon0]), np.array([self.lat0]))
        speed, east, north = (float(v[0]) for v in self.wind(0, *point))
        wind_from = float(np.rad2deg(np.arctan2(-east, -north)) % 360) if speed > 0 else 0.0
        fm = float(np.nan_to_num(self.fm100(*point)[0], nan=10.0))
        return speed, wind_from, float(np.clip(fm, 2.0, 35.0)), fm


class NpzSource:
    """Offline replay: inputs from ``$FEATURE_DIR/YYYY-MM-DD.npz`` (see feature_spec.md)."""

    name = "npz"

    def __init__(self):
        self._layers: Layers | None = None

    def _load(self, req) -> Layers:
        layers = load_features(req.day)
        if not layers.daily.has("active_fire"):
            raise ServiceError(422, "replay file has no active_fire channel")
        missing = layers.missing_for_pinn()
        if missing:
            raise ServiceError(422, f"replay file lacks channels the PINN needs: {', '.join(missing)} (see feature_spec.md)")
        return layers

    def detections(self, req, search_bbox):
        from .native_ingestion import Detections

        layers = self._load(req)
        self._layers = layers
        daily = layers.daily
        lon2, lat2 = np.meshgrid(daily.lon, daily.lat)
        fire = np.nan_to_num(daily.channels["active_fire"], nan=0.0)
        b = search_bbox
        sel = (fire >= req.additional_params.fire_threshold) & (lon2 >= b[0]) & (lon2 <= b[2]) & (lat2 >= b[1]) & (lat2 <= b[3])
        m_lat = 111132.92 - 559.82 * np.cos(2 * np.deg2rad(float(np.mean(daily.lat))))
        m_lon = 111412.84 * np.cos(np.deg2rad(float(np.mean(daily.lat))))
        half_diagonal = 0.5 * float(np.hypot(daily.dlat * m_lat, daily.dlon * m_lon))  # disks tile the pixels
        n = int(sel.sum())
        return Detections(lon2[sel], lat2[sel], np.full(n, half_diagonal), np.clip(fire[sel], 0, 1), "npz",
                          [f"replay: {n} active_fire pixels from {daily.source}"])

    def environment(self, req, lon0, lat0):
        return NpzEnvironment(self._layers or self._load(req), lon0, lat0)

    def fuel(self, lon, lat):
        return self._layers.sample("fuel_fbfm40", lon, lat, fill=0.0)

    def describe(self) -> dict:
        return {"ingestion": "npz replay", "feature_dir": os.environ.get("FEATURE_DIR", str(Path(__file__).resolve().parent / "test_features"))}
