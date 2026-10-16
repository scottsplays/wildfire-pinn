"""Load the daily feature file and sample it at arbitrary lon/lat points.

The file layout is specified in ``feature_spec.md``. Two layouts are accepted:

* named arrays: ``lat``, ``lon`` (1-D cell centres, EPSG:4326) plus one 2-D
  array per channel, shape (len(lat), len(lon));
* stacked: ``lat``, ``lon``, ``features`` with shape (H, W, C) and ``channels``,
  a length-C array of channel names.

Production reads ``gs://fightfirewai-features/daily/YYYY-MM-DD.npz``; local
development reads ``$FEATURE_DIR/YYYY-MM-DD.npz``.
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
    daily = _load_cached(date)
    static_path = os.environ.get("STATIC_LAYERS")
    return Layers(daily, _load_static(static_path) if static_path else None)


@lru_cache(maxsize=1)
def _load_static(path: str) -> FeatureGrid:
    if not Path(path).exists():
        raise ServiceError(500, f"STATIC_LAYERS points at a missing file: {path}")
    return load_npz(path, path)
