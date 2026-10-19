"""Native ingestion: the service fetches its own inputs for a bbox and reference time.

Sponsor guide rev 2 (Q4): the backend no longer writes feature files. For each
request the PINN service pulls

* current fire     NASA FIRMS area API, VIIRS S-NPP / NOAA-20 / NOAA-21 detections
                   in the last ``lookback_hours`` before the reference time
                   (needs a FIRMS map key: env FIRMS_MAP_KEY or firms_key.txt);
* hourly weather   Open-Meteo (no key): 10 m wind speed and direction, RH, temperature,
                   forecast API for recent/future times, ERA5 archive for older ones;
* fuel moisture    gridMET 100-h dead fuel moisture (THREDDS point query); if that
                   fails, an equilibrium-moisture estimate from the hourly RH/T;
* fuel, slope      LANDFIRE FBFM40 + slope, bundled once as an .npz
                   (``build_static_layers.py``) because they do not change hourly.

Everything goes through ``FETCH`` (a function url -> bytes) so tests can replay
recorded responses without network access.

    python -m pinn_service.native_ingestion --smoke     # live check of every source
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import json
import logging
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .contract import PredictionRequest, ServiceError
from .features import FeatureGrid, load_npz

log = logging.getLogger("pinn_service.ingest")

FIRMS_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{start}"
FIRMS_NRT = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
FIRMS_SP = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP")
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
GRIDMET_FM100 = "http://thredds.northwestknowledge.net:8080/thredds/ncss/MET/fm100/fm100_{year}.nc"
HOURLY_VARS = ("wind_speed_10m", "wind_direction_10m", "relative_humidity_2m", "temperature_2m")

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC = PACKAGE_DIR / "static" / "bay_area_static.npz"


# --------------------------------------------------------------------------- HTTP

def _urllib_fetch(url: str, timeout: float = 30.0) -> bytes:
    import urllib.error

    request = urllib.request.Request(url, headers={"User-Agent": "WildfirePINN/0.3 (research)"})
    last = ""
    for attempt in range(2):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            body = exc.read(300).decode("utf-8", "replace").strip().replace("\n", " ")
            last = f"HTTP {exc.code}: {body or exc.reason}"
            if 400 <= exc.code < 500:
                break  # a client error will not fix itself on retry
        except Exception as exc:  # noqa: BLE001 - network errors of every kind are retried once
            last = str(exc)
        time.sleep(0.5 * (attempt + 1))
    raise ConnectionError(f"{urllib.parse.urlsplit(url).netloc}: {last}")


FETCH: Callable[[str], bytes] = _urllib_fetch

_CACHE: dict[str, tuple[float, bytes]] = {}
CACHE_TTL_S = 600.0


def fetch_cached(url: str) -> bytes:
    hit = _CACHE.get(url)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    body = FETCH(url)
    _CACHE[url] = (time.time(), body)
    if len(_CACHE) > 256:
        _CACHE.pop(next(iter(_CACHE)))
    return body


# --------------------------------------------------------------------------- data classes

@dataclass
class Detections:
    lon: np.ndarray
    lat: np.ndarray
    radius_m: np.ndarray       # equal-area disk of the satellite pixel
    confidence: np.ndarray     # 0..1, used as horizon-0 probability
    source: str
    notes: list[str] = field(default_factory=list)
    time_s: np.ndarray | None = None   # acquisition time, Unix seconds (NaN or None = unknown)

    def __post_init__(self):
        if self.time_s is None:
            self.time_s = np.full(len(self.lon), np.nan)

    def __len__(self) -> int:
        return len(self.lon)

    def subset(self, mask: np.ndarray) -> "Detections":
        return Detections(self.lon[mask], self.lat[mask], self.radius_m[mask], self.confidence[mask], self.source,
                          self.notes, self.time_s[mask])


class Environment:
    """Inputs for one fire window. Hour i covers [t0 + i h, t0 + (i+1) h)."""

    notes: list[str]

    def static(self, name: str, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:  # fuel_fbfm40, slope_deg
        raise NotImplementedError

    def wind(self, hour: int, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(speed m/s, east, north) of the direction the wind blows toward."""
        raise NotImplementedError

    def fm100(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def conditions(self, moment: dt.datetime) -> tuple[float, float, float, float]:
        """Point conditions at the fire for any hour in the fetched window (past or future):
        (10 m wind speed m/s, wind FROM degrees, fine-fuel moisture % (Simard EMC from RH/T),
        100-h fuel moisture %). Used by the calibrated forecast."""
        raise NotImplementedError


# --------------------------------------------------------------------------- static layers

def static_layers() -> FeatureGrid:
    path = Path(os.environ.get("STATIC_LAYERS", DEFAULT_STATIC))
    return _static_cached(str(path))


_STATIC: dict[str, FeatureGrid] = {}


def _static_cached(path: str) -> FeatureGrid:
    if path not in _STATIC:
        if not Path(path).exists():
            raise ServiceError(
                503,
                f"static fuel/slope layers not found at {path}. Build them once with "
                "`python -m pinn_service.build_static_layers` (needs internet + rasterio) or set STATIC_LAYERS.",
            )
        grid = load_npz(path, path)
        missing = [n for n in ("fuel_fbfm40", "slope_deg") if not grid.has(n)]
        if missing:
            raise ServiceError(503, f"static layers {path} lack {missing}")
        _STATIC[path] = grid
    return _STATIC[path]


# --------------------------------------------------------------------------- FIRMS

def key_from_text(text: str) -> str | None:
    """First non-blank line that is not a '#' comment (same rule as wildfire_pinn.cli)."""
    for line in text.lstrip("\ufeff").splitlines():
        line = line.strip().strip('"').strip("'")
        if line and not line.startswith("#"):
            return line
    return None


def firms_key() -> str:
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    if key:
        return key
    for folder in (Path.cwd(), PACKAGE_DIR, PACKAGE_DIR.parent):
        path = folder / "firms_key.txt"
        if path.exists():
            key = key_from_text(path.read_text(encoding="utf-8-sig"))
            if key:
                return key
    raise ServiceError(
        503,
        "no NASA FIRMS map key: set FIRMS_MAP_KEY or put the key in firms_key.txt "
        "(free at https://firms.modaps.eosdis.nasa.gov/api/map_key/)",
    )


def parse_firms_csv(text: str) -> list[dict]:
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith(("Invalid", "<", "{")) or "latitude" not in stripped.splitlines()[0]:
        raise ConnectionError("FIRMS rejected the request: " + stripped[:200])
    return list(csv.DictReader(io.StringIO(text)))


def _confidence(value: str) -> float | None:
    v = str(value).strip().lower()
    if v in ("h", "high"):
        return 1.0
    if v in ("n", "nominal"):
        return 0.8
    if v in ("l", "low"):
        return None
    try:  # MODIS: 0-100
        number = float(v)
    except ValueError:
        return None
    return number / 100.0 if number >= 50 else None


def firms_detections(search_bbox: list[float], ref: dt.datetime, lookback_h: float) -> Detections:
    key = firms_key()
    start_time = ref - dt.timedelta(hours=lookback_h)
    days = min(5, (ref.date() - start_time.date()).days + 1)
    start_day = (ref.date() - dt.timedelta(days=days - 1)).isoformat()
    age_days = (dt.datetime.now(dt.timezone.utc) - ref).days
    sources = FIRMS_NRT + (FIRMS_SP if age_days > 7 else ())
    bbox_text = ",".join(f"{v:.4f}" for v in search_bbox)
    rows: list[dict] = []
    failures: list[str] = []
    for source in sources:
        url = FIRMS_URL.format(key=key, source=source, bbox=bbox_text, days=days, start=start_day)
        try:
            for row in parse_firms_csv(fetch_cached(url).decode("utf-8", "replace")):
                row["_source"] = source
                rows.append(row)
        except ConnectionError as exc:
            failures.append(f"{source}: {str(exc).replace(key, '***')}")
    if failures and len(failures) == len(sources):
        raise ServiceError(503, "FIRMS unavailable: " + "; ".join(failures))

    lon, lat, radius, conf, times = [], [], [], [], []
    seen = set()
    for row in rows:
        c = _confidence(row.get("confidence", ""))
        if c is None:
            continue
        try:
            hhmm = str(row["acq_time"]).strip().split(".")[0].zfill(4)
            when = dt.datetime.fromisoformat(f"{row['acq_date']}T{hhmm[:2]}:{hhmm[2:]}:00+00:00")
            x, y = float(row["longitude"]), float(row["latitude"])
        except (KeyError, ValueError):
            continue
        if not (start_time <= when <= ref):
            continue
        dedupe = (round(x, 4), round(y, 4), when)
        if dedupe in seen:
            continue
        seen.add(dedupe)
        scan = float(row.get("scan") or 0.375)
        track = float(row.get("track") or 0.375)
        lon.append(x)
        lat.append(y)
        radius.append(math.sqrt(scan * track / math.pi) * 1000.0)
        conf.append(c)
        times.append(when.timestamp())
    notes = [f"FIRMS {', '.join(sources)}: {len(lon)} detections (nominal/high) in the {lookback_h:g} h before {ref:%Y-%m-%dT%H:%MZ}"]
    if failures:
        notes.append("FIRMS partial: " + "; ".join(failures))
    return Detections(np.array(lon), np.array(lat), np.array(radius), np.array(conf), "firms", notes, np.array(times, dtype=float))


# --------------------------------------------------------------------------- weather

@dataclass
class HourlyWeather:
    times: list[dt.datetime]
    wind_speed: np.ndarray     # m/s
    wind_from: np.ndarray      # degrees, meteorological
    rh: np.ndarray             # %
    temp_c: np.ndarray
    source: str

    def at(self, moment: dt.datetime) -> int:
        """Index of the hourly record nearest to ``moment``."""
        target = moment.timestamp()
        return int(np.argmin([abs(t.timestamp() - target) for t in self.times]))


def open_meteo_hourly(lon: float, lat: float, start: dt.datetime, end: dt.datetime) -> HourlyWeather:
    now = dt.datetime.now(dt.timezone.utc)
    recent = (now - start).days <= 85
    base = OPEN_METEO_FORECAST if recent else OPEN_METEO_ARCHIVE
    query = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "hourly": ",".join(HOURLY_VARS),
        "wind_speed_unit": "ms",
        "timezone": "GMT",
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
    })
    try:
        payload = json.loads(fetch_cached(f"{base}?{query}"))
    except (ConnectionError, ValueError) as exc:
        raise ServiceError(503, f"weather unavailable from Open-Meteo: {exc}") from exc
    if payload.get("error"):
        raise ServiceError(503, f"Open-Meteo: {payload.get('reason', 'error')}")
    hourly = payload.get("hourly") or {}
    times = [dt.datetime.fromisoformat(t).replace(tzinfo=dt.timezone.utc) for t in hourly.get("time", [])]
    if not times:
        raise ServiceError(503, "Open-Meteo returned no hourly records")

    def column(name):
        values = np.array([np.nan if v is None else float(v) for v in hourly.get(name, [None] * len(times))])
        if np.isnan(values).all():
            raise ServiceError(503, f"Open-Meteo returned no {name}")
        return np.where(np.isnan(values), np.nanmean(values), values)

    return HourlyWeather(
        times, column("wind_speed_10m"), column("wind_direction_10m"),
        column("relative_humidity_2m"), column("temperature_2m"),
        "Open-Meteo forecast" if recent else "Open-Meteo ERA5 archive",
    )


def gridmet_fm100(lon: float, lat: float, day: dt.date) -> float:
    """Latest gridMET 100-h fuel moisture (%) on or before ``day`` at a point."""
    first = day - dt.timedelta(days=3)  # gridMET runs a day or two behind
    query = urllib.parse.urlencode({
        "var": "dead_fuel_moisture_100hr",
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "time_start": f"{first.isoformat()}T00:00:00Z",
        "time_end": f"{day.isoformat()}T00:00:00Z",
        "accept": "csv",
    })
    text = fetch_cached(f"{GRIDMET_FM100.format(year=day.year)}?{query}").decode("utf-8", "replace")
    return parse_gridmet_csv(text)


def parse_gridmet_csv(text: str) -> float:
    rows = [r for r in csv.reader(io.StringIO(text)) if r]
    preview = " ".join(text.split())[:160]
    if len(rows) < 2:
        raise ValueError(f"gridMET returned no rows: {preview!r}")
    header = [h.lower() for h in rows[0]]
    col = next((i for i, h in enumerate(header) if "moisture" in h or "fm100" in h), len(header) - 1)
    for row in reversed(rows[1:]):
        try:
            value = unpack_fm100(float(row[col]))
        except (ValueError, IndexError):
            continue
        if value is not None:
            return value
    raise ValueError(f"gridMET rows had no valid fuel moisture: {preview!r}")


def unpack_fm100(raw: float) -> float | None:
    """gridMET stores fm100 as an integer with scale_factor 0.1, and the THREDDS point-CSV
    service returns that packed integer (seen live: 149.0 'Percent' = 14.9 %, 45.0 = 4.5 %).
    So a whole number is always packed and is divided by 10. A value with decimals is already
    in percent and kept. The 32767 fill value and anything outside 1-60 % is rejected.
    (An earlier rule kept packed values <= 60 as-is, reading 4.5 % as 45 %: no spread on
    the driest days.)"""
    if raw != raw or raw <= 0:
        return None
    value = raw * 0.1 if float(raw).is_integer() else float(raw)
    return value if 1.0 <= value <= 60.0 else None


def emc_percent(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Equilibrium moisture content (Simard 1968), the NFDRS building block. T in C, RH in %."""
    t = np.asarray(temp_c) * 9 / 5 + 32
    h = np.clip(np.asarray(rh, dtype=float), 0, 100)
    return np.where(
        h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t,
        np.where(h < 50, 2.22749 + 0.160107 * h - 0.01478 * t,
                 21.0606 + 0.005565 * h ** 2 - 0.00035 * h * t - 0.483199 * h),
    )


# --------------------------------------------------------------------------- environment

class NativeEnvironment(Environment):
    """Weather from one point (the fire centroid), uniform across the window, hour by hour."""

    def __init__(self, lon0: float, lat0: float, ref: dt.datetime, hours: int, history_hours: float = 24.0):
        self.notes: list[str] = []
        self._static = static_layers()
        self.ref = ref
        weather = open_meteo_hourly(lon0, lat0, ref - dt.timedelta(hours=max(24.0, history_hours)),
                                    ref + dt.timedelta(hours=max(hours, 12) + 1))
        self._emc = np.clip(emc_percent(weather.temp_c, weather.rh), 2.0, 35.0)
        self.weather = weather
        self._hours = []
        for i in range(max(hours, 1)):
            k = weather.at(ref + dt.timedelta(hours=i + 0.5))
            toward = np.deg2rad(weather.wind_from[k]) + np.pi
            self._hours.append((float(weather.wind_speed[k]), float(np.sin(toward)), float(np.cos(toward))))
        k0 = weather.at(ref)
        self.notes.append(
            f"{weather.source} at ({lon0:.3f}, {lat0:.3f}): wind {weather.wind_speed[k0]:.1f} m/s from "
            f"{weather.wind_from[k0]:.0f} deg, RH {weather.rh[k0]:.0f} %, T {weather.temp_c[k0]:.0f} C at reference time"
        )
        try:
            self._fm100 = gridmet_fm100(lon0, lat0, ref.date())
            self.notes.append(f"gridMET FM100 {self._fm100:.1f} %")
        except Exception as exc:  # noqa: BLE001
            past = [i for i, t in enumerate(weather.times) if ref - dt.timedelta(hours=24) <= t <= ref]
            self._fm100 = float(np.mean(emc_percent(weather.temp_c[past], weather.rh[past]))) if past else 10.0
            self.notes.append(f"FM100 {self._fm100:.1f} % estimated from 24 h RH/T (gridMET unavailable: {str(exc)[:80]})")

    def static(self, name, lon, lat):
        return self._static.sample(name, lon, lat, fill=0.0)

    def wind(self, hour, lon, lat):
        speed, east, north = self._hours[min(hour, len(self._hours) - 1)]
        shape = np.shape(lon)
        return np.full(shape, speed), np.full(shape, east), np.full(shape, north)

    def fm100(self, lon, lat):
        return np.full(np.shape(lon), self._fm100)

    def conditions(self, moment):
        k = self.weather.at(moment)
        return float(self.weather.wind_speed[k]), float(self.weather.wind_from[k]), float(self._emc[k]), float(self._fm100)


class NativeSource:
    name = "native"

    def detections(self, req: PredictionRequest, search_bbox: list[float]) -> Detections:
        p = req.additional_params
        hours = max(p.lookback_hours, p.history_hours) if p.method in ("calibrated", "nowcast") else p.lookback_hours
        return firms_detections(search_bbox, req.reference_time, hours)

    def environment(self, req: PredictionRequest, lon0: float, lat0: float) -> Environment:
        p = req.additional_params
        history = max(p.lookback_hours, p.history_hours) if p.method in ("calibrated", "nowcast") else 24.0
        return NativeEnvironment(lon0, lat0, req.reference_time, max(req.horizons), history)

    def fuel(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return static_layers().sample("fuel_fbfm40", lon, lat, fill=0.0)

    def describe(self) -> dict:
        path = os.environ.get("STATIC_LAYERS", str(DEFAULT_STATIC))
        return {"ingestion": "native (FIRMS, Open-Meteo, gridMET, LANDFIRE static)", "static_layers": path,
                "static_layers_present": Path(path).exists()}


# --------------------------------------------------------------------------- smoke test

def _fm100_check(ref: dt.datetime) -> str:  # pragma: no cover - live network
    fm = gridmet_fm100(-122.22, 37.17, ref.date())
    w = open_meteo_hourly(-122.22, 37.17, ref - dt.timedelta(hours=24), ref)
    emc = float(np.mean(emc_percent(w.temp_c, w.rh)))
    return f"{fm:.1f} % (independent 24 h RH/T estimate: {emc:.1f} %; the two should be the same order of magnitude)"


def _smoke() -> int:  # pragma: no cover - live network check for a developer machine
    logging.basicConfig(level=logging.INFO)
    ref = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours=3)
    ok = True
    for name, call in (
        ("Open-Meteo", lambda: open_meteo_hourly(-122.22, 37.17, ref - dt.timedelta(hours=24), ref + dt.timedelta(hours=7)).source),
        ("gridMET FM100", lambda: _fm100_check(ref)),
        ("FIRMS (all of California, last 24 h)", lambda: f"{len(firms_detections([-124.5, 32.5, -114.0, 42.1], ref, 24))} detections"),
        ("static layers", lambda: static_layers().source),
    ):
        try:
            print(f"OK   {name}: {call()}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAIL {name}: {getattr(exc, 'detail', exc)}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    import sys

    if "--smoke" in sys.argv:
        sys.exit(_smoke())
    print(__doc__)
