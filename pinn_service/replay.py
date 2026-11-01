"""Replay source: run the deployed service on a past fire from a saved fire folder, offline.

A fire folder (written by pinn_v2/fetch_fire_data.py, e.g. data/fires/20260804_c38p11n120p73w/) holds
everything the service would otherwise fetch live:

    detections.csv   VIIRS detections (FIRMS archive), one row per pixel, with time_utc
    weather.csv      hourly 10 m wind, RH, temperature at the fire centre (Open-Meteo archive)
    fm100.csv        daily gridMET 100-h fuel moisture
    static.npz       LANDFIRE fuel (FBFM40) and slope on a local metre grid around the fire

``ReplaySource`` serves those files through the same interface as ``NativeSource``
(detections up to the reference time only, so a replayed forecast never sees the future), and the
forecast code runs unchanged. Used by the scorecard's replay mode (scorecard.py --replay).
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

from .contract import PredictionRequest
from .features import FeatureGrid
from .native_ingestion import Detections, HourlyWeather, NativeEnvironment, emc_percent


def _utc(text: str) -> dt.datetime:
    moment = dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00").replace(" ", "T"))
    return moment if moment.tzinfo else moment.replace(tzinfo=dt.timezone.utc)


def _metres_per_degree(lat0: float) -> tuple[float, float]:
    phi = math.radians(lat0)
    return (111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi),
            111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi))


def static_from_folder(path: Path) -> FeatureGrid:
    """The fire's metre-grid fuel/slope as a lon/lat grid (equirectangular around the fire centre)."""
    with np.load(path, allow_pickle=False) as z:
        xs, ys = z["xs"].astype(float), z["ys"].astype(float)
        lon0, lat0 = float(z["lon0"]), float(z["lat0"])
        fuel, slope = z["fuel_fbfm40"], z["slope_deg"]
        source = str(z["source"]) if "source" in z.files else "fire folder static.npz"
    m_lat, m_lon = _metres_per_degree(lat0)
    lat = lat0 + ys / m_lat
    lon = lon0 + xs / m_lon
    return FeatureGrid(lat=lat, lon=lon, channels={"fuel_fbfm40": fuel, "slope_deg": slope}, source=source)


class ReplayEnvironment(NativeEnvironment):
    """NativeEnvironment fed from the folder's weather.csv / fm100.csv instead of Open-Meteo / gridMET."""

    def __init__(self, src: "ReplaySource", lon0: float, lat0: float, ref: dt.datetime, hours: float):
        self.notes = []
        self._static = src.static
        self.ref = ref
        self.weather = src.weather
        self._emc = np.clip(emc_percent(src.weather.temp_c, src.weather.rh), 2.0, 35.0)
        self._hours = []
        for i in range(max(int(math.ceil(hours)), 1)):
            k = src.weather.at(ref + dt.timedelta(hours=i + 0.5))
            toward = np.deg2rad(src.weather.wind_from[k]) + np.pi
            self._hours.append((float(src.weather.wind_speed[k]), float(np.sin(toward)), float(np.cos(toward))))
        days = [d for d in sorted(src.fm100) if d <= ref.date()]
        self._fm100 = src.fm100[days[-1]] if days else float(np.median(list(src.fm100.values()) or [10.0]))
        k0 = src.weather.at(ref)
        self.notes.append(f"replay: {src.folder.name} weather at reference time: wind {src.weather.wind_speed[k0]:.1f} m/s "
                          f"from {src.weather.wind_from[k0]:.0f} deg, FM100 {self._fm100:.1f} %")


class ReplaySource:
    name = "replay"

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.meta = json.loads((self.folder / "meta.json").read_text()) if (self.folder / "meta.json").exists() else {}
        self.static = static_from_folder(self.folder / "static.npz")
        self._load_detections()
        self._load_weather()
        self.fm100 = {}
        with open(self.folder / "fm100.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    self.fm100[dt.date.fromisoformat(row["date"])] = float(row["fm100_pct"])
                except (KeyError, ValueError):
                    continue

    def _load_detections(self) -> None:
        lon, lat, radius, conf, times = [], [], [], [], []
        seen = set()
        with open(self.folder / "detections.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("instrument", "VIIRS")).upper() != "VIIRS":
                    continue
                c = {"l": None, "n": 0.6, "h": 0.9}.get(str(row.get("confidence", "")).strip().lower()[:1])
                if c is None:
                    continue
                try:
                    when = _utc(row["time_utc"])
                    x, y = float(row["longitude"]), float(row["latitude"])
                except (KeyError, ValueError):
                    continue
                key = (round(x, 4), round(y, 4), when)
                if key in seen:
                    continue
                seen.add(key)
                scan, track = float(row.get("scan") or 0.375), float(row.get("track") or 0.375)
                lon.append(x)
                lat.append(y)
                radius.append(math.sqrt(scan * track / math.pi) * 1000.0)
                conf.append(c)
                times.append(when.timestamp())
        order = np.argsort(times)
        self.all = Detections(np.array(lon)[order], np.array(lat)[order], np.array(radius)[order],
                              np.array(conf)[order], "firms", [], np.array(times, dtype=float)[order])

    def _load_weather(self) -> None:
        t, ws, wd, rh, tc = [], [], [], [], []
        with open(self.folder / "weather.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                t.append(_utc(row["time_utc"]))
                ws.append(float(row["wind_speed_m_s"]))
                wd.append(float(row["wind_from_deg"]))
                rh.append(float(row["rh_pct"]))
                tc.append(float(row["temp_c"]))
        self.weather = HourlyWeather(t, np.array(ws), np.array(wd), np.array(rh), np.array(tc), "replay weather.csv")

    # ---- the NativeSource interface
    def window(self, search_bbox: list[float], start_s: float, end_s: float) -> Detections:
        d = self.all
        keep = ((d.time_s >= start_s - 1.0) & (d.time_s <= end_s + 1.0) & (d.lon >= search_bbox[0])
                & (d.lon <= search_bbox[2]) & (d.lat >= search_bbox[1]) & (d.lat <= search_bbox[3]))
        out = d.subset(keep)
        out.notes = [f"replay {self.folder.name}: {int(keep.sum())} VIIRS detections "
                     f"{dt.datetime.fromtimestamp(start_s, dt.timezone.utc):%Y-%m-%dT%H:%MZ} to "
                     f"{dt.datetime.fromtimestamp(end_s, dt.timezone.utc):%Y-%m-%dT%H:%MZ}"]
        return out

    def detections(self, req: PredictionRequest, search_bbox: list[float]) -> Detections:
        p = req.additional_params
        hours = max(p.lookback_hours, p.history_hours) if p.method in ("calibrated", "nowcast") else p.lookback_hours
        ref = req.reference_time.timestamp()
        return self.window(search_bbox, ref - hours * 3600.0, ref)

    def environment(self, req: PredictionRequest, lon0: float, lat0: float) -> ReplayEnvironment:
        return ReplayEnvironment(self, lon0, lat0, req.reference_time, max(req.horizons))

    def fuel(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return self.static.sample("fuel_fbfm40", lon, lat, fill=0.0)

    def describe(self) -> dict:
        return {"ingestion": f"replay ({self.folder.name})", "static_layers": "fire folder static.npz"}

    def bbox(self, pad_deg: float = 0.02) -> list[float]:
        g = self.static
        return [float(g.lon.min()) + pad_deg, float(g.lat.min()) + pad_deg,
                float(g.lon.max()) - pad_deg, float(g.lat.max()) - pad_deg]
