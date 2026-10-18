"""Script 2 (stage A): per-fire data for real-fire evaluation.

    python -m pinn_v2.fetch_fire_data --fires 20191023_kincade 20200927_glass
    python -m pinn_v2.fetch_fire_data --split validation --region bay_area      # every Bay Area validation fire
    python -m pinn_v2.fetch_fire_data --split train --min-detections 500 --limit 30
    python -m pinn_v2.fetch_fire_data --split test ...  # only at the very end: the test set is scored once

Reads data/fire_catalogue.csv (Script 1) and data/firms_archive/*.csv, and writes data/fires/<fire_id>/:
  detections.csv   every FIRMS detection near the fire (VIIRS S-NPP, NOAA-20, MODIS), with UTC times
  arrival.npz      observed ARRIVAL TIME map on a 90 m grid: hours since the window start at which each
                   cell was first covered by a VIIRS 375 m detection (NaN = never seen burning).
                   The burned area at any time t is simply (arrival <= t): this is the "truth" the
                   forecasts are scored against, and an eikonal-ready target.
  passes.csv       VIIRS overpasses that saw the fire: time, satellite, detections, burned km2 so far
  weather.csv      hourly 10 m wind speed/direction, RH, temperature at the fire centre (Open-Meteo ERA5 archive)
  fm100.csv        daily gridMET 100-h fuel moisture at the fire centre
  meta.json        fire info, grid, counts, sources, warnings
Fuel/slope from the LANDFIRE version before each fire, and GOES, come in stage B / C.
Skips fires whose folder is already complete (use --force to redo). Needs numpy, pandas, scipy.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

from pinn_service import native_ingestion as ni
from pinn_service.forecaster import metres_per_degree

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RES_M = 90.0
PAD_DEG = 0.05            # look this far outside the final perimeter's box for detections
MARGIN_M = 3000.0         # grid margin around the detections
MAX_DAYS = 30             # cap very long "containment" windows; growth is early
VIIRS_RADIUS_M = 212.0    # equal-area disk of a 375 m pixel
SAT = {"N": "S-NPP", "1": "NOAA-20", "N20": "NOAA-20", "2": "NOAA-21", "N21": "NOAA-21", "Terra": "Terra", "Aqua": "Aqua", "T": "Terra", "A": "Aqua"}


# --------------------------------------------------------------------------- FIRMS archive

def load_archive(archive_dir: Path) -> pd.DataFrame:
    files = sorted(archive_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no FIRMS CSVs in {archive_dir}")
    keep = {"latitude", "longitude", "acq_date", "acq_time", "satellite", "instrument", "confidence", "frp", "daynight", "scan", "track"}
    parts = []
    for f in files:
        part = pd.read_csv(f, usecols=lambda c: c in keep, dtype={"acq_time": str, "satellite": str, "confidence": str})
        part["file"] = f.name
        parts.append(part)
        print(f"  archive {f.name}: {len(part):,} detections")
    det = pd.concat(parts, ignore_index=True)
    hhmm = det["acq_time"].str.zfill(4)
    det["time_utc"] = pd.to_datetime(det["acq_date"] + " " + hhmm.str[:2] + ":" + hhmm.str[2:], utc=True)
    det["satellite"] = det["satellite"].map(lambda s: SAT.get(str(s), str(s)))
    det["viirs"] = det["file"].str.contains("V-C2", regex=False)     # SV-C2 = S-NPP, J1V-C2 = NOAA-20, J2V-C2 = NOAA-21
    return det


# --------------------------------------------------------------------------- arrival-time map

def arrival_map(viirs: pd.DataFrame, t0: pd.Timestamp, lon0: float, lat0: float):
    """Earliest VIIRS overpass (hours after t0) covering each 90 m cell."""
    m_lat, m_lon = metres_per_degree(lat0)
    x = (viirs.longitude.to_numpy() - lon0) * m_lon
    y = (viirs.latitude.to_numpy() - lat0) * m_lat
    xmin, xmax = x.min() - MARGIN_M, x.max() + MARGIN_M
    ymin, ymax = y.min() - MARGIN_M, y.max() + MARGIN_M
    xs = np.arange(xmin, xmax + RES_M / 2, RES_M)
    ys = np.arange(ymin, ymax + RES_M / 2, RES_M)
    arrival = np.full((len(ys), len(xs)), np.inf, dtype=np.float64)
    r = int(np.ceil(VIIRS_RADIUS_M / RES_M))
    di, dj = np.mgrid[-r:r + 1, -r:r + 1]
    inside = np.hypot(di, dj) * RES_M <= VIIRS_RADIUS_M + RES_M / 2
    di, dj = di[inside], dj[inside]
    hours = ((viirs.time_utc - t0).dt.total_seconds() / 3600.0).to_numpy()
    ci = np.rint((y - ymin) / RES_M).astype(int)
    cj = np.rint((x - xmin) / RES_M).astype(int)
    rows = np.clip(ci[:, None] + di[None, :], 0, len(ys) - 1).ravel()
    cols = np.clip(cj[:, None] + dj[None, :], 0, len(xs) - 1).ravel()
    np.minimum.at(arrival, (rows, cols), np.repeat(hours, len(di)))
    arrival[np.isinf(arrival)] = np.nan
    return xs, ys, arrival.astype(np.float32)


def overpasses(viirs: pd.DataFrame, arrival: np.ndarray, t0: pd.Timestamp) -> pd.DataFrame:
    """One row per satellite pass (same satellite, detections within 20 min)."""
    rows = []
    for sat, grp in viirs.sort_values("time_utc").groupby("satellite"):
        t = grp.time_utc.to_numpy()
        breaks = np.where(np.diff(t).astype("timedelta64[m]").astype(int) > 20)[0] + 1
        for chunk in np.split(np.arange(len(grp)), breaks):
            g = grp.iloc[chunk]
            h = (g.time_utc.min() - t0).total_seconds() / 3600.0
            rows.append({"time_utc": g.time_utc.min().isoformat(), "hours": round(h, 3), "satellite": sat,
                         "detections": len(g), "daynight": g.daynight.mode().iat[0] if "daynight" in g else "",
                         "burned_km2_so_far": round(float(np.nansum(arrival <= h + 0.34)) * RES_M ** 2 / 1e6, 3)})
    return pd.DataFrame(rows).sort_values("hours").reset_index(drop=True)


# --------------------------------------------------------------------------- weather

def weather_table(lon: float, lat: float, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    w = ni.open_meteo_hourly(lon, lat, start, end)
    return pd.DataFrame({"time_utc": [t.isoformat() for t in w.times], "wind_speed_m_s": w.wind_speed,
                         "wind_from_deg": w.wind_from, "rh_pct": w.rh, "temp_c": w.temp_c, "source": w.source})


def fm100_table(lon: float, lat: float, first: dt.date, last: dt.date) -> pd.DataFrame:
    rows = []
    for year in range(first.year, last.year + 1):
        a, b = max(first, dt.date(year, 1, 1)), min(last, dt.date(year, 12, 31))
        query = urllib.parse.urlencode({
            "var": "dead_fuel_moisture_100hr", "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "time_start": f"{a.isoformat()}T00:00:00Z", "time_end": f"{b.isoformat()}T00:00:00Z", "accept": "csv"})
        text = ni.fetch_cached(f"{ni.GRIDMET_FM100.format(year=year)}?{query}").decode("utf-8", "replace")
        table = [r for r in csv.reader(io.StringIO(text)) if r]
        if len(table) < 2:
            continue
        header = [h.lower() for h in table[0]]
        col = next((i for i, h in enumerate(header) if "moisture" in h or "fm100" in h), len(header) - 1)
        tcol = next((i for i, h in enumerate(header) if "time" in h or "date" in h), 0)
        for row in table[1:]:
            try:
                value = ni.unpack_fm100(float(row[col]))
            except (ValueError, IndexError):
                continue
            if value is not None:
                rows.append({"date": row[tcol][:10], "fm100_pct": round(value, 2)})
    return pd.DataFrame(rows, columns=["date", "fm100_pct"])


# --------------------------------------------------------------------------- one fire

def fetch_fire(fire: pd.Series, det: pd.DataFrame, out_root: Path, force: bool = False) -> dict:
    out = out_root / fire.fire_id
    if (out / "meta.json").exists() and not force:
        return {"fire_id": fire.fire_id, "status": "already done"}
    out.mkdir(parents=True, exist_ok=True)
    warnings = []
    alarm = pd.Timestamp(fire.alarm_date, tz="UTC")
    cont = pd.Timestamp(fire.cont_date, tz="UTC") if isinstance(fire.cont_date, str) and fire.cont_date else alarm + pd.Timedelta(days=MAX_DAYS)
    t_start = alarm - pd.Timedelta(days=1)
    t_end = min(cont, alarm + pd.Timedelta(days=MAX_DAYS)) + pd.Timedelta(days=2)

    sel = det[(det.longitude.between(fire.min_lon - PAD_DEG, fire.max_lon + PAD_DEG))
              & (det.latitude.between(fire.min_lat - PAD_DEG, fire.max_lat + PAD_DEG))
              & (det.time_utc >= t_start) & (det.time_utc <= t_end)].copy()
    if not len(sel):
        return {"fire_id": fire.fire_id, "status": "no detections"}
    last_seen = sel.time_utc.max()
    t_end = min(t_end, last_seen + pd.Timedelta(days=1))
    sel.drop(columns=["viirs"]).to_csv(out / "detections.csv", index=False)

    viirs = sel[sel.viirs]
    if len(viirs):
        xs, ys, arrival = arrival_map(viirs, t_start, fire.lon, fire.lat)
        np.savez_compressed(out / "arrival.npz", arrival_hours=arrival, xs=xs, ys=ys, res_m=RES_M,
                            lon0=fire.lon, lat0=fire.lat, t0_utc=t_start.isoformat())
        passes = overpasses(viirs, arrival, t_start)
        passes.to_csv(out / "passes.csv", index=False)
    else:
        warnings.append("no VIIRS detections (MODIS only): no arrival map")
        passes, arrival = pd.DataFrame(), np.empty((0, 0))

    start_dt, end_dt = t_start.to_pydatetime(), t_end.to_pydatetime()
    try:
        weather_table(fire.lon, fire.lat, start_dt, end_dt).to_csv(out / "weather.csv", index=False)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"weather failed: {exc}")
    try:
        fm = fm100_table(fire.lon, fire.lat, start_dt.date(), end_dt.date())
        fm.to_csv(out / "fm100.csv", index=False)
        if not len(fm):
            warnings.append("gridMET returned no fuel moisture")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"fm100 failed: {exc}")

    meta = {
        "fire_id": fire.fire_id, "name": fire["name"], "split": fire.split, "region": fire.region,
        "alarm_date": str(fire.alarm_date), "cont_date": str(fire.cont_date), "acres_final": float(fire.acres),
        "centre_lon_lat": [float(fire.lon), float(fire.lat)],
        "window_utc": [t_start.isoformat(), t_end.isoformat()],
        "detections": {k: int(v) for k, v in sel.groupby("satellite").size().items()},
        "viirs_passes": int(len(passes)),
        "arrival_grid": {"shape": list(arrival.shape), "res_m": RES_M},
        "burned_km2_viirs": round(float(np.isfinite(arrival).sum()) * RES_M ** 2 / 1e6, 2) if arrival.size else 0.0,
        "final_perimeter_km2": round(float(fire.acres) * 0.004046856, 2),
        "warnings": warnings,
        "sources": {"detections": "NASA FIRMS archive (VIIRS 375 m S-NPP/NOAA-20, MODIS 1 km)",
                    "weather": "Open-Meteo historical archive (ERA5-based), hourly, fire centre",
                    "fm100": "gridMET 100-h dead fuel moisture, daily, fire centre"},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"fire_id": fire.fire_id, "status": "ok" if not warnings else "ok, warnings: " + "; ".join(warnings),
            "passes": meta["viirs_passes"], "viirs_km2": meta["burned_km2_viirs"], "final_km2": meta["final_perimeter_km2"]}


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch per-fire data (stage A)")
    p.add_argument("--fires", nargs="*", help="fire_ids from data/fire_catalogue.csv")
    p.add_argument("--split", choices=["train", "validation", "test"])
    p.add_argument("--region", choices=["bay_area", "norcal"])
    p.add_argument("--min-detections", type=int, default=100, help="skip fires the satellites barely saw")
    p.add_argument("--limit", type=int, default=None, help="at most this many fires (largest first)")
    p.add_argument("--force", action="store_true", help="redo fires that are already done")
    a = p.parse_args()

    cat = pd.read_csv(DATA / "fire_catalogue.csv", dtype={"cont_date": str, "alarm_date": str})
    if a.fires:
        chosen = cat[cat.fire_id.isin(a.fires)]
        missing = set(a.fires) - set(chosen.fire_id)
        if missing:
            print(f"not in the catalogue: {sorted(missing)}")
    else:
        if not a.split:
            raise SystemExit("pass --fires ... or --split (and optionally --region)")
        chosen = cat[cat.split == a.split]
        if a.region:
            chosen = chosen[chosen.region == a.region]
        if "viirs_detections" in chosen:
            chosen = chosen[chosen.viirs_detections >= a.min_detections]
        chosen = chosen.sort_values("acres", ascending=False)
        if a.limit:
            chosen = chosen.head(a.limit)
    if a.split == "test" or (len(chosen) and (chosen.split == "test").any()):
        print("NOTE: includes TEST fires - fetching is fine, but score them only once, at the very end.")
    print(f"{len(chosen)} fires selected")
    if not len(chosen):
        return

    det = load_archive(DATA / "firms_archive")
    results, started = [], time.perf_counter()
    for i, (_, fire) in enumerate(chosen.iterrows(), 1):
        print(f"[{i}/{len(chosen)}] {fire.fire_id} ({fire['name']}, {fire.acres:,.0f} acres) ...", flush=True)
        try:
            res = fetch_fire(fire, det, DATA / "fires", a.force)
        except Exception as exc:  # noqa: BLE001
            res = {"fire_id": fire.fire_id, "status": f"FAILED: {exc}"}
        print(f"    {res}", flush=True)
        results.append(res)
    pd.DataFrame(results).to_csv(DATA / "fires" / "fetch_log.csv", index=False)
    ok = sum(str(r["status"]).startswith("ok") or r["status"] == "already done" for r in results)
    print(f"\n{ok}/{len(results)} fires ready in {DATA / 'fires'} ({time.perf_counter() - started:.0f} s); log: fetch_log.csv")


if __name__ == "__main__":
    main()