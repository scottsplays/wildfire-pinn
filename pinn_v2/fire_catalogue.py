"""Script 1: candidate fires for real-fire evaluation, from the CAL FIRE (FRAP) perimeter GeoJSON.

    python -m pinn_v2.fire_catalogue                       # reads data/perimeters/frap/*.geojson
    python -m pinn_v2.fire_catalogue --min-acres 500       # include smaller fires
    python -m pinn_v2.fire_catalogue --geojson path\\to\\file.geojson

Keeps Northern California fires since VIIRS began (2012-01-20) above a size threshold and
writes data/fire_catalogue.csv with one row per fire and a proposed split BY FIRE:
  test        half of the Bay Area fires (alternating by date): never used for training or tuning
  validation  the other Bay Area half + every 6th other fire: for choosing settings
  train       everything else
If the FIRMS archive is in data/firms_archive/ (*.csv), it also counts VIIRS detections per fire,
so fires that satellites barely saw can be dropped. Needs only numpy and pandas.
"""

from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
VIIRS_START = dt.date(2012, 1, 20)
NORCAL_MIN_LAT = 36.0                               # roughly Monterey / Fresno and north
BAY_AREA = (-123.6, 36.9, -121.2, 38.9)             # Bay Area + surrounding hills (lon/lat box)
FIELDS = {                                          # FRAP column names vary a little between exports
    "name": ["FIRE_NAME", "fire_name", "FireName", "INCIDENT"],
    "alarm": ["ALARM_DATE", "alarm_date", "AlarmDate"],
    "cont": ["CONT_DATE", "cont_date", "ContDate"],
    "acres": ["GIS_ACRES", "gis_acres", "GISACRES", "ACRES"],
    "year": ["YEAR_", "YEAR", "year", "Year"],
    "cause": ["CAUSE", "cause", "Cause"],
    "inc": ["INC_NUM", "inc_num", "IRWINID", "OBJECTID"],
}


def pick(props: dict, key: str):
    for name in FIELDS[key]:
        if name in props and props[name] not in (None, ""):
            return props[name]
    return None


def to_date(value) -> dt.date | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):               # ArcGIS exports: epoch milliseconds
        return dt.datetime.fromtimestamp(value / 1000, dt.timezone.utc).date()
    text = str(value).strip()
    if text[:3].isalpha() and "," in text[:5]:        # "Fri, 01 Aug 2025 00:00:00 GMT" (CAL FIRE GeoJSON)
        return email.utils.parsedate_to_datetime(text).date()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def ring_points(geometry: dict) -> np.ndarray:
    coords = geometry.get("coordinates", [])
    if geometry.get("type") == "Polygon":
        rings = coords
    elif geometry.get("type") == "MultiPolygon":
        rings = [ring for poly in coords for ring in poly]
    else:
        return np.empty((0, 2))
    pts = [np.asarray(r, dtype=float)[:, :2] for r in rings if len(r)]
    return np.vstack(pts) if pts else np.empty((0, 2))


# ---- projected exports: CAL FIRE publishes in California Albers (EPSG:3310) or Web Mercator (EPSG:3857)

_A, _E2 = 6378137.0, 0.00669438002290            # GRS80


def _albers_consts():
    e = np.sqrt(_E2)
    q = lambda p: (1 - _E2) * (np.sin(p) / (1 - _E2 * np.sin(p) ** 2) - np.log((1 - e * np.sin(p)) / (1 + e * np.sin(p))) / (2 * e))
    m = lambda p: np.cos(p) / np.sqrt(1 - _E2 * np.sin(p) ** 2)
    p1, p2, p0 = np.radians(34.0), np.radians(40.5), 0.0
    n = (m(p1) ** 2 - m(p2) ** 2) / (q(p2) - q(p1))
    C = m(p1) ** 2 + n * q(p1)
    rho0 = _A * np.sqrt(C - n * q(p0)) / n
    return e, q, n, C, rho0


def albers3310_inverse(x, y):
    e, q, n, C, rho0 = _albers_consts()
    x, y = np.asarray(x, float), np.asarray(y, float) + 4000000.0
    rho = np.hypot(x, rho0 - y)
    th = np.arctan2(x, rho0 - y)
    qv = (C - (rho * n / _A) ** 2) / n
    phi = np.arcsin(np.clip(qv / 2, -1, 1))
    for _ in range(10):
        s = np.sin(phi)
        phi = phi + (1 - _E2 * s * s) ** 2 / (2 * np.cos(phi)) * (
            qv / (1 - _E2) - s / (1 - _E2 * s * s) + np.log((1 - e * s) / (1 + e * s)) / (2 * e))
    return np.degrees(th / n) - 120.0, np.degrees(phi)


def mercator3857_inverse(x, y):
    return np.degrees(np.asarray(x) / _A), np.degrees(2 * np.arctan(np.exp(np.asarray(y) / _A)) - np.pi / 2)


def to_lonlat(x, y, epsg: int):
    try:
        from pyproj import Transformer

        return Transformer.from_crs(epsg, 4326, always_xy=True).transform(x, y)
    except ImportError:
        if epsg == 3310:
            return albers3310_inverse(x, y)
        if epsg == 3857:
            return mercator3857_inverse(x, y)
        raise SystemExit(f"GeoJSON is in EPSG:{epsg}; pip install pyproj to convert it")


def guess_epsg(data: dict, sample_x: float) -> int:
    name = str((data.get("crs") or {}).get("properties", {}).get("name", ""))
    digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
    if digits and digits[-1] not in ("4326", "84"):
        return int(digits[-1])
    return 3857 if abs(sample_x) > 2.0e6 else 3310   # Web Mercator x ~ -1.3e7 in California; Albers ~ +-5e5


def load_frap(path: Path) -> pd.DataFrame:
    print(f"reading {path} ({path.stat().st_size / 1e6:.0f} MB) ...")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    feats = data.get("features", [])
    if feats:
        print("  columns:", ", ".join(list(feats[0].get("properties", {}))[:30]))
    rows, epsg = [], None
    for f in feats:
        props, geom = f.get("properties") or {}, f.get("geometry") or {}
        pts = ring_points(geom)
        if not len(pts):
            continue
        if abs(pts[:, 0]).max() > 180 or abs(pts[:, 1]).max() > 90:   # projected metres -> lon/lat
            if epsg is None:
                epsg = guess_epsg(data, float(pts[0, 0]))
                print(f"  coordinates are projected (EPSG:{epsg}); converting to lon/lat")
            lon, lat = to_lonlat(pts[:, 0], pts[:, 1], epsg)
            pts = np.column_stack([lon, lat])
        alarm = to_date(pick(props, "alarm"))
        cont = to_date(pick(props, "cont"))
        rows.append({
            "name": str(pick(props, "name") or "UNNAMED").strip().title(),
            "inc_num": pick(props, "inc"),
            "year": pick(props, "year"),
            "alarm_date": alarm, "cont_date": cont,
            "acres": float(pick(props, "acres") or 0.0),
            "cause": pick(props, "cause"),
            "complex_name": props.get("COMPLEX_NAME") or "",
            "lon": float((pts[:, 0].min() + pts[:, 0].max()) / 2), "lat": float((pts[:, 1].min() + pts[:, 1].max()) / 2),
            "min_lon": float(pts[:, 0].min()), "min_lat": float(pts[:, 1].min()),
            "max_lon": float(pts[:, 0].max()), "max_lat": float(pts[:, 1].max()),
        })
    skipped = len(feats) - len(rows)
    if skipped:
        print(f"  skipped {skipped} features without lon/lat polygons")
    if feats and not rows:
        raise SystemExit("no usable polygons: is the GeoJSON in lon/lat (EPSG:4326)? Re-download as GeoJSON.")
    return pd.DataFrame(rows)


def count_firms(cat: pd.DataFrame, archive_dir: Path, pad_deg: float = 0.02, pad_days: int = 2) -> pd.Series | None:
    files = sorted(archive_dir.glob("*.csv"))
    if not files:
        return None
    parts = []
    for f in files:
        print(f"  FIRMS archive: {f.name}")
        parts.append(pd.read_csv(f, usecols=lambda c: c in {"latitude", "longitude", "acq_date"}))
    det = pd.concat(parts, ignore_index=True)
    det["acq_date"] = pd.to_datetime(det["acq_date"]).dt.date
    counts = []
    for r in cat.itertuples():
        end = r.cont_date or (r.alarm_date + dt.timedelta(days=30))
        sel = ((det.longitude >= r.min_lon - pad_deg) & (det.longitude <= r.max_lon + pad_deg)
               & (det.latitude >= r.min_lat - pad_deg) & (det.latitude <= r.max_lat + pad_deg)
               & (det.acq_date >= r.alarm_date - dt.timedelta(days=pad_days)) & (det.acq_date <= end + dt.timedelta(days=pad_days)))
        counts.append(int(sel.sum()))
    return pd.Series(counts, index=cat.index)


USED_BEFORE = {"kincade"}            # already looked at (persistence comparison): never in the test set


def assign_events(cat: pd.DataFrame, days: int = 3, km: float = 150.0) -> list[int]:
    """Fires in the same complex, or starting within `days` and `km` of each other, are one event
    (same wind / humidity / dryness). Single-linkage, so chains of nearby fires join too."""
    n = len(cat)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lat, lon = cat.lat.to_numpy(), cat.lon.to_numpy()
    dates = cat.alarm_date.to_numpy()
    comp = cat.complex_name.fillna("").str.strip().str.upper().to_numpy()
    for i in range(n):
        for j in range(i + 1, n):
            gap = (dates[j] - dates[i]).days
            if gap > days and not comp[i]:
                break                                   # sorted by date
            same_complex = comp[i] and comp[i] == comp[j]
            dist = 111.0 * np.hypot(lat[i] - lat[j], (lon[i] - lon[j]) * np.cos(np.radians(lat[i])))
            if same_complex or (abs(gap) <= days and dist <= km):
                parent[find(j)] = find(i)
    roots = [find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return [ids[r] for r in roots]


def split_by_event(cat: pd.DataFrame) -> list[str]:
    """Whole events go to one split. Bay Area events alternate test / validation by date; any event
    with a fire we have already used goes to validation; other events: every 6th -> validation."""
    split_of = {}
    bay_i = other_i = 0
    for ev, grp in cat.groupby("event", sort=False):
        names = {"".join(ch for ch in nm.lower() if ch.isalnum()) for nm in grp.name}
        if (grp.region == "bay_area").any():
            if names & USED_BEFORE:
                split_of[ev] = "validation"
            else:
                split_of[ev] = "test" if bay_i % 2 == 0 else "validation"
                bay_i += 1
        else:
            split_of[ev] = "validation" if other_i % 6 == 5 else "train"
            other_i += 1
    return [split_of[e] for e in cat.event]

def main() -> None:
    p = argparse.ArgumentParser(description="Build the candidate fire catalogue")
    p.add_argument("--geojson", default=None)
    p.add_argument("--min-acres", type=float, default=1000.0)
    p.add_argument("--out", default=str(REPO / "data" / "fire_catalogue.csv"))
    a = p.parse_args()

    if a.geojson:
        path = Path(a.geojson)
    else:
        found = sorted((REPO / "data" / "perimeters" / "frap").glob("*.geojson"))
        if not found:
            raise SystemExit("no .geojson in data/perimeters/frap/ - move the CAL FIRE download there or pass --geojson")
        path = found[0]
    frap = load_frap(path)
    print(f"  {len(frap)} perimeters in the file")

    cat = frap[(frap.alarm_date.notna()) & (frap.alarm_date >= VIIRS_START)
               & (frap.acres >= a.min_acres) & (frap.lat >= NORCAL_MIN_LAT)].copy()
    cat["duration_days"] = [((c - s).days + 1) if c and s and c >= s else None for s, c in zip(cat.alarm_date, cat.cont_date)]
    w, s, e, n = BAY_AREA
    cat["region"] = np.where((cat.lon >= w) & (cat.lon <= e) & (cat.lat >= s) & (cat.lat <= n), "bay_area", "norcal")
    cat = cat.sort_values("alarm_date").reset_index(drop=True)
    cat.insert(0, "fire_id", [f"{d:%Y%m%d}_{''.join(ch for ch in nm if ch.isalnum())[:20].lower() or 'fire'}" for d, nm in zip(cat.alarm_date, cat.name)])

    cat["event"] = assign_events(cat)
    cat["split"] = split_by_event(cat)

    counts = count_firms(cat, REPO / "data" / "firms_archive")
    if counts is not None:
        cat["viirs_detections"] = counts
    else:
        print("  (no FIRMS archive in data/firms_archive/ yet: detection counts skipped)")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(out, index=False)

    print(f"\n{len(cat)} Northern California fires since {VIIRS_START}, >= {a.min_acres:g} acres -> {out}")
    print(cat.groupby(["split", "region"]).size().unstack(fill_value=0).to_string())
    print(f"\nby year: {cat.groupby(cat.alarm_date.map(lambda d: d.year)).size().to_dict()}")
    cols = ["fire_id", "name", "alarm_date", "acres", "region", "split"] + (["viirs_detections"] if counts is not None else [])
    print("\nBay Area fires:")
    print(cat[cat.region == "bay_area"][cols].to_string(index=False))
    print("\n15 largest other fires:")
    print(cat[cat.region != "bay_area"].nlargest(15, "acres")[cols].to_string(index=False))


if __name__ == "__main__":
    main()