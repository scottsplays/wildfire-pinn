"""Script 2 stage B: fuel + slope for each fire, from the LANDFIRE version BEFORE the fire.

    python -m pinn_v2.fetch_fuel                  # every fire folder in data/fires/ that has arrival.npz
    python -m pinn_v2.fetch_fuel --fires 20191023_kincade --force

Writes data/fires/<fire_id>/static.npz on exactly the same 90 m grid as arrival.npz:
  fuel_fbfm40 (int16, Scott & Burgan codes), slope_deg (float32), fuel_version, fuel_after_fire (bool)
Why the version matters: a fuel map made after the fire shows the burn scar as bare ground, which
quietly tells a model where the fire went. Versions on the LANDFIRE server: LF2016, LF2022-LF2025.
  fires 2017-2022 -> LF2016, 2023 -> LF2022, 2024 -> LF2023, 2025 -> LF2024
  fires 2012-2016 -> LF2016 too (nothing older is online): flagged fuel_after_fire = True
Slope: LANDFIRE LF2020 topography (terrain does not change). Needs rasterio (as build_static_layers).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pinn_service.build_static_layers import export
from pinn_service.forecaster import metres_per_degree

REPO = Path(__file__).resolve().parent.parent
FIRES = REPO / "data" / "fires"
SLOPE = "Landfire_Topo/LF2020_SlpD_CONUS"
MAX_PIXELS = 4000                     # ImageServer export limit per side


def fuel_service(fire_year: int) -> tuple[str, bool]:
    if fire_year >= 2025:
        v = 2024
    elif fire_year == 2024:
        v = 2023
    elif fire_year == 2023:
        v = 2022
    else:
        v = 2016
    return f"Landfire_LF{v}/LF{v}_FBFM40_CONUS", fire_year <= 2016


def on_grid(lon_src, lat_src, band, lon_c, lat_c):
    """Nearest-neighbour sample of a north-up raster at the fire grid's cell centres."""
    order = np.argsort(lat_src)
    lat_sorted, band_sorted = lat_src[order], band[order]
    i = np.clip(np.searchsorted(lat_sorted, lat_c), 1, len(lat_sorted) - 1)
    i -= (lat_c - lat_sorted[i - 1]) < (lat_sorted[i] - lat_c)
    j = np.clip(np.searchsorted(lon_src, lon_c), 1, len(lon_src) - 1)
    j -= (lon_c - lon_src[j - 1]) < (lon_src[j] - lon_c)
    return band_sorted[np.ix_(i, j)]


def fetch_one(folder: Path, force: bool = False) -> dict:
    out = folder / "static.npz"
    if out.exists() and not force:
        return {"fire_id": folder.name, "status": "already done"}
    arrival_file = folder / "arrival.npz"
    if not arrival_file.exists():
        return {"fire_id": folder.name, "status": "skipped: no arrival.npz"}
    meta = json.loads((folder / "meta.json").read_text())
    z = np.load(arrival_file)
    xs, ys, lon0, lat0 = z["xs"], z["ys"], float(z["lon0"]), float(z["lat0"])
    m_lat, m_lon = metres_per_degree(lat0)
    lon_c, lat_c = lon0 + xs / m_lon, lat0 + ys / m_lat
    dlon, dlat = (xs[1] - xs[0]) / m_lon, (ys[1] - ys[0]) / m_lat
    bbox = (lon_c[0] - dlon / 2, lat_c[0] - dlat / 2, lon_c[-1] + dlon / 2, lat_c[-1] + dlat / 2)
    width, height = min(len(xs), MAX_PIXELS), min(len(ys), MAX_PIXELS)

    year = int(meta["alarm_date"][:4])
    service, after_fire = fuel_service(year)
    lon_f, lat_f, fuel = export(service, bbox, width, height, fuel=True)
    lon_s, lat_s, slope = export(SLOPE, bbox, width, height, fuel=False)
    fuel = np.nan_to_num(on_grid(lon_f, lat_f, fuel, lon_c, lat_c), nan=0).astype(np.int16)
    slope = np.nan_to_num(on_grid(lon_s, lat_s, slope, lon_c, lat_c), nan=0).astype(np.float32)
    version = service.split("/")[0].replace("Landfire_", "")
    np.savez_compressed(out, fuel_fbfm40=fuel, slope_deg=slope, xs=xs, ys=ys, lon0=lon0, lat0=lat0,
                        fuel_version=np.array(version), fuel_after_fire=np.array(after_fire),
                        source=np.array(f"LANDFIRE {version} FBFM40 + LF2020 slope (degrees)"))
    nonburn = {91, 92, 93, 98, 99}
    burnable = float((~np.isin(fuel, list(nonburn)) & (fuel > 0)).mean())
    meta["fuel"] = {"version": version, "fuel_after_fire": after_fire, "burnable_fraction": round(burnable, 3)}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"fire_id": folder.name, "status": "ok", "version": version, "after_fire": after_fire,
            "burnable": round(burnable, 2), "grid": list(fuel.shape)}


def main() -> None:
    p = argparse.ArgumentParser(description="Pre-fire LANDFIRE fuel + slope per fire")
    p.add_argument("--fires", nargs="*")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    folders = [FIRES / f for f in a.fires] if a.fires else sorted(d for d in FIRES.iterdir() if d.is_dir())
    print(f"{len(folders)} fire folders")
    ok = 0
    for i, folder in enumerate(folders, 1):
        print(f"[{i}/{len(folders)}] {folder.name} ...", flush=True)
        try:
            res = fetch_one(folder, a.force)
        except Exception as exc:  # noqa: BLE001
            res = {"fire_id": folder.name, "status": f"FAILED: {exc}"}
        ok += res["status"] in ("ok", "already done")
        print(f"    {res}", flush=True)
    print(f"\n{ok}/{len(folders)} fires have static.npz")


if __name__ == "__main__":
    main()