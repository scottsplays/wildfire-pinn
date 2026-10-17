"""Write SYNTHETIC daily feature files for local integration tests.

    python pinn_service/make_test_features.py

Writes to pinn_service/test_features/:
  2026-10-16.npz      replay file: one synthetic fire in the Santa Cruz Mountains
                      (near the 2020 CZU footprint), NW wind 7 m/s, dry fuels
  2026-10-17.npz      replay file: no fire anywhere in the box (the common case)
  static_synthetic.npz  synthetic fuel + slope in the static-layer format, so the
                      native-ingestion path can be tested without LANDFIRE

Every value is made up. These files exercise the code path and the contract;
they say nothing about real fire behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BBOX = (-122.9, 36.9, -121.5, 38.3)
STEP = 0.0035  # degrees, ~375 m (VIIRS-like); the U-Net grid may differ, the service resamples


def build(fire: bool, seed: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    lat = np.arange(BBOX[1] + STEP / 2, BBOX[3], STEP)
    lon = np.arange(BBOX[0] + STEP / 2, BBOX[2], STEP)
    lon2, lat2 = np.meshgrid(lon, lat)

    # terrain: smooth synthetic hills, metres
    elev = (
        400 * np.sin(lon2 * 9.0) * np.cos(lat2 * 7.0)
        + 300 * np.exp(-((lon2 + 122.2) ** 2 + (lat2 - 37.2) ** 2) / 0.02)
        + 350
    ).clip(0)
    water = (lon2 < -122.45 + 0.5 * (lat2 - 37.0)) & (lat2 < 37.7)  # crude Pacific edge
    water |= (lon2 > -122.45) & (lon2 < -122.1) & (lat2 > 37.45) & (lat2 < 38.05)  # crude Bay

    # FBFM40 fuel: grass/shrub low, timber understory/litter on hills, 98 = water
    fuel = np.where(elev > 500, 165, np.where(elev > 250, 145, 102)).astype(np.int16)
    patches = rng.random(fuel.shape) < 0.08
    fuel = np.where(patches, 183, fuel)
    fuel = np.where(water, 98, fuel)

    fm100 = np.full(lat2.shape, 7.0) + rng.normal(0, 0.5, lat2.shape)
    wind_u = np.full(lat2.shape, 4.5)    # blowing toward the east ...
    wind_v = np.full(lat2.shape, -5.4)   # ... and south: a NW wind, ~7 m/s
    temp = np.full(lat2.shape, 31.0)
    rh = np.full(lat2.shape, 14.0)
    ndvi = np.clip(0.2 + elev / 2000 + rng.normal(0, 0.05, lat2.shape), -0.1, 0.9)

    active = np.zeros(lat2.shape, dtype=np.float32)
    if fire:
        centre = (-122.22, 37.17)
        d = np.hypot((lon2 - centre[0]) * 88.6, (lat2 - centre[1]) * 111.0)  # km
        active[(d < 1.6) & ~water] = 1.0
        active[(d >= 1.6) & (d < 2.0) & ~water & (rng.random(d.shape) < 0.5)] = 0.7

    return {
        "lat": lat.astype(np.float64),
        "lon": lon.astype(np.float64),
        "active_fire": active,
        "fuel_fbfm40": fuel,
        "elevation_m": elev.astype(np.float32),
        "fm100_pct": fm100.astype(np.float32),
        "wind_u": wind_u.astype(np.float32),
        "wind_v": wind_v.astype(np.float32),
        "temp_c": temp.astype(np.float32),
        "humidity_pct": rh.astype(np.float32),
        "ndvi": ndvi.astype(np.float32),
        "synthetic": np.array(True),
    }


def main() -> None:
    out = Path(__file__).resolve().parent / "test_features"
    out.mkdir(exist_ok=True)
    np.savez_compressed(out / "2026-10-16.npz", **build(fire=True))
    np.savez_compressed(out / "2026-10-17.npz", **build(fire=False))
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow running as a plain script
    from pinn_service.features import _slope_from_elevation

    day = build(fire=False)
    slope = _slope_from_elevation(day["elevation_m"].astype(float), day["lat"], day["lon"]).astype(np.float32)
    np.savez_compressed(out / "static_synthetic.npz", lat=day["lat"], lon=day["lon"],
                        fuel_fbfm40=day["fuel_fbfm40"], slope_deg=slope, synthetic=np.array(True))
    print(f"wrote synthetic feature files to {out}")


if __name__ == "__main__":
    main()
